from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.traits.options import Options

from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

if TYPE_CHECKING:
    from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")


class TripoEndpoint(StrEnum):
    """The Tripo task types this library exposes as nodes."""

    TEXT = "text_to_model"
    IMAGE = "image_to_model"
    MULTIVIEW = "multiview_to_model"


class TripoCapability(StrEnum):
    """An optional request field whose support varies by model version and by endpoint.

    ``TEXTURE`` covers the ``texture`` / ``pbr`` / ``texture_quality`` trio, which
    Tripo documents and accepts as a set rather than individually.
    """

    TEXTURE = "texture"
    GEOMETRY_QUALITY = "geometry_quality"
    TEXTURE_ALIGNMENT = "texture_alignment"
    NEGATIVE_PROMPT = "negative_prompt"


@dataclass(frozen=True)
class TripoModelVersion:
    """One Tripo model version: the wire value, how to describe it, and where it works.

    ``capabilities`` is what the model itself supports. The fields a node actually
    sends are that set intersected with the endpoint's own capabilities, because a
    capability needs both a model that implements it and an endpoint that has a
    parameter for it -- ``NEGATIVE_PROMPT`` is meaningless without a prompt, and
    ``TEXTURE_ALIGNMENT`` without a source image.
    """

    value: str
    label: str
    subtitle: str
    endpoints: frozenset[TripoEndpoint]
    capabilities: frozenset[TripoCapability]


_EVERY_ENDPOINT = frozenset(TripoEndpoint)

# Optional fields each endpoint exposes as a node parameter at all.
_ENDPOINT_CAPABILITIES: dict[TripoEndpoint, frozenset[TripoCapability]] = {
    TripoEndpoint.TEXT: frozenset(
        {TripoCapability.TEXTURE, TripoCapability.GEOMETRY_QUALITY, TripoCapability.NEGATIVE_PROMPT}
    ),
    TripoEndpoint.IMAGE: frozenset(
        {TripoCapability.TEXTURE, TripoCapability.GEOMETRY_QUALITY, TripoCapability.TEXTURE_ALIGNMENT}
    ),
    TripoEndpoint.MULTIVIEW: frozenset(
        {TripoCapability.TEXTURE, TripoCapability.GEOMETRY_QUALITY, TripoCapability.TEXTURE_ALIGNMENT}
    ),
}

_DOCS_URLS: dict[TripoEndpoint, str] = {
    TripoEndpoint.TEXT: "https://docs.tripo3d.ai/model-generation/text-to-model-p1-20260311.html",
    TripoEndpoint.IMAGE: "https://docs.tripo3d.ai/model-generation/image-to-model-p1-20260311.html",
    TripoEndpoint.MULTIVIEW: "https://docs.tripo3d.ai/model-generation/multiview-to-model-p1-20260311.html",
}

# The versions Tripo currently generates on, newest family first. `label` stays short
# because it is what the collapsed dropdown shows; the build date and the tradeoff go
# in `subtitle`, which the UI renders as a secondary line on the open row. P1 and the
# H3 (v3.x) line are separate model families rather than successive versions, so the
# labels name the family instead of leaving the dates to imply an ordering.
MODEL_VERSIONS: tuple[TripoModelVersion, ...] = (
    TripoModelVersion(
        value="P1-20260311",
        label="P1",
        subtitle="Premium; low-poly, stable topology (2026-03-11)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {TripoCapability.TEXTURE, TripoCapability.TEXTURE_ALIGNMENT, TripoCapability.NEGATIVE_PROMPT}
        ),
    ),
    TripoModelVersion(
        value="v3.1-20260211",
        label="H3 v3.1",
        subtitle="High detail, geometry control (2026-02-11)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {
                TripoCapability.TEXTURE,
                TripoCapability.GEOMETRY_QUALITY,
                TripoCapability.TEXTURE_ALIGNMENT,
                TripoCapability.NEGATIVE_PROMPT,
            }
        ),
    ),
    TripoModelVersion(
        value="v3.0-20250812",
        label="H3 v3.0",
        subtitle="High detail, geometry control (2025-08-12)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {
                TripoCapability.TEXTURE,
                TripoCapability.GEOMETRY_QUALITY,
                TripoCapability.TEXTURE_ALIGNMENT,
                TripoCapability.NEGATIVE_PROMPT,
            }
        ),
    ),
    TripoModelVersion(
        value="v2.5-20250123",
        label="H2 v2.5",
        subtitle="Standard quality (2025-01-23)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {TripoCapability.TEXTURE, TripoCapability.TEXTURE_ALIGNMENT, TripoCapability.NEGATIVE_PROMPT}
        ),
    ),
    TripoModelVersion(
        value="v1.4-20240625",
        label="v1.4",
        subtitle="Legacy; basic parameters only (2024-06-25)",
        # Rejected on multiview_to_model with Tripo code 2015 ("the version has been
        # deprecated"), though still generating on the text and image endpoints.
        endpoints=frozenset({TripoEndpoint.TEXT, TripoEndpoint.IMAGE}),
        capabilities=frozenset(),
    ),
)

DEFAULT_MODEL_VERSION = "v3.1-20260211"

# Versions Tripo has retired, mapped to the live version a stored value becomes.
# Tripo answers a retired version with code 2015 and no generation, so leaving one
# selectable only offers a guaranteed failure. Keyed per endpoint because retirement
# is per endpoint: v1.4 still generates on text and image.
_RETIRED_VERSIONS: dict[TripoEndpoint, dict[str, str]] = {
    TripoEndpoint.TEXT: {
        "Turbo-v1.0-20250506": DEFAULT_MODEL_VERSION,
        "v2.0-20240919": "v2.5-20250123",
    },
    TripoEndpoint.IMAGE: {
        "Turbo-v1.0-20250506": DEFAULT_MODEL_VERSION,
        "v2.0-20240919": "v2.5-20250123",
    },
    TripoEndpoint.MULTIVIEW: {
        "v2.0-20240919": "v2.5-20250123",
        "v1.4-20240625": DEFAULT_MODEL_VERSION,
    },
}


def versions_for(endpoint: TripoEndpoint) -> tuple[TripoModelVersion, ...]:
    """The live versions this endpoint generates on, in display order."""
    return tuple(version for version in MODEL_VERSIONS if endpoint in version.endpoints)


def default_version(endpoint: TripoEndpoint) -> str:
    """The version a fresh node selects.

    Shared across endpoints so two Tripo nodes in one workflow don't start on
    models from different families.
    """
    offered = {version.value for version in versions_for(endpoint)}
    if DEFAULT_MODEL_VERSION not in offered:
        msg = f"{DEFAULT_MODEL_VERSION} is not offered on {endpoint}"
        raise ValueError(msg)
    return DEFAULT_MODEL_VERSION


def version_choices(endpoint: TripoEndpoint) -> list[str]:
    """Values the ``model_version`` parameter accepts, live plus retired.

    ``Options`` rewrites an assigned value that is outside ``choices`` to
    ``choices[0]``, so a retired value has to be here for a saved workflow's stored
    value to survive long enough for ``migrate_version`` to translate it. Retired
    values are deliberately absent from ``dropdown_row_data``, which is what the UI
    offers.
    """
    return [version.value for version in versions_for(endpoint)] + list(_RETIRED_VERSIONS[endpoint])


def dropdown_row_data(endpoint: TripoEndpoint) -> list[dict[str, str]]:
    """Rich dropdown rows for the live versions, newest family first."""
    return [
        {"name": version.value, "label": version.label, "subtitle": version.subtitle}
        for version in versions_for(endpoint)
    ]


def supports(endpoint: TripoEndpoint, model_version: str, capability: TripoCapability) -> bool:
    """Whether this endpoint sends ``capability`` for ``model_version``.

    An unknown version supports nothing, so a version Tripo adds is inert until it
    is declared in ``MODEL_VERSIONS`` rather than silently inheriting fields the
    model may reject.
    """
    if capability not in _ENDPOINT_CAPABILITIES[endpoint]:
        return False
    for version in versions_for(endpoint):
        if version.value == model_version:
            return capability in version.capabilities
    return False


def migrate_version(endpoint: TripoEndpoint, value: Any) -> str | None:
    """The live version a retired ``value`` becomes, or None if it needs no migration."""
    if isinstance(value, str):
        return _RETIRED_VERSIONS[endpoint].get(value)
    return None


def badge_message(endpoint: TripoEndpoint) -> str:
    """The parameter badge describing every version this endpoint offers.

    Built from the version table so the badge cannot drift from the dropdown, or
    describe the same version differently on two nodes.
    """
    lines = [f"**{version.label}** (`{version.value}`): {version.subtitle}" for version in versions_for(endpoint)]
    lines.append(f"\n[Model docs]({_DOCS_URLS[endpoint]})")
    return "\n".join(lines)


def add_model_version_parameter(node: GriptapeProxyNode, endpoint: TripoEndpoint) -> ParameterString:
    """Add the shared ``model_version`` dropdown to a Tripo node.

    Installs the migration as a converter rather than in ``before_value_set``:
    ``set_parameter_value`` runs converters on every assignment path including
    workflow load, which passes ``skip_before_value_set=True`` and so is the one
    path a stored retired value would otherwise slip through untranslated.
    """
    parameter = ParameterString(
        name="model_version",
        default_value=default_version(endpoint),
        tooltip="Tripo model version. See badge for details on what each version supports.",
        allow_output=False,
        traits={Options(choices=version_choices(endpoint))},
        ui_options={"display_name": "Model Version"},
    )
    parameter.set_badge(variant="info", title="Model Versions", message=badge_message(endpoint))
    node.add_parameter(parameter)

    # Must follow add_parameter; rows pushed before the parameter is registered are dropped.
    parameter.update_ui_options(
        {
            "data": dropdown_row_data(endpoint),
            "dropdown_row_subtitles": True,
        }
    )
    parameter.add_converter(lambda value: migrate_version(endpoint, value) or value)

    migrated = migrate_version(endpoint, node.get_parameter_value(parameter.name))
    if migrated is not None:
        node.set_parameter_value(parameter.name, migrated, initial_setup=True)

    return parameter


def _extract_model_url(data: dict[str, Any]) -> str | None:
    """Find the best 3D model URL in a Tripo task payload's data block."""
    result = data.get("result") or {}
    if isinstance(result.get("pbr_model"), dict):
        url = result["pbr_model"].get("url")
        if url:
            return url
    if isinstance(result.get("model"), dict):
        url = result["model"].get("url")
        if url:
            return url

    output = data.get("output") or {}
    return output.get("pbr_model") or output.get("model") or output.get("base_model")


def _extract_preview_url(data: dict[str, Any]) -> str | None:
    """Find the best preview image URL in a Tripo task payload's data block."""
    result = data.get("result") or {}
    if isinstance(result.get("rendered_image"), dict):
        url = result["rendered_image"].get("url")
        if url:
            return url

    output = data.get("output") or {}
    return output.get("rendered_image") or output.get("generated_image")


async def parse_tripo_task_result(node: GriptapeProxyNode, result_json: dict[str, Any]) -> None:
    """Parse a completed Tripo task payload, saving the GLB and preview to project files.

    The proxy's `fetch_completed_generation` returns Tripo's raw task response:
        {"code": 0,
         "data": {"status": "success",
                  "output": {"pbr_model": "<signed URL>", "rendered_image": "<signed URL>"},
                  "result": {"pbr_model": {"url": "...", "type": "glb"}, ...},
                  "consumed_credit": 20}}

    Tripo's signed URLs expire within 5 minutes, so we download the bytes
    immediately and save them as project files rather than exposing the
    expiring URLs downstream.
    """
    data = result_json.get("data") if isinstance(result_json, dict) else None
    if not isinstance(data, dict):
        data = result_json if isinstance(result_json, dict) else {}

    model_url = _extract_model_url(data)
    if not model_url:
        node._set_safe_defaults()
        node._set_status_results(
            was_successful=False,
            result_details="Tripo task completed but no model URL was present in the response.",
        )
        return

    model_bytes = await node._download_bytes_from_url(model_url)
    if not model_bytes:
        node._set_safe_defaults()
        node._set_status_results(
            was_successful=False,
            result_details="Failed to download the generated 3D model from Tripo's signed URL.",
        )
        return

    output_file_value = node.get_parameter_value("output_file") or "tripo_model.glb"
    model_path = Path(output_file_value)
    if model_path.suffix.lower() != ".glb":
        model_path = model_path.with_suffix(".glb")
    model_dest = ProjectFileDestination.from_situation(
        filename=str(model_path),
        situation="save_node_output",
        node_name=node.name,
    )
    saved_model = await model_dest.awrite_bytes(model_bytes)
    node.parameter_output_values["model_url"] = ThreeDUrlArtifact(
        value=saved_model.location,
        meta={"filename": saved_model.name, "format": "glb"},
    )

    preview_url = _extract_preview_url(data)
    if preview_url:
        preview_bytes = await node._download_bytes_from_url(preview_url)
        if preview_bytes:
            preview_path = model_path.with_suffix(".webp")
            preview_dest = ProjectFileDestination.from_situation(
                filename=str(preview_path),
                situation="save_node_output",
                node_name=node.name,
            )
            saved_preview = await preview_dest.awrite_bytes(preview_bytes)
            node.parameter_output_values["preview_image"] = ImageUrlArtifact(
                value=saved_preview.location,
                meta={"filename": saved_preview.name},
            )

    consumed = data.get("consumed_credit")
    detail = "3D model generated successfully."
    if consumed:
        detail += f" Tripo charged {consumed} credits."
    node._set_status_results(was_successful=True, result_details=detail)
