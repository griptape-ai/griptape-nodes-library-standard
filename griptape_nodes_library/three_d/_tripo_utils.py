from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import ArtifactKind
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
    QUAD = "quad"
    FACE_LIMIT = "face_limit"


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
        {
            TripoCapability.TEXTURE,
            TripoCapability.GEOMETRY_QUALITY,
            TripoCapability.NEGATIVE_PROMPT,
            TripoCapability.QUAD,
            TripoCapability.FACE_LIMIT,
        }
    ),
    TripoEndpoint.IMAGE: frozenset(
        {
            TripoCapability.TEXTURE,
            TripoCapability.GEOMETRY_QUALITY,
            TripoCapability.TEXTURE_ALIGNMENT,
            TripoCapability.QUAD,
            TripoCapability.FACE_LIMIT,
        }
    ),
    TripoEndpoint.MULTIVIEW: frozenset(
        {
            TripoCapability.TEXTURE,
            TripoCapability.GEOMETRY_QUALITY,
            TripoCapability.TEXTURE_ALIGNMENT,
            TripoCapability.QUAD,
            TripoCapability.FACE_LIMIT,
        }
    ),
}

_DOCS_URLS: dict[TripoEndpoint, str] = {
    TripoEndpoint.TEXT: "https://developers.tripo3d.ai/en/docs/generation-text-to-model/p",
    TripoEndpoint.IMAGE: "https://developers.tripo3d.ai/en/docs/generation-image-to-model/p",
    TripoEndpoint.MULTIVIEW: "https://developers.tripo3d.ai/en/docs/generation-multiview-to-model/p",
}

P2_MODEL_VERSION = "P2-20260801"

# P2 face_limit ranges (Tripo v3 docs, P Series). Checked client-side so a bad value
# fails with a clear message rather than an opaque 400 from the provider.
P2_FACE_LIMIT_MIN = 48
P2_FACE_LIMIT_MAX_TRIANGLE = 50000
P2_FACE_LIMIT_MAX_QUAD = 25000

# The versions Tripo currently generates on, newest family first. `label` stays short
# because it is what the collapsed dropdown shows; the build date and the tradeoff go
# in `subtitle`, which the UI renders as a secondary line on the open row. P1 and the
# H3 (v3.x) line are separate model families rather than successive versions, so the
# labels name the family instead of leaving the dates to imply an ordering.
MODEL_VERSIONS: tuple[TripoModelVersion, ...] = (
    TripoModelVersion(
        value=P2_MODEL_VERSION,
        label="P2",
        subtitle="Preview; low-poly with quad mesh output (2026-08-01)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {
                TripoCapability.TEXTURE,
                TripoCapability.TEXTURE_ALIGNMENT,
                TripoCapability.NEGATIVE_PROMPT,
                TripoCapability.QUAD,
                TripoCapability.FACE_LIMIT,
            }
        ),
    ),
    TripoModelVersion(
        value="P1-20260311",
        label="P1",
        subtitle="Premium; low-poly, stable topology (2026-03-11)",
        endpoints=_EVERY_ENDPOINT,
        capabilities=frozenset(
            {
                TripoCapability.TEXTURE,
                TripoCapability.TEXTURE_ALIGNMENT,
                TripoCapability.NEGATIVE_PROMPT,
                TripoCapability.FACE_LIMIT,
            }
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

# One default for every endpoint, so two Tripo nodes in the same workflow don't start on
# models from different families. `_raise_if_misconfigured` enforces that each endpoint
# actually offers it.
DEFAULT_MODEL_VERSION = "v3.1-20260211"

# Versions Tripo has deprecated, mapped to the current version a stored value becomes.
# Same shape and contract as ``ModelAccessComponent``'s ``deprecated_values``: a key is
# accepted wherever a value is assigned, migrated to its canonical choice, and never
# offered as a fresh selection.
#
# Tripo answers a deprecated version with code 2015 and no generation, so leaving one
# selectable only offers a guaranteed failure. Keyed per endpoint because deprecation
# is per endpoint: v1.4 still generates on text and image.
DEPRECATED_VERSIONS: dict[TripoEndpoint, dict[str, str]] = {
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


def version_choices(endpoint: TripoEndpoint) -> list[str]:
    """Values the ``model_version`` parameter accepts, current plus deprecated.

    ``Options`` rewrites an assigned value that is outside ``choices`` to
    ``choices[0]``, so a deprecated value has to be here for a saved workflow's stored
    value to survive long enough for ``migrate_version`` to translate it. Deprecated
    values are deliberately absent from ``dropdown_row_data``, which is what the UI
    offers.
    """
    return [version.value for version in versions_for(endpoint)] + list(DEPRECATED_VERSIONS[endpoint])


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
    """The canonical version ``value`` migrates to if it is a deprecated key, else None.

    Non-``str`` input returns None: a connected upstream value isn't a dropdown token
    this table covers.
    """
    if not isinstance(value, str):
        return None
    return DEPRECATED_VERSIONS[endpoint].get(value)


def _raise_if_misconfigured(endpoint: TripoEndpoint) -> None:
    """Raise on any table misuse, so a bad edit fails loudly instead of shipping.

    Mirrors ``ModelAccessComponent``'s preconditions: every deprecated value must
    migrate to a current choice, no deprecated key may itself still be offered, and
    the default must be a current choice. Without this a mistyped replacement snaps
    the parameter to ``choices[0]`` on load, silently changing the model a saved
    workflow generates on.
    """
    choice_set = {version.value for version in versions_for(endpoint)}
    deprecated = DEPRECATED_VERSIONS[endpoint]

    problems = []
    invalid_values = sorted({canonical for canonical in deprecated.values() if canonical not in choice_set})
    if invalid_values:
        problems.append(f"value(s) not offered on this endpoint: {', '.join(repr(v) for v in invalid_values)}")
    colliding_keys = sorted(legacy for legacy in deprecated if legacy in choice_set)
    if colliding_keys:
        problems.append(f"key(s) already a current choice: {', '.join(repr(k) for k in colliding_keys)}")
    if problems:
        msg = f"Tripo {endpoint} DEPRECATED_VERSIONS is invalid: {'; '.join(problems)}."
        raise ValueError(msg)

    if DEFAULT_MODEL_VERSION not in choice_set:
        msg = (
            f"Tripo {endpoint} declares DEFAULT_MODEL_VERSION {DEFAULT_MODEL_VERSION!r}, which this "
            "endpoint does not offer. Point it at a current version -- a deprecated one belongs in "
            "DEPRECATED_VERSIONS, not the default."
        )
        raise ValueError(msg)


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
    path a stored deprecated value would otherwise slip through untranslated.
    """
    _raise_if_misconfigured(endpoint)

    parameter = ParameterString(
        name="model_version",
        default_value=DEFAULT_MODEL_VERSION,
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

    return parameter


def add_mesh_topology_parameters(node: GriptapeProxyNode) -> None:
    """Add the ``quad`` and ``face_limit`` parameters, hidden until a version supports them."""
    node.add_parameter(
        ParameterBool(
            name="quad",
            default_value=False,
            tooltip="Output a quad mesh instead of triangles (P2 only). Produces an FBX file instead of GLB.",
            allow_output=False,
            hide=True,
            ui_options={"display_name": "Quad Mesh"},
        )
    )
    node.add_parameter(
        ParameterInt(
            name="face_limit",
            default_value=0,
            tooltip=(
                "Maximum polycount for the output mesh. 0 lets Tripo decide. "
                f"P2 triangle: {P2_FACE_LIMIT_MIN}-{P2_FACE_LIMIT_MAX_TRIANGLE}; "
                f"P2 quad: {P2_FACE_LIMIT_MIN}-{P2_FACE_LIMIT_MAX_QUAD}."
            ),
            allow_output=False,
            hide=True,
            ui_options={"display_name": "Face Limit"},
        )
    )


def update_mesh_topology_visibility(node: GriptapeProxyNode, endpoint: TripoEndpoint, model_version: str) -> None:
    for name, capability in (("quad", TripoCapability.QUAD), ("face_limit", TripoCapability.FACE_LIMIT)):
        if supports(endpoint, model_version, capability):
            node.show_parameter_by_name(name)
        else:
            node.hide_parameter_by_name(name)


def add_mesh_topology_fields(
    node: GriptapeProxyNode, endpoint: TripoEndpoint, model_version: str, payload: dict[str, Any]
) -> None:
    """Add ``quad`` and a non-zero ``face_limit`` to ``payload`` where the version supports them.

    Raises:
        ValueError: If ``face_limit`` is outside P2's documented range for the chosen topology.
    """
    quad = False
    if supports(endpoint, model_version, TripoCapability.QUAD):
        quad = bool(node.get_parameter_value("quad"))
        payload["quad"] = quad

    if not supports(endpoint, model_version, TripoCapability.FACE_LIMIT):
        return
    face_limit = int(node.get_parameter_value("face_limit") or 0)
    if not face_limit:
        return
    if model_version == P2_MODEL_VERSION:
        maximum = P2_FACE_LIMIT_MAX_QUAD if quad else P2_FACE_LIMIT_MAX_TRIANGLE
        if not P2_FACE_LIMIT_MIN <= face_limit <= maximum:
            topology = "quad" if quad else "triangle"
            msg = f"Face limit must be {P2_FACE_LIMIT_MIN}-{maximum} for P2 {topology} output, got {face_limit}."
            raise ValueError(msg)
    payload["face_limit"] = face_limit


# Tripo returns GLB normally and FBX when quad=true; the proxy states which in the
# hosted artifact's content type.
_MODEL_EXTENSIONS_BY_CONTENT_TYPE = {
    "model/vnd.fbx": "fbx",
    "model/gltf-binary": "glb",
}


def _model_extension(content_type: str | None) -> str:
    if not content_type:
        return "glb"
    return _MODEL_EXTENSIONS_BY_CONTENT_TYPE.get(content_type.split(";")[0].strip().lower(), "glb")


async def parse_tripo_task_result(node: GriptapeProxyNode, result_json: dict[str, Any], generation_id: str) -> None:
    """Save a completed Tripo task's mesh and preview image as project files.

    The proxy hosts both, the mesh first, so neither of Tripo's signed URLs (they
    expire within five minutes) is ever handed downstream. The task payload is still
    read for the credits Tripo charged:
        {"code": 0, "data": {"status": "success", "credits_consumed": 20.0, ...}}
    """
    try:
        model_artifact = await node._hosted_artifact(generation_id, kind=ArtifactKind.MODEL_3D)
        model_bytes = await node._download_artifact(model_artifact)
    except Exception as e:
        node._set_safe_defaults()
        node._set_status_results(
            was_successful=False,
            result_details=f"Tripo task completed but its 3D model could not be retrieved: {e}",
        )
        return

    extension = _model_extension(model_artifact.content_type)
    output_file_value = node.get_parameter_value("output_file") or f"tripo_model.{extension}"
    model_path = Path(output_file_value).with_suffix(f".{extension}")
    model_dest = ProjectFileDestination.from_situation(
        filename=str(model_path),
        situation="save_node_output",
        node_name=node.name,
    )
    saved_model = await model_dest.awrite_bytes(model_bytes)
    node.parameter_output_values["model_url"] = ThreeDUrlArtifact(
        value=saved_model.location,
        meta={"filename": saved_model.name, "format": extension},
    )

    # Tripo renders a preview for most task types but not all, so a missing one
    # leaves the model output standing rather than failing the node.
    try:
        preview_bytes = await node._load_generated_media(generation_id, kind=ArtifactKind.IMAGE)
    except Exception as e:
        logger.info("%s has no preview image to save: %s", node.name, e)
    else:
        preview_dest = ProjectFileDestination.from_situation(
            filename=str(model_path.with_suffix(".webp")),
            situation="save_node_output",
            node_name=node.name,
        )
        saved_preview = await preview_dest.awrite_bytes(preview_bytes)
        node.parameter_output_values["preview_image"] = ImageUrlArtifact(
            value=saved_preview.location,
            meta={"filename": saved_preview.name},
        )

    data = result_json.get("data") if isinstance(result_json, dict) else None
    consumed = (data.get("credits_consumed") or data.get("consumed_credit")) if isinstance(data, dict) else None
    detail = "3D model generated successfully."
    if consumed:
        detail += f" Tripo charged {consumed} credits."
    node._set_status_results(was_successful=True, result_details=detail)
