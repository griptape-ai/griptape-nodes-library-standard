"""The three Tripo nodes share one model-version table, and it matches the live API.

Every version and capability asserted here was probed against Tripo through the
Griptape proxy: a version Tripo has retired answers with code 2015 and no
generation, so a retired version left in a dropdown offers only a guaranteed
failure. The retirements are per endpoint -- `v1.4-20240625` still generates on
text and image while multiview rejects it -- which is why the table keys them by
endpoint rather than globally.

Capability gating matters beyond tidiness: the per-version sets decide which
fields reach the proxy, and a field the model does not accept fails the request
(issue 275 was `geometry_quality` sent on P1).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from griptape_nodes_library.three_d._tripo_utils import (
    DEFAULT_MODEL_VERSION,
    MODEL_VERSIONS,
    TripoCapability,
    TripoEndpoint,
    badge_message,
    default_version,
    dropdown_row_data,
    migrate_version,
    supports,
    version_choices,
    versions_for,
)
from griptape_nodes_library.three_d.tripo_image_to_3d_generation import TripoImageTo3DGeneration
from griptape_nodes_library.three_d.tripo_multiview_to_3d_generation import TripoMultiviewTo3DGeneration
from griptape_nodes_library.three_d.tripo_text_to_3d_generation import TripoTextTo3DGeneration

if TYPE_CHECKING:
    from griptape_nodes.exe_types.core_types import Parameter

# What the live probe found each endpoint generates on. An explicit table rather
# than a re-derivation of the registry, so a future edit that quietly drops a
# working version or revives a retired one fails here.
LIVE_VERSIONS: dict[TripoEndpoint, list[str]] = {
    TripoEndpoint.TEXT: [
        "P1-20260311",
        "v3.1-20260211",
        "v3.0-20250812",
        "v2.5-20250123",
        "v1.4-20240625",
    ],
    TripoEndpoint.IMAGE: [
        "P1-20260311",
        "v3.1-20260211",
        "v3.0-20250812",
        "v2.5-20250123",
        "v1.4-20240625",
    ],
    TripoEndpoint.MULTIVIEW: [
        "P1-20260311",
        "v3.1-20260211",
        "v3.0-20250812",
        "v2.5-20250123",
    ],
}

# Retired by Tripo (code 2015), with the live version a stored value becomes.
RETIRED_VERSIONS: dict[TripoEndpoint, dict[str, str]] = {
    TripoEndpoint.TEXT: {"Turbo-v1.0-20250506": "v3.1-20260211", "v2.0-20240919": "v2.5-20250123"},
    TripoEndpoint.IMAGE: {"Turbo-v1.0-20250506": "v3.1-20260211", "v2.0-20240919": "v2.5-20250123"},
    TripoEndpoint.MULTIVIEW: {"v2.0-20240919": "v2.5-20250123", "v1.4-20240625": "v3.1-20260211"},
}

NODE_CLASSES: dict[TripoEndpoint, Any] = {
    TripoEndpoint.TEXT: TripoTextTo3DGeneration,
    TripoEndpoint.IMAGE: TripoImageTo3DGeneration,
    TripoEndpoint.MULTIVIEW: TripoMultiviewTo3DGeneration,
}

ENDPOINTS = list(TripoEndpoint)

FAKE_DATA_URI = "data:image/png;base64,iVBORw0KGgo="


def _make_node(endpoint: TripoEndpoint) -> Any:
    return NODE_CLASSES[endpoint](name=f"Tripo{endpoint.name}")


def _parameter(node: Any, name: str) -> Parameter:
    return next(parameter for parameter in node.parameters if parameter.name == name)


def _offered_rows(node: Any) -> list[dict[str, str]]:
    return list(_parameter(node, "model_version").ui_options["data"])


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_dropdown_offers_exactly_the_live_versions(endpoint: TripoEndpoint) -> None:
    assert [version.value for version in versions_for(endpoint)] == LIVE_VERSIONS[endpoint]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_node_dropdown_rows_offer_only_live_versions(endpoint: TripoEndpoint) -> None:
    rows = _offered_rows(_make_node(endpoint))

    assert [row["name"] for row in rows] == LIVE_VERSIONS[endpoint]
    for retired in RETIRED_VERSIONS[endpoint]:
        assert retired not in {row["name"] for row in rows}, f"{retired} can only fail; it must not be offered"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_retired_versions_stay_assignable_so_saved_workflows_can_migrate(endpoint: TripoEndpoint) -> None:
    """`Options` snaps an out-of-choices value to choices[0], which would lose the migration."""
    choices = version_choices(endpoint)

    for retired in RETIRED_VERSIONS[endpoint]:
        assert retired in choices


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_retired_value_migrates_to_a_live_version_on_assignment(endpoint: TripoEndpoint) -> None:
    for retired, expected in RETIRED_VERSIONS[endpoint].items():
        node = _make_node(endpoint)
        node.set_parameter_value("model_version", retired)

        assert node.get_parameter_value("model_version") == expected


def test_v1_4_survives_where_it_still_generates() -> None:
    """v1.4 is retired on multiview only, so the other endpoints must keep it selectable."""
    for endpoint in (TripoEndpoint.TEXT, TripoEndpoint.IMAGE):
        node = _make_node(endpoint)
        node.set_parameter_value("model_version", "v1.4-20240625")

        assert node.get_parameter_value("model_version") == "v1.4-20240625"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_every_node_defaults_to_the_shared_version(endpoint: TripoEndpoint) -> None:
    node = _make_node(endpoint)

    assert node.get_parameter_value("model_version") == DEFAULT_MODEL_VERSION
    assert default_version(endpoint) == DEFAULT_MODEL_VERSION
    assert DEFAULT_MODEL_VERSION in LIVE_VERSIONS[endpoint]


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_dropdown_rows_carry_a_label_and_subtitle(endpoint: TripoEndpoint) -> None:
    node = _make_node(endpoint)
    parameter = _parameter(node, "model_version")

    assert parameter.ui_options["dropdown_row_subtitles"] is True
    for row in _offered_rows(node):
        assert row["label"], f"{row['name']} has no display label"
        assert row["subtitle"], f"{row['name']} has no subtitle"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_labels_are_shorter_than_the_raw_build_string(endpoint: TripoEndpoint) -> None:
    """The label is what the collapsed control shows, so it must not crowd the node."""
    for row in dropdown_row_data(endpoint):
        assert len(row["label"]) < len(row["name"])


def test_labels_separate_the_p1_and_h3_families() -> None:
    """P1 and v3.x are different model families; nothing in a bare date says so."""
    labels = {version.value: version.label for version in MODEL_VERSIONS}

    assert labels["P1-20260311"] == "P1"
    assert labels["v3.1-20260211"].startswith("H3")
    assert labels["v3.0-20250812"].startswith("H3")


def test_every_node_describes_a_version_identically() -> None:
    """The same version must not read differently depending on which node you opened."""
    for version in MODEL_VERSIONS:
        descriptions = {
            row["subtitle"]
            for endpoint in version.endpoints
            for row in dropdown_row_data(endpoint)
            if row["name"] == version.value
        }

        assert len(descriptions) == 1, f"{version.value} is described {len(descriptions)} different ways"


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_badge_documents_every_offered_version(endpoint: TripoEndpoint) -> None:
    message = badge_message(endpoint)

    for version in versions_for(endpoint):
        assert version.value in message
    for retired in RETIRED_VERSIONS[endpoint]:
        assert retired not in message


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_registry_retirement_table_is_coherent(endpoint: TripoEndpoint) -> None:
    """Mirrors ModelAccessComponent's preconditions: targets are live, keys are not."""
    live = set(LIVE_VERSIONS[endpoint])

    for retired, replacement in RETIRED_VERSIONS[endpoint].items():
        assert replacement in live, f"{retired} migrates to {replacement}, which this endpoint does not offer"
        assert retired not in live, f"{retired} is both retired and offered"


def test_geometry_quality_is_gated_to_the_h3_line() -> None:
    """P1 rejects geometry_quality (issue 275); v3.x accepts it on all three endpoints."""
    for endpoint in ENDPOINTS:
        assert supports(endpoint, "v3.1-20260211", TripoCapability.GEOMETRY_QUALITY)
        assert supports(endpoint, "v3.0-20250812", TripoCapability.GEOMETRY_QUALITY)
        assert not supports(endpoint, "P1-20260311", TripoCapability.GEOMETRY_QUALITY)
        assert not supports(endpoint, "v2.5-20250123", TripoCapability.GEOMETRY_QUALITY)


def test_capabilities_need_both_a_model_and_an_endpoint_that_has_them() -> None:
    # No source image on text-to-model, so texture_alignment has no meaning there.
    assert not supports(TripoEndpoint.TEXT, "P1-20260311", TripoCapability.TEXTURE_ALIGNMENT)
    assert supports(TripoEndpoint.IMAGE, "P1-20260311", TripoCapability.TEXTURE_ALIGNMENT)
    # No prompt on the image endpoints, so negative_prompt has none either.
    assert not supports(TripoEndpoint.IMAGE, "v3.1-20260211", TripoCapability.NEGATIVE_PROMPT)
    assert supports(TripoEndpoint.TEXT, "v3.1-20260211", TripoCapability.NEGATIVE_PROMPT)


def test_legacy_v1_4_takes_no_optional_fields() -> None:
    for capability in TripoCapability:
        assert not supports(TripoEndpoint.TEXT, "v1.4-20240625", capability)
        assert not supports(TripoEndpoint.IMAGE, "v1.4-20240625", capability)


def test_an_undeclared_version_supports_nothing() -> None:
    """A version Tripo adds stays inert until declared, rather than inheriting fields."""
    for capability in TripoCapability:
        assert not supports(TripoEndpoint.IMAGE, "v4.0-20270101", capability)


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_geometry_quality_visibility_follows_the_selected_version(endpoint: TripoEndpoint) -> None:
    node = _make_node(endpoint)

    node.set_parameter_value("model_version", "v3.1-20260211")
    assert _parameter(node, "geometry_quality").hide is False

    node.set_parameter_value("model_version", "P1-20260311")
    assert _parameter(node, "geometry_quality").hide is True


@pytest.mark.asyncio
async def test_multiview_payload_sends_geometry_quality_only_on_the_h3_line(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.three_d.tripo_multiview_to_3d_generation.prepare_media_data_uri",
        _stub_media_data_uri,
    )
    node = TripoMultiviewTo3DGeneration(name="TripoMultiview")
    node.set_parameter_value("front_image", FAKE_DATA_URI)
    node.set_parameter_value("left_image", FAKE_DATA_URI)

    node.set_parameter_value("model_version", "v3.1-20260211")
    payload = await node._build_payload()
    assert payload["geometry_quality"] == "standard"
    assert payload["texture_alignment"] == "original_image"
    assert payload["texture"] is True

    node.set_parameter_value("model_version", "P1-20260311")
    payload = await node._build_payload()
    assert "geometry_quality" not in payload
    assert payload["texture_alignment"] == "original_image"


@pytest.mark.asyncio
async def test_image_payload_omits_geometry_quality_on_p1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.three_d.tripo_image_to_3d_generation.prepare_media_data_uri",
        _stub_media_data_uri,
    )
    node = TripoImageTo3DGeneration(name="TripoImage")
    node.set_parameter_value("image", FAKE_DATA_URI)

    node.set_parameter_value("model_version", "P1-20260311")
    payload = await node._build_payload()
    assert "geometry_quality" not in payload

    node.set_parameter_value("model_version", "v3.1-20260211")
    payload = await node._build_payload()
    assert payload["geometry_quality"] == "standard"


@pytest.mark.asyncio
async def test_image_payload_drops_every_optional_field_on_v1_4(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.three_d.tripo_image_to_3d_generation.prepare_media_data_uri",
        _stub_media_data_uri,
    )
    node = TripoImageTo3DGeneration(name="TripoImage")
    node.set_parameter_value("image", FAKE_DATA_URI)
    node.set_parameter_value("model_version", "v1.4-20240625")

    payload = await node._build_payload()

    for field in ("texture", "pbr", "texture_quality", "texture_alignment", "geometry_quality"):
        assert field not in payload
    assert payload["model_version"] == "v1.4-20240625"


@pytest.mark.asyncio
async def test_text_payload_gates_negative_prompt_and_geometry_quality() -> None:
    node = TripoTextTo3DGeneration(name="TripoText")
    node.set_parameter_value("prompt", "a small red cube")
    node.set_parameter_value("negative_prompt", "blurry")

    node.set_parameter_value("model_version", "v3.1-20260211")
    payload = await node._build_payload()
    assert payload["negative_prompt"] == "blurry"
    assert payload["geometry_quality"] == "standard"

    node.set_parameter_value("model_version", "v1.4-20240625")
    payload = await node._build_payload()
    assert "negative_prompt" not in payload
    assert "geometry_quality" not in payload
    assert "texture" not in payload


@pytest.mark.asyncio
async def test_retired_stored_version_never_reaches_the_proxy() -> None:
    """A workflow saved on a retired version generates on its replacement instead of failing."""
    node = TripoTextTo3DGeneration(name="TripoText")
    node.set_parameter_value("prompt", "a small red cube")
    node.set_parameter_value("model_version", "Turbo-v1.0-20250506")

    payload = await node._build_payload()

    assert payload["model_version"] == DEFAULT_MODEL_VERSION


def test_migrate_version_passes_through_live_and_unknown_values() -> None:
    assert migrate_version(TripoEndpoint.IMAGE, "v3.1-20260211") is None
    assert migrate_version(TripoEndpoint.IMAGE, "something-else") is None
    assert migrate_version(TripoEndpoint.IMAGE, None) is None


async def _stub_media_data_uri(*_args: Any, **_kwargs: Any) -> str:
    return FAKE_DATA_URI
