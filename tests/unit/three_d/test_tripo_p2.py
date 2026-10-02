"""P2's quad mesh and face limit, and saving the FBX a quad request produces.

P2-20260801 was verified live through the Tripo v3 API: a quad request returned an
FBX model URL where every other request returns GLB, so the saved file has to take
its extension from what the proxy hosted rather than assume GLB.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifact
from griptape_nodes_library.three_d._tripo_utils import (
    P2_FACE_LIMIT_MAX_QUAD,
    P2_FACE_LIMIT_MAX_TRIANGLE,
    P2_FACE_LIMIT_MIN,
    P2_MODEL_VERSION,
    TripoCapability,
    TripoEndpoint,
    parse_tripo_task_result,
    supports,
)
from griptape_nodes_library.three_d.tripo_image_to_3d_generation import TripoImageTo3DGeneration
from griptape_nodes_library.three_d.tripo_multiview_to_3d_generation import TripoMultiviewTo3DGeneration
from griptape_nodes_library.three_d.tripo_text_to_3d_generation import TripoTextTo3DGeneration

ENDPOINTS = list(TripoEndpoint)
FAKE_DATA_URI = "data:image/png;base64,iVBORw0KGgo="


async def _stub_media_data_uri(*_args: Any, **_kwargs: Any) -> str:
    return FAKE_DATA_URI


def _text_node(model_version: str = P2_MODEL_VERSION) -> TripoTextTo3DGeneration:
    node = TripoTextTo3DGeneration(name="TripoText")
    node.set_parameter_value("prompt", "a wooden chair")
    node.set_parameter_value("model_version", model_version)
    return node


def _hidden(node: Any, name: str) -> bool:
    return next(parameter for parameter in node.parameters if parameter.name == name).hide is True


@pytest.mark.parametrize("endpoint", ENDPOINTS)
def test_quad_is_p2_only_and_face_limit_covers_the_p_series(endpoint: TripoEndpoint) -> None:
    assert supports(endpoint, P2_MODEL_VERSION, TripoCapability.QUAD)
    assert not supports(endpoint, "P1-20260311", TripoCapability.QUAD)
    assert not supports(endpoint, "v3.1-20260211", TripoCapability.QUAD)

    assert supports(endpoint, P2_MODEL_VERSION, TripoCapability.FACE_LIMIT)
    assert supports(endpoint, "P1-20260311", TripoCapability.FACE_LIMIT)
    assert not supports(endpoint, "v3.1-20260211", TripoCapability.FACE_LIMIT)


def test_p2_takes_no_geometry_quality() -> None:
    """Like P1, the P Series rejects geometry_quality."""
    for endpoint in ENDPOINTS:
        assert not supports(endpoint, P2_MODEL_VERSION, TripoCapability.GEOMETRY_QUALITY)


@pytest.mark.parametrize(
    "node_class", [TripoTextTo3DGeneration, TripoImageTo3DGeneration, TripoMultiviewTo3DGeneration]
)
def test_topology_parameter_visibility_follows_the_selected_version(node_class: Any) -> None:
    node = node_class(name="Tripo")

    node.set_parameter_value("model_version", P2_MODEL_VERSION)
    assert not _hidden(node, "quad")
    assert not _hidden(node, "face_limit")

    node.set_parameter_value("model_version", "P1-20260311")
    assert _hidden(node, "quad")
    assert not _hidden(node, "face_limit")

    node.set_parameter_value("model_version", "v3.1-20260211")
    assert _hidden(node, "quad")
    assert _hidden(node, "face_limit")


@pytest.mark.asyncio
async def test_p2_defaults_send_quad_false_and_no_face_limit() -> None:
    payload = await _text_node()._build_payload()

    assert payload["model_version"] == P2_MODEL_VERSION
    assert payload["quad"] is False
    assert "face_limit" not in payload  # 0 means adaptive, so the field is omitted


@pytest.mark.asyncio
async def test_p2_sends_quad_and_face_limit_when_set() -> None:
    node = _text_node()
    node.set_parameter_value("quad", True)
    node.set_parameter_value("face_limit", 5000)

    payload = await node._build_payload()

    assert payload["quad"] is True
    assert payload["face_limit"] == 5000


@pytest.mark.asyncio
async def test_p2_accepts_the_range_boundaries() -> None:
    for quad, maximum in ((False, P2_FACE_LIMIT_MAX_TRIANGLE), (True, P2_FACE_LIMIT_MAX_QUAD)):
        for face_limit in (P2_FACE_LIMIT_MIN, maximum):
            node = _text_node()
            node.set_parameter_value("quad", quad)
            node.set_parameter_value("face_limit", face_limit)

            assert (await node._build_payload())["face_limit"] == face_limit


@pytest.mark.asyncio
async def test_p2_rejects_a_triangle_face_limit_on_quad_output() -> None:
    node = _text_node()
    node.set_parameter_value("quad", True)
    node.set_parameter_value("face_limit", P2_FACE_LIMIT_MAX_TRIANGLE)

    with pytest.raises(ValueError, match="quad"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_p2_rejects_a_face_limit_below_the_minimum() -> None:
    node = _text_node()
    node.set_parameter_value("face_limit", P2_FACE_LIMIT_MIN - 1)

    with pytest.raises(ValueError, match="triangle"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_p1_sends_face_limit_without_p2_range_checks_and_never_quad() -> None:
    node = _text_node("P1-20260311")
    node.set_parameter_value("quad", True)
    node.set_parameter_value("face_limit", 20000)

    payload = await node._build_payload()

    assert payload["face_limit"] == 20000
    assert "quad" not in payload


@pytest.mark.asyncio
async def test_h3_sends_neither_topology_field() -> None:
    node = _text_node("v3.1-20260211")
    node.set_parameter_value("quad", True)
    node.set_parameter_value("face_limit", 5000)

    payload = await node._build_payload()

    assert "quad" not in payload
    assert "face_limit" not in payload


@pytest.mark.asyncio
async def test_multiview_payload_carries_topology_fields_on_p2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.three_d.tripo_multiview_to_3d_generation.prepare_media_data_uri",
        _stub_media_data_uri,
    )
    node = TripoMultiviewTo3DGeneration(name="TripoMultiview")
    node.set_parameter_value("front_image", FAKE_DATA_URI)
    node.set_parameter_value("left_image", FAKE_DATA_URI)
    node.set_parameter_value("model_version", P2_MODEL_VERSION)
    node.set_parameter_value("quad", True)
    node.set_parameter_value("face_limit", 2000)

    payload = await node._build_payload()

    assert payload["quad"] is True
    assert payload["face_limit"] == 2000
    assert payload["texture_alignment"] == "original_image"


# ----- saving the hosted model --------------------------------------------------------


class _FakeDestination:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def awrite_bytes(self, data: bytes) -> SimpleNamespace:
        self._path.write_bytes(data)
        return SimpleNamespace(location=str(self._path), name=self._path.name)


def _stub_hosting(
    node: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifacts: list[HostedArtifact]
) -> list[str]:
    saved: list[str] = []

    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return artifacts

    async def fake_download_artifact(artifact: HostedArtifact) -> bytes:
        return f"bytes-{artifact.index}".encode()

    def fake_from_situation(*, filename: str, **_kwargs: Any) -> _FakeDestination:
        saved.append(filename)
        return _FakeDestination(tmp_path / Path(filename).name)

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)
    monkeypatch.setattr(
        "griptape_nodes_library.three_d._tripo_utils.ProjectFileDestination.from_situation", fake_from_situation
    )
    return saved


@pytest.mark.asyncio
async def test_quad_output_is_saved_as_fbx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = TripoTextTo3DGeneration(name="TripoText")
    saved = _stub_hosting(
        node,
        monkeypatch,
        tmp_path,
        [HostedArtifact(index=0, kind="model_3d", url="https://example/0", content_type="model/vnd.fbx")],
    )

    await parse_tripo_task_result(node, {"data": {"credits_consumed": 110.0}}, "gen-1")

    assert saved[0].endswith(".fbx")
    assert node.parameter_output_values["model_url"].meta["format"] == "fbx"


@pytest.mark.asyncio
async def test_glb_output_is_saved_as_glb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    node = TripoTextTo3DGeneration(name="TripoText")
    saved = _stub_hosting(
        node,
        monkeypatch,
        tmp_path,
        [
            HostedArtifact(index=0, kind="model_3d", url="https://example/0", content_type="model/gltf-binary"),
            HostedArtifact(index=1, kind="image", url="https://example/1", content_type="image/webp"),
        ],
    )

    await parse_tripo_task_result(node, {"data": {"credits_consumed": 100.0}}, "gen-1")

    assert saved[0].endswith(".glb")
    assert node.parameter_output_values["model_url"].meta["format"] == "glb"
    assert "preview_image" in node.parameter_output_values


@pytest.mark.asyncio
async def test_untyped_model_falls_back_to_glb(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A proxy deployed before it typed Tripo models hosts them untyped; GLB was the old assumption."""
    node = TripoTextTo3DGeneration(name="TripoText")
    saved = _stub_hosting(
        node, monkeypatch, tmp_path, [HostedArtifact(index=0, kind="model_3d", url="https://example/0")]
    )

    await parse_tripo_task_result(node, {}, "gen-1")

    assert saved[0].endswith(".glb")


@pytest.mark.asyncio
async def test_status_reports_credits_from_either_field_name(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for data in ({"credits_consumed": 110.0}, {"consumed_credit": 20}):
        node = TripoTextTo3DGeneration(name="TripoText")
        _stub_hosting(node, monkeypatch, tmp_path, [HostedArtifact(index=0, kind="model_3d", url="https://example/0")])

        await parse_tripo_task_result(node, {"data": data}, "gen-1")

        details = node.get_parameter_value("result_details") or ""
        assert str(next(iter(data.values()))) in details
