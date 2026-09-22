"""Tests for how WorldLabsWorldGeneration maps the proxy's hosted artifact list

onto World Labs's assets (splats, collider mesh, panorama).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import pytest

from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifactError
from griptape_nodes_library.splat.world_labs_world_generation import WorldLabsWorldGeneration


@dataclass(frozen=True)
class _FakeArtifact:
    index: int
    kind: str
    url: str = "https://example/artifact"
    content_type: str | None = None
    size_bytes: int | None = None


def _fake_parse_assets(unsaved: list[str]) -> Callable[..., Awaitable[list[str]]]:
    async def _parse_assets(_assets: dict[str, Any], _world_id: str, _generation_id: str) -> list[str]:
        return unsaved

    return _parse_assets


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> WorldLabsWorldGeneration:
    instance = WorldLabsWorldGeneration.__new__(WorldLabsWorldGeneration)
    instance.name = "WorldLabs"  # type: ignore[misc]
    monkeypatch.setattr(instance, "_log", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(instance, "get_parameter_value", lambda _name: None, raising=False)
    instance.parameter_output_values = {}  # type: ignore[assignment]
    return instance


def test_expected_asset_slots_orders_splats_then_mesh_then_panorama() -> None:
    assets = {
        "splats": {"spz_urls": {"full_res": "https://a", "100k": "https://b"}},
        "mesh": {"collider_mesh_url": "https://c"},
        "imagery": {"pano_url": "https://d"},
    }

    slots = WorldLabsWorldGeneration._expected_asset_slots(assets)

    assert [key for key, _filename in slots] == ["splat_100k", "splat_full_res", "mesh", "panorama"]


def test_expected_asset_slots_skips_absent_assets() -> None:
    assets = {"mesh": {"collider_mesh_url": "https://c"}}

    slots = WorldLabsWorldGeneration._expected_asset_slots(assets)

    assert [key for key, _filename in slots] == ["mesh"]


def test_expected_asset_slots_is_empty_for_no_assets() -> None:
    assert WorldLabsWorldGeneration._expected_asset_slots({}) == []


@pytest.mark.asyncio
async def test_parse_assets_saves_the_prefix_the_proxy_hosted(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = {"mesh": {"collider_mesh_url": "https://c"}, "imagery": {"pano_url": "https://d"}}

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        # The proxy truncates a set by dropping its tail, so one hosted artifact for two
        # declared assets is the mesh, and saving it beats discarding it.
        return [_FakeArtifact(index=0, kind="model_3d")]

    async def fake_download_artifact(_artifact: _FakeArtifact) -> bytes:
        return b"mesh-bytes"

    class _SavedFile:
        location = "project/files/collider_mesh.glb"
        name = "collider_mesh.glb"

    class _Dest:
        async def awrite_bytes(self, _data: bytes) -> _SavedFile:
            return _SavedFile()

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)
    monkeypatch.setattr(
        "griptape_nodes_library.splat.world_labs_world_generation.ProjectFileDestination.from_situation",
        staticmethod(lambda *, filename, situation: _Dest()),  # noqa: ARG005
    )

    saved = await node._parse_assets(assets, "world-1", "gen-1")

    assert saved == ["panorama"]
    assert node.parameter_output_values["mesh"].value == "project/files/collider_mesh.glb"
    assert node.parameter_output_values.get("panorama") is None


@pytest.mark.asyncio
async def test_parse_assets_refuses_more_hosted_than_declared(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = {"mesh": {"collider_mesh_url": "https://c"}}

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        # More hosted than declared means the two lists cannot be paired by position.
        return [_FakeArtifact(index=0, kind="model_3d"), _FakeArtifact(index=1, kind="image")]

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)

    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)
    monkeypatch.setattr(node, "_set_safe_defaults", lambda: None, raising=False)

    saved = await node._parse_assets(assets, "world-1", "gen-1")

    assert saved is None
    assert status_calls[0]["was_successful"] is False
    assert "1 asset(s)" in status_calls[0]["result_details"]
    assert "hosts 2" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_parse_assets_reports_failure_rather_than_mispair_on_an_untrustworthy_list(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A gap or a dropped entry makes _hosted_artifacts raise rather than return a list
    # this node could slice-and-zip against the wrong bytes; the existing try/except
    # around _hosted_artifacts must turn that into a failure, not a mispair.
    assets = {"mesh": {"collider_mesh_url": "https://c"}, "imagery": {"pano_url": "https://d"}}

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        raise HostedArtifactError("surviving indices [0, 2]")

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)

    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)
    monkeypatch.setattr(node, "_set_safe_defaults", lambda: None, raising=False)

    saved = await node._parse_assets(assets, "world-1", "gen-1")

    assert saved is None
    assert status_calls[0]["was_successful"] is False
    assert node.parameter_output_values.get("mesh") is None
    assert node.parameter_output_values.get("panorama") is None


@pytest.mark.asyncio
async def test_parse_assets_reports_failure_when_every_download_fails(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A billed world whose every asset download failed is a failure, not a success
    # with empty asset outputs.
    assets = {"mesh": {"collider_mesh_url": "https://c"}, "imagery": {"pano_url": "https://d"}}

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        return [_FakeArtifact(index=0, kind="model_3d"), _FakeArtifact(index=1, kind="image")]

    async def fake_download_artifact(_artifact: _FakeArtifact) -> bytes:
        msg = "artifact gone"
        raise RuntimeError(msg)

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)

    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)
    monkeypatch.setattr(node, "_set_safe_defaults", lambda: None, raising=False)

    saved = await node._parse_assets(assets, "world-1", "gen-1")

    assert saved is None
    assert status_calls[0]["was_successful"] is False
    assert "none of its 2 asset(s) could be retrieved" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_handle_success_notes_the_assets_that_could_not_be_retrieved(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A partial world still succeeds, but the status has to name what is missing.
    monkeypatch.setattr(node, "_parse_assets", _fake_parse_assets(["panorama"]), raising=False)

    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)

    await node._handle_success({"world_id": "w1", "assets": {}}, "gen-1")

    assert status_calls[0]["was_successful"] is True
    assert "1 asset(s) could not be retrieved: panorama" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_handle_success_stays_silent_when_every_asset_landed(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "_parse_assets", _fake_parse_assets([]), raising=False)

    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)

    await node._handle_success({"world_id": "w1", "assets": {}}, "gen-1")

    assert status_calls[0] == {"was_successful": True, "result_details": "Successfully generated 3D world: w1"}


@pytest.mark.asyncio
async def test_parse_assets_saves_mesh_and_panorama_as_project_files(
    node: WorldLabsWorldGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = {"mesh": {"collider_mesh_url": "https://c"}, "imagery": {"pano_url": "https://d"}}

    mesh_artifact = _FakeArtifact(index=0, kind="model_3d")
    pano_artifact = _FakeArtifact(index=1, kind="image")

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        return [mesh_artifact, pano_artifact]

    async def fake_download_artifact(artifact: _FakeArtifact) -> bytes:
        return b"mesh-bytes" if artifact is mesh_artifact else b"pano-bytes"

    class _SavedFile:
        def __init__(self, location: str) -> None:
            self.location = location
            self.name = location.rsplit("/", maxsplit=1)[-1]

    class _Dest:
        def __init__(self, filename: str) -> None:
            self._filename = filename

        async def awrite_bytes(self, _data: bytes) -> _SavedFile:
            return _SavedFile(f"project/files/{self._filename}")

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)
    monkeypatch.setattr(
        "griptape_nodes_library.splat.world_labs_world_generation.ProjectFileDestination.from_situation",
        staticmethod(lambda *, filename, situation: _Dest(filename)),  # noqa: ARG005
    )

    saved = await node._parse_assets(assets, "world-1", "gen-1")

    assert saved == []
    mesh_out = node.parameter_output_values["mesh"]
    assert mesh_out.value == "project/files/collider_mesh.glb"
    panorama_out = node.parameter_output_values["panorama"]
    assert panorama_out.value == "project/files/panorama.jpg"
