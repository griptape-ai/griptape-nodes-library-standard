"""Tests for how Rodin23DGeneration maps the proxy's hosted artifact list

onto Rodin's reported download files.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pytest

from griptape_nodes_library.three_d.rodin_2_3d_generation import Rodin23DGeneration


@dataclass(frozen=True)
class _FakeArtifact:
    index: int
    kind: str = "model_3d"
    url: str = "https://example/artifact"
    content_type: str | None = None
    size_bytes: int | None = None


class _SavedFile:
    def __init__(self, location: str) -> None:
        self.location = location
        self.name = location.rsplit("/", maxsplit=1)[-1]


class _Dest:
    def __init__(self, filename: str) -> None:
        self._filename = filename

    async def awrite_bytes(self, _data: bytes) -> _SavedFile:
        return _SavedFile(f"project/files/{self._filename}")


def _stub_destinations(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "griptape_nodes_library.three_d.rodin_2_3d_generation.ProjectFileDestination.from_situation",
        staticmethod(lambda *, filename, situation: _Dest(filename)),  # noqa: ARG005
    )


def _stub_status(node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs), raising=False)
    return status_calls


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Rodin23DGeneration:
    instance = Rodin23DGeneration.__new__(Rodin23DGeneration)
    instance.name = "Rodin"  # type: ignore[misc]
    monkeypatch.setattr(instance, "_log", lambda *_args, **_kwargs: None, raising=False)
    monkeypatch.setattr(instance, "get_parameter_value", lambda _name: None, raising=False)
    instance.parameter_output_values = {}  # type: ignore[assignment]
    return instance


@pytest.mark.asyncio
async def test_save_model_files_saves_model_and_preview_at_matching_count(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"name": "model.glb", "url": "https://a"},
        {"name": "preview.webp", "url": "https://b"},
    ]
    model_artifact = _FakeArtifact(index=0)
    preview_artifact = _FakeArtifact(index=1, kind="image")

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        return [model_artifact, preview_artifact]

    async def fake_download_artifact(artifact: _FakeArtifact) -> bytes:
        return b"model-bytes" if artifact is model_artifact else b"preview-bytes"

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)
    _stub_destinations(monkeypatch)
    status_calls = _stub_status(node, monkeypatch)

    await node._save_model_files(files, {"geometry_file_format": "glb"}, "gen-1")

    assert status_calls[-1]["was_successful"] is True
    model_out = node.parameter_output_values["model_url"]
    assert model_out.value == "project/files/model.glb"
    assert model_out.meta["filename"] == "model.glb"
    preview_out = node.parameter_output_values["preview_image"]
    assert preview_out.value == "project/files/preview.webp"


@pytest.mark.asyncio
async def test_save_model_files_refuses_more_hosted_than_reported(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [{"name": "model.glb", "url": "https://a"}]

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        # More hosted than reported means the two lists cannot be paired by position.
        return [_FakeArtifact(index=0), _FakeArtifact(index=1)]

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    status_calls = _stub_status(node, monkeypatch)

    await node._save_model_files(files, {"geometry_file_format": "glb"}, "gen-1")

    assert status_calls[-1]["was_successful"] is False
    assert "refusing to guess" in status_calls[-1]["result_details"]
    assert node.parameter_output_values["model_url"] is None


@pytest.mark.asyncio
async def test_save_model_files_saves_the_prefix_when_it_keeps_the_requested_format(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    files = [
        {"name": "model.glb", "url": "https://a"},
        {"name": "texture.png", "url": "https://b"},
    ]
    model_artifact = _FakeArtifact(index=0)

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        # The proxy truncates by dropping a tail, so one hosted artifact for two
        # reported files is the model, and saving it beats discarding it.
        return [model_artifact]

    async def fake_download_artifact(_artifact: _FakeArtifact) -> bytes:
        return b"model-bytes"

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)
    _stub_destinations(monkeypatch)
    status_calls = _stub_status(node, monkeypatch)

    with caplog.at_level(logging.WARNING, logger="griptape_nodes"):
        await node._save_model_files(files, {"geometry_file_format": "glb"}, "gen-1")

    assert status_calls[-1]["was_successful"] is True
    model_out = node.parameter_output_values["model_url"]
    assert model_out.value == "project/files/model.glb"
    assert node.parameter_output_values["preview_image"] is None
    assert any("hosts 1 of Rodin's 2 file(s)" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_save_model_files_refuses_when_the_prefix_drops_the_requested_format(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"name": "preview.webp", "url": "https://a"},
        {"name": "model.glb", "url": "https://b"},
    ]

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        # Truncation keeps only the preview; the requested .glb falls off the tail.
        return [_FakeArtifact(index=0, kind="image")]

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    status_calls = _stub_status(node, monkeypatch)

    await node._save_model_files(files, {"geometry_file_format": "glb"}, "gen-1")

    assert status_calls[-1]["was_successful"] is False
    assert "none of them that one" in status_calls[-1]["result_details"]
    assert node.parameter_output_values["model_url"] is None


@pytest.mark.asyncio
async def test_save_model_files_picks_the_largest_file_as_primary(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = [
        {"name": "model_a.glb", "url": "https://a"},
        {"name": "model_b.glb", "url": "https://b"},
    ]
    small_artifact = _FakeArtifact(index=0)
    large_artifact = _FakeArtifact(index=1)

    async def fake_hosted_artifacts(_generation_id: str) -> list[_FakeArtifact]:
        return [small_artifact, large_artifact]

    async def fake_download_artifact(artifact: _FakeArtifact) -> bytes:
        return b"x" * 10 if artifact is small_artifact else b"x" * 30

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts, raising=False)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact, raising=False)
    _stub_destinations(monkeypatch)
    status_calls = _stub_status(node, monkeypatch)

    await node._save_model_files(files, {"geometry_file_format": "glb"}, "gen-1")

    assert status_calls[-1]["was_successful"] is True
    model_out = node.parameter_output_values["model_url"]
    assert model_out.value == "project/files/model_b.glb"
    assert model_out.meta["filename"] == "model_b.glb"
