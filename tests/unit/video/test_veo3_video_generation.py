"""Tests for how Veo3VideoGeneration maps a hosted video's position onto its

output-file index.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.proxy import ArtifactKind
from griptape_nodes_library.video.veo3_video_generation import Veo3VideoGeneration


class _SavedFile:
    def __init__(self, name: str) -> None:
        self.name = name
        self.location = f"project/files/{name}"


class _FakeDest:
    def __init__(self, index: int) -> None:
        self._index = index

    async def awrite_bytes(self, _data: bytes) -> _SavedFile:
        return _SavedFile(f"video_{self._index}.mp4")


class _FakeOutputFile:
    def __init__(self) -> None:
        self.built_indexes: list[int] = []

    def build_file(self, *, _index: int) -> _FakeDest:
        self.built_indexes.append(_index)
        return _FakeDest(_index)


@pytest.fixture
def node() -> Veo3VideoGeneration:
    instance = Veo3VideoGeneration.__new__(Veo3VideoGeneration)
    instance.name = "Veo3"  # type: ignore[misc]
    instance._output_file = _FakeOutputFile()  # type: ignore[assignment]
    return instance


@pytest.mark.asyncio
async def test_save_videos_maps_position_to_a_one_based_output_index(
    node: Veo3VideoGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # _save_videos reads hosted videos at positions 0..count-1 and writes each to
    # the output file at index position + 1, in the same order.
    requests: list[dict[str, Any]] = []

    async def fake_load_generated_media(generation_id: str, *, kind: str | None = None, position: int = 0) -> bytes:
        requests.append({"generation_id": generation_id, "kind": kind, "position": position})
        return f"bytes-{position}".encode()

    monkeypatch.setattr(node, "_load_generated_media", fake_load_generated_media, raising=False)

    video_artifacts = await node._save_videos("gen-1", 2)

    assert [request["position"] for request in requests] == [0, 1]
    assert all(request["kind"] == ArtifactKind.VIDEO for request in requests)
    assert all(request["generation_id"] == "gen-1" for request in requests)
    assert cast("_FakeOutputFile", node._output_file).built_indexes == [1, 2]
    assert [artifact.name if artifact else None for artifact in video_artifacts] == ["video_1.mp4", "video_2.mp4"]


@pytest.mark.asyncio
async def test_save_videos_leaves_a_failed_position_empty(
    node: Veo3VideoGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A failure reading one position must not stop the others from being read at
    # their own, unshifted position and output index, and must not pull a later
    # video into the failed video's output slot.
    async def fake_load_generated_media(_generation_id: str, *, kind: str | None = None, position: int = 0) -> bytes:
        if position == 0:
            msg = "not hosted"
            raise RuntimeError(msg)
        return f"bytes-{position}".encode()

    monkeypatch.setattr(node, "_load_generated_media", fake_load_generated_media, raising=False)

    video_artifacts = await node._save_videos("gen-1", 2)

    assert cast("_FakeOutputFile", node._output_file).built_indexes == [2]
    assert [artifact.name if artifact else None for artifact in video_artifacts] == [None, "video_2.mp4"]


def test_set_video_output_parameters_keeps_a_failed_slot_empty(
    node: Veo3VideoGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    shown: list[int] = []
    outputs: dict[str, Any] = {}
    monkeypatch.setattr(node, "parameter_output_values", outputs, raising=False)
    monkeypatch.setattr(node, "_show_video_output_parameters", shown.append, raising=False)
    monkeypatch.setattr(
        node,
        "_set_status_results",
        lambda *, was_successful, result_details: outputs.update(
            {"was_successful": was_successful, "result_details": result_details}
        ),
        raising=False,
    )
    second = VideoUrlArtifact(value="project/files/video_2.mp4", name="video_2.mp4")

    node._set_video_output_parameters([None, second])

    assert shown == [2]
    assert outputs["video_url"] is None
    assert outputs["video_url_2"] is second
    assert outputs["was_successful"] is True
    assert "Video(s) 1 could not be retrieved" in outputs["result_details"]
