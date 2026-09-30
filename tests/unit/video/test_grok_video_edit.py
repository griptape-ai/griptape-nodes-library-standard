from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files import file as file_module
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.video.grok_video_edit import GrokVideoEdit

RESOLVED = "data:video/mp4;base64,RESOLVED"


def _node() -> GrokVideoEdit:
    node = GrokVideoEdit.__new__(GrokVideoEdit)
    node.name = "GrokVideoEdit"
    return node


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    paths: list[str] = []
    original_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: str, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        original_init(self, path, *args, **kwargs)

    async def fake_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:  # noqa: ARG001
        return RESOLVED

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "aread_data_uri", fake_aread)
    return paths


@pytest.mark.asyncio
class TestPrepareVideoDataUri:
    @pytest.mark.parametrize(
        "video_input",
        [
            "{outputs}/clip.mp4",
            VideoUrlArtifact(value="{outputs}/clip.mp4"),
            {"type": "VideoUrlArtifact", "value": "{outputs}/clip.mp4"},
        ],
        ids=["string", "artifact", "serialized-dict"],
    )
    async def test_macro_path_is_handed_to_file(self, video_input: Any, captured_paths: list[str]) -> None:
        assert await _node()._prepare_video_data_uri(video_input) == RESOLVED
        assert captured_paths == ["{outputs}/clip.mp4"]

    async def test_data_uri_passes_through(self, captured_paths: list[str]) -> None:
        assert await _node()._prepare_video_data_uri("data:video/mp4;base64,ABC") == "data:video/mp4;base64,ABC"
        assert captured_paths == []

    async def test_load_failure_raises_with_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing_aread(self: file_module.File, fallback_mime: str = "application/octet-stream") -> str:  # noqa: ARG001
            raise FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "file not found")

        monkeypatch.setattr(file_module.File, "aread_data_uri", failing_aread)

        with pytest.raises(ValueError, match=r"\{outputs\}/missing\.mp4"):
            await _node()._prepare_video_data_uri("{outputs}/missing.mp4")

    async def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="no usable value"):
            await _node()._prepare_video_data_uri(VideoUrlArtifact(value=""))
