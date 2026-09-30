from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files import file as file_module
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.video.grok_video_edit import GrokVideoEdit

RESOLVED = "data:video/mp4;base64,RESOLVED"


def _node(duration: float | None = 5.0) -> GrokVideoEdit:
    node = GrokVideoEdit.__new__(GrokVideoEdit)
    node.name = "GrokVideoEdit"
    node._probe_duration = lambda _value: duration  # type: ignore[method-assign]
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


@pytest.mark.asyncio
class TestInputLimits:
    @pytest.mark.parametrize(
        "video_input",
        ["{inputs}/videos/IMG_2689.mov", "data:video/quicktime;base64,ABC", "https://example.com/clip.webm?sig=1"],
        ids=["mov-path", "quicktime-data-uri", "webm-url"],
    )
    async def test_non_mp4_container_raises(self, video_input: str, captured_paths: list[str]) -> None:
        with pytest.raises(
            ValueError,
            match=r"^Grok can't edit this video\. It's a \.\w+ file\. Grok needs \.mp4\. Use a Trim Video node",
        ):
            await _node()._prepare_video_data_uri(video_input)
        assert captured_paths == []

    async def test_url_without_extension_is_allowed(self, captured_paths: list[str]) -> None:
        assert await _node()._prepare_video_data_uri("https://example.com/signed/abc123") == RESOLVED
        assert captured_paths == ["https://example.com/signed/abc123"]

    async def test_over_long_video_raises_with_duration(self, captured_paths: list[str]) -> None:
        with pytest.raises(ValueError, match=r"It's 8\.80 seconds long\. Grok's limit is 8\.7 seconds\."):
            await _node(duration=8.798)._prepare_video_data_uri("{inputs}/clip.mp4")
        assert captured_paths == []

    async def test_wrong_container_and_too_long_are_reported_together(self, captured_paths: list[str]) -> None:
        with pytest.raises(ValueError, match=r"It's a \.mov file\. Grok needs \.mp4\. It's 8\.80 seconds long\."):
            await _node(duration=8.798)._prepare_video_data_uri("{inputs}/IMG_2689.mov")
        assert captured_paths == []

    async def test_video_at_limit_is_allowed(self, captured_paths: list[str]) -> None:
        assert await _node(duration=8.7)._prepare_video_data_uri("{inputs}/clip.mp4") == RESOLVED
        assert captured_paths == ["{inputs}/clip.mp4"]

    async def test_unknown_duration_is_allowed(self, captured_paths: list[str]) -> None:
        assert await _node(duration=None)._prepare_video_data_uri("{inputs}/clip.mp4") == RESOLVED
        assert captured_paths == ["{inputs}/clip.mp4"]


def test_probe_failure_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_resolve(self: file_module.File) -> str:
        raise FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "no project")

    monkeypatch.setattr(file_module.File, "resolve", failing_resolve)
    node = GrokVideoEdit.__new__(GrokVideoEdit)
    node.name = "GrokVideoEdit"
    assert node._probe_duration("{inputs}/clip.mp4") is None
