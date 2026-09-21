from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor
from griptape_nodes_library.video.resize_video import ResizeVideo
from griptape_nodes_library.video.split_video import SplitVideo
from griptape_nodes_library.video.trim_video import TrimVideo
from griptape_nodes_library.video.video_color_match import VideoColorMatch

# The shape the editor sends for a video: an artifact-like dict, not a VideoUrlArtifact.
# User-defined parameters (the "Add parameter" slots on start/end nodes) are plain
# Parameters with no converters, so this dict reaches the consuming node verbatim.
EDITOR_VIDEO_DICT = {
    "type": "VideoUrlArtifact",
    "value": "http://example.com/clip.mp4",
    "name": "clip.mp4",
    "width": 1920,
    "height": 1080,
    "duration": 12,
}


class _StubVideoProcessor(BaseVideoProcessor):
    """Minimal concrete subclass so the base class's video input can be tested without a real ffmpeg pipeline."""

    def _setup_custom_parameters(self) -> None:
        pass

    def _get_processing_description(self) -> str:
        return "stub processing"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        return []


VIDEO_INPUT_NODES = [
    (ResizeVideo, "video"),
    (TrimVideo, "video"),
    (SplitVideo, "video"),
    (VideoColorMatch, "target_video"),
    (_StubVideoProcessor, "video"),
]


@pytest.mark.parametrize(("node_class", "parameter_name"), VIDEO_INPUT_NODES)
def test_editor_video_dict_is_converted_to_artifact(node_class: type, parameter_name: str) -> None:
    node = node_class(name=node_class.__name__)

    node.set_parameter_value(parameter_name, EDITOR_VIDEO_DICT)

    value = node.parameter_values[parameter_name]
    assert isinstance(value, VideoUrlArtifact)
    assert value.value == EDITOR_VIDEO_DICT["value"]


@pytest.mark.parametrize(("node_class", "parameter_name"), VIDEO_INPUT_NODES)
@pytest.mark.parametrize(
    "incoming",
    [
        pytest.param(VideoUrlArtifact("http://example.com/clip.mp4"), id="artifact"),
        pytest.param("http://example.com/clip.mp4", id="url-string"),
    ],
)
def test_already_normalized_video_input_is_preserved(node_class: type, parameter_name: str, incoming: Any) -> None:
    node = node_class(name=node_class.__name__)

    node.set_parameter_value(parameter_name, incoming)

    value = node.parameter_values[parameter_name]
    assert isinstance(value, VideoUrlArtifact)
    assert value.value == "http://example.com/clip.mp4"


@pytest.mark.parametrize("node_class", [ResizeVideo, TrimVideo, SplitVideo])
def test_editor_video_dict_passes_video_validation(node_class: type) -> None:
    node = node_class(name=node_class.__name__)
    node.set_parameter_value("video", EDITOR_VIDEO_DICT)

    exceptions = node.validate_before_node_run() or []

    assert [str(e) for e in exceptions if "Video parameter" in str(e)] == []
