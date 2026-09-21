from __future__ import annotations

import base64
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


# A minimal MP4 header is enough: the conversion only base64-decodes the payload and
# writes it out, it never parses the container.
INLINE_VIDEO_B64 = base64.b64encode(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 64).decode()

# Dict shapes the conversion understands. The editor sends the first for a stored video
# and the base64 variants for an inline one, so all of these must keep converting.
CONVERTIBLE_VIDEO_DICTS = [
    pytest.param(EDITOR_VIDEO_DICT, id="artifact-dict"),
    pytest.param({"type": "video/mp4", "value": INLINE_VIDEO_B64}, id="base64-mime-dict"),
    pytest.param({"type": "video/mp4", "value": f"data:video/mp4;base64,{INLINE_VIDEO_B64}"}, id="base64-data-url"),
    pytest.param({"value": INLINE_VIDEO_B64}, id="base64-no-type"),
]

# ParameterVideo accepts any input type, so a dict that isn't a video at all can arrive
# from a connection. The conversion cannot handle these — it would raise KeyError on a
# missing "value" or a base64 error on a non-video "type" — so they must pass through
# untouched and be reported by the node's own validation instead.
UNCONVERTIBLE_DICTS = [
    pytest.param({"foo": "bar"}, id="no-value-key"),
    pytest.param({}, id="empty"),
    pytest.param({"type": "ImageUrlArtifact", "value": "http://example.com/x.png"}, id="image-dict"),
    pytest.param({"value": "/opt/media/clip.mp4"}, id="path-without-type"),
]


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


@pytest.mark.parametrize(("node_class", "parameter_name"), VIDEO_INPUT_NODES)
@pytest.mark.parametrize("video_dict", CONVERTIBLE_VIDEO_DICTS)
def test_convertible_video_dicts_are_converted(node_class: type, parameter_name: str, video_dict: dict) -> None:
    """Every dict shape the conversion understands still converts, base64 variants included."""
    node = node_class(name=node_class.__name__)

    node.set_parameter_value(parameter_name, dict(video_dict))

    assert isinstance(node.parameter_values[parameter_name], VideoUrlArtifact)


@pytest.mark.parametrize(("node_class", "parameter_name"), VIDEO_INPUT_NODES)
@pytest.mark.parametrize("foreign_dict", UNCONVERTIBLE_DICTS)
def test_unconvertible_dict_passes_through_unchanged(node_class: type, parameter_name: str, foreign_dict: dict) -> None:
    """A dict that isn't a video must not raise out of the converter.

    Before the converter was added these nodes stored the dict and let validation
    reject it. Raising here would surface a bare KeyError/base64 error at connect time.
    """
    node = node_class(name=node_class.__name__)

    node.set_parameter_value(parameter_name, dict(foreign_dict))

    assert node.parameter_values[parameter_name] == foreign_dict


@pytest.mark.parametrize("node_class", [ResizeVideo, TrimVideo, SplitVideo, _StubVideoProcessor])
@pytest.mark.parametrize("foreign_dict", UNCONVERTIBLE_DICTS)
def test_unconvertible_dict_is_reported_by_validation(node_class: type, foreign_dict: dict) -> None:
    """The pass-through hands the type mismatch to validation, which reports it as a video error.

    An empty dict is falsy, so some nodes report it as missing rather than mistyped;
    either message is a clear rejection, which is what the converter must preserve.
    """
    node = node_class(name=node_class.__name__)
    node.set_parameter_value("video", dict(foreign_dict))

    exceptions = node.validate_before_node_run() or []

    assert [str(e) for e in exceptions if "Video parameter" in str(e)] != []
