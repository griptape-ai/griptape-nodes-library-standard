"""Tests for the Kling nodes' pre-flight media checks against Kling's documented limits."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from PIL import Image

from griptape_nodes_library.proxy import preflight
from griptape_nodes_library.proxy.preflight import ImageInfo, VideoProbe
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_motion_control import KlingMotionControl
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration


def _image(tmp_path: Path, size: tuple[int, int] = (1280, 720), image_format: str = "PNG", name: str = "") -> str:
    path = tmp_path / (name or f"{size[0]}x{size[1]}.{image_format.lower()}")
    Image.new("RGB", size).save(path, format=image_format)
    return str(path)


def _messages(node: Any) -> list[str]:
    return [str(e) for e in node.validate_before_node_run() or [] if isinstance(e, NodeError)]


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A local video whose probed frame size and duration a test can set, for any Kling node."""
    path = tmp_path / "reference.mp4"
    path.write_bytes(b"video")
    probe: dict[str, Any] = {"path": str(path), "info": ImageInfo(1280, 720, None), "seconds": 5.0}
    # video_errors probes through preflight's own reader, so patch it there.
    monkeypatch.setattr(
        preflight, "local_video_probe", lambda _value: VideoProbe(info=probe["info"], seconds=probe["seconds"])
    )
    return probe


class TestImageToVideo:
    @pytest.fixture
    def node(self) -> KlingImageToVideoGeneration:
        node = KlingImageToVideoGeneration(name="KlingImageToVideo")
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_a_valid_start_frame(self, node: KlingImageToVideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("image", _image(tmp_path, image_format="JPEG"))
        assert _messages(node) == []

    @pytest.mark.parametrize(
        ("size", "image_format", "expected"),
        [
            (
                (1280, 720),
                "GIF",
                "The image in 'Start Frame' is a GIF file. Convert it to JPEG or PNG.",
            ),
            (
                (200, 400),
                "PNG",
                "The image in 'Start Frame' is 200x400 and each side must be at least 300 pixels. Use a larger one.",
            ),
            (
                (3000, 1000),
                "PNG",
                "The image in 'Start Frame' is 3000x1000, which is too wide. "
                "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
            ),
        ],
        ids=["format", "too-small", "too-wide"],
    )
    def test_rejects_a_start_frame_outside_the_limits(
        self,
        node: KlingImageToVideoGeneration,
        tmp_path: Path,
        size: tuple[int, int],
        image_format: str,
        expected: str,
    ) -> None:
        node.set_parameter_value("image", _image(tmp_path, size, image_format))
        assert _messages(node) == [expected]

    def test_rejects_an_oversized_start_frame(
        self, node: KlingImageToVideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[KlingImageToVideoGeneration.__module__], "MAX_IMAGE_BYTES", 10)
        node.set_parameter_value("image", _image(tmp_path))
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The image in 'Start Frame' is ")
        assert "and the limit is" in messages[0]

    def test_does_not_check_the_end_frames_aspect_ratio(
        self, node: KlingImageToVideoGeneration, tmp_path: Path
    ) -> None:
        node.set_parameter_value("mode", "pro")
        node.set_parameter_value("image_tail", _image(tmp_path, (3000, 1000)))
        assert not any("aspect ratio" in message for message in _messages(node))

    def test_checks_the_end_frames_format_and_size(self, node: KlingImageToVideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("image_tail", _image(tmp_path, (200, 400), "GIF"))
        messages = _messages(node)
        assert "The image in 'End Frame' is a GIF file. Convert it to JPEG or PNG." in messages
        assert (
            "The image in 'End Frame' is 200x400 and each side must be at least 300 pixels. Use a larger one."
            in messages
        )

    def test_errors_link_klings_page(self, node: KlingImageToVideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("image", _image(tmp_path, image_format="GIF"))
        errors = [e for e in node.validate_before_node_run() or [] if isinstance(e, NodeError)]
        assert errors[0].links
        assert errors[0].links[0].url == "https://kling.ai/document-api/api/video/image-to-video"

    def test_skips_a_remote_frame(self, node: KlingImageToVideoGeneration) -> None:
        node.set_parameter_value("image", "https://example.com/frame.gif")
        assert _messages(node) == []


class TestOmni:
    @pytest.fixture
    def node(self, monkeypatch: pytest.MonkeyPatch) -> KlingOmniVideoGeneration:
        node = KlingOmniVideoGeneration(name="KlingOmni")
        node.set_parameter_value("prompt", "a cat")
        # A ParameterList reads its items from child parameters; with no connections in a unit
        # test, read the value set on the list itself.
        monkeypatch.setattr(node, "get_parameter_list_value", lambda name: node.parameter_values.get(name) or [])
        return node

    def test_accepts_valid_media(self, node: KlingOmniVideoGeneration, tmp_path: Path, video: dict[str, Any]) -> None:
        node.set_parameter_value("first_frame_image", _image(tmp_path, name="first.png"))
        node.set_parameter_value("reference_images", [_image(tmp_path, image_format="JPEG", name="ref.jpeg")])
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == []

    def test_rejects_an_invalid_first_frame(self, node: KlingOmniVideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("first_frame_image", _image(tmp_path, (200, 400)))
        assert _messages(node) == [
            "The image in 'first frame' is 200x400 and each side must be at least 300 pixels. Use a larger one."
        ]

    def test_names_the_position_of_an_invalid_reference_image(
        self, node: KlingOmniVideoGeneration, tmp_path: Path
    ) -> None:
        node.set_parameter_value(
            "reference_images",
            [_image(tmp_path, name="ok.png"), _image(tmp_path, (1280, 720), "GIF", name="bad.gif")],
        )
        assert _messages(node) == ["Image 2 in 'reference images' is a GIF file. Convert it to JPEG or PNG."]

    @pytest.mark.parametrize(
        ("info", "seconds", "expected"),
        [
            (
                ImageInfo(1280, 720, None),
                2.0,
                "The video in 'reference_video' is 2.00 seconds long and the minimum is 3. Use a longer clip.",
            ),
            (
                ImageInfo(1280, 720, None),
                20.0,
                "The video in 'reference_video' is 20.00 seconds long and the limit is 15.5. "
                "Use a Trim Video node to shorten it.",
            ),
            (
                ImageInfo(640, 480, None),
                5.0,
                "The video in 'reference_video' is 640x480 and each side must be at least 700 pixels. "
                "Use a larger one.",
            ),
            (
                ImageInfo(3000, 1000, None),
                5.0,
                "The video in 'reference_video' is 3000x1000, which is too wide. "
                "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
            ),
        ],
        ids=["too-short", "too-long", "too-small", "too-wide"],
    )
    def test_rejects_a_reference_video_outside_the_limits(
        self,
        node: KlingOmniVideoGeneration,
        video: dict[str, Any],
        info: ImageInfo,
        seconds: float,
        expected: str,
    ) -> None:
        video["info"], video["seconds"] = info, seconds
        node.set_parameter_value("reference_video", video["path"])
        assert expected in _messages(node)

    def test_accepts_a_reference_video_inside_both_kling_pages_bounds(
        self, node: KlingOmniVideoGeneration, video: dict[str, Any]
    ) -> None:
        # 3000 px is over the O1 page's 2160 px but under the 3.0 Omni page's 4553 px; the pages
        # disagree, so only what both refuse is checked.
        video["info"] = ImageInfo(3000, 2000, None)
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == []

    def test_accepts_a_reference_video_exactly_on_the_area_limit(
        self, node: KlingOmniVideoGeneration, video: dict[str, Any]
    ) -> None:
        # 3840x2160 is exactly 8,294,400 px, the most the 3.0 Omni page allows.
        video["info"] = ImageInfo(3840, 2160, None)
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == []

    @pytest.mark.parametrize(
        ("info", "expected"),
        [
            (ImageInfo(4000, 2100, None), "megapixels"),
            (ImageInfo(4600, 2000, None), "each side must be at most 4553 pixels"),
        ],
        ids=["over-the-area-limit", "side-over-4553"],
    )
    def test_rejects_a_reference_video_over_the_size_limits(
        self, node: KlingOmniVideoGeneration, video: dict[str, Any], info: ImageInfo, expected: str
    ) -> None:
        video["info"] = info
        node.set_parameter_value("reference_video", video["path"])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The video in 'reference_video' is ")
        assert expected in messages[0]

    def test_rejects_an_oversized_reference_image(
        self, node: KlingOmniVideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[KlingOmniVideoGeneration.__module__], "MAX_IMAGE_BYTES", 10)
        node.set_parameter_value("reference_images", [_image(tmp_path)])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("Image 1 in 'reference images' is ")
        assert "and the limit is" in messages[0]

    def test_rejects_an_oversized_reference_video(
        self, node: KlingOmniVideoGeneration, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[KlingOmniVideoGeneration.__module__], "MAX_VIDEO_BYTES", 1)
        node.set_parameter_value("reference_video", video["path"])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The video in 'reference_video' is ")
        assert "and the limit is" in messages[0]


class TestMotionControl:
    @pytest.fixture
    def node(self) -> KlingMotionControl:
        return KlingMotionControl(name="KlingMotionControl")

    def test_accepts_valid_media(self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any]) -> None:
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == []

    def test_rejects_an_invalid_reference_image(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any]
    ) -> None:
        node.set_parameter_value("reference_image", _image(tmp_path, (3000, 1000), "GIF"))
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == [
            "The image in 'reference_image' is a GIF file. Convert it to JPEG or PNG.",
            "The image in 'reference_image' is 3000x1000, which is too wide. "
            "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
        ]

    @pytest.mark.parametrize(
        ("orientation", "seconds", "expected"),
        [
            ("video", 25.0, None),
            ("video", 31.0, "is 31.00 seconds long and the limit is 30."),
            ("image", 11.0, "is 11.00 seconds long and the limit is 10."),
        ],
        ids=["video-orientation-25s", "video-orientation-31s", "image-orientation-11s"],
    )
    def test_the_longest_clip_depends_on_character_orientation(
        self,
        node: KlingMotionControl,
        tmp_path: Path,
        video: dict[str, Any],
        orientation: str,
        seconds: float,
        expected: str | None,
    ) -> None:
        video["seconds"] = seconds
        node.set_parameter_value("character_orientation", orientation)
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        messages = _messages(node)
        if expected is None:
            assert messages == []
        else:
            assert len(messages) == 1
            assert expected in messages[0]

    def test_rejects_a_reference_video_outside_the_pixel_limits(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any]
    ) -> None:
        video["info"] = ImageInfo(160, 120, None)
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == [
            "The video in 'reference_video' is 160x120 and each side must be at least 340 pixels. Use a larger one."
        ]

    def test_rejects_an_oversized_reference_image(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[KlingMotionControl.__module__], "MAX_IMAGE_BYTES", 10)
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The image in 'reference_image' is ")
        assert "and the limit is" in messages[0]

    def test_rejects_a_reference_image_over_the_largest_side(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A real 65537 px image is too large to build in a test; lower the limit instead.
        monkeypatch.setattr(sys.modules[KlingMotionControl.__module__], "MAX_IMAGE_SIDE", 1000)
        node.set_parameter_value("reference_image", _image(tmp_path, (1200, 800)))
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == [
            "The image in 'reference_image' is 1200x800 and each side must be at most 1000 pixels. Resize it smaller."
        ]

    def test_rejects_an_oversized_reference_video(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[KlingMotionControl.__module__], "MAX_VIDEO_BYTES", 1)
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The video in 'reference_video' is ")
        assert "and the limit is" in messages[0]

    def test_rejects_a_reference_video_over_3850_pixels(
        self, node: KlingMotionControl, tmp_path: Path, video: dict[str, Any]
    ) -> None:
        video["info"] = ImageInfo(3851, 2000, None)
        node.set_parameter_value("reference_image", _image(tmp_path))
        node.set_parameter_value("reference_video", video["path"])
        assert _messages(node) == [
            "The video in 'reference_video' is 3851x2000 and each side must be at most 3850 pixels. Resize it smaller."
        ]

    def test_skips_remote_inputs(self, node: KlingMotionControl, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("a remote video must not be probed"))
        node.set_parameter_value("reference_image", "https://example.com/person.gif")
        node.set_parameter_value("reference_video", "https://example.com/dance.mp4")
        assert _messages(node) == []
