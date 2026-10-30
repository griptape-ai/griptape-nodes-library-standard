"""Tests for Wan Animate's pre-flight input checks against DashScope's documented limits."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.exe_types.validators._runner import validate_node
from griptape_nodes.utils import media_probe
from griptape_nodes.utils.media_probe import ImageInfo, VideoProbe
from PIL import Image

from griptape_nodes_library.video.wan_animate_generation import WanAnimateGeneration


@pytest.fixture
def node() -> WanAnimateGeneration:
    return WanAnimateGeneration(name="WanAnimate")


@pytest.fixture
def video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A local video whose probed size and duration a test can set."""
    path = tmp_path / "reference.mp4"
    path.write_bytes(b"video")
    probe: dict[str, Any] = {
        "path": str(path),
        "info": ImageInfo(1280, 720, None),
        "seconds": 5.0,
        "mime_type": "video/mp4",
    }
    # The validators probe through the engine's own reader, so patch it there.
    monkeypatch.setattr(
        media_probe,
        "local_video_probe",
        lambda _value: VideoProbe(info=probe["info"], seconds=probe["seconds"], mime_type=probe["mime_type"]),
    )
    return probe


def _image(tmp_path: Path, size: tuple[int, int] = (800, 600), image_format: str = "PNG") -> str:
    path = tmp_path / f"source.{image_format.lower()}"
    Image.new("RGB", size).save(path, format=image_format)
    return str(path)


def _all_errors(node: WanAnimateGeneration) -> list[Exception]:
    """What the engine reports at the node's turn: its pre-run validators', then its own checks'."""
    return validate_node(node)


def _messages(node: WanAnimateGeneration) -> list[str]:
    return [str(e) for e in _all_errors(node) if isinstance(e, NodeError)]


def _oversized(monkeypatch: pytest.MonkeyPatch, suffix: str) -> None:
    """Make local files ending in `suffix` read as 1 GB, over each DashScope size limit, without writing one."""
    read_size = media_probe.local_size_bytes
    monkeypatch.setattr(
        media_probe,
        "local_size_bytes",
        lambda value, *, kind: 1024**3 if str(value).endswith(suffix) else read_size(value, kind=kind),
    )


def test_accepts_inputs_within_every_limit(node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any]) -> None:
    node.set_parameter_value("image_url", _image(tmp_path, image_format="WEBP"))
    node.set_parameter_value("video_url", video["path"])
    assert _all_errors(node) == []


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (
            (150, 300),
            "The image in 'Image URL' is 150x300 and each side must be at least 200 pixels. Use a larger one.",
        ),
        (
            (5000, 2000),
            "The image in 'Image URL' is 5000x2000 and each side must be at most 4096 pixels. Resize it smaller.",
        ),
        (
            (1000, 300),
            "The image in 'Image URL' is 1000x300, which is too wide. The aspect ratio must be between 1:3 and 3:1. "
            "Crop it.",
        ),
    ],
    ids=["too-small", "too-large", "too-wide"],
)
def test_rejects_an_image_outside_the_pixel_limits(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], size: tuple[int, int], expected: str
) -> None:
    node.set_parameter_value("image_url", _image(tmp_path, size))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == [expected]


def test_rejects_an_unsupported_image_format(node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any]) -> None:
    node.set_parameter_value("image_url", _image(tmp_path, image_format="GIF"))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == ["The image in 'Image URL' is in GIF format. Convert it to JPEG, PNG, BMP or WEBP."]


def test_rejects_an_oversized_image(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _oversized(monkeypatch, ".png")
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    messages = _messages(node)
    assert len(messages) == 1
    assert messages[0].startswith("The image in 'Image URL' is ")
    assert "and the limit is" in messages[0]


@pytest.mark.parametrize("mime_type", ["video/mp4", "video/x-msvideo", "video/quicktime"])
def test_accepts_each_documented_video_format(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], mime_type: str
) -> None:
    video["mime_type"] = mime_type
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == []


def test_rejects_an_unsupported_video_format(node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any]) -> None:
    video["mime_type"] = "video/webm"
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == ["The video in 'Video URL' is in WebM format. Convert it to MP4, AVI or MOV."]


def test_rejects_an_oversized_video(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _oversized(monkeypatch, ".mp4")
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    messages = _messages(node)
    assert len(messages) == 1
    assert messages[0].startswith("The video in 'Video URL' is ")
    assert "and the limit is" in messages[0]


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (1.5, "The video in 'Video URL' is 1.50 seconds long and the minimum is 2. Use a longer clip."),
        (
            31.0,
            "The video in 'Video URL' is 31.00 seconds long and the limit is 30. Shorten it.",
        ),
    ],
    ids=["too-short", "too-long"],
)
def test_rejects_a_video_outside_the_duration_limits(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], seconds: float, expected: str
) -> None:
    video["seconds"] = seconds
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == [expected]


def test_rejects_a_video_outside_the_pixel_limits(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any]
) -> None:
    video["info"] = ImageInfo(3840, 2160, None)
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == [
        "The video in 'Video URL' is 3840x2160 and each side must be at most 2048 pixels. Resize it smaller."
    ]


def test_rejects_a_video_that_is_too_tall(node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any]) -> None:
    video["info"] = ImageInfo(300, 1000, None)
    node.set_parameter_value("image_url", _image(tmp_path))
    node.set_parameter_value("video_url", video["path"])
    assert _messages(node) == [
        "The video in 'Video URL' is 300x1000, which is too tall. The aspect ratio must be between 1:3 and 3:1. "
        "Crop it."
    ]


def test_skips_remote_inputs(node: WanAnimateGeneration, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("a remote video must not be probed"))
    node.set_parameter_value("image_url", "https://example.com/source.gif")
    node.set_parameter_value("video_url", "https://example.com/reference.mp4")
    assert _all_errors(node) == []


@pytest.mark.parametrize(
    ("model", "page"),
    [("wan2.2-animate-mix", "wan-animate-mix-api"), ("wan2.2-animate-move", "wan-animate-move-api")],
)
def test_errors_link_the_selected_models_page(
    node: WanAnimateGeneration, tmp_path: Path, video: dict[str, Any], model: str, page: str
) -> None:
    node.set_parameter_value("model", model)
    node.set_parameter_value("image_url", _image(tmp_path, image_format="GIF"))
    node.set_parameter_value("video_url", video["path"])
    errors = [e for e in _all_errors(node) if isinstance(e, NodeError)]
    assert len(errors) == 1
    assert errors[0].links
    assert errors[0].links[0].url.endswith(page)
