"""Tests for the Seedance nodes' pre-flight media checks against BytePlus's documented limits."""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError, ParameterList
from griptape_nodes.exe_types.input_checks import ImageInfo, VideoProbe
from PIL import Image

from griptape_nodes_library.assets import ASSET_KIND_IMAGE, ASSET_KIND_VIDEO, create_provider_asset_reference
from griptape_nodes_library.video import seedance_common
from griptape_nodes_library.video.seedance_2_0_video_generation import Seedance20VideoGeneration
from griptape_nodes_library.video.seedance_2_5_video_generation import Seedance25VideoGeneration, SeedanceTask
from griptape_nodes_library.video.seedance_video_generation import SeedanceVideoGeneration

SEEDANCE_1_5_PRO = "seedance-1-5-pro-251215"
SEEDANCE_1_0_PRO = "seedance-1-0-pro-250528"
SEEDANCE_1_0_PRO_FAST = "seedance-1-0-pro-fast-251015"
CREATE_TASK_PAGE = "https://docs.byteplus.com/en/docs/ModelArk/1520757"


def _image(tmp_path: Path, size: tuple[int, int] = (1280, 720), image_format: str = "PNG", name: str = "") -> str:
    path = tmp_path / (name or f"{size[0]}x{size[1]}.{image_format.lower()}")
    Image.new("RGB", size).save(path, format=image_format)
    return str(path)


def _audio(tmp_path: Path) -> str:
    """A one-second silent WAV file."""
    path = tmp_path / "voice.wav"
    with wave.open(str(path), "wb") as clip:
        clip.setnchannels(1)
        clip.setsampwidth(2)
        clip.setframerate(8000)
        clip.writeframes(b"\x00\x00" * 8000)
    return str(path)


def _errors(node: Any) -> list[NodeError]:
    return [e for e in node.validate_before_node_run() or [] if isinstance(e, NodeError)]


def _messages(node: Any) -> list[str]:
    return [str(e) for e in _errors(node)]


def _set_list(node: Any, parameter_name: str, values: list[object]) -> None:
    """Connect each value to its own child of a ParameterList, as the editor does."""
    parameter_list = next(
        parameter
        for parameter in node.parameters
        if isinstance(parameter, ParameterList) and parameter.name == parameter_name
    )
    parameter_list.clear_list()
    for value in values:
        node.set_parameter_value(parameter_list.add_child_parameter().name, value)


@pytest.fixture
def videos(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, VideoProbe | None]:
    """Local reference videos, each probed as the VideoProbe a test sets for its file name.

    A name a test hasn't set probes as a valid 5 s 1280x720 30 fps clip; one set to None reads as
    unreadable, as when ffprobe isn't installed.
    """
    probes: dict[str, VideoProbe | None] = {}
    valid = VideoProbe(info=ImageInfo(1280, 720, None), seconds=5.0, fps=30.0)

    def fake_probe(value: Any) -> VideoProbe | None:
        return probes.get(Path(str(value)).name, valid)

    # The Seedance checks probe through seedance_common's import of the reader, so patch it there.
    monkeypatch.setattr(seedance_common, "local_video_probe", fake_probe)
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        (tmp_path / name).write_bytes(b"video")
    return probes


class TestSeedanceVideoGeneration:
    @pytest.fixture
    def node(self) -> SeedanceVideoGeneration:
        return SeedanceVideoGeneration(name="Seedance")

    @pytest.mark.parametrize("image_format", ["JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"])
    def test_accepts_frames_in_every_documented_format(
        self, node: SeedanceVideoGeneration, tmp_path: Path, image_format: str
    ) -> None:
        node.set_parameter_value("first_frame", _image(tmp_path, image_format=image_format, name="first"))
        node.set_parameter_value("last_frame", _image(tmp_path, image_format=image_format, name="last"))
        assert _messages(node) == []

    @pytest.mark.parametrize(
        ("size", "image_format", "expected"),
        [
            (
                (1280, 720),
                "PPM",
                "The image in 'First Frame' is in PPM format. "
                "Convert it to JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC or HEIF.",
            ),
            (
                (200, 400),
                "PNG",
                "The image in 'First Frame' is 200x400 and each side must be at least 300 pixels. Use a larger one.",
            ),
            (
                (3000, 1000),
                "PNG",
                "The image in 'First Frame' is 3000x1000, which is too wide. "
                "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
            ),
            (
                (400, 1100),
                "PNG",
                "The image in 'First Frame' is 400x1100, which is too tall. "
                "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
            ),
        ],
        ids=["format", "too-small", "too-wide", "too-tall"],
    )
    def test_rejects_a_first_frame_outside_the_limits(
        self,
        node: SeedanceVideoGeneration,
        tmp_path: Path,
        size: tuple[int, int],
        image_format: str,
        expected: str,
    ) -> None:
        node.set_parameter_value("first_frame", _image(tmp_path, size, image_format))
        assert _messages(node) == [expected]

    def test_rejects_a_frame_over_the_largest_side(
        self, node: SeedanceVideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A real 6001 px image is slow to build in a test; lower the limit instead.
        monkeypatch.setattr(sys.modules[SeedanceVideoGeneration.__module__], "SEEDANCE_MAX_IMAGE_DIMENSION_PX", 1000)
        node.set_parameter_value("first_frame", _image(tmp_path, (1200, 800)))
        assert _messages(node) == [
            "The image in 'First Frame' is 1200x800 and each side must be at most 1000 pixels. Resize it smaller."
        ]

    def test_rejects_an_oversized_frame(
        self, node: SeedanceVideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(sys.modules[SeedanceVideoGeneration.__module__], "MAX_IMAGE_BYTES", 10)
        node.set_parameter_value("first_frame", _image(tmp_path))
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("The image in 'First Frame' is ")
        assert "and the limit is" in messages[0]

    @pytest.mark.parametrize("model", [SEEDANCE_1_5_PRO, SEEDANCE_1_0_PRO])
    def test_checks_the_last_frame_for_the_models_that_send_it(
        self, node: SeedanceVideoGeneration, tmp_path: Path, model: str
    ) -> None:
        node.set_parameter_value("model_id", model)
        node.set_parameter_value("last_frame", _image(tmp_path, (200, 400)))
        assert _messages(node) == [
            "The image in 'Last Frame' is 200x400 and each side must be at least 300 pixels. Use a larger one."
        ]

    def test_does_not_check_a_last_frame_the_request_does_not_send(
        self, node: SeedanceVideoGeneration, tmp_path: Path
    ) -> None:
        node.set_parameter_value("model_id", SEEDANCE_1_0_PRO_FAST)
        node.set_parameter_value("last_frame", _image(tmp_path, (200, 400), "PPM"))
        assert _messages(node) == []

    @pytest.mark.parametrize(
        ("model", "formats"),
        [
            (SEEDANCE_1_5_PRO, "JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC or HEIF"),
            (SEEDANCE_1_0_PRO, "JPEG, PNG, WEBP, BMP, TIFF or GIF"),
            (SEEDANCE_1_0_PRO_FAST, "JPEG, PNG, WEBP, BMP, TIFF or GIF"),
        ],
    )
    def test_lists_heic_and_heif_only_for_the_models_that_take_them(
        self, node: SeedanceVideoGeneration, tmp_path: Path, model: str, formats: str
    ) -> None:
        node.set_parameter_value("model_id", model)
        node.set_parameter_value("first_frame", _image(tmp_path, image_format="PPM"))
        assert _messages(node) == [f"The image in 'First Frame' is in PPM format. Convert it to {formats}."]

    def test_errors_link_byteplus_page(self, node: SeedanceVideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("first_frame", _image(tmp_path, image_format="PPM"))
        errors = _errors(node)
        assert errors[0].links
        assert errors[0].links[0].url == CREATE_TASK_PAGE

    def test_skips_a_frame_pillow_cannot_read(self, node: SeedanceVideoGeneration, tmp_path: Path) -> None:
        # Pillow has no HEIC decoder, so neither its format nor its size in pixels can be read.
        path = tmp_path / "first.heic"
        path.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00")
        node.set_parameter_value("first_frame", str(path))
        assert _messages(node) == []

    def test_skips_remote_frames(self, node: SeedanceVideoGeneration) -> None:
        node.set_parameter_value("first_frame", "https://example.com/first.ppm")
        node.set_parameter_value("last_frame", "https://example.com/last.ppm")
        assert _messages(node) == []


class TestSeedance20:
    @pytest.fixture
    def node(self) -> Seedance20VideoGeneration:
        return Seedance20VideoGeneration(name="Seedance20")

    @pytest.fixture
    def references(self, node: Seedance20VideoGeneration) -> Seedance20VideoGeneration:
        node.set_parameter_value("input_mode", "Multimodal References")
        return node

    def test_accepts_valid_frames(self, node: Seedance20VideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("input_mode", "First/Last Frame")
        node.set_parameter_value("first_frame", _image(tmp_path, image_format="WEBP", name="first"))
        node.set_parameter_value("last_frame", _image(tmp_path, image_format="TIFF", name="last"))
        assert _messages(node) == []

    def test_accepts_valid_references(
        self,
        references: Seedance20VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],  # noqa: ARG002
    ) -> None:
        _set_list(references, "reference_images", [_image(tmp_path, image_format="BMP")])
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        _set_list(references, "reference_audio", [_audio(tmp_path)])
        assert _messages(references) == []

    def test_rejects_invalid_frames(self, node: Seedance20VideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("input_mode", "First/Last Frame")
        node.set_parameter_value("first_frame", _image(tmp_path, image_format="PPM", name="first"))
        node.set_parameter_value("last_frame", _image(tmp_path, (3000, 1000), name="last"))
        assert _messages(node) == [
            "The image in 'First Frame' is in PPM format. Convert it to JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC or HEIF.",
            "The image in 'Last Frame' is 3000x1000, which is too wide. "
            "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
        ]

    def test_names_the_position_of_an_invalid_reference_image(
        self, references: Seedance20VideoGeneration, tmp_path: Path
    ) -> None:
        _set_list(
            references, "reference_images", [_image(tmp_path, name="ok.png"), _image(tmp_path, (200, 400), name="bad")]
        )
        assert _messages(references) == [
            "Image 2 in 'Reference Images' is 200x400 and each side must be at least 300 pixels. Use a larger one."
        ]

    def test_rejects_an_oversized_reference_image(
        self, references: Seedance20VideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(seedance_common, "MAX_IMAGE_BYTES", 10)
        _set_list(references, "reference_images", [_image(tmp_path)])
        messages = _messages(references)
        assert len(messages) == 1
        assert messages[0].startswith("Image 1 in 'Reference Images' is ")
        assert "and the limit is" in messages[0]

    def test_checks_only_the_shortest_side_and_shape_of_an_image_it_will_downscale(
        self, references: Seedance20VideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The node downscales a local image over 6000 px and re-encodes it before sending, so the
        # original's format and file size don't reach BytePlus. Lower the cap to keep the test fast.
        monkeypatch.setattr(seedance_common, "SEEDANCE_MAX_IMAGE_DIMENSION_PX", 1000)
        monkeypatch.setattr(seedance_common, "MAX_IMAGE_BYTES", 10)
        _set_list(
            references,
            "reference_images",
            [_image(tmp_path, (1200, 800), "PPM", name="wide"), _image(tmp_path, (3000, 1000), name="too-wide")],
        )
        assert _messages(references) == [
            "Image 2 in 'Reference Images' is 3000x1000, which is too wide. "
            "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it."
        ]

    @pytest.mark.parametrize(
        ("probe", "expected"),
        [
            (
                VideoProbe(info=ImageInfo(1280, 720, None), seconds=1.5, fps=30.0),
                "The video in 'Reference Video 1' is 1.50 seconds long and the minimum is 2. Use a longer clip.",
            ),
            (
                VideoProbe(info=ImageInfo(1280, 720, None), seconds=16.0, fps=30.0),
                "The video in 'Reference Video 1' is 16.00 seconds long and the limit is 15. "
                "Use a Trim Video node to shorten it.",
            ),
            (
                VideoProbe(info=ImageInfo(640, 480, None), seconds=5.0, fps=30.0),
                "The video in 'Reference Video 1' is 640x480 (0.3 megapixels) and the minimum is 0.407696 "
                "megapixels. Use a larger one.",
            ),
            (
                VideoProbe(info=ImageInfo(4000, 2100, None), seconds=5.0, fps=30.0),
                "The video in 'Reference Video 1' is 4000x2100 (8.4 megapixels) and the limit is 8.29504 "
                "megapixels. Resize it smaller.",
            ),
            (
                VideoProbe(info=ImageInfo(1500, 500, None), seconds=5.0, fps=30.0),
                "The video in 'Reference Video 1' is 1500x500, which is too wide. "
                "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
            ),
            (
                VideoProbe(info=ImageInfo(1280, 720, None), seconds=5.0, fps=15.0),
                "The video in 'Reference Video 1' is 15 fps and the minimum is 24 fps. "
                "Re-export it at a higher frame rate.",
            ),
            (
                VideoProbe(info=ImageInfo(1280, 720, None), seconds=5.0, fps=120.0),
                "The video in 'Reference Video 1' is 120 fps and the maximum is 60 fps. "
                "Re-export it at a lower frame rate.",
            ),
        ],
        ids=[
            "too-short",
            "too-long",
            "under-min-area",
            "over-max-area",
            "too-wide",
            "fps-low",
            "fps-high",
        ],
    )
    def test_rejects_a_reference_video_outside_the_limits(
        self,
        references: Seedance20VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],
        probe: VideoProbe,
        expected: str,
    ) -> None:
        videos["a.mp4"] = probe
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        assert _messages(references) == [expected]

    def test_rejects_a_reference_video_side_under_300_pixels(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        # With the 407,696 px area minimum, a side under 300 px needs the other at least 1364 px, which is
        # always too wide or tall as well.
        videos["a.mp4"] = VideoProbe(info=ImageInfo(2000, 280, None), seconds=5.0, fps=30.0)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        assert _messages(references) == [
            "The video in 'Reference Video 1' is 2000x280 and each side must be at least 300 pixels. Use a larger one.",
            "The video in 'Reference Video 1' is 2000x280, which is too wide. "
            "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
        ]

    def test_accepts_a_reference_video_exactly_on_the_area_limits(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(614, 664, None), seconds=5.0, fps=30.0)
        videos["b.mp4"] = VideoProbe(info=ImageInfo(3326, 2494, None), seconds=5.0, fps=30.0)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        assert _messages(references) == []

    def test_rejects_an_oversized_reference_video(
        self,
        references: Seedance20VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],  # noqa: ARG002
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(seedance_common, "MAX_VIDEO_BYTES", 1)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        messages = _messages(references)
        assert len(messages) == 1
        assert messages[0].startswith("The video in 'Reference Video 1' is ")
        assert "and the limit is" in messages[0]

    def test_rejects_reference_videos_over_the_total_duration(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            videos[name] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=6.0, fps=30.0)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        references.set_parameter_value("reference_video_3", str(tmp_path / "c.mp4"))
        assert _messages(references) == [
            "Together, the videos in 'Reference Video 1', 'Reference Video 2' and 'Reference Video 3' are "
            "18.00 seconds long and the limit is 15 seconds in total. Use a Trim Video node to shorten one, "
            "or remove one."
        ]

    def test_accepts_reference_videos_exactly_on_the_total_duration(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            videos[name] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=5.0, fps=30.0)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        references.set_parameter_value("reference_video_3", str(tmp_path / "c.mp4"))
        assert _messages(references) == []

    def test_sums_each_videos_shorter_reading(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        # Each clip has 7 s of video in a 9 s container (a longer audio track): 14 s together, not 18.
        for name in ("a.mp4", "b.mp4"):
            videos[name] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=7.0, fps=30.0, container_seconds=9.0)
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        assert _messages(references) == []

    def test_skips_the_total_when_a_duration_cannot_be_read(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=12.0, fps=30.0)
        videos["b.mp4"] = None
        references.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        references.set_parameter_value("reference_video_2", str(tmp_path / "b.mp4"))
        references.set_parameter_value("reference_video_3", "https://example.com/c.mp4")
        assert _messages(references) == []

    def test_rejects_an_oversized_reference_audio_clip(
        self, references: Seedance20VideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(seedance_common, "MAX_AUDIO_BYTES", 10)
        _set_list(references, "reference_images", [_image(tmp_path)])
        _set_list(references, "reference_audio", [_audio(tmp_path)])
        messages = _messages(references)
        assert len(messages) == 1
        assert messages[0].startswith("Audio clip 1 in 'Reference Audio' is ")
        assert "and the limit is" in messages[0]

    def test_does_not_check_frames_the_mode_does_not_send(
        self, references: Seedance20VideoGeneration, tmp_path: Path
    ) -> None:
        references.set_parameter_value("first_frame", _image(tmp_path, (200, 400), "PPM"))
        # _validate_parameters reports the frame as unused in this mode; no limit is checked on it.
        assert _messages(references) == []

    def test_does_not_check_references_the_mode_does_not_send(
        self,
        node: Seedance20VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(640, 480, None), seconds=30.0, fps=15.0)
        node.set_parameter_value("input_mode", "First/Last Frame")
        _set_list(node, "reference_images", [_image(tmp_path, (200, 400), "PPM")])
        node.set_parameter_value("reference_video_1", str(tmp_path / "a.mp4"))
        assert _messages(node) == []

    def test_does_not_check_private_asset_references(
        self, references: Seedance20VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(640, 480, None), seconds=30.0, fps=15.0)
        _set_list(
            references,
            "reference_images",
            [create_provider_asset_reference(_image(tmp_path, (200, 400), "PPM"), ASSET_KIND_IMAGE)],
        )
        references.set_parameter_value(
            "reference_video_1", create_provider_asset_reference(str(tmp_path / "a.mp4"), ASSET_KIND_VIDEO)
        )
        assert _messages(references) == []

    def test_skips_remote_references(
        self, references: Seedance20VideoGeneration, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("a remote video must not be probed"))
        _set_list(references, "reference_images", ["https://example.com/ref.ppm"])
        references.set_parameter_value("reference_video_1", "https://example.com/a.mp4")
        _set_list(references, "reference_audio", ["https://example.com/a.wav"])
        assert _messages(references) == []

    def test_errors_link_byteplus_page(self, references: Seedance20VideoGeneration, tmp_path: Path) -> None:
        _set_list(references, "reference_images", [_image(tmp_path, image_format="PPM")])
        errors = _errors(references)
        assert errors[0].links
        assert errors[0].links[0].url == CREATE_TASK_PAGE


class TestSeedance25:
    @pytest.fixture
    def node(self) -> Seedance25VideoGeneration:
        return Seedance25VideoGeneration(name="Seedance25")

    def _references(
        self, node: Seedance25VideoGeneration, task: SeedanceTask = SeedanceTask.REFERENCE_TO_VIDEO
    ) -> Seedance25VideoGeneration:
        node.set_parameter_value("task", task)
        node.set_parameter_value("prompt", "Edit @Video 1 and extend it")
        return node

    def test_accepts_valid_frames(self, node: Seedance25VideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("task", SeedanceTask.FIRST_LAST_FRAME)
        node.set_parameter_value("first_frame", _image(tmp_path, image_format="JPEG", name="first"))
        node.set_parameter_value("last_frame", _image(tmp_path, image_format="GIF", name="last"))
        assert _messages(node) == []

    def test_accepts_valid_references(
        self,
        node: Seedance25VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],  # noqa: ARG002
    ) -> None:
        self._references(node)
        _set_list(node, "reference_images", [_image(tmp_path, image_format="WEBP")])
        _set_list(node, "reference_videos", [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")])
        _set_list(node, "reference_audio", [_audio(tmp_path)])
        assert _messages(node) == []

    def test_rejects_invalid_frames(self, node: Seedance25VideoGeneration, tmp_path: Path) -> None:
        node.set_parameter_value("task", SeedanceTask.FIRST_LAST_FRAME)
        node.set_parameter_value("first_frame", _image(tmp_path, (200, 400), name="first"))
        node.set_parameter_value("last_frame", _image(tmp_path, (400, 1100), "PPM", name="last"))
        assert _messages(node) == [
            "The image in 'First Frame' is 200x400 and each side must be at least 300 pixels. Use a larger one.",
            "The image in 'Last Frame' is in PPM format. Convert it to JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC or HEIF.",
            "The image in 'Last Frame' is 400x1100, which is too tall. "
            "The aspect ratio must be between 1:2.5 and 2.5:1. Crop it.",
        ]

    def test_names_the_position_of_an_invalid_reference_image(
        self, node: Seedance25VideoGeneration, tmp_path: Path
    ) -> None:
        self._references(node)
        _set_list(node, "reference_images", [_image(tmp_path, name="ok.png"), _image(tmp_path, image_format="PPM")])
        assert _messages(node) == [
            "Image 2 in 'Reference Images' is in PPM format. Convert it to JPEG, PNG, WEBP, BMP, TIFF, GIF, HEIC or HEIF."
        ]

    @pytest.mark.parametrize(
        ("task", "seconds", "expected"),
        [
            (SeedanceTask.REFERENCE_TO_VIDEO, 3.0, None),
            (SeedanceTask.VIDEO_EXTENSION, 3.0, None),
            (
                SeedanceTask.REFERENCE_TO_VIDEO,
                1.5,
                "Video 1 in 'Reference Videos' is 1.50 seconds long and the minimum is 2. Use a longer clip.",
            ),
            (
                SeedanceTask.VIDEO_EDITING,
                3.0,
                "Video 1 in 'Reference Videos' is 3.00 seconds long and the minimum is 4. Use a longer clip.",
            ),
            (
                SeedanceTask.VIDEO_EXTENSION,
                31.0,
                "Video 1 in 'Reference Videos' is 31.00 seconds long and the limit is 30. "
                "Use a Trim Video node to shorten it.",
            ),
        ],
        ids=["reference-3s", "extension-3s", "reference-1.5s", "editing-3s", "extension-31s"],
    )
    def test_the_shortest_reference_video_depends_on_the_task(
        self,
        node: Seedance25VideoGeneration,
        tmp_path: Path,
        videos: dict[str, VideoProbe | None],
        task: SeedanceTask,
        seconds: float,
        expected: str | None,
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=seconds, fps=30.0)
        self._references(node, task)
        _set_list(node, "reference_videos", [str(tmp_path / "a.mp4")])
        assert _messages(node) == ([] if expected is None else [expected])

    def test_names_the_position_of_an_invalid_reference_video(
        self, node: Seedance25VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["b.mp4"] = VideoProbe(info=ImageInfo(640, 480, None), seconds=5.0, fps=30.0)
        self._references(node)
        _set_list(node, "reference_videos", [str(tmp_path / "a.mp4"), str(tmp_path / "b.mp4")])
        assert _messages(node) == [
            "Video 2 in 'Reference Videos' is 640x480 (0.3 megapixels) and the minimum is 0.407696 megapixels. "
            "Use a larger one."
        ]

    def test_rejects_reference_videos_over_the_total_duration(
        self, node: Seedance25VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            videos[name] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=10.5, fps=30.0)
        self._references(node)
        _set_list(node, "reference_videos", [str(tmp_path / name) for name in ("a.mp4", "b.mp4", "c.mp4")])
        assert _messages(node) == [
            "Together, the videos in 'Reference Videos' are 31.50 seconds long and the limit is 30 seconds in "
            "total. Use a Trim Video node to shorten one, or remove one."
        ]

    def test_accepts_reference_videos_exactly_on_the_total_duration(
        self, node: Seedance25VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        for name in ("a.mp4", "b.mp4", "c.mp4"):
            videos[name] = VideoProbe(info=ImageInfo(1280, 720, None), seconds=10.0, fps=30.0)
        self._references(node)
        _set_list(node, "reference_videos", [str(tmp_path / name) for name in ("a.mp4", "b.mp4", "c.mp4")])
        assert _messages(node) == []

    def test_rejects_an_oversized_reference_audio_clip(
        self, node: Seedance25VideoGeneration, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(seedance_common, "MAX_AUDIO_BYTES", 10)
        self._references(node)
        _set_list(node, "reference_audio", [_audio(tmp_path)])
        messages = _messages(node)
        assert len(messages) == 1
        assert messages[0].startswith("Audio clip 1 in 'Reference Audio' is ")
        assert "and the limit is" in messages[0]

    def test_does_not_check_frames_the_task_does_not_send(
        self, node: Seedance25VideoGeneration, tmp_path: Path
    ) -> None:
        self._references(node)
        _set_list(node, "reference_images", [_image(tmp_path)])
        node.set_parameter_value("first_frame", _image(tmp_path, (200, 400), "PPM"))
        # _validate_parameters reports the frame as unused by this task; no limit is checked on it.
        assert _messages(node) == []

    def test_does_not_check_references_the_task_does_not_send(
        self, node: Seedance25VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(640, 480, None), seconds=45.0, fps=15.0)
        node.set_parameter_value("task", SeedanceTask.FIRST_LAST_FRAME)
        _set_list(node, "reference_images", [_image(tmp_path, (200, 400), "PPM")])
        _set_list(node, "reference_videos", [str(tmp_path / "a.mp4")])
        assert _messages(node) == []

    def test_does_not_check_private_asset_references(
        self, node: Seedance25VideoGeneration, tmp_path: Path, videos: dict[str, VideoProbe | None]
    ) -> None:
        videos["a.mp4"] = VideoProbe(info=ImageInfo(640, 480, None), seconds=45.0, fps=15.0)
        self._references(node)
        _set_list(
            node,
            "reference_images",
            [create_provider_asset_reference(_image(tmp_path, (200, 400), "PPM"), ASSET_KIND_IMAGE)],
        )
        _set_list(
            node, "reference_videos", [create_provider_asset_reference(str(tmp_path / "a.mp4"), ASSET_KIND_VIDEO)]
        )
        assert _messages(node) == []

    def test_skips_remote_references(self, node: Seedance25VideoGeneration, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("a remote video must not be probed"))
        self._references(node)
        _set_list(node, "reference_images", ["https://example.com/ref.ppm"])
        _set_list(node, "reference_videos", ["https://example.com/a.mp4", "asset://asset-123"])
        _set_list(node, "reference_audio", ["https://example.com/a.wav"])
        assert _messages(node) == []

    def test_errors_link_byteplus_page(self, node: Seedance25VideoGeneration, tmp_path: Path) -> None:
        self._references(node)
        _set_list(node, "reference_images", [_image(tmp_path, image_format="PPM")])
        errors = _errors(node)
        assert errors[0].links
        assert errors[0].links[0].url == CREATE_TASK_PAGE
