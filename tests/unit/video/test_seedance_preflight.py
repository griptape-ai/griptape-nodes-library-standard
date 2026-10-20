"""Tests for the Seedance nodes' pre-flight media checks against BytePlus's documented limits."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from PIL import Image

from griptape_nodes_library.video.seedance_video_generation import SeedanceVideoGeneration

SEEDANCE_1_5_PRO = "seedance-1-5-pro-251215"
SEEDANCE_1_0_PRO = "seedance-1-0-pro-250528"
SEEDANCE_1_0_PRO_FAST = "seedance-1-0-pro-fast-251015"
CREATE_TASK_PAGE = "https://docs.byteplus.com/en/docs/ModelArk/1520757"


def _image(tmp_path: Path, size: tuple[int, int] = (1280, 720), image_format: str = "PNG", name: str = "") -> str:
    path = tmp_path / (name or f"{size[0]}x{size[1]}.{image_format.lower()}")
    Image.new("RGB", size).save(path, format=image_format)
    return str(path)


def _errors(node: Any) -> list[NodeError]:
    return [e for e in node.validate_before_node_run() or [] if isinstance(e, NodeError)]


def _messages(node: Any) -> list[str]:
    return [str(e) for e in _errors(node)]


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
