"""Tests that image nodes apply their quality setting to lossy formats."""

from __future__ import annotations

import io
import random
from types import SimpleNamespace

import pytest
from PIL import Image

from griptape_nodes_library.image.crop_image import CropImage
from griptape_nodes_library.image.flip_image import FlipImage


@pytest.fixture
def noisy_image() -> Image.Image:
    """A gradient with noise, so lossy encoders have detail to discard."""
    rng = random.Random(0)
    img = Image.new("RGB", (128, 128))
    img.putdata([(x * 2, y * 2, rng.randrange(256)) for y in range(128) for x in range(128)])
    return img


def _capture_output(node: FlipImage | CropImage) -> list[bytes]:
    written: list[bytes] = []

    def write_bytes(data: bytes) -> SimpleNamespace:
        written.append(data)
        return SimpleNamespace(location="{outputs}/output")

    node._output_file = SimpleNamespace(build_file=lambda: SimpleNamespace(write_bytes=write_bytes))  # type: ignore[assignment]
    return written


@pytest.mark.parametrize("output_format", ["JPEG", "WEBP"])
def test_base_processor_save_applies_quality(noisy_image: Image.Image, output_format: str) -> None:
    node = FlipImage(name="test_flip_image")
    written = _capture_output(node)

    for quality in ("50", "100"):
        node.set_parameter_value("quality", quality)
        node._save_image_artifact(noisy_image, output_format)

    low, high = written
    assert len(low) < len(high)
    assert Image.open(io.BytesIO(low)).format == output_format


@pytest.mark.parametrize("output_format", ["JPEG", "WEBP"])
def test_crop_image_save_applies_quality(noisy_image: Image.Image, output_format: str) -> None:
    node = CropImage(name="test_crop_image")
    written = _capture_output(node)

    for quality in (0.1, 0.95):
        node._save_cropped_image(noisy_image, {"output_format": output_format, "output_quality": quality})

    low, high = written
    assert len(low) < len(high) / 2
    assert Image.open(io.BytesIO(low)).format == output_format


def test_base_processor_png_save_ignores_quality(noisy_image: Image.Image) -> None:
    node = FlipImage(name="test_flip_image")
    written = _capture_output(node)
    node.set_parameter_value("quality", "high")

    node._save_image_artifact(noisy_image, "png")

    assert Image.open(io.BytesIO(written[0])).format == "PNG"
