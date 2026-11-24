"""Tests for ImageBlendCompositor's blend modes and output saving."""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

import pytest
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

from griptape_nodes_library.image import image_blend_compositor
from griptape_nodes_library.image.image_blend_compositor import BlendMode, ImageBlendCompositor

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


@pytest.fixture
def node() -> ImageBlendCompositor:
    return ImageBlendCompositor(name="test_image_blend_compositor")


def two_pixel(first: tuple[int, int, int], second: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (2, 1))
    img.putpixel((0, 0), first)
    img.putpixel((1, 0), second)
    return img


def blend(node: ImageBlendCompositor, base: Image.Image, layer: Image.Image, mode: str) -> list:
    result = node._process_images(base, layer, blend_mode=mode, opacity=1.0)
    rgb = result.convert("RGB")
    return [rgb.getpixel((x, 0)) for x in range(rgb.width)]


@pytest.mark.parametrize("mode", list(BlendMode))
def test_every_blend_mode_runs_on_rgb_images(node: ImageBlendCompositor, mode: BlendMode) -> None:
    base = two_pixel((200, 100, 50), (10, 240, 128))
    layer = two_pixel((100, 150, 60), (250, 30, 128))

    result = node._process_images(base, layer, blend_mode=mode, opacity=1.0)

    assert result.size == base.size


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (BlendMode.LOGICAL_AND, [WHITE, BLACK, BLACK]),
        (BlendMode.LOGICAL_OR, [WHITE, WHITE, BLACK]),
        (BlendMode.LOGICAL_XOR, [BLACK, WHITE, BLACK]),
    ],
)
def test_logical_modes_combine_thresholded_pixels(
    node: ImageBlendCompositor, mode: BlendMode, expected: list[tuple[int, int, int]]
) -> None:
    base = Image.new("RGB", (3, 1))
    layer = Image.new("RGB", (3, 1))
    for x, (base_px, layer_px) in enumerate([(WHITE, WHITE), (WHITE, BLACK), (BLACK, BLACK)]):
        base.putpixel((x, 0), base_px)
        layer.putpixel((x, 0), layer_px)

    assert blend(node, base, layer, mode) == expected


def test_logical_and_does_not_return_the_blend_image(node: ImageBlendCompositor) -> None:
    base = two_pixel((200, 100, 50), (10, 240, 128))
    layer = two_pixel((100, 150, 60), (250, 30, 128))

    assert blend(node, base, layer, BlendMode.LOGICAL_AND) != blend(node, base, layer, BlendMode.NORMAL)


def test_exclusion_is_base_plus_blend_minus_twice_their_product(node: ImageBlendCompositor) -> None:
    base = two_pixel((200, 100, 50), WHITE)
    layer = two_pixel((100, 150, 60), (10, 240, 128))

    assert blend(node, base, layer, BlendMode.EXCLUSION) == [(143, 132, 86), (245, 15, 127)]


def test_unknown_blend_mode_raises(node: ImageBlendCompositor) -> None:
    base = two_pixel(WHITE, BLACK)

    with pytest.raises(ValueError, match="Unknown blend mode"):
        node._process_images(base, base, blend_mode="logicaland", opacity=1.0)


def saved_bytes(node: ImageBlendCompositor, monkeypatch: pytest.MonkeyPatch, base_format: str) -> bytes:
    """Run the node on a base image of the given format and return the bytes it saved."""
    base = Image.new("RGB", (8, 8), (200, 100, 50))
    buffer = io.BytesIO()
    base.save(buffer, format=base_format)
    images = {"base": Image.open(io.BytesIO(buffer.getvalue())), "layer": Image.new("RGB", (4, 4), WHITE)}
    written: list[bytes] = []

    def write_bytes(data: bytes) -> SimpleNamespace:
        written.append(data)
        return SimpleNamespace(location="output")

    monkeypatch.setattr(image_blend_compositor, "load_pil_from_url", lambda url: images[url])
    monkeypatch.setattr(node._output_file, "build_file", lambda: SimpleNamespace(write_bytes=write_bytes))
    node.set_parameter_value("input_image", ImageUrlArtifact("base"))
    node.set_parameter_value("blend_image", ImageUrlArtifact("layer"))
    written.clear()

    asyncio.run(node.aprocess())

    return written[-1]


@pytest.mark.parametrize(("output_format", "expected"), [("JPEG", "JPEG"), ("WEBP", "WEBP"), ("PNG", "PNG")])
def test_output_format_sets_the_saved_format(
    node: ImageBlendCompositor, monkeypatch: pytest.MonkeyPatch, output_format: str, expected: str
) -> None:
    node.set_parameter_value("output_format", output_format)

    data = saved_bytes(node, monkeypatch, "PNG")

    assert Image.open(io.BytesIO(data)).format == expected


def test_auto_output_format_keeps_the_base_image_format(
    node: ImageBlendCompositor, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = saved_bytes(node, monkeypatch, "JPEG")

    assert Image.open(io.BytesIO(data)).format == "JPEG"


def test_quality_changes_lossy_output(node: ImageBlendCompositor, monkeypatch: pytest.MonkeyPatch) -> None:
    node.set_parameter_value("output_format", "JPEG")
    node.set_parameter_value("quality", "50")
    low = saved_bytes(node, monkeypatch, "PNG")
    node.set_parameter_value("quality", "100")
    high = saved_bytes(node, monkeypatch, "PNG")

    assert len(high) > len(low)


def test_auto_output_format_falls_back_to_png_for_unwritable_formats(
    node: ImageBlendCompositor, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = saved_bytes(node, monkeypatch, "GIF")

    assert Image.open(io.BytesIO(data)).format == "PNG"


def test_non_normal_modes_cover_the_whole_base_when_centered(node: ImageBlendCompositor) -> None:
    base = Image.new("RGB", (100, 100), BLACK)
    layer = Image.new("RGB", (20, 20), WHITE)

    result = node._process_images(base, layer, blend_mode=BlendMode.SCREEN, opacity=1.0)

    assert result.convert("RGB").getpixel((0, 0)) == WHITE
    assert result.convert("RGB").getpixel((99, 99)) == WHITE
