"""Tests for ExtendCanvas's layout, upscaling and mask."""

from __future__ import annotations

import pytest
from PIL import Image

from griptape_nodes_library.image.extend_canvas import ExtendCanvas

RED = (255, 0, 0, 255)
BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
TRANSPARENT = (0, 0, 0, 0)


@pytest.fixture
def node() -> ExtendCanvas:
    return ExtendCanvas(name="test_extend_canvas")


@pytest.fixture
def red() -> Image.Image:
    return Image.new("RGBA", (100, 50), RED)


def red_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """Bounding box of the red image on a black canvas."""
    return image.getchannel("R").getbbox()


def set_custom(node: ExtendCanvas, *, left: int, top: int, right: int, bottom: int) -> None:
    node.set_parameter_value("aspect_ratio_preset", "custom")
    node.set_parameter_value("left", left)
    node.set_parameter_value("top", top)
    node.set_parameter_value("right", right)
    node.set_parameter_value("bottom", bottom)


def test_custom_margins_place_image(node: ExtendCanvas, red: Image.Image) -> None:
    set_custom(node, left=20, top=10, right=20, bottom=10)

    image, mask = node._extend_image(red)

    assert image.size == (140, 70)
    assert red_bbox(image) == (20, 10, 120, 60)
    assert mask.getpixel((20, 10)) == BLACK
    assert mask.getpixel((0, 0)) == WHITE


def test_upscale_scales_image_and_offsets_with_canvas(node: ExtendCanvas, red: Image.Image) -> None:
    set_custom(node, left=20, top=10, right=20, bottom=10)
    node.set_parameter_value("upscale_factor", 2.0)

    image, mask = node._extend_image(red)

    assert image.size == (280, 140)
    assert red_bbox(image) == (40, 20, 240, 120)
    assert image.getpixel((40, 20)) == RED
    assert image.getpixel((239, 119)) == RED
    assert mask.getpixel((40, 20)) == BLACK
    assert mask.getpixel((239, 119)) == BLACK
    assert mask.getpixel((39, 20)) == WHITE
    assert mask.getpixel((240, 119)) == WHITE


def test_upscale_with_preset_keeps_ratio_and_scales_image(node: ExtendCanvas, red: Image.Image) -> None:
    node.set_parameter_value("aspect_ratio_preset", "1:1 square")
    node.set_parameter_value("upscale_factor", 2.0)

    image, _ = node._extend_image(red)

    assert image.size == (200, 200)
    assert red_bbox(image) == (0, 50, 200, 150)


def test_transparent_background_leaves_mask_transparent(node: ExtendCanvas, red: Image.Image) -> None:
    set_custom(node, left=10, top=0, right=0, bottom=0)
    node.set_parameter_value("background_color", "transparent")

    image, mask = node._extend_image(red)

    assert image.getpixel((0, 0)) == TRANSPARENT
    assert mask.getpixel((0, 0)) == TRANSPARENT
    assert mask.getpixel((10, 0)) == BLACK


@pytest.mark.parametrize("factor", [0.0, -1.0])
def test_validation_rejects_non_positive_upscale(node: ExtendCanvas, factor: float) -> None:
    node.set_parameter_value("upscale_factor", factor)

    exceptions = node.validate_before_node_run() or []

    assert any("Upscale Factor" in str(e) for e in exceptions)


def test_extend_rejects_non_positive_upscale(node: ExtendCanvas, red: Image.Image) -> None:
    node.set_parameter_value("upscale_factor", 0.0)

    with pytest.raises(ValueError, match="Upscale Factor"):
        node._extend_image(red)


def test_upscale_below_one_shrinks_image_and_margins(node: ExtendCanvas, red: Image.Image) -> None:
    set_custom(node, left=20, top=10, right=20, bottom=10)
    node.set_parameter_value("upscale_factor", 0.5)

    image, _ = node._extend_image(red)

    assert image.size == (70, 35)
    assert red_bbox(image) == (10, 5, 60, 30)
