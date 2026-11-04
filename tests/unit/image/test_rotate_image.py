"""Tests for RotateImage's rotation math."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest
from PIL import Image, ImageOps

from griptape_nodes_library.image.rotate_image import (
    ROTATION_ICONS,
    RotateImage,
    Rotation,
    exif_transpose_array,
    expanded_size,
    is_sixteen_bit,
    rotate_array,
)

RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def node() -> RotateImage:
    return RotateImage(name="test_rotate_image")


@pytest.fixture
def landscape() -> Image.Image:
    """A 4x2 image, red with a blue top-left pixel."""
    img = Image.new("RGB", (4, 2), RED)
    img.putpixel((0, 0), BLUE)
    return img


@pytest.mark.parametrize(
    ("rotation", "size", "blue_at"),
    [
        (Rotation.CLOCKWISE_90, (2, 4), (1, 0)),
        (Rotation.ROTATE_180, (4, 2), (3, 1)),
        (Rotation.COUNTER_CLOCKWISE_90, (2, 4), (0, 3)),
    ],
)
def test_quarter_turns_are_lossless(
    node: RotateImage,
    landscape: Image.Image,
    rotation: Rotation,
    size: tuple[int, int],
    blue_at: tuple[int, int],
) -> None:
    result = node._process_image(landscape, rotation=rotation)

    assert result.size == size
    assert result.mode == "RGB"
    assert result.getpixel(blue_at) == BLUE
    assert sorted(result.getcolors() or []) == sorted(landscape.getcolors() or [])


def test_quarter_turn_accepts_plain_string(node: RotateImage, landscape: Image.Image) -> None:
    assert node._process_image(landscape, rotation="90° clockwise").size == (2, 4)


def test_custom_angle_matches_quarter_turn_direction(node: RotateImage) -> None:
    img = Image.new("RGB", (40, 20), RED)
    img.paste(BLUE, (0, 0, 10, 10))

    result = node._process_image(
        img, rotation=Rotation.CUSTOM, angle=90.0, expand_canvas=True, background_color="#000000ff"
    )

    assert result.size == (20, 40)
    # Clockwise: the top-left corner moves to the top-right.
    assert result.getpixel((15, 5)) == BLUE


def test_custom_angle_expands_canvas_with_transparent_corners(node: RotateImage) -> None:
    img = Image.new("RGB", (100, 50), RED)

    result = node._process_image(
        img, rotation=Rotation.CUSTOM, angle=45.0, expand_canvas=True, background_color="#00000000"
    )

    assert result.width > img.width
    assert result.height > img.height
    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (0, 0, 0, 0)
    assert result.getpixel((result.width // 2, result.height // 2)) == (*RED, 255)


def test_custom_angle_without_expand_keeps_size(node: RotateImage) -> None:
    img = Image.new("RGB", (100, 50), RED)

    result = node._process_image(
        img, rotation=Rotation.CUSTOM, angle=30.0, expand_canvas=False, background_color="#00ff00ff"
    )

    assert result.size == img.size
    assert result.mode == "RGB"
    assert result.getpixel((0, 0)) == (0, 255, 0)


def test_custom_angle_converts_grayscale_for_color_fill(node: RotateImage) -> None:
    img = Image.new("L", (100, 50), 128)

    result = node._process_image(
        img, rotation=Rotation.CUSTOM, angle=10.0, expand_canvas=True, background_color="#ff0000ff"
    )

    assert result.mode == "RGBA"
    assert result.getpixel((0, 0)) == (255, 0, 0, 255)


def test_unknown_rotation_raises(node: RotateImage, landscape: Image.Image) -> None:
    with pytest.raises(ValueError, match="Unknown rotation"):
        node._process_image(landscape, rotation="sideways")


def test_custom_parameters_are_hidden_until_custom_is_picked(node: RotateImage) -> None:
    angle = node.get_parameter_by_name("angle")
    assert angle is not None
    assert angle.ui_options.get("hide") is True

    node.set_parameter_value("rotation", Rotation.CUSTOM)
    assert angle.ui_options.get("hide") is False

    node.set_parameter_value("rotation", Rotation.ROTATE_180)
    assert angle.ui_options.get("hide") is True


def test_rotates_from_exif_orientation(node: RotateImage) -> None:
    """A sideways phone photo is rotated as it is displayed, not as it is stored."""
    stored = Image.new("RGB", (40, 20), RED)
    exif = stored.getexif()
    exif[0x0112] = 6  # displayed rotated 90° clockwise, as 20x40
    buffer = io.BytesIO()
    stored.save(buffer, format="JPEG", exif=exif)
    photo = Image.open(io.BytesIO(buffer.getvalue()))

    result = node._process_image(photo, rotation=Rotation.COUNTER_CLOCKWISE_90)

    assert result.size == (40, 20)


def test_dropdown_shows_an_icon_for_every_choice(node: RotateImage) -> None:
    rotation = node.get_parameter_by_name("rotation")
    assert rotation is not None
    assert rotation.ui_options["dropdown_row_icons"] is True
    assert [row["name"] for row in rotation.ui_options["data"]] == [r.value for r in Rotation]
    assert set(ROTATION_ICONS) == set(Rotation)


def _sixteen_bit_png(pixels: np.ndarray) -> Image.Image:
    ok, buffer = cv2.imencode(".png", pixels)
    assert ok
    return Image.open(io.BytesIO(buffer.tobytes()))


@pytest.fixture
def sixteen_bit_bgr() -> np.ndarray:
    """A 6x4 16-bit image with values no 8-bit image can hold."""
    return np.random.default_rng(0).integers(0, 65536, (4, 6, 3), dtype=np.uint16)


@pytest.mark.parametrize("channels", [None, 3, 4])
def test_detects_sixteen_bit_pngs(channels: int | None) -> None:
    shape = (4, 6) if channels is None else (4, 6, channels)
    assert is_sixteen_bit(_sixteen_bit_png(np.zeros(shape, dtype=np.uint16)))
    assert not is_sixteen_bit(_sixteen_bit_png(np.zeros(shape, dtype=np.uint8)))


@pytest.mark.parametrize(
    ("rotation", "k"),
    [(Rotation.CLOCKWISE_90, -1), (Rotation.ROTATE_180, 2), (Rotation.COUNTER_CLOCKWISE_90, 1)],
)
def test_sixteen_bit_quarter_turns_keep_every_value(sixteen_bit_bgr: np.ndarray, rotation: Rotation, k: int) -> None:
    result = rotate_array(sixteen_bit_bgr, rotation, angle=0.0, expand_canvas=True, background_color="#00000000")

    assert result.dtype == np.uint16
    np.testing.assert_array_equal(result, np.rot90(sixteen_bit_bgr, k))


def test_sixteen_bit_quarter_turn_matches_eight_bit_direction(node: RotateImage, landscape: Image.Image) -> None:
    bgr = cv2.cvtColor(np.asarray(landscape), cv2.COLOR_RGB2BGR).astype(np.uint16) * 257

    result = rotate_array(bgr, Rotation.CLOCKWISE_90, angle=0.0, expand_canvas=True, background_color="#00000000")

    expected = np.asarray(node._process_image(landscape, rotation=Rotation.CLOCKWISE_90))
    np.testing.assert_array_equal(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), expected.astype(np.uint16) * 257)


@pytest.mark.parametrize("angle", [10.0, 30.0, 45.0, 90.0, -37.5, 180.0])
def test_expanded_size_matches_pil(angle: float) -> None:
    img = Image.new("RGB", (101, 50))
    assert expanded_size(101, 50, angle) == img.rotate(-angle, expand=True).size


def test_sixteen_bit_custom_angle_keeps_bit_depth() -> None:
    pixels = np.full((50, 100, 3), 1000, dtype=np.uint16)

    result = rotate_array(pixels, Rotation.CUSTOM, angle=45.0, expand_canvas=True, background_color="#00000000")

    assert result.dtype == np.uint16
    assert result.shape == (*expanded_size(100, 50, 45.0)[::-1], 4)
    assert tuple(result[0, 0]) == (0, 0, 0, 0)
    center = result[result.shape[0] // 2, result.shape[1] // 2]
    assert tuple(center) == (1000, 1000, 1000, 65535)


def test_sixteen_bit_custom_angle_turns_clockwise() -> None:
    pixels = np.zeros((20, 40, 3), dtype=np.uint16)
    pixels[0:10, 0:10] = (60000, 0, 0)  # blue in BGR

    result = rotate_array(pixels, Rotation.CUSTOM, angle=90.0, expand_canvas=True, background_color="#000000ff")

    assert result.shape == (40, 20, 3)
    # Clockwise: the top-left corner moves to the top-right.
    assert tuple(result[5, 15]) == (60000, 0, 0)


def test_sixteen_bit_custom_angle_fills_opaque_color_without_alpha() -> None:
    pixels = np.zeros((50, 100, 3), dtype=np.uint16)

    result = rotate_array(pixels, Rotation.CUSTOM, angle=30.0, expand_canvas=False, background_color="#00ff00ff")

    assert result.shape == pixels.shape
    assert tuple(result[0, 0]) == (0, 65535, 0)


def test_sixteen_bit_grayscale_custom_angle_gets_color_fill() -> None:
    pixels = np.full((50, 100), 30000, dtype=np.uint16)

    result = rotate_array(pixels, Rotation.CUSTOM, angle=10.0, expand_canvas=True, background_color="#ff0000ff")

    assert result.dtype == np.uint16
    assert tuple(result[0, 0]) == (0, 0, 65535, 65535)


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_transpose_array_matches_pil(orientation: int) -> None:
    pixels = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    img = Image.fromarray(pixels)
    img.getexif()[0x0112] = orientation

    expected = np.asarray(ImageOps.exif_transpose(img))

    np.testing.assert_array_equal(exif_transpose_array(pixels, orientation), expected)
