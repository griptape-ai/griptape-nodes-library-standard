"""Tests for RotateImage's rotation math."""

from __future__ import annotations

import io

import pytest
from PIL import Image

from griptape_nodes_library.image.rotate_image import ROTATION_ICONS, RotateImage, Rotation

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
