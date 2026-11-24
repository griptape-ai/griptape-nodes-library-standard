"""Tests for RescaleImage."""

from __future__ import annotations

import pytest
from PIL import Image

from griptape_nodes_library.image.rescale_image import RescaleImage


def test_divisible_by_resizes_to_multiples() -> None:
    node = RescaleImage(name="test_rescale_image")

    result = node._process_image(
        Image.new("RGB", (4970, 3727)),
        resize_mode=RescaleImage.RESIZE_MODE_DIVISIBLE,
        divisor=128,
        rounding="round",
        preserve_aspect_ratio=True,
        anchor_dimension="auto",
    )

    assert result.size == (4992, 3712)


def _hidden(node: RescaleImage, name: str) -> bool:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None, name
    return parameter.ui_options.get("hide") is True


def test_divisible_by_shows_only_its_parameters() -> None:
    node = RescaleImage(name="test_rescale_image")

    node.set_parameter_value("resize_mode", RescaleImage.RESIZE_MODE_DIVISIBLE)

    for name in RescaleImage.DIVISIBLE_PARAMETERS:
        assert not _hidden(node, name), name
    assert _hidden(node, "percentage_scale")

    node.set_parameter_value("resize_mode", RescaleImage.RESIZE_MODE_PERCENTAGE)

    for name in RescaleImage.DIVISIBLE_PARAMETERS:
        assert _hidden(node, name), name


def test_divisor_is_clamped_to_its_range() -> None:
    node = RescaleImage(name="test_rescale_image")

    node.set_parameter_value("divisor", 0)
    assert node.get_parameter_value("divisor") == RescaleImage.MIN_DIVISOR
    node.set_parameter_value("divisor", 4096)
    assert node.get_parameter_value("divisor") == RescaleImage.MAX_DIVISOR


def test_anchor_dimension_shown_only_when_preserving_aspect_ratio() -> None:
    node = RescaleImage(name="test_rescale_image")
    node.set_parameter_value("resize_mode", RescaleImage.RESIZE_MODE_DIVISIBLE)

    node.set_parameter_value("preserve_aspect_ratio", False)
    assert _hidden(node, "anchor_dimension")
    assert not _hidden(node, "preserve_aspect_ratio")

    node.set_parameter_value("preserve_aspect_ratio", True)
    assert not _hidden(node, "anchor_dimension")

    node.set_parameter_value("preserve_aspect_ratio", False)
    node.set_parameter_value("resize_mode", RescaleImage.RESIZE_MODE_PERCENTAGE)
    node.set_parameter_value("resize_mode", RescaleImage.RESIZE_MODE_DIVISIBLE)
    assert _hidden(node, "anchor_dimension")


@pytest.mark.parametrize(
    ("size", "percentage_scale", "expected"),
    [
        ((10, 10), 5, (1, 1)),
        ((30, 200), 2, (1, 4)),
        ((3, 3), 10, (1, 1)),
        ((100, 100), 1, (1, 1)),
    ],
)
def test_percentage_keeps_each_side_at_least_one_pixel(
    size: tuple[int, int], percentage_scale: int, expected: tuple[int, int]
) -> None:
    node = RescaleImage(name="test_rescale_image")

    result = node._process_image(
        Image.new("RGB", size),
        resize_mode=RescaleImage.RESIZE_MODE_PERCENTAGE,
        percentage_scale=percentage_scale,
    )

    assert result.size == expected


@pytest.mark.parametrize(
    ("resize_mode", "size", "expected"),
    [
        (RescaleImage.RESIZE_MODE_WIDTH, (200, 10), (1, 1)),
        (RescaleImage.RESIZE_MODE_HEIGHT, (10, 200), (1, 1)),
    ],
)
def test_width_and_height_keep_derived_side_at_least_one_pixel(
    resize_mode: str, size: tuple[int, int], expected: tuple[int, int]
) -> None:
    node = RescaleImage(name="test_rescale_image")

    result = node._process_image(Image.new("RGB", size), resize_mode=resize_mode, target_size=1)

    assert result.size == expected
