"""Tests for RescaleImage's divisible by mode."""

from __future__ import annotations

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
