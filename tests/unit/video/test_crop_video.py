"""Tests for CropVideo's crop editor widget and the settings it syncs."""

from __future__ import annotations

import pytest

from griptape_nodes_library.video.crop_video import CROP_PARAMETERS, CropVideo


def test_the_editor_syncs_the_crop_settings() -> None:
    node = CropVideo(name="test_crop_video")

    editor = node.get_parameter_by_name("crop_editor")
    assert editor is not None
    assert editor.ui_options["synced_parameters"] == CROP_PARAMETERS
    for name in CROP_PARAMETERS:
        assert node.get_parameter_by_name(name) is not None, name


def test_the_editor_value_holds_only_the_video() -> None:
    node = CropVideo(name="test_crop_video")

    assert set(node.get_parameter_value("crop_editor")) == {
        "video_url",
        "video_path",
        "video_width",
        "video_height",
        "total_frames",
    }


def test_choosing_custom_shows_the_custom_fields() -> None:
    node = CropVideo(name="test_crop_video")

    node.set_parameter_value("crop_position", "Custom")

    custom_left = node.get_parameter_by_name("custom_left")
    assert custom_left is not None
    assert custom_left.ui_options.get("hide") is not True


@pytest.mark.parametrize(("width", "height"), [(0, 0), (100, 0), (0, 100), (-10, 100)])
def test_a_custom_size_of_zero_fails_validation(width: int, height: int) -> None:
    node = CropVideo(name="test_crop_video")
    node.set_parameter_value("custom_width", width)
    node.set_parameter_value("custom_height", height)

    errors = node._validate_custom_parameters()

    assert errors is not None
    assert f"crop size is {width}x{height}" in str(errors[0])


def test_the_default_settings_fail_validation_before_the_run() -> None:
    node = CropVideo(name="test_crop_video")

    errors = node.validate_before_node_run() or []

    assert any("crop size is 0x0" in str(error) for error in errors)


def test_an_unrecognized_crop_size_fails_validation() -> None:
    node = CropVideo(name="test_crop_video")
    node.parameter_values["crop_size"] = "wide"

    errors = node._validate_custom_parameters()

    assert errors is not None
    assert "'wide'" in str(errors[0])


@pytest.mark.parametrize(
    ("values"),
    [
        {"custom_width": 100, "custom_height": 100},
        {"crop_size": "720p (1280x720)"},
    ],
)
def test_a_crop_size_above_zero_passes_validation(values: dict[str, object]) -> None:
    node = CropVideo(name="test_crop_video")
    for name, value in values.items():
        node.set_parameter_value(name, value)

    assert node._validate_custom_parameters() is None


def test_the_defaults_are_unchanged() -> None:
    node = CropVideo(name="test_crop_video")

    assert node.get_parameter_value("crop_size") == "Custom"
    assert node.get_parameter_value("custom_width") == 0
    assert node.get_parameter_value("custom_height") == 0
