"""Tests for CropVideo's crop editor widget and the settings it syncs."""

from __future__ import annotations

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
