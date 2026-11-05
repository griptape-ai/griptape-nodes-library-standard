"""Tests for AdjustMaskSize's adjustment parameter and its preview widget."""

from __future__ import annotations

import pytest

from griptape_nodes_library.video.adjust_mask_size import AdjustMaskSize


@pytest.fixture
def node() -> AdjustMaskSize:
    return AdjustMaskSize(name="test_adjust_mask_size")


def test_adjustment_is_a_parameter_the_widget_syncs(node: AdjustMaskSize) -> None:
    adjustment = node.get_parameter_by_name("adjustment")
    assert adjustment is not None
    assert node.get_parameter_value("adjustment") == 0

    preview = node.get_parameter_by_name("preview")
    assert preview is not None
    assert preview.ui_options["synced_parameters"] == ["adjustment"]
    assert "adjustment" not in node.get_parameter_value("preview")


def test_an_adjustment_saved_in_the_preview_moves_to_the_parameter(node: AdjustMaskSize) -> None:
    saved_preview = {
        "original_video_url": "",
        "mask_video_url": "",
        "adjustment": -7,
        "current_frame": 12,
        "total_frames": 100,
    }

    node.set_parameter_value("preview", saved_preview)

    assert node.get_parameter_value("adjustment") == -7
    assert node.get_parameter_value("preview") == {
        "original_video_url": "",
        "mask_video_url": "",
        "current_frame": 12,
        "total_frames": 100,
    }


def test_a_saved_adjustment_out_of_range_is_clamped(node: AdjustMaskSize) -> None:
    node.set_parameter_value("preview", {"adjustment": 99})

    assert node.get_parameter_value("adjustment") == AdjustMaskSize.MAX_ADJUSTMENT
