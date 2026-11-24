"""Tests for ImageBash canvas dimensions."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.image.image_bash import CANVAS_DIMENSIONS, ImageBash


@pytest.fixture
def node() -> ImageBash:
    return ImageBash(name="test_image_bash")


@pytest.mark.parametrize("canvas_size", [None, "bogus", "custom", "HD"])
def test_unknown_canvas_size_raises(node: ImageBash, canvas_size: str | None) -> None:
    node.parameter_values["canvas_size"] = canvas_size

    with pytest.raises(NodeError) as exc_info:
        node._get_canvas_dimensions()

    message = str(exc_info.value)
    assert f"It's set to {canvas_size!r}." in message
    assert ", ".join(CANVAS_DIMENSIONS) in message


def test_preset_canvas_size_uses_preset_dimensions(node: ImageBash) -> None:
    node.parameter_values["canvas_size"] = "4K Scope"

    assert node._get_canvas_dimensions() == (4096, 1716)


def test_custom_canvas_size_uses_width_and_height(node: ImageBash) -> None:
    node.parameter_values["canvas_size"] = "Custom"
    node.parameter_values["width"] = 640
    node.parameter_values["height"] = 480

    assert node._get_canvas_dimensions() == (640, 480)
