"""Tests for the live preview BaseImageProcessor gives subclasses that name a preview operation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact

from griptape_nodes_library.image.adjust_image_eq import AdjustImageEQ
from griptape_nodes_library.image.adjust_image_levels import AdjustImageLevels
from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.image.bloom_effect import BloomEffect
from griptape_nodes_library.image.flip_image import FlipImage
from griptape_nodes_library.image.gaussian_blur_image import GaussianBlurImage
from griptape_nodes_library.image.grayscale_image import GrayscaleImage
from griptape_nodes_library.image.image_blend_compositor import ImageBlendCompositor
from griptape_nodes_library.image.rescale_image import RescaleImage
from griptape_nodes_library.image.rotate_image import RotateImage

PREVIEWED = [
    AdjustImageEQ,
    AdjustImageLevels,
    BloomEffect,
    FlipImage,
    GaussianBlurImage,
    GrayscaleImage,
    RescaleImage,
    RotateImage,
]

OPERATIONS_JS = Path(__file__).parents[3] / "widgets" / "ImageProcessorPreview" / "_operations.js"


@pytest.fixture(autouse=True)
def no_immediate_processing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the nodes' engine-side previews from fetching the test image URL."""
    for node_class in PREVIEWED:
        monkeypatch.setattr(node_class, "_process_image_immediately", lambda *_args, **_kwargs: None, raising=False)


@pytest.mark.parametrize("node_class", PREVIEWED)
def test_preview_syncs_the_nodes_own_settings(node_class: type[BaseImageProcessor]) -> None:
    node = node_class(name="test_node")

    preview = node.get_parameter_by_name("preview")
    assert preview is not None
    assert preview.ui_options["widget"] == "ImageProcessorPreview"
    assert preview.ui_options.get("synced_parameters", []) == node_class.PREVIEW_PARAMETERS
    for name in node_class.PREVIEW_PARAMETERS:
        assert node.get_parameter_by_name(name) is not None, name


@pytest.mark.parametrize("node_class", PREVIEWED)
def test_widget_knows_the_operation(node_class: type[BaseImageProcessor]) -> None:
    exported = re.search(r"export const OPERATIONS = \{(.*?)\};", OPERATIONS_JS.read_text(), re.DOTALL)
    assert exported is not None
    keys = {line.strip().split(":")[0].rstrip(",") for line in exported.group(1).splitlines() if line.strip()}

    assert node_class.PREVIEW_OPERATION in keys


@pytest.mark.parametrize("node_class", PREVIEWED)
def test_setting_the_input_image_gives_the_preview_its_url(node_class: type[BaseImageProcessor]) -> None:
    node = node_class(name="test_node")

    node.set_parameter_value("input_image", ImageUrlArtifact(value="https://example.com/cat.png"))

    assert node.get_parameter_value("preview") == {
        "operation": node_class.PREVIEW_OPERATION,
        "image_url": "https://example.com/cat.png",
    }


def test_a_node_without_an_operation_has_no_preview() -> None:
    node = ImageBlendCompositor(name="test_node")

    assert node.get_parameter_by_name("preview") is None
