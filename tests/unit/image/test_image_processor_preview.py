"""Tests for the live preview ImagePreviewMixin gives image and video nodes that name a preview operation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from PIL import Image

from griptape_nodes_library.image import image_grid_splitter, image_preview
from griptape_nodes_library.image.add_text_to_existing_image import AddTextToExistingImage
from griptape_nodes_library.image.add_text_to_image import AddTextToImage
from griptape_nodes_library.image.adjust_image_eq import AdjustImageEQ
from griptape_nodes_library.image.adjust_image_levels import AdjustImageLevels
from griptape_nodes_library.image.bloom_effect import BloomEffect
from griptape_nodes_library.image.create_color_bars import CreateColorBars
from griptape_nodes_library.image.display_channel import DisplayChannel
from griptape_nodes_library.image.display_mask import DisplayMask
from griptape_nodes_library.image.extend_canvas import ExtendCanvas
from griptape_nodes_library.image.flip_image import FlipImage
from griptape_nodes_library.image.gaussian_blur_image import GaussianBlurImage
from griptape_nodes_library.image.grayscale_image import GrayscaleImage
from griptape_nodes_library.image.image_blend_compositor import ImageBlendCompositor
from griptape_nodes_library.image.image_grid_splitter import ImageGridSplitter
from griptape_nodes_library.image.image_preview import ImagePreviewMixin
from griptape_nodes_library.image.invert_image import InvertImage
from griptape_nodes_library.image.invert_mask import InvertMask
from griptape_nodes_library.image.rescale_image import RescaleImage
from griptape_nodes_library.image.rotate_image import RotateImage
from griptape_nodes_library.image.set_color_to_transparent import SetColorToTransparent
from griptape_nodes_library.video.add_color_curves import AddColorCurves
from griptape_nodes_library.video.add_film_grain import AddFilmGrain
from griptape_nodes_library.video.add_rgb_shift import AddRGBShift
from griptape_nodes_library.video.add_vignette import AddVignette
from griptape_nodes_library.video.adjust_video_eq import AdjustVideoEQ
from griptape_nodes_library.video.blur_video import BlurVideo
from griptape_nodes_library.video.flip_video import FlipVideo

PREVIEWED = [
    AdjustImageEQ,
    AdjustImageLevels,
    BloomEffect,
    ExtendCanvas,
    FlipImage,
    GaussianBlurImage,
    GrayscaleImage,
    RescaleImage,
    RotateImage,
    AddColorCurves,
    AddFilmGrain,
    AddRGBShift,
    AddVignette,
    AdjustVideoEQ,
    BlurVideo,
    FlipVideo,
    AddTextToImage,
    AddTextToExistingImage,
    CreateColorBars,
    InvertImage,
    InvertMask,
    DisplayMask,
    DisplayChannel,
    SetColorToTransparent,
    ImageGridSplitter,
]

WITH_SOURCE = [node_class for node_class in PREVIEWED if node_class.PREVIEW_IMAGES]

WITH_MODE = [InvertImage, InvertMask, DisplayMask, DisplayChannel]

OPERATIONS_JS = Path(__file__).parents[3] / "widgets" / "ImageProcessorPreview" / "_operations.js"


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nodes that read an image's mode or detect a grid load the image; give them one without fetching."""

    def load(_url: str) -> Image.Image:
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(image_preview, "load_pil_from_url", load)
    monkeypatch.setattr(image_grid_splitter, "load_pil_from_url", load)


@pytest.mark.parametrize("node_class", PREVIEWED)
def test_preview_syncs_the_nodes_own_settings(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")

    preview = node.get_parameter_by_name(node_class.PREVIEW_PARAMETER_NAME)
    assert preview is not None
    assert preview.ui_options["widget"] == "ImageProcessorPreview"
    assert preview.ui_options.get("synced_parameters", []) == node_class.PREVIEW_PARAMETERS
    for name in node_class.PREVIEW_PARAMETERS:
        assert node.get_parameter_by_name(name) is not None, name


@pytest.mark.parametrize("node_class", PREVIEWED)
def test_widget_knows_the_operation(node_class: type[ImagePreviewMixin]) -> None:
    exported = re.search(r"export const OPERATIONS = \{(.*?)\};", OPERATIONS_JS.read_text(), re.DOTALL)
    assert exported is not None
    keys = {line.strip().split(":")[0].rstrip(",") for line in exported.group(1).splitlines() if line.strip()}

    assert node_class.PREVIEW_OPERATION in keys


@pytest.mark.parametrize("node_class", WITH_SOURCE)
def test_setting_the_source_gives_the_preview_its_url(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")
    [(parameter_name, key)] = node_class.PREVIEW_IMAGES.items()
    if key == "video_url":
        media = VideoUrlArtifact(value="https://example.com/cat.mp4")
    else:
        media = ImageUrlArtifact(value="https://example.com/cat.png")

    node.set_parameter_value(parameter_name, media)

    preview = node.get_parameter_value(node_class.PREVIEW_PARAMETER_NAME)
    assert preview["operation"] == node_class.PREVIEW_OPERATION
    assert preview[key] == media.value


@pytest.mark.parametrize("node_class", [AddTextToImage, AddTextToExistingImage])
def test_text_previews_carry_the_font_pillow_draws_with(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")

    assert node.get_parameter_value("preview")["font_url"]


@pytest.mark.parametrize("node_class", WITH_MODE)
def test_mode_sensitive_previews_carry_the_image_mode(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")
    [parameter_name] = node_class.PREVIEW_IMAGES

    node.set_parameter_value(parameter_name, ImageUrlArtifact(value="https://example.com/cat.png"))

    assert node.get_parameter_value("preview")["image_mode"] == "RGB"


def test_grid_splitter_sends_the_detected_grid() -> None:
    node = ImageGridSplitter(name="test_node")

    node.set_parameter_value("input_image", ImageUrlArtifact(value="https://example.com/cat.png"))

    preview = node.get_parameter_value("grid_preview")
    assert preview["detected_rows"] >= 1
    assert preview["detected_cols"] >= 1


def test_a_node_without_an_operation_has_no_preview() -> None:
    node = ImageBlendCompositor(name="test_node")

    assert node.get_parameter_by_name("preview") is None
