"""Tests for the live preview ImagePreviewMixin gives image and video nodes that name a preview operation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList
from PIL import Image

from griptape_nodes_library.files.extract_pdf_images import ExtractPdfImages
from griptape_nodes_library.image import image_grid_splitter, image_preview
from griptape_nodes_library.image.add_text_to_existing_image import AddTextToExistingImage
from griptape_nodes_library.image.add_text_to_image import AddTextToImage
from griptape_nodes_library.image.adjust_image_eq import AdjustImageEQ
from griptape_nodes_library.image.adjust_image_levels import AdjustImageLevels
from griptape_nodes_library.image.bloom_effect import BloomEffect
from griptape_nodes_library.image.combine_masks import CombineMasks
from griptape_nodes_library.image.create_color_bars import CreateColorBars
from griptape_nodes_library.image.display_channel import DisplayChannel
from griptape_nodes_library.image.display_mask import DisplayMask
from griptape_nodes_library.image.extend_canvas import ExtendCanvas
from griptape_nodes_library.image.flip_image import FlipImage
from griptape_nodes_library.image.gaussian_blur_image import GaussianBlurImage
from griptape_nodes_library.image.gaussian_edge_fade import GaussianEdgeFade
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
from griptape_nodes_library.video.resize_video import ResizeVideo

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
    ExtractPdfImages,
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
    GaussianEdgeFade,
    CombineMasks,
    ResizeVideo,
]

WITH_SOURCE = [node_class for node_class in PREVIEWED if node_class.PREVIEW_IMAGES]

WITH_MODE = [InvertImage, InvertMask, DisplayMask, DisplayChannel, GaussianEdgeFade]

OPERATIONS_TS = Path(__file__).parents[3] / "widgets" / "src" / "ImageProcessorPreview" / "_operations.ts"


@pytest.fixture(autouse=True)
def no_downloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nodes that read an image's mode or detect a grid load the image; give them one without fetching."""

    def load(_url: str) -> Image.Image:
        return Image.new("RGB", (8, 8))

    monkeypatch.setattr(image_preview, "load_pil_from_url", load)
    monkeypatch.setattr(image_grid_splitter, "load_pil_from_url", load)


@pytest.fixture(autouse=True)
def no_static_server(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking the engine for a static URL to a file that exists segfaults the interpreter at exit.

    The text nodes ask for the URL of Pillow's font while they are built, so answer with the path.
    """
    monkeypatch.setattr(ImagePreviewMixin, "_static_url", lambda _self, path: path)


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
    exported = re.search(r"export const OPERATIONS = \{(.*?)\};", OPERATIONS_TS.read_text(), re.DOTALL)
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


@pytest.mark.parametrize("node_class", WITH_SOURCE)
def test_a_loaded_source_reaches_the_preview(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")
    [(parameter_name, key)] = node_class.PREVIEW_IMAGES.items()
    if key == "video_url":
        media = VideoUrlArtifact(value="https://example.com/cat.mp4")
    else:
        media = ImageUrlArtifact(value="https://example.com/cat.png")

    # As a workflow loads: initial_setup skips after_value_set, which used to leave the preview empty.
    node.set_parameter_value(parameter_name, media, initial_setup=True)

    assert node.get_parameter_value(node_class.PREVIEW_PARAMETER_NAME)[key] == media.value


@pytest.mark.parametrize("node_class", WITH_SOURCE)
def test_the_preview_says_which_kind_of_media_it_shows(node_class: type[ImagePreviewMixin]) -> None:
    node = node_class(name="test_node")
    [(parameter_name, key)] = node_class.PREVIEW_IMAGES.items()
    expected = "video" if key == "video_url" else "image"

    # Before anything is connected, so the widget asks for the right kind of media. A video node has no video_url
    # in its default value, so the widget can't tell from that.
    assert node.get_parameter_value(node_class.PREVIEW_PARAMETER_NAME)["media"] == expected

    # Setting the source to nothing rebuilds the preview, which still says it.
    node.set_parameter_value(parameter_name, None)
    assert node.get_parameter_value(node_class.PREVIEW_PARAMETER_NAME)["media"] == expected


def test_combine_masks_items_feed_the_preview() -> None:
    node = CombineMasks(name="test_node")
    masks = node.get_parameter_by_name("masks")
    assert isinstance(masks, ParameterList)
    item = masks.add_child_parameter()

    assert node._is_preview_source("masks")
    assert node._is_preview_source(item.name)
    assert not node._is_preview_source("output_mask")


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


@pytest.mark.parametrize("initial_setup", [False, True])
def test_grid_splitter_with_an_unreadable_image_has_an_empty_preview(
    monkeypatch: pytest.MonkeyPatch, *, initial_setup: bool
) -> None:
    def unreadable(_url: str) -> Image.Image:
        msg = "the file is gone"
        raise ValueError(msg)

    monkeypatch.setattr(image_grid_splitter, "load_pil_from_url", unreadable)
    node = ImageGridSplitter(name="test_node")

    # Loading a workflow whose image was moved or whose URL expired must not fail the value set.
    node.set_parameter_value(
        "input_image", ImageUrlArtifact(value="https://example.com/gone.png"), initial_setup=initial_setup
    )

    preview = node.get_parameter_value("grid_preview")
    assert preview["operation"] == ImageGridSplitter.PREVIEW_OPERATION
    assert "detected_rows" not in preview


def test_combine_masks_sends_every_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    node = CombineMasks(name="test_node")
    masks = [ImageUrlArtifact(value="https://example.com/a.png"), ImageUrlArtifact(value="https://example.com/b.png")]
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: masks)

    extras = node._preview_extras()

    assert extras["images"] == [
        {"url": "https://example.com/a.png", "mode": "RGB"},
        {"url": "https://example.com/b.png", "mode": "RGB"},
    ]
    assert extras["skipped"] == 0


def test_combine_masks_keeps_modes_with_their_masks_when_one_cant_be_previewed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = CombineMasks(name="test_node")
    no_file = ImageUrlArtifact(value="")
    masks = [no_file, ImageUrlArtifact(value="https://example.com/b.png")]
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: masks)

    extras = node._preview_extras()

    assert extras["images"] == [{"url": "https://example.com/b.png", "mode": "RGB"}]
    assert extras["skipped"] == 1


@pytest.mark.parametrize("node_class", [AddTextToImage, AddTextToExistingImage])
def test_a_saved_font_url_is_replaced_with_this_sessions(
    node_class: type[ImagePreviewMixin], monkeypatch: pytest.MonkeyPatch
) -> None:
    node = node_class(name="test_node")
    monkeypatch.setattr(node, "_pillow_font_url", lambda: "http://localhost:8124/fresh-font.ttf")
    saved = {**node.get_parameter_value("preview"), "font_url": "http://elsewhere:8124/stale-font.ttf"}

    # Loading a workflow sets values with initial_setup, which skips after_value_set.
    node.set_parameter_value("preview", saved, initial_setup=True)

    assert node.get_parameter_value("preview")["font_url"] == "http://localhost:8124/fresh-font.ttf"


def test_a_node_without_an_operation_has_no_preview() -> None:
    node = ImageBlendCompositor(name="test_node")

    assert node.get_parameter_by_name("preview") is None
