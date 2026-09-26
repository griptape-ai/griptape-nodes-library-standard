import io
from enum import StrEnum
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list
from PIL import Image

from griptape_nodes_library.utils.image_utils import (
    composite_over,
    dict_to_image_url_artifact,
    image_to_bytes,
    load_pil_from_url,
    scale_alpha,
    unpremultiply_rgba,
)


class Layout(StrEnum):
    """How the input images are arranged in the merged result."""

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    GRID = "grid"
    COMPOSITE = "quick composite"


class AlphaConvention(StrEnum):
    """Whether an input image's colour channels have been scaled by its alpha."""

    STRAIGHT = "straight"
    PREMULTIPLIED = "premultiplied"


class Background(StrEnum):
    """What fills the areas of the merged result no image covers."""

    TRANSPARENT = "transparent"
    WHITE = "white"
    BLACK = "black"


BACKGROUND_COLORS = {
    Background.WHITE: (255, 255, 255, 255),
    Background.BLACK: (0, 0, 0, 255),
}


class MergeImages(ControlNode):
    """Node for merging images together in different layouts with grid options and dynamic image input list."""

    MAX_COLUMNS = 3
    FIT_IMAGES = True

    MIN_OPACITY = 0.0
    MAX_OPACITY = 1.0
    DEFAULT_OPACITY = 1.0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Add ParameterList for images (dynamic add/remove, max 4)
        self.add_parameter(
            ParameterList(
                name="Images",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                default_value=None,
                tooltip="Images to merge (add up to 4)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"max": 4, "min": 1, "display_name": "Images"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="layout",
                tooltip="Select how to arrange the images",
                default_value=Layout.HORIZONTAL,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=list(Layout))},
            )
        )

        self.add_parameter(
            ParameterString(
                name="input_alpha",
                tooltip=(
                    "Alpha convention of the input images. Most generators emit straight alpha; "
                    "renderers and keyers often emit premultiplied. Choosing the wrong one fringes "
                    "soft edges — dark if premultiplied images are treated as straight, bright the other way."
                ),
                default_value=AlphaConvention.STRAIGHT,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=list(AlphaConvention))},
            )
        )

        opacity_parameter = ParameterFloat(
            name="opacity",
            tooltip=(
                f"Opacity of each image as it is laid down ({self.MIN_OPACITY} = transparent, "
                f"{self.MAX_OPACITY} = fully opaque). In 'quick composite' the bottom image stays "
                "fully opaque and this fades the ones stacked over it; in the other layouts it fades "
                "each image toward the background."
            ),
            default_value=self.DEFAULT_OPACITY,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        opacity_parameter.add_trait(Slider(min_val=self.MIN_OPACITY, max_val=self.MAX_OPACITY))
        self.add_parameter(opacity_parameter)

        self.add_parameter(
            ParameterString(
                name="background",
                tooltip="What fills the areas no image covers, and whether the result keeps its alpha channel",
                default_value=Background.WHITE,
                allowed_modes={ParameterMode.PROPERTY},
                traits={Options(choices=list(Background))},
            )
        )

        # Add output parameter
        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="The merged image",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="merged.png")
        self._output_file.add_parameter()

    def get_images(self) -> list:
        images = self.get_parameter_value("Images")
        if images:
            if not isinstance(images, list):
                images = [images]
            # Normalize string paths to ImageUrlArtifact
            images = normalize_artifact_list(images, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            return images[:4]  # Enforce max 4
        return []

    def _convert_to_pil_image(self, img: Any) -> Image.Image:
        """Convert various image types to PIL Image."""
        if isinstance(img, dict):
            img = dict_to_image_url_artifact(img)
        if isinstance(img, ImageArtifact):
            # ImageArtifact has base64 data
            img = Image.open(io.BytesIO(img.to_bytes()))
        elif isinstance(img, ImageUrlArtifact):
            # Use load_pil_from_url so macro paths (e.g. "{inputs}/x.png") resolve via File.
            img = load_pil_from_url(img.value)
        return img

    def _prepare_image(self, img: Image.Image, alpha_convention: AlphaConvention) -> Image.Image:
        """Bring an input image into RGBA straight alpha, the space every layout works in."""
        rgba = img if img.mode == "RGBA" else img.convert("RGBA")
        match alpha_convention:
            case AlphaConvention.STRAIGHT:
                return rgba
            case AlphaConvention.PREMULTIPLIED:
                return unpremultiply_rgba(rgba)
            case _:
                msg = f"Unknown alpha convention: {alpha_convention!r}"
                raise ValueError(msg)

    def _resize_image(self, img: Image.Image, target_width: int, target_height: int) -> Image.Image:
        """Resize image while preserving aspect ratio."""
        img_ratio = img.width / img.height
        target_ratio = target_width / target_height

        if img_ratio > target_ratio:
            new_width = target_width
            new_height = int(target_width / img_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * img_ratio)

        # Pillow premultiplies RGBA internally while resampling, so soft edges don't pick
        # up colour from transparent pixels.
        return img.resize((max(new_width, 1), max(new_height, 1)), Image.Resampling.LANCZOS)

    def _process_horizontal_layout(self, images: list[Image.Image]) -> Image.Image:
        # Resize all images to the same height (max height), preserving aspect ratio
        max_height = max(img.height for img in images)
        resized_images = [
            self._resize_image(img, int(img.width * max_height / img.height), max_height) for img in images
        ]
        total_width = sum(img.width for img in resized_images)
        merged = Image.new("RGBA", (total_width, max_height), (0, 0, 0, 0))
        x_offset = 0
        for img in resized_images:
            merged.paste(img, (x_offset, 0))
            x_offset += img.width
        return merged

    def _process_vertical_layout(self, images: list[Image.Image]) -> Image.Image:
        # Resize all images to the same width (max width), preserving aspect ratio
        max_width = max(img.width for img in images)
        resized_images = [self._resize_image(img, max_width, int(img.height * max_width / img.width)) for img in images]
        total_height = sum(img.height for img in resized_images)
        merged = Image.new("RGBA", (max_width, total_height), (0, 0, 0, 0))
        y_offset = 0
        for img in resized_images:
            merged.paste(img, (0, y_offset))
            y_offset += img.height
        return merged

    def _process_grid_layout(self, images: list[Image.Image]) -> Image.Image:
        n_images = len(images)
        columns = n_images if n_images <= self.MAX_COLUMNS else 2
        rows = (n_images + columns - 1) // columns

        cell_width = int(sum(img.width for img in images) / n_images)
        cell_height = int(sum(img.height for img in images) / n_images)

        merged = Image.new("RGBA", (cell_width * columns, cell_height * rows), (0, 0, 0, 0))

        for idx, img in enumerate(images):
            row = idx // columns
            col = idx % columns
            resized_img = self._resize_image(img, cell_width, cell_height)
            x_offset = col * cell_width + (cell_width - resized_img.width) // 2
            y_offset = row * cell_height + (cell_height - resized_img.height) // 2
            merged.paste(resized_img, (x_offset, y_offset))

        return merged

    def _process_composite_layout(self, images: list[Image.Image]) -> Image.Image:
        if not images:
            msg = "No images provided for composite layout"
            raise ValueError(msg)
        # Reverse the order so the last image is at the bottom (like a stack)
        reversed_images = list(reversed(images))
        base = reversed_images[0].copy()
        base_width, base_height = base.size
        for img in reversed_images[1:]:
            resized_overlay = self._resize_image(img, base_width, base_height)
            # Center the overlay on the base
            x_offset = (base_width - resized_overlay.width) // 2
            y_offset = (base_height - resized_overlay.height) // 2
            base = composite_over(base, resized_overlay, (x_offset, y_offset))
        return base

    def _apply_background(self, merged: Image.Image, background: Background) -> Image.Image:
        """Flatten the merged result onto a solid background, or leave its alpha intact."""
        match background:
            case Background.TRANSPARENT:
                return merged
            case Background.WHITE | Background.BLACK:
                canvas = Image.new("RGBA", merged.size, BACKGROUND_COLORS[background])
                return Image.alpha_composite(canvas, merged).convert("RGB")
            case _:
                msg = f"Unknown background: {background!r}"
                raise ValueError(msg)

    def _merge(self, images: list[Image.Image], layout: Layout) -> Image.Image:
        match layout:
            case Layout.HORIZONTAL:
                return self._process_horizontal_layout(images)
            case Layout.VERTICAL:
                return self._process_vertical_layout(images)
            case Layout.GRID:
                return self._process_grid_layout(images)
            case Layout.COMPOSITE:
                return self._process_composite_layout(images)
            case _:
                msg = f"Unknown layout: {layout!r}"
                raise ValueError(msg)

    def _apply_opacity(self, images: list[Image.Image], opacity: float, layout: Layout) -> list[Image.Image]:
        """Fade the images opacity applies to for this layout.

        A stack needs something to show through to, so the bottom of a composite keeps
        its own alpha; every other layout puts each image straight onto the background.
        """
        if opacity >= self.MAX_OPACITY:
            return images
        faded = [scale_alpha(img, opacity) for img in images]
        if layout is Layout.COMPOSITE and images:
            faded[-1] = images[-1]
        return faded

    def process(self) -> None:
        self.parameter_output_values["output"] = None
        alpha_convention = AlphaConvention(self.get_parameter_value("input_alpha"))
        images = [
            self._prepare_image(self._convert_to_pil_image(img), alpha_convention)
            for img in self.get_images()
            if img is not None
        ]

        if not images:
            return

        layout = Layout(self.get_parameter_value("layout"))
        opacity = min(max(float(self.get_parameter_value("opacity")), self.MIN_OPACITY), self.MAX_OPACITY)
        background = Background(self.get_parameter_value("background"))

        merged = self._merge(self._apply_opacity(images, opacity, layout), layout)
        merged = self._apply_background(merged, background)

        # Save output image
        image_bytes = image_to_bytes(merged, "PNG")
        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)
        url_artifact = ImageUrlArtifact(saved.location)
        self.set_parameter_value("output", url_artifact)
        self.parameter_output_values["output"] = url_artifact
