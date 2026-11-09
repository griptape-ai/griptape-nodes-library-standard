from enum import StrEnum
from typing import Any, ClassVar

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline, ResizeFilter
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    MAX_UINT8,
    ImageMode,
    composite_over,
    converted,
    dict_to_image_url_artifact,
    edited_pixels,
    load_image_from_url,
    pixels_of,
)

# Channel values below this are the dark half for overlay and hard light
_HALF_LEVEL = 128
# The divisor Pillow's overlay and hard light scale their products by
_OVERLAY_DIVISOR = 127
# The divisor Pillow's soft light scales its first term by
_SOFT_LIGHT_DIVISOR = 65536


class BlendMode(StrEnum):
    """How the blend image's color combines with the base image's."""

    NORMAL = "normal"
    MULTIPLY = "multiply"
    SCREEN = "screen"
    OVERLAY = "overlay"
    SOFT_LIGHT = "soft_light"
    HARD_LIGHT = "hard_light"
    DARKEN = "darken"
    LIGHTEN = "lighten"
    DIFFERENCE = "difference"
    EXCLUSION = "exclusion"
    ADD = "add"
    SUBTRACT = "subtract"
    LOGICAL_AND = "logical_and"
    LOGICAL_OR = "logical_or"
    LOGICAL_XOR = "logical_xor"


def _blend_pixels(base: np.ndarray, blend: np.ndarray, mode: BlendMode) -> np.ndarray:
    """Combine two same-size uint8 RGB arrays with ``mode``, using the integer math of Pillow's ``ImageChops``.

    The logical modes work per channel, treating any nonzero value as on, as ``ImageChops`` does
    for 1-bit images.

    Raises:
        ValueError: If the mode is unknown
    """
    a = base.astype(np.int32)
    b = blend.astype(np.int32)
    match mode:
        case BlendMode.NORMAL:
            result = b
        case BlendMode.MULTIPLY:
            # Float math rather than integer, as this node always computed multiply
            product = (a.astype(np.float32) / 255.0) * (b.astype(np.float32) / 255.0)
            return np.clip(product * 255.0, 0, MAX_UINT8).astype(np.uint8)
        case BlendMode.SCREEN:
            result = MAX_UINT8 - (MAX_UINT8 - a) * (MAX_UINT8 - b) // MAX_UINT8
        case BlendMode.OVERLAY:
            dark = a * b // _OVERLAY_DIVISOR
            light = MAX_UINT8 - (MAX_UINT8 - a) * (MAX_UINT8 - b) // _OVERLAY_DIVISOR
            result = np.where(a < _HALF_LEVEL, dark, light)
        case BlendMode.SOFT_LIGHT:
            result = (MAX_UINT8 - a) * (a * b) // _SOFT_LIGHT_DIVISOR + (
                a * (MAX_UINT8 - (MAX_UINT8 - a) * (MAX_UINT8 - b) // MAX_UINT8)
            ) // MAX_UINT8
        case BlendMode.HARD_LIGHT:
            dark = a * b // _OVERLAY_DIVISOR
            light = MAX_UINT8 - (MAX_UINT8 - b) * (MAX_UINT8 - a) // _OVERLAY_DIVISOR
            result = np.where(b < _HALF_LEVEL, dark, light)
        case BlendMode.DARKEN:
            result = np.minimum(a, b)
        case BlendMode.LIGHTEN:
            result = np.maximum(a, b)
        case BlendMode.DIFFERENCE:
            result = np.abs(a - b)
        case BlendMode.EXCLUSION:
            result = a + b - 2 * a * b // MAX_UINT8
        case BlendMode.ADD:
            result = a + b
        case BlendMode.SUBTRACT:
            result = a - b
        case BlendMode.LOGICAL_AND:
            result = np.where((a != 0) & (b != 0), MAX_UINT8, 0)
        case BlendMode.LOGICAL_OR:
            result = np.where((a != 0) | (b != 0), MAX_UINT8, 0)
        case BlendMode.LOGICAL_XOR:
            result = np.where((a != 0) ^ (b != 0), MAX_UINT8, 0)
        case _:
            msg = f"Unknown blend mode: {mode!r}"
            raise ValueError(msg)
    return np.clip(result, 0, MAX_UINT8).astype(np.uint8)


def _with_scaled_alpha(image: GriptapeImage, opacity: float) -> GriptapeImage:
    """Return an RGBA image with its alpha multiplied by ``opacity``, rounded down."""

    def scaled(pixels: np.ndarray) -> np.ndarray:
        result = pixels.copy()
        result[:, :, 3] = (pixels[:, :, 3] * opacity).astype(np.uint8)
        return result

    return edited_pixels(image, ImageMode.RGBA, scaled)


class ImageBlendCompositor(BaseImageProcessor):
    """Compose two images using various blend modes with positioning and advanced compositing options."""

    # Blend mode options
    BLEND_MODE_OPTIONS: ClassVar[list[str]] = list(BlendMode)

    # Opacity constants
    MIN_OPACITY = 0.0
    MAX_OPACITY = 1.0
    DEFAULT_OPACITY = 1.0

    # Position constants
    DEFAULT_POSITION = 0

    def _setup_custom_parameters(self) -> None:
        """Setup blend compositor parameters."""
        # Input parameters
        with ParameterGroup(name="inputs", ui_options={"collapsed": False}) as inputs_group:
            blend_image_param = ParameterImage(
                name="blend_image",
                tooltip="The image to blend/composite onto the base image",
            )
            self.add_parameter(blend_image_param)

        self.add_node_element(inputs_group)

        # Blend settings
        with ParameterGroup(name="blend_settings", ui_options={"collapsed": False}) as blend_group:
            # Blend mode
            blend_mode_param = ParameterString(
                name="blend_mode",
                default_value="normal",
                tooltip="The blend mode to use for compositing",
            )
            blend_mode_param.add_trait(Options(choices=self.BLEND_MODE_OPTIONS))
            self.add_parameter(blend_mode_param)

            # Opacity
            opacity_param = ParameterFloat(
                name="opacity",
                default_value=self.DEFAULT_OPACITY,
                tooltip=f"Opacity of the blend image ({self.MIN_OPACITY}-{self.MAX_OPACITY}, 0.0 = transparent, 1.0 = fully opaque)",
            )
            opacity_param.add_trait(Slider(min_val=self.MIN_OPACITY, max_val=self.MAX_OPACITY))
            self.add_parameter(opacity_param)

        self.add_node_element(blend_group)

        # Position and sizing
        with ParameterGroup(name="position_and_sizing", ui_options={"collapsed": False}) as position_group:
            # Blend position X
            blend_position_x_param = ParameterInt(
                name="blend_position_x",
                default_value=self.DEFAULT_POSITION,
                tooltip="X-coordinate position of the blend image (0 = center, negative = left, positive = right)",
            )
            self.add_parameter(blend_position_x_param)

            # Blend position Y
            blend_position_y_param = ParameterInt(
                name="blend_position_y",
                default_value=self.DEFAULT_POSITION,
                tooltip="Y-coordinate position of the blend image (0 = center, negative = up, positive = down)",
            )
            self.add_parameter(blend_position_y_param)

            # Resize blend to fit
            resize_blend_to_fit_param = ParameterBool(
                name="resize_blend_to_fit",
                default_value=False,
                tooltip="Whether to resize the blend image to match the base image dimensions",
            )
            self.add_parameter(resize_blend_to_fit_param)

        self.add_node_element(position_group)

        # Advanced options
        with ParameterGroup(name="advanced_options", ui_options={"collapsed": True}) as advanced_group:
            # Preserve alpha
            preserve_alpha_param = ParameterBool(
                name="preserve_alpha",
                default_value=True,
                tooltip="Whether to preserve the alpha channel of the blend image",
            )
            self.add_parameter(preserve_alpha_param)

            # Invert blend
            invert_blend_param = ParameterBool(
                name="invert_blend",
                default_value=False,
                tooltip="Whether to invert the blend image before compositing",
            )
            self.add_parameter(invert_blend_param)

        self.add_node_element(advanced_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Process image automatically when inputs or parameters change."""
        if parameter.name in ["input_image", "blend_image"] and value is not None:
            # Check if both images are available
            image = self.get_parameter_value("input_image")
            blend_image = self.get_parameter_value("blend_image")
            if image is not None and blend_image is not None:
                self._process_images_immediately(image, blend_image)
        elif parameter.name in [
            "blend_mode",
            "opacity",
            "blend_position_x",
            "blend_position_y",
            "resize_blend_to_fit",
            "preserve_alpha",
            "invert_blend",
        ]:
            # Process when blend parameters change (for live preview)
            image = self.get_parameter_value("input_image")
            blend_image = self.get_parameter_value("blend_image")
            if image is not None and blend_image is not None:
                self._process_images_immediately(image, blend_image)
        return super().after_value_set(parameter, value)

    def _process_images_immediately(self, image_value: Any, blend_image_value: Any) -> None:
        """Process images immediately for live preview."""
        try:
            # Convert to ImageUrlArtifact if needed
            if isinstance(image_value, dict):
                image_artifact = dict_to_image_url_artifact(image_value)
            else:
                image_artifact = image_value

            if isinstance(blend_image_value, dict):
                blend_image_artifact = dict_to_image_url_artifact(blend_image_value)
            else:
                blend_image_artifact = blend_image_value

            # Load images
            image = load_image_from_url(image_artifact.value)
            blend = load_image_from_url(blend_image_artifact.value)

            # Process with current settings
            processed_image = self._process_images(image, blend, **self._get_custom_parameters())

            # Save and set output
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            # Log error but don't fail the node
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image blend compositing"

    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:  # noqa: ARG002
        """Process a single image (required by BaseImageProcessor)."""
        # This method is required by the abstract base class but not used for this node
        # since we work with two images. Return the input image unchanged.
        return image

    def _process_images(self, image: GriptapeImage, blend: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the images by compositing them together."""
        blend_mode = kwargs.get("blend_mode", BlendMode.NORMAL)
        opacity = kwargs.get("opacity", self.DEFAULT_OPACITY)
        blend_position_x = kwargs.get("blend_position_x", self.DEFAULT_POSITION)
        blend_position_y = kwargs.get("blend_position_y", self.DEFAULT_POSITION)
        resize_blend_to_fit = kwargs.get("resize_blend_to_fit", False)
        preserve_alpha = kwargs.get("preserve_alpha", True)
        invert_blend = kwargs.get("invert_blend", False)

        # Convert images to RGBA for alpha channel support
        image = converted(image, ImageMode.RGBA)
        blend = converted(blend, ImageMode.RGBA)

        # Invert blend image if requested (preserve alpha channel)
        if invert_blend:
            blend = GriptapeImagePipeline(blend).invert().end()

        # Resize blend image if requested (a same-size resize would only soften it)
        if resize_blend_to_fit and (blend.width, blend.height) != (image.width, image.height):
            blend = GriptapeImagePipeline(blend).resize(image.width, image.height, ResizeFilter.LANCZOS).end()

        # Calculate position (center blend image by default)
        if blend_position_x == 0 and blend_position_y == 0:
            # Center the blend image
            x = (image.width - blend.width) // 2
            y = (image.height - blend.height) // 2
        else:
            # Use specified position
            x = blend_position_x
            y = blend_position_y

        # Apply blend mode
        if blend_mode == BlendMode.NORMAL:
            # Simple alpha compositing - multiply existing alpha by opacity
            # to preserve gradient masks (like blurred edges)
            if opacity < 1.0:
                blend = _with_scaled_alpha(blend, opacity)
            blended_image = blend
        else:
            # Apply custom blend mode with proper alpha compositing
            blended_image = self._apply_blend_mode(image, blend, BlendMode(blend_mode), opacity)

        # Apply the blended image with proper alpha handling
        return self._apply_blended_image(image, blended_image, (x, y), preserve_alpha=preserve_alpha)

    def _apply_blended_image(
        self,
        result: GriptapeImage,
        blended_image: GriptapeImage,
        position: tuple[int, int],
        *,
        preserve_alpha: bool,
    ) -> GriptapeImage:
        """Apply the blended image to the result with proper alpha compositing.

        Uses Porter-Duff "over" compositing for correct handling of semi-transparent
        areas (like blurred mask edges).
        """
        if not preserve_alpha:
            # Paste RGB only, ignoring alpha: an opaque layer replaces what it covers
            blended_image = converted(converted(blended_image, ImageMode.RGB), ImageMode.RGBA)
        return composite_over(result, blended_image, position)

    def _apply_blend_mode(
        self, base: GriptapeImage, blend: GriptapeImage, mode: BlendMode, opacity: float
    ) -> GriptapeImage:
        """Apply a specific blend mode to two RGBA images.

        Returns the blend result with alpha channel set (blend's alpha * opacity).
        The actual compositing onto the base is done by _apply_blended_image.
        This avoids double-application of alpha which causes darkening artifacts.
        """
        # Ensure both images are the same size for blending
        if (base.width, base.height) != (blend.width, blend.height):
            blend = GriptapeImagePipeline(blend).resize(base.width, base.height, ResizeFilter.LANCZOS).end()

        base_rgb = pixels_of(base, ImageMode.RGB)

        # Blend the color, then set alpha: blend's alpha multiplied by opacity.
        # This preserves gradient masks (like blurred edges)
        def blended(pixels: np.ndarray) -> np.ndarray:
            rgb = _blend_pixels(base_rgb, pixels[:, :, :3], mode)
            alpha = pixels[:, :, 3]
            if opacity < 1.0:
                # Scale alpha by opacity factor, preserving the gradient
                alpha = (alpha * opacity).astype(np.uint8)
            return np.dstack([rgb, alpha])

        return edited_pixels(blend, ImageMode.RGBA, blended)

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate blend compositor parameters."""
        exceptions = []

        # Check that both images are provided
        input_image = self.get_parameter_value("input_image")
        blend_image = self.get_parameter_value("blend_image")
        if input_image is None:
            exceptions.append(ValueError("Connect an image to 'Input Image'."))
        if blend_image is None:
            exceptions.append(ValueError("Connect an image to 'Blend Image'."))

        # Validate opacity
        opacity = self.get_parameter_value("opacity")
        if opacity is not None and (opacity < self.MIN_OPACITY or opacity > self.MAX_OPACITY):
            msg = f"'Opacity' must be between {self.MIN_OPACITY} and {self.MAX_OPACITY}, got {opacity}."
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get blend compositor parameters."""
        return {
            "blend_mode": self.get_parameter_value("blend_mode"),
            "opacity": self.get_parameter_value("opacity"),
            "blend_position_x": self.get_parameter_value("blend_position_x"),
            "blend_position_y": self.get_parameter_value("blend_position_y"),
            "resize_blend_to_fit": self.get_parameter_value("resize_blend_to_fit"),
            "preserve_alpha": self.get_parameter_value("preserve_alpha"),
            "invert_blend": self.get_parameter_value("invert_blend"),
        }

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_blend"

    async def aprocess(self) -> None:
        """Main workflow execution method."""
        # Reset execution state and clear status
        self._clear_execution_status()

        # Get input images
        base_image = self.get_parameter_value("input_image")
        blend_image = self.get_parameter_value("blend_image")

        if base_image is None or blend_image is None:
            return

        # Process the images directly for workflow execution
        try:
            # Convert to ImageUrlArtifact if needed
            if isinstance(base_image, dict):
                base_image = dict_to_image_url_artifact(base_image)

            if isinstance(blend_image, dict):
                blend_image = dict_to_image_url_artifact(blend_image)

            # Load images
            image = load_image_from_url(base_image.value)
            blend = load_image_from_url(blend_image.value)

            # Process with current settings
            processed_image = self._process_images(image, blend, **self._get_custom_parameters())

            # Save and set output
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

            # Set success status with detailed information
            success_details = (
                f"Successfully composed images: {self._get_processing_description()}\n"
                f"Base: {image.width}x{image.height}\n"
                f"Blend: {blend.width}x{blend.height}\n"
                f"Output: {processed_image.width}x{processed_image.height}"
            )
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")

        except Exception as e:
            error_message = str(e)

            # Set failure status with detailed error information
            failure_details = f"Image composition failed: {self._get_processing_description()}\nError: {error_message}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")

            # Handle failure based on whether failure output is connected
            self._handle_failure_exception(e)
