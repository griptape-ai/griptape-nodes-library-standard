from enum import StrEnum
from typing import Any

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image, ImageChops

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_pil_from_url,
)


class BlendMode(StrEnum):
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


class ImageBlendCompositor(BaseImageProcessor):
    """Compose two images using various blend modes with positioning and advanced compositing options."""

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
                default_value=BlendMode.NORMAL.value,
                tooltip="The blend mode to use for compositing",
            )
            blend_mode_param.add_trait(Options(choices=[mode.value for mode in BlendMode]))
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
                tooltip="X pixel of the blend image's left edge on the base image. When X and Y are both 0, the blend image is centered.",
            )
            self.add_parameter(blend_position_x_param)

            # Blend position Y
            blend_position_y_param = ParameterInt(
                name="blend_position_y",
                default_value=self.DEFAULT_POSITION,
                tooltip="Y pixel of the blend image's top edge on the base image. When X and Y are both 0, the blend image is centered.",
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

            # Load PIL images
            image_pil = load_pil_from_url(image_artifact.value)
            blend_pil = load_pil_from_url(blend_image_artifact.value)

            # Process with current settings
            processed_image = self._process_images(image_pil, blend_pil, **self._get_custom_parameters())

            # Save and set output
            output_format = self._get_output_format(self._auto_output_format(image_pil))
            output_artifact = self._save_image_artifact(processed_image, output_format)

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            # Log error but don't fail the node
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _auto_output_format(self, image_pil: Image.Image) -> str:
        """The base image's format if it can be saved, otherwise PNG."""
        detected_format = self._detect_image_format(image_pil)
        if detected_format in self.IMAGE_FORMAT_OPTIONS or detected_format in self.EXTENDED_FORMAT_OPTIONS:
            return detected_format
        return "PNG"

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image blend compositing"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:  # noqa: ARG002
        """Process a single image (required by BaseImageProcessor)."""
        # This method is required by the abstract base class but not used for this node
        # since we work with two images. Return the input image unchanged.
        return pil_image

    def _process_images(self, image_pil: Image.Image, blend_pil: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL images by compositing them together."""
        blend_mode = kwargs.get("blend_mode", BlendMode.NORMAL)
        opacity = kwargs.get("opacity", self.DEFAULT_OPACITY)
        blend_position_x = kwargs.get("blend_position_x", self.DEFAULT_POSITION)
        blend_position_y = kwargs.get("blend_position_y", self.DEFAULT_POSITION)
        resize_blend_to_fit = kwargs.get("resize_blend_to_fit", False)
        preserve_alpha = kwargs.get("preserve_alpha", True)
        invert_blend = kwargs.get("invert_blend", False)

        # Convert images to RGBA for alpha channel support
        if image_pil.mode != "RGBA":
            image_pil = image_pil.convert("RGBA")
        if blend_pil.mode != "RGBA":
            blend_pil = blend_pil.convert("RGBA")

        # Invert blend image if requested (preserve alpha channel)
        if invert_blend:
            # Extract alpha channel before inverting
            alpha_channel = blend_pil.getchannel("A") if blend_pil.mode == "RGBA" else None

            # Invert only the RGB channels
            rgb_channels = blend_pil.convert("RGB")
            inverted_rgb = ImageChops.invert(rgb_channels)

            # Reconstruct the image with original alpha
            if alpha_channel is not None:
                blend_pil = Image.merge("RGBA", (*inverted_rgb.split(), alpha_channel))
            else:
                blend_pil = inverted_rgb.convert("RGBA")

        # Non-normal modes blend pixel by pixel against the whole base, so they need the blend at the
        # base's size before the position is worked out
        if resize_blend_to_fit or blend_mode != BlendMode.NORMAL:
            blend_pil = blend_pil.resize(image_pil.size, Image.Resampling.LANCZOS)

        # Calculate position (center blend image by default)
        if blend_position_x == 0 and blend_position_y == 0:
            # Center the blend image
            x = (image_pil.width - blend_pil.width) // 2
            y = (image_pil.height - blend_pil.height) // 2
        else:
            # Use specified position
            x = blend_position_x
            y = blend_position_y

        # Create a new image with the base image
        result = image_pil.copy()

        # Apply blend mode
        if blend_mode == BlendMode.NORMAL:
            # Simple alpha compositing - multiply existing alpha by opacity
            # to preserve gradient masks (like blurred edges)
            if opacity < 1.0:
                alpha = blend_pil.getchannel("A")
                # Scale alpha by opacity factor, preserving the gradient
                alpha = alpha.point(lambda a: int(a * opacity))
                blend_pil = blend_pil.copy()
                blend_pil.putalpha(alpha)
            blended_image = blend_pil
        else:
            # Apply custom blend mode with proper alpha compositing
            blended_image = self._apply_blend_mode(image_pil, blend_pil, blend_mode, opacity)

        # Apply the blended image with proper alpha handling
        result = self._apply_blended_image(result, blended_image, (x, y), preserve_alpha=preserve_alpha)

        return result

    def _apply_blended_image(
        self,
        result: Image.Image,
        blended_image: Image.Image,
        position: tuple[int, int],
        *,
        preserve_alpha: bool,
    ) -> Image.Image:
        """Apply the blended image to the result with proper alpha compositing.

        Uses Porter-Duff "over" compositing via alpha_composite for correct
        handling of semi-transparent areas (like blurred mask edges).
        """
        if preserve_alpha and blended_image.mode == "RGBA":
            # Use alpha_composite for proper Porter-Duff "over" compositing
            # This correctly handles semi-transparent areas without darkening
            # Create a transparent canvas the same size as result
            overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
            overlay.paste(blended_image, position)
            result = Image.alpha_composite(result, overlay)
        elif blended_image.mode == "RGBA" and not preserve_alpha:
            # Paste RGB only, ignoring alpha
            result.paste(blended_image.convert("RGB"), position)
        else:
            result.paste(blended_image, position)
        return result

    def _apply_blend_mode(self, base: Image.Image, blend: Image.Image, mode: str, opacity: float) -> Image.Image:
        """Apply a specific blend mode to two images.

        Returns the blend result with alpha channel set (blend's alpha * opacity).
        The actual compositing onto the base is done by _apply_blended_image.
        This avoids double-application of alpha which causes darkening artifacts.
        """
        # Ensure both images are the same size for blending
        if base.size != blend.size:
            blend = blend.resize(base.size, Image.Resampling.LANCZOS)

        # Convert to RGB for mathematical operations (ImageChops works with RGB)
        base_rgb = base.convert("RGB")
        blend_rgb = blend.convert("RGB")

        # Apply the blend mode using a mapping approach
        blend_result = self._get_blend_result(base_rgb, blend_rgb, mode)

        # Convert to RGBA
        result_rgba = blend_result.convert("RGBA")

        # Set alpha channel: blend's alpha multiplied by opacity
        # This preserves gradient masks (like blurred edges)
        if blend.mode == "RGBA":
            blend_alpha = blend.getchannel("A")
            if opacity < 1.0:
                # Scale alpha by opacity factor, preserving the gradient
                blend_alpha = blend_alpha.point(lambda a: int(a * opacity))
            result_rgba.putalpha(blend_alpha)
        else:
            # No alpha in blend image, use opacity as uniform alpha
            result_rgba.putalpha(int(255 * opacity))

        return result_rgba

    def _get_blend_result(self, base_rgb: Image.Image, blend_rgb: Image.Image, mode: str) -> Image.Image:
        """Get the result of applying a specific blend mode."""
        match mode:
            case BlendMode.NORMAL:
                return blend_rgb
            case BlendMode.MULTIPLY:
                return self._custom_multiply(base_rgb, blend_rgb)
            case BlendMode.SCREEN:
                return ImageChops.screen(base_rgb, blend_rgb)
            case BlendMode.OVERLAY:
                return ImageChops.overlay(base_rgb, blend_rgb)
            case BlendMode.SOFT_LIGHT:
                return ImageChops.soft_light(base_rgb, blend_rgb)
            case BlendMode.HARD_LIGHT:
                return ImageChops.hard_light(base_rgb, blend_rgb)
            case BlendMode.DARKEN:
                return ImageChops.darker(base_rgb, blend_rgb)
            case BlendMode.LIGHTEN:
                return ImageChops.lighter(base_rgb, blend_rgb)
            case BlendMode.DIFFERENCE:
                return ImageChops.difference(base_rgb, blend_rgb)
            case BlendMode.EXCLUSION:
                return self._exclusion(base_rgb, blend_rgb)
            case BlendMode.ADD:
                return ImageChops.add(base_rgb, blend_rgb)
            case BlendMode.SUBTRACT:
                return ImageChops.subtract(base_rgb, blend_rgb)
            case BlendMode.LOGICAL_AND:
                return ImageChops.logical_and(self._to_bitmap(base_rgb), self._to_bitmap(blend_rgb)).convert("RGB")
            case BlendMode.LOGICAL_OR:
                return ImageChops.logical_or(self._to_bitmap(base_rgb), self._to_bitmap(blend_rgb)).convert("RGB")
            case BlendMode.LOGICAL_XOR:
                return ImageChops.logical_xor(self._to_bitmap(base_rgb), self._to_bitmap(blend_rgb)).convert("RGB")
            case _:
                msg = f"Unknown blend mode {mode!r}. Pick one of: {', '.join(BlendMode)}."
                raise ValueError(msg)

    def _to_bitmap(self, image: Image.Image) -> Image.Image:
        """Threshold an image to 1-bit at mid-gray, the mode PIL's logical operations require."""
        return image.convert("1", dither=Image.Dither.NONE)

    def _exclusion(self, base: Image.Image, blend: Image.Image) -> Image.Image:
        """Exclusion blend: base + blend - 2 * base * blend, on 0-1 values."""
        base_norm = np.array(base, dtype=np.float32) / 255.0
        blend_norm = np.array(blend, dtype=np.float32) / 255.0
        result_norm = base_norm + blend_norm - 2.0 * base_norm * blend_norm
        return Image.fromarray(np.clip(result_norm * 255.0 + 0.5, 0, 255).astype(np.uint8))

    def _custom_multiply(self, base: Image.Image, blend: Image.Image) -> Image.Image:
        """Custom multiply implementation to ensure correct math."""
        # Convert PIL images to numpy arrays (0-255 range)
        base_array = np.array(base, dtype=np.float32)
        blend_array = np.array(blend, dtype=np.float32)

        # Normalize to 0-1 range for proper multiply math
        base_norm = base_array / 255.0
        blend_norm = blend_array / 255.0

        # Apply multiply: result = base * blend
        result_norm = base_norm * blend_norm

        # Convert back to 0-255 range and ensure proper data type
        result_array = np.clip(result_norm * 255.0, 0, 255).astype(np.uint8)

        # Convert back to PIL image
        return Image.fromarray(result_array)

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

            # Load PIL images
            image_pil = load_pil_from_url(base_image.value)
            blend_pil = load_pil_from_url(blend_image.value)

            # Process with current settings
            processed_image = self._process_images(image_pil, blend_pil, **self._get_custom_parameters())

            # Save and set output
            output_format = self._get_output_format(self._auto_output_format(image_pil))
            output_artifact = self._save_image_artifact(processed_image, output_format)

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

            # Set success status with detailed information
            success_details = (
                f"Successfully composed images: {self._get_processing_description()}\n"
                f"Base: {image_pil.width}x{image_pil.height}\n"
                f"Blend: {blend_pil.width}x{blend_pil.height}\n"
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
