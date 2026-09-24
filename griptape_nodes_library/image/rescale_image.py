from math import gcd
from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor


class RescaleImage(BaseImageProcessor):
    """Rescale an image with different resize modes and resample filters."""

    # Resize mode constants
    RESIZE_MODE_WIDTH = "width"
    RESIZE_MODE_HEIGHT = "height"
    RESIZE_MODE_PERCENTAGE = "percentage"
    RESIZE_MODE_WIDTH_HEIGHT = "width and height"
    RESIZE_MODE_ASPECT_RATIO = "aspect ratio"

    # Target size constants (for width/height modes)
    MIN_TARGET_SIZE = 1
    MAX_TARGET_SIZE = 8000  # Reasonable max for most use cases
    DEFAULT_TARGET_SIZE = 1000

    # Percentage scale constants
    MIN_PERCENTAGE_SCALE = 1
    MAX_PERCENTAGE_SCALE = 500  # 500% = 5x size
    DEFAULT_PERCENTAGE_SCALE = 100  # 100% = original size

    # Fit mode constants
    FIT_MODE_FIT = "fit"
    FIT_MODE_FILL = "fill"
    FIT_MODE_STRETCH = "stretch"

    # Hex color format constants
    HEX_SHORT_LENGTH = 3
    HEX_FULL_LENGTH = 6

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "resize_mode":
            if value == self.RESIZE_MODE_PERCENTAGE:
                self.show_parameter_by_name("percentage_scale")
                self.hide_parameter_by_name("target_size")
                self.hide_parameter_by_name("target_width")
                self.hide_parameter_by_name("target_height")
                self.hide_parameter_by_name("aspect_ratio")
                self.hide_parameter_by_name("fit_mode")
                self.hide_parameter_by_name("background_color")
            elif value == self.RESIZE_MODE_WIDTH_HEIGHT:
                self.hide_parameter_by_name("percentage_scale")
                self.hide_parameter_by_name("target_size")
                self.show_parameter_by_name("target_width")
                self.show_parameter_by_name("target_height")
                self.hide_parameter_by_name("aspect_ratio")
                self.show_parameter_by_name("fit_mode")
                # Background color visibility will be controlled by fit_mode
                self.hide_parameter_by_name("background_color")
            elif value == self.RESIZE_MODE_ASPECT_RATIO:
                self.hide_parameter_by_name("percentage_scale")
                self.hide_parameter_by_name("target_size")
                self.hide_parameter_by_name("target_width")
                self.hide_parameter_by_name("target_height")
                self.show_parameter_by_name("aspect_ratio")
                self.show_parameter_by_name("fit_mode")
                # Background color visibility will be controlled by fit_mode
                self.hide_parameter_by_name("background_color")
            else:
                self.hide_parameter_by_name("percentage_scale")
                self.show_parameter_by_name("target_size")
                self.hide_parameter_by_name("target_width")
                self.hide_parameter_by_name("target_height")
                self.hide_parameter_by_name("aspect_ratio")
                self.hide_parameter_by_name("fit_mode")
                self.hide_parameter_by_name("background_color")
        elif parameter.name == "fit_mode":
            # Show background color only for fit mode
            if value == self.FIT_MODE_FIT:
                self.show_parameter_by_name("background_color")
            else:
                self.hide_parameter_by_name("background_color")
        return super().after_value_set(parameter, value)

    def _setup_custom_parameters(self) -> None:
        """Setup rescale-specific parameters."""
        with ParameterGroup(name="rescale_settings", ui_options={"collapsed": False}) as rescale_group:
            # Resize mode parameter
            resize_mode_param = ParameterString(
                name="resize_mode",
                default_value=self.RESIZE_MODE_PERCENTAGE,
                tooltip="How to resize the image: by width, height, percentage, or width and height",
            )
            resize_mode_param.add_trait(
                Options(
                    choices=[
                        self.RESIZE_MODE_WIDTH,
                        self.RESIZE_MODE_HEIGHT,
                        self.RESIZE_MODE_WIDTH_HEIGHT,
                        self.RESIZE_MODE_PERCENTAGE,
                        self.RESIZE_MODE_ASPECT_RATIO,
                    ]
                )
            )

            # Target size parameter (for width/height modes)
            target_size_param = ParameterInt(
                name="target_size",
                default_value=self.DEFAULT_TARGET_SIZE,
                tooltip=f"Target size in pixels for width/height modes ({self.MIN_TARGET_SIZE}-{self.MAX_TARGET_SIZE})",
            )
            target_size_param.add_trait(Slider(min_val=self.MIN_TARGET_SIZE, max_val=self.MAX_TARGET_SIZE))

            # Percentage scale parameter (for percentage mode)
            percentage_scale_param = ParameterInt(
                name="percentage_scale",
                default_value=self.DEFAULT_PERCENTAGE_SCALE,
                tooltip=f"Scale factor as percentage ({self.MIN_PERCENTAGE_SCALE}-{self.MAX_PERCENTAGE_SCALE}%, 100% = original size)",
            )
            percentage_scale_param.add_trait(
                Slider(min_val=self.MIN_PERCENTAGE_SCALE, max_val=self.MAX_PERCENTAGE_SCALE)
            )

            # Target width parameter (for width and height mode)
            target_width_param = ParameterInt(
                name="target_width",
                max_val=self.MAX_TARGET_SIZE,
                default_value=self.DEFAULT_TARGET_SIZE,
                tooltip=f"Target width in pixels ({self.MIN_TARGET_SIZE}-{self.MAX_TARGET_SIZE})",
            )
            target_width_param.add_trait(Slider(min_val=self.MIN_TARGET_SIZE, max_val=self.MAX_TARGET_SIZE))

            # Target height parameter (for width and height mode)
            target_height_param = ParameterInt(
                name="target_height",
                max_val=self.MAX_TARGET_SIZE,
                default_value=self.DEFAULT_TARGET_SIZE,
                tooltip=f"Target height in pixels ({self.MIN_TARGET_SIZE}-{self.MAX_TARGET_SIZE})",
            )
            target_height_param.add_trait(Slider(min_val=self.MIN_TARGET_SIZE, max_val=self.MAX_TARGET_SIZE))

            # Aspect ratio parameter (for aspect ratio mode)
            ParameterString(
                name="aspect_ratio",
                default_value="16:9",
                placeholder_text="16:9",
                tooltip=(
                    "Target aspect ratio. Use ratio notation (e.g. 16:9) or a decimal scalar (e.g. 1.77). "
                    "The image is resized to match this ratio using the selected fit mode."
                ),
                traits={
                    Button(
                        icon="scan",
                        size="icon",
                        variant="secondary",
                        on_click=self._on_detect_ratio_clicked,
                        tooltip="Detect aspect ratio from input image",
                    )
                },
            )

            # Fit mode parameter (for width and height mode
            fit_mode_param = ParameterString(
                name="fit_mode",
                default_value=self.FIT_MODE_FIT,
                tooltip="How to fit the image within the target dimensions",
            )
            fit_mode_param.add_trait(
                Options(
                    choices=[
                        self.FIT_MODE_FIT,
                        self.FIT_MODE_FILL,
                        self.FIT_MODE_STRETCH,
                    ]
                )
            )

            # Background color parameter (for width and height mode with fit/fill)
            background_color_param = ParameterString(
                name="background_color",
                default_value="#000000",
                tooltip="Background color for letterboxing/matting",
            )
            background_color_param.add_trait(ColorPicker(format="hex"))

            # Resample filter parameter
            resample_filter_param = ParameterString(
                name="resample_filter",
                default_value="lanczos",
                tooltip="Resample filter for resizing (higher quality = slower processing)",
            )
            resample_filter_param.add_trait(
                Options(choices=["nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"])
            )

        self.add_node_element(rescale_group)

        # Hide the correct parameters
        self.hide_parameter_by_name("target_size")
        self.hide_parameter_by_name("target_width")
        self.hide_parameter_by_name("target_height")
        self.hide_parameter_by_name("aspect_ratio")
        self.hide_parameter_by_name("fit_mode")
        self.hide_parameter_by_name("background_color")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image rescaling"

    def _parse_aspect_ratio(self, ratio_str: str) -> float:
        """Parse an aspect ratio string to a float scalar.

        Accepts ratio notation (e.g. '16:9') or a decimal scalar (e.g. '1.77').
        """
        if ":" in ratio_str:
            parts = ratio_str.split(":")
            if len(parts) != 2:  # noqa: PLR2004
                msg = f"Invalid ratio '{ratio_str}'. Use ratio notation like '16:9' or a scalar like '1.77'."
                raise ValueError(msg)
            try:
                w = float(parts[0].strip())
                h = float(parts[1].strip())
            except ValueError:
                msg = f"Invalid ratio '{ratio_str}'. Use ratio notation like '16:9' or a scalar like '1.77'."
                raise ValueError(msg) from None
            if h == 0:
                msg = f"Invalid ratio '{ratio_str}': denominator cannot be zero."
                raise ValueError(msg)
            return w / h
        try:
            return float(ratio_str.strip())
        except ValueError:
            msg = f"Invalid aspect ratio '{ratio_str}'. Provide a ratio (e.g. '16:9') or a scalar (e.g. '1.77')."
            raise ValueError(msg) from None

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL image by rescaling it."""
        resize_mode = kwargs.get("resize_mode", self.RESIZE_MODE_PERCENTAGE)
        target_size = kwargs.get("target_size", self.DEFAULT_TARGET_SIZE)
        percentage_scale = kwargs.get("percentage_scale", self.DEFAULT_PERCENTAGE_SCALE)
        target_width = kwargs.get("target_width", self.DEFAULT_TARGET_SIZE)
        target_height = kwargs.get("target_height", self.DEFAULT_TARGET_SIZE)
        aspect_ratio_str = kwargs.get("aspect_ratio", "16:9")
        fit_mode = kwargs.get("fit_mode", self.FIT_MODE_FIT)
        background_color = kwargs.get("background_color", "transparent")
        resample_filter = kwargs.get("resample_filter", "lanczos")

        # Get the resample filter constant
        resample_constant = self._get_resample_constant(resample_filter)

        # Calculate new dimensions based on resize mode
        if resize_mode == self.RESIZE_MODE_WIDTH:
            # Resize by width, maintain aspect ratio
            ratio = target_size / pil_image.width
            new_width = target_size
            new_height = int(pil_image.height * ratio)
            resized_image = pil_image.resize((new_width, new_height), resample_constant)
        elif resize_mode == self.RESIZE_MODE_HEIGHT:
            # Resize by height, maintain aspect ratio
            ratio = target_size / pil_image.height
            new_width = int(pil_image.width * ratio)
            new_height = target_size
            resized_image = pil_image.resize((new_width, new_height), resample_constant)
        elif resize_mode == self.RESIZE_MODE_PERCENTAGE:
            # Resize by percentage scale
            scale_factor = percentage_scale / 100.0
            new_width = int(pil_image.width * scale_factor)
            new_height = int(pil_image.height * scale_factor)
            resized_image = pil_image.resize((new_width, new_height), resample_constant)
        elif resize_mode == self.RESIZE_MODE_WIDTH_HEIGHT:
            # Resize to specific width and height with fit mode
            resize_config = {
                "target_width": target_width,
                "target_height": target_height,
                "fit_mode": fit_mode,
                "background_color": background_color,
                "resample_constant": resample_constant,
            }
            resized_image = self._resize_with_fit_mode(pil_image, resize_config)
        elif resize_mode == self.RESIZE_MODE_ASPECT_RATIO:
            # Derive target dimensions from the desired ratio, then apply fit mode
            target_ratio = self._parse_aspect_ratio(aspect_ratio_str)
            if target_ratio >= 1:
                # Landscape/square — keep width, compute height
                tw = pil_image.width
                th = max(1, round(pil_image.width / target_ratio))
            else:
                # Portrait — keep height, compute width
                th = pil_image.height
                tw = max(1, round(pil_image.height * target_ratio))
            resize_config = {
                "target_width": tw,
                "target_height": th,
                "fit_mode": fit_mode,
                "background_color": background_color,
                "resample_constant": resample_constant,
            }
            resized_image = self._resize_with_fit_mode(pil_image, resize_config)
        else:
            msg = f"{self.name} - Invalid resize mode: {resize_mode}"
            raise ValueError(msg)

        return resized_image

    def _get_resample_constant(self, filter_name: str) -> int:
        """Get the PIL resample constant for the given filter name."""
        filter_map = {
            "nearest": Image.Resampling.NEAREST,
            "box": Image.Resampling.BOX,
            "bilinear": Image.Resampling.BILINEAR,
            "hamming": Image.Resampling.HAMMING,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }
        return filter_map.get(filter_name, Image.Resampling.LANCZOS)

    def _resize_with_fit_mode(self, pil_image: Image.Image, config: dict) -> Image.Image:
        """Resize image to target dimensions using the specified fit mode."""
        target_width = max(1, config["target_width"])
        target_height = max(1, config["target_height"])
        fit_mode = config["fit_mode"]
        background_color = config["background_color"]
        resample_constant = config["resample_constant"]

        if fit_mode == self.FIT_MODE_STRETCH:
            # Stretch to exact dimensions, ignoring aspect ratio
            return pil_image.resize((target_width, target_height), resample_constant)

        # Calculate scale factors for fit and fill modes
        scale_x = target_width / pil_image.width
        scale_y = target_height / pil_image.height

        if fit_mode == self.FIT_MODE_FIT:
            # Fit mode: maintain aspect ratio, fit entirely within target, output exactly target_w x target_h
            scale = min(scale_x, scale_y)
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)

            # Ensure minimum dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            # Resize the image
            resized_image = pil_image.resize((new_width, new_height), resample_constant)

            # Create canvas with target dimensions and background color
            canvas = self._create_canvas(target_width, target_height, background_color)

            # Calculate padding to center the image
            pad_x = (target_width - new_width) // 2
            pad_y = (target_height - new_height) // 2

            # Ensure canvas mode matches image mode for proper blending
            if resized_image.mode == "RGBA" and canvas.mode != "RGBA":
                canvas = canvas.convert("RGBA")

            # Paste the resized image onto the canvas
            if resized_image.mode == "RGBA":
                canvas.paste(resized_image, (pad_x, pad_y), resized_image)
            else:
                canvas.paste(resized_image, (pad_x, pad_y))

            return canvas

        if fit_mode == self.FIT_MODE_FILL:
            # Fill mode: maintain aspect ratio, fill target dimensions (may crop)
            scale = max(scale_x, scale_y)
            new_width = int(pil_image.width * scale)
            new_height = int(pil_image.height * scale)

            # Ensure minimum dimensions
            new_width = max(1, new_width)
            new_height = max(1, new_height)

            # Resize the image
            resized_image = pil_image.resize((new_width, new_height), resample_constant)

            # Crop to target dimensions from center
            if new_width != target_width or new_height != target_height:
                left = (new_width - target_width) // 2
                top = (new_height - target_height) // 2
                right = left + target_width
                bottom = top + target_height
                return resized_image.crop((left, top, right, bottom))

            return resized_image
        msg = f"{self.name} - Invalid fit mode: {fit_mode}"
        raise ValueError(msg)

    def _create_canvas(self, width: int, height: int, background_color: str) -> Image.Image:
        """Create a canvas with the specified background color."""
        if background_color == "transparent":
            return Image.new("RGBA", (width, height), (0, 0, 0, 0))

        # Parse hex color
        if background_color.startswith("#"):
            hex_color = background_color[1:]
            if len(hex_color) == self.HEX_SHORT_LENGTH:
                # Short hex format (#RGB -> #RRGGBB)
                hex_color = "".join([c * 2 for c in hex_color])
            elif len(hex_color) == self.HEX_FULL_LENGTH:
                # Full hex format
                pass
            else:
                # Invalid hex, default to white
                return Image.new("RGB", (width, height), (255, 255, 255))

            try:
                r = int(hex_color[0:2], 16)
                g = int(hex_color[2:4], 16)
                b = int(hex_color[4:6], 16)
                return Image.new("RGB", (width, height), (r, g, b))
            except ValueError:
                # Invalid hex, default to white
                return Image.new("RGB", (width, height), (255, 255, 255))

        # Default to white for other formats
        return Image.new("RGB", (width, height), (255, 255, 255))

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate rescale parameters."""
        exceptions = []

        resize_mode = self.get_parameter_value("resize_mode")
        target_size = self.get_parameter_value("target_size")
        percentage_scale = self.get_parameter_value("percentage_scale")
        target_width = self.get_parameter_value("target_width")
        target_height = self.get_parameter_value("target_height")

        # Validate target_size for width/height modes
        if (
            resize_mode in [self.RESIZE_MODE_WIDTH, self.RESIZE_MODE_HEIGHT]
            and target_size is not None
            and (target_size < self.MIN_TARGET_SIZE or target_size > self.MAX_TARGET_SIZE)
        ):
            msg = f"{self.name} - Target size must be between {self.MIN_TARGET_SIZE} and {self.MAX_TARGET_SIZE}, got {target_size}"
            exceptions.append(ValueError(msg))

        # Validate percentage_scale for percentage mode
        if (
            resize_mode == self.RESIZE_MODE_PERCENTAGE
            and percentage_scale is not None
            and (percentage_scale < self.MIN_PERCENTAGE_SCALE or percentage_scale > self.MAX_PERCENTAGE_SCALE)
        ):
            msg = f"{self.name} - Percentage scale must be between {self.MIN_PERCENTAGE_SCALE} and {self.MAX_PERCENTAGE_SCALE}, got {percentage_scale}"
            exceptions.append(ValueError(msg))

        # Validate target_width and target_height for width and height mode
        if resize_mode == self.RESIZE_MODE_WIDTH_HEIGHT:
            if target_width is not None and (
                target_width < self.MIN_TARGET_SIZE or target_width > self.MAX_TARGET_SIZE
            ):
                msg = f"{self.name} - Target width must be between {self.MIN_TARGET_SIZE} and {self.MAX_TARGET_SIZE}, got {target_width}"
                exceptions.append(ValueError(msg))

            if target_height is not None and (
                target_height < self.MIN_TARGET_SIZE or target_height > self.MAX_TARGET_SIZE
            ):
                msg = f"{self.name} - Target height must be between {self.MIN_TARGET_SIZE} and {self.MAX_TARGET_SIZE}, got {target_height}"
                exceptions.append(ValueError(msg))

        # Validate aspect ratio string for aspect ratio mode
        if resize_mode == self.RESIZE_MODE_ASPECT_RATIO:
            aspect_ratio_str = self.get_parameter_value("aspect_ratio")
            if aspect_ratio_str is not None:
                try:
                    self._parse_aspect_ratio(aspect_ratio_str)
                except ValueError as e:
                    exceptions.append(e)

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get rescale parameters."""
        return {
            "resize_mode": self.get_parameter_value("resize_mode"),
            "target_size": self.get_parameter_value("target_size"),
            "percentage_scale": self.get_parameter_value("percentage_scale"),
            "target_width": self.get_parameter_value("target_width"),
            "target_height": self.get_parameter_value("target_height"),
            "aspect_ratio": self.get_parameter_value("aspect_ratio"),
            "fit_mode": self.get_parameter_value("fit_mode"),
            "background_color": self.get_parameter_value("background_color"),
            "resample_filter": self.get_parameter_value("resample_filter"),
        }

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_rescaled"

    def _on_detect_ratio_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,
    ) -> NodeMessageResult | None:
        """Detect the input image's aspect ratio and populate the aspect_ratio parameter."""
        from griptape_nodes_library.utils.image_utils import load_pil_from_url

        image = self.get_parameter_value("input_image")
        if image is None:
            return NodeMessageResult(success=False, details="No input image connected.", response=button_details)

        try:
            if hasattr(image, "to_dict"):
                from griptape_nodes_library.utils.image_utils import dict_to_image_url_artifact

                image = dict_to_image_url_artifact(image.to_dict())

            pil_image = load_pil_from_url(image.value)
            w, h = pil_image.width, pil_image.height
            divisor = gcd(w, h)
            ratio_str = f"{w // divisor}:{h // divisor}"
            self.set_parameter_value("aspect_ratio", ratio_str)
            self.publish_update_to_parameter("aspect_ratio", ratio_str)
            return NodeMessageResult(success=True, details=f"Detected ratio: {ratio_str}", response=button_details)
        except Exception as e:
            return NodeMessageResult(success=False, details=f"Failed to detect ratio: {e}", response=button_details)
