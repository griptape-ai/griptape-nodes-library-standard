from typing import Any

import numpy as np
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    ImageMode,
    dict_to_image_url_artifact,
    edited_pixels,
    load_image_from_url,
)

# Pillow-style mode for each channel count
_MODES_BY_CHANNELS = {1: ImageMode.GRAY, 2: ImageMode.GRAY_ALPHA, 3: ImageMode.RGB, 4: ImageMode.RGBA}
_ALPHA_MODES = frozenset({ImageMode.GRAY_ALPHA, ImageMode.RGBA})


def _apply_lut(image: GriptapeImage, table: list[int]) -> GriptapeImage:
    """Map the color channels of ``image`` (as 8-bit) through a 256-entry table, keeping any alpha channel."""
    mode = _MODES_BY_CHANNELS[image.channels]
    lut = np.array(table, dtype=np.uint8)
    color = slice(0, -1) if mode in _ALPHA_MODES else slice(None)

    def mapped(pixels: np.ndarray) -> np.ndarray:
        result = pixels.copy()
        result[:, :, color] = lut[pixels[:, :, color]]
        return result

    return edited_pixels(image, mode, mapped)


class AdjustImageEQ(BaseImageProcessor):
    """Adjust image brightness and contrast and color."""

    # Brightness constants
    MIN_BRIGHTNESS = 0.0
    MAX_BRIGHTNESS = 3.0
    DEFAULT_BRIGHTNESS = 1.0

    # Contrast constants
    MIN_CONTRAST = 0.0
    MAX_CONTRAST = 3.0
    DEFAULT_CONTRAST = 1.0

    # Saturation constants
    MIN_SATURATION = 0.0
    MAX_SATURATION = 3.0
    DEFAULT_SATURATION = 1.0

    # Gamma constants
    MIN_GAMMA = 0.1
    MAX_GAMMA = 10.0
    DEFAULT_GAMMA = 1.0

    def _setup_custom_parameters(self) -> None:
        """Setup brightness/contrast-specific parameters."""
        with ParameterGroup(name="adjustment_settings", ui_options={"collapsed": False}) as adjustment_group:
            # Brightness parameter
            brightness_param = ParameterFloat(
                name="brightness",
                default_value=self.DEFAULT_BRIGHTNESS,
                tooltip=f"Brightness adjustment ({self.MIN_BRIGHTNESS}-{self.MAX_BRIGHTNESS}, 1.0 = normal)",
            )
            brightness_param.add_trait(Slider(min_val=self.MIN_BRIGHTNESS, max_val=self.MAX_BRIGHTNESS))
            self.add_parameter(brightness_param)

            # Contrast parameter
            contrast_param = ParameterFloat(
                name="contrast",
                default_value=self.DEFAULT_CONTRAST,
                tooltip=f"Contrast adjustment ({self.MIN_CONTRAST}-{self.MAX_CONTRAST}, 1.0 = normal)",
            )
            contrast_param.add_trait(Slider(min_val=self.MIN_CONTRAST, max_val=self.MAX_CONTRAST))
            self.add_parameter(contrast_param)

            # Saturation parameter
            saturation_param = ParameterFloat(
                name="saturation",
                default_value=self.DEFAULT_SATURATION,
                tooltip=f"Saturation adjustment ({self.MIN_SATURATION}-{self.MAX_SATURATION}, 1.0 = normal)",
            )
            saturation_param.add_trait(Slider(min_val=self.MIN_SATURATION, max_val=self.MAX_SATURATION))
            self.add_parameter(saturation_param)

            # Gamma parameter
            gamma_param = ParameterFloat(
                name="gamma",
                default_value=self.DEFAULT_GAMMA,
                tooltip=f"Gamma adjustment ({self.MIN_GAMMA}-{self.MAX_GAMMA}, 1.0 = normal)",
            )
            gamma_param.add_trait(Slider(min_val=self.MIN_GAMMA, max_val=self.MAX_GAMMA))
            self.add_parameter(gamma_param)

        self.add_node_element(adjustment_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Process image automatically when connected or when EQ parameters change."""
        if parameter.name == "input_image" and value is not None:
            # Process the image immediately when connected
            self._process_image_immediately(value)
        elif parameter.name in ["brightness", "contrast", "saturation", "gamma"]:
            # Process the image when EQ parameters change (for live preview)
            image_value = self.get_parameter_value("input_image")
            if image_value is not None:
                self._process_image_immediately(image_value)
        return super().after_value_set(parameter, value)

    def _process_image_immediately(self, image_value: Any) -> None:
        """Process image immediately for live preview."""
        try:
            # Convert to ImageUrlArtifact if needed
            if isinstance(image_value, dict):
                image_artifact = dict_to_image_url_artifact(image_value)
            else:
                image_artifact = image_value

            # Load image
            image = load_image_from_url(image_artifact.value)

            # Process with current EQ settings
            processed_image = self._process_image(image, **self._get_custom_parameters())

            # Save and set output
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            # Log error but don't fail the node
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image EQ adjustment (brightness, contrast, saturation, gamma)"

    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the image by adjusting brightness, contrast, saturation, and gamma."""
        brightness = kwargs.get("brightness", self.DEFAULT_BRIGHTNESS)
        contrast = kwargs.get("contrast", self.DEFAULT_CONTRAST)
        saturation = kwargs.get("saturation", self.DEFAULT_SATURATION)
        gamma = kwargs.get("gamma", self.DEFAULT_GAMMA)

        # Debug logging
        logger.debug(
            f"{self.name}: Processing image with brightness={brightness}, contrast={contrast}, saturation={saturation}, gamma={gamma}"
        )

        # Work on the 8-bit pixels the gamma table expects
        pipeline = GriptapeImagePipeline(image).to_uint8()

        # Apply brightness adjustment
        if brightness != 1.0:
            pipeline.brightness(brightness)

        # Apply contrast adjustment
        if contrast != 1.0:
            pipeline.contrast(contrast)

        # Apply saturation adjustment
        if saturation != 1.0:
            pipeline.saturation(saturation)

        image = pipeline.end()

        # Apply gamma adjustment
        if gamma != 1.0:
            # Debug logging
            logger.debug(f"{self.name}: Applying gamma correction with gamma={gamma}, channels={image.channels}")

            # Gamma correction lookup table
            # Based on: https://linuxtut.com/en/145d858eff3c505c100a/
            def create_gamma_table(gamma_value: float, gain: float = 1.0) -> list[int]:
                """Create gamma correction lookup table."""
                return [min(255, int((x / 255.0) ** (1.0 / gamma_value) * gain * 255.0)) for x in range(256)]

            # Create lookup table for gamma correction
            gamma_table = create_gamma_table(gamma, gain=1.0)

            # Debug: Check if lookup table is actually changing values
            logger.debug(
                f"{self.name}: Gamma table sample - input 0->{gamma_table[0]}, 128->{gamma_table[128]}, 255->{gamma_table[255]}"
            )

            # Apply to the color channels, preserving any alpha
            image = _apply_lut(image, gamma_table)

        return image

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate EQ parameters."""
        exceptions = []

        brightness = self.get_parameter_value("brightness")
        if brightness is not None and (brightness < self.MIN_BRIGHTNESS or brightness > self.MAX_BRIGHTNESS):
            msg = f"Brightness must be between {self.MIN_BRIGHTNESS} and {self.MAX_BRIGHTNESS}, got {brightness}"
            exceptions.append(ValueError(msg))

        contrast = self.get_parameter_value("contrast")
        if contrast is not None and (contrast < self.MIN_CONTRAST or contrast > self.MAX_CONTRAST):
            msg = f"Contrast must be between {self.MIN_CONTRAST} and {self.MAX_CONTRAST}, got {contrast}"
            exceptions.append(ValueError(msg))

        saturation = self.get_parameter_value("saturation")
        if saturation is not None and (saturation < self.MIN_SATURATION or saturation > self.MAX_SATURATION):
            msg = f"Saturation must be between {self.MIN_SATURATION} and {self.MAX_SATURATION}, got {saturation}"
            exceptions.append(ValueError(msg))

        gamma = self.get_parameter_value("gamma")
        if gamma is not None and (gamma < self.MIN_GAMMA or gamma > self.MAX_GAMMA):
            msg = f"Gamma must be between {self.MIN_GAMMA} and {self.MAX_GAMMA}, got {gamma}"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get EQ parameters."""
        return {
            "brightness": self.get_parameter_value("brightness"),
            "contrast": self.get_parameter_value("contrast"),
            "saturation": self.get_parameter_value("saturation"),
            "gamma": self.get_parameter_value("gamma"),
        }

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_eq"
