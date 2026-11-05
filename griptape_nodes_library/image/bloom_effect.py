from typing import Any, ClassVar

import numpy as np
from griptape_nodes.exe_types.core_types import ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    MAX_UINT8,
    ImageMode,
    converted,
    pixels_of,
)


class BloomEffect(BaseImageProcessor):
    """Apply a bloom/glow effect to images with configurable intensity and radius."""

    PREVIEW_OPERATION = "bloom"
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["bloom_amount", "bloom_radius"]

    # Bloom amount constants
    MIN_BLOOM_AMOUNT = 0.0
    MAX_BLOOM_AMOUNT = 2.0
    DEFAULT_BLOOM_AMOUNT = 0.5

    # Bloom radius constants
    MIN_BLOOM_RADIUS = 1
    MAX_BLOOM_RADIUS = 20
    DEFAULT_BLOOM_RADIUS = 5

    def _setup_custom_parameters(self) -> None:
        """Setup bloom-specific parameters."""
        with ParameterGroup(name="bloom_settings", ui_options={"collapsed": False}) as bloom_group:
            # Bloom amount parameter
            bloom_amount_param = ParameterFloat(
                name="bloom_amount",
                default_value=self.DEFAULT_BLOOM_AMOUNT,
                tooltip=f"Bloom intensity ({self.MIN_BLOOM_AMOUNT}-{self.MAX_BLOOM_AMOUNT}, 0.0 = no effect, 2.0 = maximum glow)",
            )
            bloom_amount_param.add_trait(Slider(min_val=self.MIN_BLOOM_AMOUNT, max_val=self.MAX_BLOOM_AMOUNT))
            self.add_parameter(bloom_amount_param)

            # Bloom radius parameter
            bloom_radius_param = ParameterInt(
                name="bloom_radius",
                default_value=self.DEFAULT_BLOOM_RADIUS,
                tooltip=f"Bloom radius in pixels ({self.MIN_BLOOM_RADIUS}-{self.MAX_BLOOM_RADIUS}, higher = softer glow)",
            )
            bloom_radius_param.add_trait(Slider(min_val=self.MIN_BLOOM_RADIUS, max_val=self.MAX_BLOOM_RADIUS))
            self.add_parameter(bloom_radius_param)

        self.add_node_element(bloom_group)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image bloom effect"

    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the image by applying bloom effect."""
        bloom_amount = kwargs.get("bloom_amount", self.DEFAULT_BLOOM_AMOUNT)
        bloom_radius = kwargs.get("bloom_radius", self.DEFAULT_BLOOM_RADIUS)

        if bloom_amount <= 0.0:
            return image  # No bloom effect

        # Convert to RGB (bloom works best with RGB)
        rgb_image = converted(image, ImageMode.RGB)

        # Create a brightened, blurred version for bloom
        bloomed = GriptapeImagePipeline(rgb_image).brightness(1.0 + bloom_amount).gaussian_blur(bloom_radius).end()
        bloomed_pixels = pixels_of(bloomed, ImageMode.RGB).astype(np.float32)

        # Blend the original image with the bloomed version
        # Use a weighted blend based on bloom amount
        blend_factor = min(bloom_amount, 1.0)  # Cap at 1.0 for reasonable blending

        # Create the final image by blending, truncating as Pillow's blend did
        def blend(pixels: np.ndarray) -> np.ndarray:
            original = pixels.astype(np.float32)
            blended = np.trunc(original + blend_factor * (bloomed_pixels - original))
            return np.clip(blended, 0, MAX_UINT8).astype(np.uint8)

        return GriptapeImagePipeline(rgb_image).apply(blend).end()

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate bloom parameters."""
        exceptions = []

        bloom_amount = self.get_parameter_value("bloom_amount")
        if bloom_amount is not None and (bloom_amount < self.MIN_BLOOM_AMOUNT or bloom_amount > self.MAX_BLOOM_AMOUNT):
            msg = (
                f"Bloom amount must be between {self.MIN_BLOOM_AMOUNT} and {self.MAX_BLOOM_AMOUNT}, got {bloom_amount}"
            )
            exceptions.append(ValueError(msg))

        bloom_radius = self.get_parameter_value("bloom_radius")
        if bloom_radius is not None and (bloom_radius < self.MIN_BLOOM_RADIUS or bloom_radius > self.MAX_BLOOM_RADIUS):
            msg = (
                f"Bloom radius must be between {self.MIN_BLOOM_RADIUS} and {self.MAX_BLOOM_RADIUS}, got {bloom_radius}"
            )
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get bloom parameters."""
        return {
            "bloom_amount": self.get_parameter_value("bloom_amount"),
            "bloom_radius": self.get_parameter_value("bloom_radius"),
        }

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_bloom"
