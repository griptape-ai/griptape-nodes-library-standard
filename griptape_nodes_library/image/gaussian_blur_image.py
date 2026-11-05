from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.slider import Slider
from PIL import Image, ImageFilter

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor


class GaussianBlurImage(BaseImageProcessor):
    """Apply Gaussian blur to an image using PIL's ImageFilter."""

    PREVIEW_OPERATION = "gaussian_blur"
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["radius"]

    # Radius constants
    MIN_RADIUS = 0.0
    MAX_RADIUS = 50.0
    DEFAULT_RADIUS = 5.0

    def _setup_custom_parameters(self) -> None:
        """Setup blur-specific parameters."""
        with ParameterGroup(name="blur_settings", ui_options={"collapsed": False}) as blur_group:
            # Radius parameter
            radius_param = ParameterFloat(
                name="radius",
                default_value=self.DEFAULT_RADIUS,
                tooltip=f"Blur radius ({self.MIN_RADIUS}-{self.MAX_RADIUS}, higher values create more blur)",
            )
            radius_param.add_trait(Slider(min_val=self.MIN_RADIUS, max_val=self.MAX_RADIUS))
            self.add_parameter(radius_param)

        self.add_node_element(blur_group)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "Gaussian blur"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL image by applying Gaussian blur."""
        radius = kwargs.get("radius", self.DEFAULT_RADIUS)

        # Debug logging
        logger.debug(f"{self.name}: Processing image with blur radius={radius}")

        # Apply Gaussian blur
        if radius > 0:
            blurred_image = pil_image.filter(ImageFilter.GaussianBlur(radius=radius))
            return blurred_image
        # If radius is 0, return original image (no blur)
        return pil_image

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate blur parameters."""
        exceptions = []

        radius = self.get_parameter_value("radius")
        if radius is not None and (radius < self.MIN_RADIUS or radius > self.MAX_RADIUS):
            msg = f"Radius must be between {self.MIN_RADIUS} and {self.MAX_RADIUS}, got {radius}"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get blur parameters."""
        return {
            "radius": self.get_parameter_value("radius"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        radius = kwargs.get("radius", self.DEFAULT_RADIUS)
        return f"_blur_{radius:.1f}"
