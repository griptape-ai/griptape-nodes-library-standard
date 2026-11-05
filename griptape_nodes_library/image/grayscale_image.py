from griptape_nodes.retained_mode.griptape_nodes import logger
from PIL import Image, ImageEnhance

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor


class GrayscaleImage(BaseImageProcessor):
    """Convert image to grayscale using PIL's ImageEnhance."""

    PREVIEW_OPERATION = "grayscale"

    def _setup_custom_parameters(self) -> None:
        """Setup brightness/contrast-specific parameters."""

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "grayscale conversion"

    def _process_image(self, pil_image: Image.Image) -> Image.Image:
        """Process the PIL image by converting to grayscale."""
        saturation = 0

        logger.debug(f"{self.name}: Processing image with saturation={saturation}")

        # Apply saturation adjustment
        enhancer = ImageEnhance.Color(pil_image)
        pil_image = enhancer.enhance(saturation)

        return pil_image

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate custom parameters."""
        return None

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_grayscale"
