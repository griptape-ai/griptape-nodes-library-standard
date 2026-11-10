from typing import ClassVar

from griptape_nodes.exe_types.core_types import ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from PIL import Image, ImageOps

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor


class FlipImage(BaseImageProcessor):
    """Flip image horizontally or vertically."""

    PREVIEW_OPERATION = "flip"
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["direction"]

    def _setup_custom_parameters(self) -> None:
        """Setup flip-specific parameters."""
        with ParameterGroup(name="flip_settings", ui_options={"collapsed": False}) as flip_group:
            # Flip direction parameter
            direction_parameter = ParameterString(
                name="direction",
                default_value="horizontal",
                tooltip="Flip direction: horizontal, vertical, or both",
            )
            direction_parameter.add_trait(Options(choices=["horizontal", "vertical", "both"]))
            self.add_parameter(direction_parameter)

        self.add_node_element(flip_group)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image flipping"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL image by flipping it."""
        direction = kwargs.get("direction", "horizontal")

        logger.debug(f"{self.name}: Flipping image with direction={direction}")

        # Apply flip based on direction
        if direction == "horizontal":
            pil_image = ImageOps.mirror(pil_image)
        elif direction == "vertical":
            pil_image = ImageOps.flip(pil_image)
        elif direction == "both":
            pil_image = ImageOps.mirror(pil_image)
            pil_image = ImageOps.flip(pil_image)

        return pil_image

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate flip parameters."""
        exceptions = []

        direction = self.get_parameter_value("direction")
        valid_choices = ["horizontal", "vertical", "both"]
        if direction is not None and direction not in valid_choices:
            msg = f"Direction must be one of {valid_choices}, got {direction}"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, str]:
        """Get flip parameters."""
        return {
            "direction": self.get_parameter_value("direction"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        direction = kwargs.get("direction", "horizontal")
        return f"_flipped_{direction}"
