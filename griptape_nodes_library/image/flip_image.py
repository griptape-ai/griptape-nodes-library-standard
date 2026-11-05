from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_image_from_url,
)


class FlipDirection(StrEnum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    BOTH = "both"


class FlipImage(BaseImageProcessor):
    """Flip image horizontally or vertically."""

    def _setup_custom_parameters(self) -> None:
        """Setup flip-specific parameters."""
        with ParameterGroup(name="flip_settings", ui_options={"collapsed": False}) as flip_group:
            # Flip direction parameter
            direction_parameter = ParameterString(
                name="direction",
                default_value=FlipDirection.HORIZONTAL,
                tooltip="Flip direction: horizontal, vertical, or both",
            )
            direction_parameter.add_trait(Options(choices=[d.value for d in FlipDirection]))
            self.add_parameter(direction_parameter)

        self.add_node_element(flip_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Process image automatically when connected or when flip parameters change."""
        if parameter.name == "input_image" and value is not None:
            # Process the image immediately when connected
            self._process_image_immediately(value)
        elif parameter.name == "direction":
            # Re-process when direction changes
            input_image = self.get_parameter_value("input_image")
            if input_image is not None:
                self._process_image_immediately(input_image)
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

            # Get current direction
            direction = self.get_parameter_value("direction") or FlipDirection.HORIZONTAL

            # Process with current flip settings
            processed_image = self._process_image(image, direction=direction)

            # Save and set output
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            # Log error but don't fail the node
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image flipping"

    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the image by flipping it."""
        direction = kwargs.get("direction") or FlipDirection.HORIZONTAL

        logger.debug(f"{self.name}: Flipping image with direction={direction}")

        # Apply flip based on direction
        pipeline = GriptapeImagePipeline(image)
        match direction:
            case FlipDirection.HORIZONTAL:
                pipeline.flip_horizontal()
            case FlipDirection.VERTICAL:
                pipeline.flip_vertical()
            case FlipDirection.BOTH:
                pipeline.flip_horizontal().flip_vertical()
            case _:
                msg = f"Direction must be one of {[d.value for d in FlipDirection]}, got {direction}"
                raise ValueError(msg)

        return pipeline.end()

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate flip parameters."""
        exceptions = []

        direction = self.get_parameter_value("direction")
        valid_choices = [d.value for d in FlipDirection]
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
        direction = kwargs.get("direction", FlipDirection.HORIZONTAL)
        return f"_flipped_{direction}"
