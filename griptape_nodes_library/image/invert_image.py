from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.image import GriptapeImagePipeline
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.image_utils import (
    ImageMode,
    converted,
    dict_to_image_url_artifact,
    image_to_bytes,
    load_image_from_url,
)

_RGBA_CHANNELS = 4


class InvertImage(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterImage(
                name="input_image",
                default_value=None,
                tooltip="The image to invert",
                hide_property=True,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="Inverted image.",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="inverted.png")
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "input_image" and value is not None:
            if isinstance(value, dict):
                value = dict_to_image_url_artifact(value)

            # Invert the image
            self._invert_image(value, raise_on_failure=False)

    def process(self) -> None:
        # Get input image
        input_image = self.get_parameter_value("input_image")

        if input_image is None:
            msg = "Connect an image to 'Input Image'."
            raise ValueError(msg)

        # Normalize input to ImageUrlArtifact
        if isinstance(input_image, dict):
            input_image = dict_to_image_url_artifact(input_image)

        # Invert the image
        self._invert_image(input_image, raise_on_failure=True)

    def _invert_image(self, image_artifact: ImageUrlArtifact, *, raise_on_failure: bool) -> None:
        """Invert the input image and set it as output.

        raise_on_failure is True on the process() path so failures fail the node. It is False from
        after_value_set(), which only logs.
        """
        try:
            # Load image
            image = load_image_from_url(image_artifact.value)

            # Invert the image based on its channels: RGBA keeps its alpha, other modes become RGB
            mode = ImageMode.RGBA if image.channels == _RGBA_CHANNELS else ImageMode.RGB
            image_to_invert = converted(image, mode)
            inverted_image = GriptapeImagePipeline(image_to_invert).invert().end()

            # Save output image and create URL artifact
            image_bytes = image_to_bytes(inverted_image, "PNG")
            dest = self._output_file.build_file()
            saved = dest.write_bytes(image_bytes)
            output_artifact = ImageUrlArtifact(saved.location)
            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            # Log the error and set a meaningful error message
            error_msg = f"Failed to invert image: {e!s}"
            logger.error(f"{self.name}: {error_msg}")
            if raise_on_failure:
                msg = f"Could not invert the image: {e}"
                raise RuntimeError(msg) from e
