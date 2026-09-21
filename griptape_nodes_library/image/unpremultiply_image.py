from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    image_to_bytes,
    load_pil_from_url,
    unpremultiply_rgba,
)


class UnpremultiplyImage(DataNode):
    """Recover straight (unweighted) RGB values from a premultiplied image.

    Renders from 3D software or EXR files typically deliver premultiplied output:
    RGB has already been scaled by alpha, so semi-transparent areas appear darker
    than their true colour. Run this node first before colour-grading, keying,
    or further compositing — otherwise corrections are applied to the weighted values
    rather than the real ones.

    Equivalent to Nuke's Unpremult node: C_out = C_in / A.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterImage(
                name="input_image",
                default_value=None,
                tooltip="The premultiplied RGBA image to unpremultiply.",
                hide_property=True,
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="invert",
                default_value=False,
                tooltip="Divide by (1 - alpha) instead of alpha.",
            )
        )

        self.add_parameter(
            ParameterImage(
                name="output",
                tooltip="Unpremultiplied (straight alpha) RGBA image.",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="unpremultiplied.png",
        )
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in {"input_image", "invert"} and value is not None:
            input_image = self.get_parameter_value("input_image")
            if input_image is not None:
                if isinstance(input_image, dict):
                    input_image = dict_to_image_url_artifact(input_image)
                self._run(input_image)

    def process(self) -> None:
        input_image = self.get_parameter_value("input_image")
        if input_image is None:
            return
        if isinstance(input_image, dict):
            input_image = dict_to_image_url_artifact(input_image)
        self._run(input_image)

    def _run(self, image_artifact: ImageUrlArtifact) -> None:
        try:
            pil_image = load_pil_from_url(image_artifact.value)
            invert = bool(self.get_parameter_value("invert"))
            result = unpremultiply_rgba(pil_image, invert=invert)

            image_bytes = image_to_bytes(result, "PNG")
            dest = self._output_file.build_file()
            saved = dest.write_bytes(image_bytes)
            output_artifact = ImageUrlArtifact(saved.location)
            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)
        except Exception as e:
            logger.error(f"{self.name}: Failed to unpremultiply image: {e!s}")
