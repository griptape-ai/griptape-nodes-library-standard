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

        # Inputs of the last successful run, so process() can skip work the live preview already did.
        self._last_run_key: tuple[str, bool] | None = None

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
        super().after_value_set(parameter, value)
        if parameter.name not in {"input_image", "invert"} or self.get_parameter_value("input_image") is None:
            return
        # Live preview: a failure here is reported but must not break setting the value.
        try:
            self._run()
        except Exception as e:
            logger.error(f"{self.name}: Failed to unpremultiply image: {e!s}")

    def process(self) -> None:
        if self.get_parameter_value("input_image") is None:
            return
        self._run()

    def _run(self) -> None:
        input_image = self.get_parameter_value("input_image")
        if isinstance(input_image, dict):
            input_image = dict_to_image_url_artifact(input_image)
        invert = bool(self.get_parameter_value("invert"))

        run_key = (input_image.value, invert)
        if run_key == self._last_run_key and self.get_parameter_value("output") is not None:
            return

        result = unpremultiply_rgba(load_pil_from_url(input_image.value), invert=invert)
        saved = self._output_file.build_file().write_bytes(image_to_bytes(result, "PNG"))
        output_artifact = ImageUrlArtifact(saved.location)
        self.set_parameter_value("output", output_artifact)
        self.publish_update_to_parameter("output", output_artifact)
        self._last_run_key = run_key
