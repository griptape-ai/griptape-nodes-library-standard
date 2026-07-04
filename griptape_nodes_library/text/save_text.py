from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.situation_utils import add_situation_parameter


class SaveText(ControlNode):
    """Save text to a file."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add text input parameter
        self.add_parameter(
            Parameter(
                name="text",
                input_types=["str"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="The text content to save to file",
                ui_options={"multiline": True, "placeholder_text": "Text to save to a file..."},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="griptape_output.txt",
        )
        add_situation_parameter(self, self._output_file)
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: object, **kwargs: object) -> None:
        if parameter.name == "situation":
            self._output_file._situation_name = str(value)
        super().after_value_set(parameter, value, **kwargs)

    def process(self) -> None:
        """Process the node by saving text to a file."""
        text = self.parameter_values.get("text", "")

        try:
            dest = self._output_file.build_file()
            saved = dest.write_bytes(text.encode("utf-8"))
            saved_path = saved.location
            logger.info("Saved file: %s", saved_path)

        except Exception as e:
            error_message = str(e)
            msg = f"Error saving file: {error_message}"
            raise ValueError(msg) from e
