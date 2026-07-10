from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.situation_utils import (
    add_situation_parameter,
    on_output_file_connected,
    on_output_file_disconnected,
)


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

    def process(self) -> None:
        """Process the node by saving text to a file."""
        self._output_file._situation_name = self.get_parameter_value("situation")
        text = self.parameter_values.get("text", "")

        try:
            dest = self._output_file.build_file()
            saved = dest.write_bytes(text.encode("utf-8"))
            saved_path = saved.location
            logger.info("Saved file: %s", saved_path)
            # Do NOT write saved_path back to parameter_output_values["output_file"] —
            # that clobbers the user's filename with the resolved macro path. No other
            # node using ProjectFileParameter does this; the path is log-only here.

        except Exception as e:
            error_message = str(e)
            msg = f"Error saving file: {error_message}"
            raise ValueError(msg) from e

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        on_output_file_connected(self, target_parameter)
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        on_output_file_disconnected(self, target_parameter)
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)
