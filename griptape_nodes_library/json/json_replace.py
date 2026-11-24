import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode

from griptape_nodes_library.json.json_utils import parse_json_input
from griptape_nodes_library.utils.dot_path import set_value_at_path

logger = logging.getLogger("griptape_nodes")


class JsonReplace(ControlNode):
    """Replace a value in JSON using dot notation path."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add parameter for input JSON
        self.add_parameter(
            Parameter(
                name="json",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=["json", "str", "dict"],
                type="json",
                default_value="{}",
                tooltip="Input JSON data to modify",
            )
        )

        # Add parameter for the path to replace
        self.add_parameter(
            Parameter(
                name="path",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Dot notation path to replace (e.g., 'user.name', 'items[0].title')",
                ui_options={"placeholder_text": "Dot notation path to replace (e.g., 'user.name', 'items[0].title')"},
            )
        )

        # Add parameter for the replacement value
        self.add_parameter(
            Parameter(
                name="replacement_value",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=["json", "str", "dict", "float", "int", "bool"],
                type="json",
                default_value="",
                tooltip="The new value to put at the specified path",
            )
        )

        # Add output parameter
        self.add_parameter(
            Parameter(
                name="output",
                type="json",
                tooltip="The modified JSON with the replacement value",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"placeholder_text": "The modified JSON with the replacement value"},
            )
        )

    def _get_input_parameters(self) -> tuple[Any, str, Any]:
        """Get the input parameters for the replacement operation."""
        json_data = self.get_parameter_value("json")
        path = self.get_parameter_value("path")
        replacement_value = self.get_parameter_value("replacement_value")

        json_data = parse_json_input(self.name, json_data)

        return json_data, path, replacement_value

    def _update_output_parameter(self, result: Any) -> None:
        """Update the output parameter with the result."""
        self.set_parameter_value("output", result)
        self.publish_update_to_parameter("output", result)

    def _perform_replacement(self) -> None:
        """Perform the JSON replacement and set the output value."""
        # Get input parameters
        json_data, path, replacement_value = self._get_input_parameters()

        # Perform the replacement operation
        result = set_value_at_path(json_data, path, replacement_value)

        # Update the output parameter
        self._update_output_parameter(result)

    def _preview_replacement(self) -> None:
        """Update the output while the user edits. Input that cannot be parsed or a path that cannot be followed clears the output.

        Raising here would leave the new input stored on a node that still looks resolved, so these
        errors are caught. process() raises the same error on a real run.
        """
        try:
            self._perform_replacement()
        except ValueError as e:
            # Clear the output so it never shows a result for input it no longer matches.
            self._update_output_parameter(None)
            logger.debug("%s: skipped preview: %s", self.name, e)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in ["json", "path", "replacement_value"]:
            self._preview_replacement()

        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Process the node by replacing the value at the specified path."""
        self._perform_replacement()
