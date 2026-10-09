from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.split_text_utils import SplitOptions


class SplitText(ControlNode):
    """SplitText Node that can either split text by delimiter or parse it as a list.

    This node provides two modes of operation:
    1. Split mode: Splits text by a specified delimiter (original behavior)
    2. Parse mode: Intelligently parses text as JSON/Python lists with fallback to delimiter splitting

    Key Features:
    - Multiple delimiter options (comma, semicolon, pipe, etc.)
    - Intelligent list parsing (JSON, Python literals, comma-separated)
    - Whitespace trimming and delimiter inclusion options
    - Robust error handling and fallback mechanisms
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        # Add input text parameter
        self.text_input = ParameterString(
            name="text",
            tooltip="Text string to split",
            allow_output=False,
            multiline=True,
        )
        self.add_parameter(self.text_input)

        # remove_empty is off by default so existing workflows keep their empty items
        self.split_options = SplitOptions.create(remove_empty_default=False, allow_toggle_input=True)
        for param in self.split_options.parameters:
            self.add_parameter(param)

        # Add output parameter
        self.output = Parameter(
            name="output",
            tooltip="List of text items",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

        self.split_options.update_delimiter_visibility(self)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in {self.text_input.name, *self.split_options.names}:
            self._process_text(raise_on_failure=False)

        if parameter.name == self.split_options.split_mode.name:
            self.split_options.update_delimiter_visibility(self)

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception]:
        exceptions = []
        text = self.get_parameter_value(self.text_input.name)
        if text is None:
            exceptions.append(Exception("Connect text to 'Text' to split."))
        elif not isinstance(text, str):
            exceptions.append(Exception(f"'Text' must be a string, got {type(text).__name__}."))
        return exceptions

    def _process_text(self, *, raise_on_failure: bool) -> None:
        """Process the text input according to the selected mode (split or parse).

        raise_on_failure is True on the process() path so failures fail the node. It is False from
        after_value_set(), which only logs and clears the output.
        """
        text = self.get_parameter_value(self.text_input.name)

        # Ensure text is a string
        if not isinstance(text, str):
            text = ""

        try:
            split_result = self.split_options.split(self, text)

            self.parameter_output_values[self.output.name] = split_result
            self.publish_update_to_parameter(self.output.name, split_result)
        except (TypeError, ValueError) as e:
            self.parameter_output_values[self.output.name] = []
            self.publish_update_to_parameter(self.output.name, [])
            if raise_on_failure:
                msg = f"Could not split the text: {e}"
                raise ValueError(msg) from e
            logger.warning("%s: Could not split the text: %s", self.name, e)

    def process(self) -> None:
        self._process_text(raise_on_failure=True)
