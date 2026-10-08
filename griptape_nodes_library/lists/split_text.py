from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.split_text_utils import DELIMITER_MAP, SplitMode, split_text


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

        # Add split mode parameter (moved up to control visibility of other parameters)
        self.split_mode = ParameterString(
            name="split_mode",
            tooltip="How to process the text: split by delimiter or parse as list",
            allow_output=False,
            allow_input=False,
            default_value=SplitMode.SPLIT.value,
        )
        self.add_parameter(self.split_mode)
        self.split_mode.add_trait(Options(choices=[mode.value for mode in SplitMode]))

        # Add delimiter type parameter
        self.delimiter_type = ParameterString(
            name="delimiter_type",
            tooltip="Type of delimiter to use for splitting",
            allow_output=False,
            allow_input=False,
            default_value="newlines",
        )
        self.add_parameter(self.delimiter_type)
        self.delimiter_type.add_trait(Options(choices=list(DELIMITER_MAP.keys())))

        # Add include delimiter option
        self.include_delimiter = ParameterBool(
            name="include_delimiter",
            tooltip="Whether to include the delimiter in the split results",
            allow_output=False,
            default_value=False,
        )
        self.add_parameter(self.include_delimiter)

        # Add trim whitespace option
        self.trim_whitespace = ParameterBool(
            name="trim_whitespace",
            tooltip="Whether to trim leading and trailing whitespace from each item",
            on_label="trim",
            off_label="keep",
            allow_output=False,
            default_value=False,
        )
        self.add_parameter(self.trim_whitespace)

        # Off by default so existing workflows keep their empty items
        self.remove_empty = ParameterBool(
            name="remove_empty",
            tooltip="Whether to drop blank or whitespace-only items from the split results",
            allow_output=False,
            default_value=False,
        )
        self.add_parameter(self.remove_empty)

        # Add output parameter
        self.output = Parameter(
            name="output",
            tooltip="List of text items",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

        # Set initial parameter visibility
        self._update_parameter_visibility()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in [
            self.text_input.name,
            self.delimiter_type.name,
            self.include_delimiter.name,
            self.trim_whitespace.name,
            self.remove_empty.name,
            self.split_mode.name,
        ]:
            self._process_text()

        # Control parameter visibility based on split_mode
        if parameter.name == self.split_mode.name:
            if value == SplitMode.PARSE_LIST:
                self.hide_parameter_by_name("delimiter_type")
                self.hide_parameter_by_name("include_delimiter")
            else:
                self.show_parameter_by_name("delimiter_type")
                self.show_parameter_by_name("include_delimiter")

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception]:
        exceptions = []
        text = self.get_parameter_value(self.text_input.name)
        if text is None:
            exceptions.append(Exception(f"{self.name}: Text is required to split"))
        elif not isinstance(text, str):
            exceptions.append(Exception(f"{self.name}: Text must be a string"))
        return exceptions

    def _process_text(self) -> None:
        """Process the text input according to the selected mode (split or parse)."""
        # Get all input parameters
        text = self.get_parameter_value(self.text_input.name)
        split_mode = self.get_parameter_value(self.split_mode.name)
        delimiter_type = self.get_parameter_value(self.delimiter_type.name)
        include_delimiter = self.get_parameter_value(self.include_delimiter.name)
        trim_whitespace = self.get_parameter_value(self.trim_whitespace.name)
        remove_empty = self.get_parameter_value(self.remove_empty.name)

        # Ensure text is a string
        if not isinstance(text, str):
            text = ""

        try:
            split_result = split_text(
                text,
                split_mode,
                delimiter_type,
                include_delimiter=include_delimiter,
                trim_whitespace=trim_whitespace,
                remove_empty=remove_empty,
            )

            self.parameter_output_values[self.output.name] = split_result
            self.publish_update_to_parameter(self.output.name, split_result)
        except (TypeError, ValueError) as e:
            # Handle type or value errors
            msg = f"{self.name}: Error processing text: {e}"
            logger.error(msg)
            self.parameter_output_values[self.output.name] = []
            self.publish_update_to_parameter(self.output.name, [])

    def _update_parameter_visibility(self) -> None:
        """Update parameter visibility based on split_mode."""
        split_mode = self.get_parameter_value(self.split_mode.name)

        if split_mode == SplitMode.PARSE_LIST:
            # Hide delimiter-specific parameters when in parse_list mode
            self.hide_parameter_by_name("delimiter_type")
            self.hide_parameter_by_name("include_delimiter")
            # Keep trim_whitespace visible as it's still useful for parsing
        else:
            # Show all parameters when in split mode
            self.show_parameter_by_name("delimiter_type")
            self.show_parameter_by_name("include_delimiter")

    def process(self) -> None:
        self._process_text()
