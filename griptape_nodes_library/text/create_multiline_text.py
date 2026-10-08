from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.split_text_utils import (
    DEFAULT_DELIMITER_TYPE,
    DELIMITER_MAP,
    SplitMode,
    split_text,
)


class TextInput(DataNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: str = "",
    ) -> None:
        super().__init__(name, metadata)

        # Add output parameter for the string
        self.add_parameter(
            Parameter(
                name="text",
                default_value=value,
                input_types=["str"],
                output_type="str",
                type="str",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                tooltip="The text content to pass to another node.",
                ui_options={"multiline": True},
            ),
        )

        self.split_text = ParameterBool(
            name="split_text",
            tooltip="Also split the text into a list, output from output_split",
            allow_input=False,
            allow_output=False,
            default_value=False,
        )
        self.add_parameter(self.split_text)

        with ParameterGroup(name="split_text_options") as self.split_text_options:
            self.split_mode = ParameterString(
                name="split_mode",
                tooltip="How to process the text: split by delimiter or parse as list",
                allow_output=False,
                allow_input=False,
                default_value=SplitMode.SPLIT.value,
            )
            self.split_mode.add_trait(Options(choices=[mode.value for mode in SplitMode]))

            self.delimiter_type = ParameterString(
                name="delimiter_type",
                tooltip="Type of delimiter to use for splitting",
                allow_output=False,
                allow_input=False,
                default_value=DEFAULT_DELIMITER_TYPE,
            )
            self.delimiter_type.add_trait(Options(choices=list(DELIMITER_MAP.keys())))

            self.include_delimiter = ParameterBool(
                name="include_delimiter",
                tooltip="Whether to include the delimiter in the split results",
                allow_input=False,
                allow_output=False,
                default_value=False,
            )

            self.trim_whitespace = ParameterBool(
                name="trim_whitespace",
                tooltip="Whether to trim leading and trailing whitespace from each item",
                on_label="trim",
                off_label="keep",
                allow_input=False,
                allow_output=False,
                default_value=False,
            )

            self.remove_empty = ParameterBool(
                name="remove_empty",
                tooltip="Whether to drop blank or whitespace-only items from the split results",
                allow_input=False,
                allow_output=False,
                default_value=True,
            )
        self.add_node_element(self.split_text_options)

        # Hidden rather than removed when splitting is off, so its connections survive toggling
        self.output_split = Parameter(
            name="output_split",
            tooltip="List of text items split from the text",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output_split)

        self._update_parameter_visibility()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in {self.split_text.name, self.split_mode.name}:
            self._update_parameter_visibility()

        if parameter.name in {
            "text",
            self.split_text.name,
            self.split_mode.name,
            self.delimiter_type.name,
            self.include_delimiter.name,
            self.trim_whitespace.name,
            self.remove_empty.name,
        }:
            self._update_split_output()

        return super().after_value_set(parameter, value)

    def _update_parameter_visibility(self) -> None:
        if not self.get_parameter_value(self.split_text.name):
            self.split_text_options.update_ui_options({"hide": True})
            self.hide_parameter_by_name(self.output_split.name)
            return

        self.split_text_options.update_ui_options({"hide": False})
        self.show_parameter_by_name(self.output_split.name)

        # Delimiter options don't apply when parsing as a list
        delimiter_params = [self.delimiter_type.name, self.include_delimiter.name]
        if self.get_parameter_value(self.split_mode.name) == SplitMode.PARSE_LIST:
            self.hide_parameter_by_name(delimiter_params)
        else:
            self.show_parameter_by_name(delimiter_params)

    def _update_split_output(self) -> None:
        if not self.get_parameter_value(self.split_text.name):
            # Set rather than removed so the cleared value reaches anything still connected
            self.parameter_output_values[self.output_split.name] = None
            return

        text = self.get_parameter_value("text")
        if not isinstance(text, str):
            text = ""

        try:
            split_result = split_text(
                text,
                self.get_parameter_value(self.split_mode.name),
                self.get_parameter_value(self.delimiter_type.name),
                include_delimiter=self.get_parameter_value(self.include_delimiter.name),
                trim_whitespace=self.get_parameter_value(self.trim_whitespace.name),
                remove_empty=self.get_parameter_value(self.remove_empty.name),
            )
        except (TypeError, ValueError) as e:
            logger.error("%s: Error splitting text: %s", self.name, e)
            split_result = []

        self.parameter_output_values[self.output_split.name] = split_result
        self.publish_update_to_parameter(self.output_split.name, split_result)

    def process(self) -> None:
        # Simply output the default value or any updated property value
        self.parameter_output_values["text"] = self.get_parameter_value("text")
        self._update_split_output()
