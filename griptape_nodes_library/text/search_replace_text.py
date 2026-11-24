import re
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class SearchReplaceText(DataNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add input text parameter
        self.add_parameter(
            ParameterString(
                name="input_text",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                multiline=True,
                placeholder_text="Replace this text: wombat...",
                tooltip="The multiline text to perform search and replace on.",
            )
        )

        # Add search pattern parameter
        self.add_parameter(
            ParameterString(
                name="search_pattern",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                placeholder_text="wombat",
                tooltip="The text or pattern to search for. Can include newlines when using regex mode.",
            )
        )

        # Add replacement text parameter
        self.add_parameter(
            ParameterString(
                name="replacement_text",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                placeholder_text="capybara",
                tooltip="The text to replace the search pattern with. Inserted as-is unless regex mode is on, where \\1 or \\g<name> inserts a captured group.",
            )
        )

        # Create options group
        with ParameterGroup(name="Options", ui_options={"hide": True}) as options_group:
            # Add case sensitive option
            ParameterBool(
                name="case_sensitive",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=True,
                tooltip="Whether the search should be case sensitive.",
            )

            # Add regex option
            ParameterBool(
                name="use_regex",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=False,
                tooltip="Whether to treat the search pattern as a regular expression. When enabled, you can use patterns like \\n for newlines.",
            )

            # Add replace all option
            ParameterBool(
                name="replace_all",
                allowed_modes={ParameterMode.PROPERTY},
                default_value=True,
                tooltip="Whether to replace all occurrences or just the first one.",
            )

        self.add_node_element(options_group)

        # Add output parameter
        self.add_parameter(
            ParameterString(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                allow_input=False,
                allow_property=False,
                multiline=True,
                placeholder_text="The text after performing search and replace.",
            )
        )

    def _search_replace(self) -> str:
        """Perform search and replace using regex under the hood.

        Raises:
            ValueError: If the search pattern is an invalid regex, or the replacement
                text has an invalid group reference or escape in regex mode.
        """
        input_text = self.get_parameter_value("input_text")
        search_pattern = self.get_parameter_value("search_pattern")
        replacement_text = self.get_parameter_value("replacement_text") or ""
        case_sensitive = self.get_parameter_value("case_sensitive")
        use_regex = self.get_parameter_value("use_regex")
        replace_all = self.get_parameter_value("replace_all")

        if not input_text or not search_pattern:
            return input_text

        pattern = search_pattern if use_regex else re.escape(search_pattern)
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            compiled = re.compile(pattern, flags)
        except re.error as e:
            msg = f"'search_pattern' {search_pattern!r} is not a valid regular expression: {e}"
            raise ValueError(msg) from e

        # A callable replacement is inserted as-is, so plain mode never interprets backslashes.
        replacement = replacement_text if use_regex else lambda _match: replacement_text
        try:
            return compiled.sub(replacement, input_text, count=0 if replace_all else 1)
        except re.error as e:
            msg = f"'replacement_text' {replacement_text!r} is not valid for this pattern: {e}"
            raise ValueError(msg) from e

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name != "output":
            try:
                result = self._search_replace()
            except ValueError:
                # The pattern may be half-typed; running the node reports the error.
                result = self.get_parameter_value("input_text")
            self.parameter_output_values["output"] = result
            self.set_parameter_value("output", result)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        # Perform the search and replace
        result = self._search_replace()

        # Set the output
        self.parameter_output_values["output"] = result
