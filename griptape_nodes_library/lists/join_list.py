import re
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

DEFAULT_SEPARATOR = "\\n"

# Escape sequences typed into the separator, prefix, and suffix fields, mapped to the characters they stand for.
# A doubled backslash stands for one literal backslash, so a typed \\n stays a backslash followed by "n".
ESCAPE_SEQUENCES = {"\\\\": "\\", "\\n": "\n", "\\t": "\t"}
ESCAPE_PATTERN = re.compile(r"\\\\|\\n|\\t")


def _unescape(text: str) -> str:
    return ESCAPE_PATTERN.sub(lambda match: ESCAPE_SEQUENCES[match.group(0)], text)


class JoinList(DataNode):
    """Join the items of a list into one text, with a separator between items and an optional prefix and suffix on each."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.items = Parameter(
            name="items",
            tooltip="List of items to join. Items that aren't text are converted to text; empty (None) items become empty text.",
            input_types=["list"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items)

        self.separator = ParameterString(
            name="separator",
            tooltip="Text placed between items. Use \\n for a new line, \\t for a tab, and \\\\ for a backslash.",
            default_value=DEFAULT_SEPARATOR,
            allow_output=False,
            placeholder_text="separator",
        )
        self.add_parameter(self.separator)

        self.item_prefix = ParameterString(
            name="item_prefix",
            tooltip="Text added before each item, e.g. '- ' for a bulleted list. Use \\n for a new line, \\t for a tab, and \\\\ for a backslash.",
            default_value="",
            allow_output=False,
            placeholder_text="prefix",
        )
        self.add_parameter(self.item_prefix)

        self.item_suffix = ParameterString(
            name="item_suffix",
            tooltip="Text added after each item. Use \\n for a new line, \\t for a tab, and \\\\ for a backslash.",
            default_value="",
            allow_output=False,
            placeholder_text="suffix",
        )
        self.add_parameter(self.item_suffix)

        self.output = ParameterString(
            name="output",
            tooltip="The joined text.",
            allowed_modes={ParameterMode.OUTPUT},
            multiline=True,
            placeholder_text="The joined text.",
        )
        self.add_parameter(self.output)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in {self.items.name, self.separator.name, self.item_prefix.name, self.item_suffix.name}:
            self._join()
        return super().after_value_set(parameter, value)

    def _join(self) -> None:
        items = self.get_parameter_value(self.items.name)
        if items is None:
            items = []
        elif not isinstance(items, list):
            items = [items]
        separator = _unescape(self.get_parameter_value(self.separator.name) or "")
        prefix = _unescape(self.get_parameter_value(self.item_prefix.name) or "")
        suffix = _unescape(self.get_parameter_value(self.item_suffix.name) or "")

        texts = ["" if item is None else str(item) for item in items]
        joined = separator.join(f"{prefix}{text}{suffix}" for text in texts)

        self.parameter_output_values[self.output.name] = joined
        self.publish_update_to_parameter(self.output.name, joined)

    def process(self) -> None:
        self._join()
