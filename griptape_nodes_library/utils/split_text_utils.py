"""Splitting and list-parsing of text, and the split option parameters, shared by the Split Text and Text Input nodes."""

import ast
import json
from dataclasses import dataclass
from enum import StrEnum

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options


class SplitMode(StrEnum):
    SPLIT = "split"
    PARSE_LIST = "parse_list"


DELIMITER_MAP: dict[str, str] = {
    "newlines": "\n",
    "double_newline": "\n\n",
    "space": " ",
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "tab": "\t",
    "pipe": "|",
    "dash": "-",
    "underscore": "_",
    "period": ".",
    "slash": "/",
    "backslash": "\\",
    "at": "@",
    "hash": "#",
    "ampersand": "&",
    "equals": "=",
    "question mark": "?",
}

DEFAULT_DELIMITER_TYPE = "newlines"


@dataclass(frozen=True)
class SplitOptions:
    """The split option parameters, defined once so every node that splits text offers the same options."""

    split_mode: ParameterString
    delimiter_type: ParameterString
    include_delimiter: ParameterBool
    trim_whitespace: ParameterBool
    remove_empty: ParameterBool

    @classmethod
    def create(cls, *, remove_empty_default: bool, allow_toggle_input: bool) -> "SplitOptions":
        """Create the parameters. Created inside a ParameterGroup context, they join that group.

        Args:
            remove_empty_default: Default for remove_empty.
            allow_toggle_input: Whether the boolean options accept input connections.
        """
        split_mode = ParameterString(
            name="split_mode",
            tooltip="How to process the text: split by delimiter or parse as list",
            allow_output=False,
            allow_input=False,
            default_value=SplitMode.SPLIT.value,
        )
        split_mode.add_trait(Options(choices=[mode.value for mode in SplitMode]))

        delimiter_type = ParameterString(
            name="delimiter_type",
            tooltip="Type of delimiter to use for splitting",
            allow_output=False,
            allow_input=False,
            default_value=DEFAULT_DELIMITER_TYPE,
        )
        delimiter_type.add_trait(Options(choices=list(DELIMITER_MAP.keys())))

        include_delimiter = ParameterBool(
            name="include_delimiter",
            tooltip="Whether to include the delimiter in the split results",
            allow_input=allow_toggle_input,
            allow_output=False,
            default_value=False,
        )

        trim_whitespace = ParameterBool(
            name="trim_whitespace",
            tooltip="Whether to trim leading and trailing whitespace from each item",
            on_label="trim",
            off_label="keep",
            allow_input=allow_toggle_input,
            allow_output=False,
            default_value=False,
        )

        remove_empty = ParameterBool(
            name="remove_empty",
            tooltip="Whether to drop blank or whitespace-only items from the split results",
            allow_input=allow_toggle_input,
            allow_output=False,
            default_value=remove_empty_default,
        )

        return cls(split_mode, delimiter_type, include_delimiter, trim_whitespace, remove_empty)

    @property
    def parameters(self) -> list[Parameter]:
        return [self.split_mode, self.delimiter_type, self.include_delimiter, self.trim_whitespace, self.remove_empty]

    @property
    def names(self) -> set[str]:
        return {param.name for param in self.parameters}

    def update_delimiter_visibility(self, node: BaseNode) -> None:
        """Hide the delimiter options when parsing as a list, where they don't apply."""
        delimiter_names = [self.delimiter_type.name, self.include_delimiter.name]
        if node.get_parameter_value(self.split_mode.name) == SplitMode.PARSE_LIST:
            node.hide_parameter_by_name(delimiter_names)
        else:
            node.show_parameter_by_name(delimiter_names)

    def split(self, node: BaseNode, text: str) -> list[str]:
        """Split text using the option values currently set on node."""
        return split_text(
            text,
            node.get_parameter_value(self.split_mode.name),
            node.get_parameter_value(self.delimiter_type.name),
            include_delimiter=node.get_parameter_value(self.include_delimiter.name),
            trim_whitespace=node.get_parameter_value(self.trim_whitespace.name),
            remove_empty=node.get_parameter_value(self.remove_empty.name),
        )


def split_text(
    text: str,
    split_mode: str,
    delimiter_type: str,
    *,
    include_delimiter: bool,
    trim_whitespace: bool,
    remove_empty: bool = False,
) -> list[str]:
    """Split text by delimiter, or parse it as a list, according to split_mode.

    With remove_empty, items that are blank or whitespace-only are dropped from the result.
    """
    match split_mode:
        case SplitMode.SPLIT:
            return split_by_delimiter(
                text,
                delimiter_type,
                include_delimiter=include_delimiter,
                trim_whitespace=trim_whitespace,
                remove_empty=remove_empty,
            )
        case SplitMode.PARSE_LIST:
            result = parse_as_list(text, delimiter_type, trim_whitespace=trim_whitespace)
            if remove_empty:
                result = [item for item in result if item.strip()]
            return result
        case _:
            msg = f"Unknown split mode: {split_mode!r}"
            raise ValueError(msg)


def split_by_delimiter(
    text: str, delimiter_type: str, *, include_delimiter: bool, trim_whitespace: bool, remove_empty: bool = False
) -> list[str]:
    """Split text by the delimiter named by delimiter_type."""
    actual_delimiter = DELIMITER_MAP.get(delimiter_type, DELIMITER_MAP[DEFAULT_DELIMITER_TYPE])

    items = text.split(actual_delimiter)
    # Trim before appending the delimiter so whitespace delimiters like newlines survive the strip
    if trim_whitespace:
        items = [item.strip() for item in items]

    # Check emptiness before the delimiter is appended, or a non-whitespace delimiter makes an empty item look full.
    # The delimiter goes on every item but the last of the original split, so removed items don't shift it.
    last_index = len(items) - 1
    split_result = []
    for i, item in enumerate(items):
        if remove_empty and not item.strip():
            continue
        if include_delimiter and i < last_index:
            item += actual_delimiter
        split_result.append(item)

    return split_result


def parse_as_list(text: str, delimiter_type: str, *, trim_whitespace: bool) -> list[str]:
    """Parse text as a JSON or Python list, falling back to comma, then delimiter, splitting."""
    # Try JSON parsing first (for double-quoted lists like '["one", "two"]')
    try:
        parsed_list = json.loads(text)
        if isinstance(parsed_list, list):
            return [str(item) for item in parsed_list]
        return [str(parsed_list)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal evaluation (for single-quoted lists like "['one', 'two']")
    try:
        parsed_list = ast.literal_eval(text)
        if isinstance(parsed_list, list):
            return [str(item) for item in parsed_list]
        return [str(parsed_list)]
    except (ValueError, SyntaxError):
        pass

    # Try comma-separated parsing (for cases like "one, two, three")
    # Comma items are always stripped, regardless of trim_whitespace
    if "," in text:
        return [item.strip() for item in text.split(",")]

    # Fallback to delimiter splitting
    actual_delimiter = DELIMITER_MAP.get(delimiter_type, DELIMITER_MAP[DEFAULT_DELIMITER_TYPE])
    result = text.split(actual_delimiter)
    if trim_whitespace:
        result = [item.strip() for item in result]
    return result
