"""Splitting and list-parsing of text, shared by the Split Text and Text Input nodes."""

import ast
import json
from enum import StrEnum


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
            result = split_by_delimiter(
                text, delimiter_type, include_delimiter=include_delimiter, trim_whitespace=trim_whitespace
            )
        case SplitMode.PARSE_LIST:
            result = parse_as_list(text, delimiter_type, trim_whitespace=trim_whitespace)
        case _:
            msg = f"Unknown split mode: {split_mode!r}"
            raise ValueError(msg)

    if remove_empty:
        result = [item for item in result if item.strip()]
    return result


def split_by_delimiter(text: str, delimiter_type: str, *, include_delimiter: bool, trim_whitespace: bool) -> list[str]:
    """Split text by the delimiter named by delimiter_type."""
    actual_delimiter = DELIMITER_MAP.get(delimiter_type, DELIMITER_MAP[DEFAULT_DELIMITER_TYPE])

    split_result = text.split(actual_delimiter)
    # Trim before appending the delimiter so whitespace delimiters like newlines survive the strip
    if trim_whitespace:
        split_result = [item.strip() for item in split_result]

    if include_delimiter:
        # Append the delimiter to every element except the last one
        for i in range(len(split_result) - 1):
            split_result[i] += actual_delimiter

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
