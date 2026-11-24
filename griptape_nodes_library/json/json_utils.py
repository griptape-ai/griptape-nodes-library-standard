import json
from typing import Any

from json_repair import repair_json


def parse_json_input(node_name: str, value: Any) -> Any:
    """Return ``value`` parsed as JSON if it is a string, otherwise return it unchanged.

    Raises:
        ValueError: If ``value`` is a string that is not valid JSON.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as e:
        msg = f"{node_name}: Invalid JSON string provided. Failed to parse JSON: {e}. Input was: {value[:200]!r}"
        raise ValueError(msg) from e


JSON_NATIVE_TYPES = (dict, list, int, float, bool, type(None))


def parse_or_repair_json(node_name: str, text: str) -> Any:
    """Parse ``text`` as JSON, repairing it first if it is malformed. Blank text parses to ``None``.

    Raises:
        ValueError: If ``text`` can't be read as JSON even after repair.
    """
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        repaired = repair_json(text, return_objects=True)
    except Exception as e:
        msg = f"{node_name}: Failed to repair JSON: {e}. Input was: {text[:200]!r}"
        raise ValueError(msg) from e
    # repair_json gives back an empty string when nothing in the text is JSON.
    if repaired == "":
        msg = f"{node_name}: Could not read the input as JSON. Input was: {text[:200]!r}"
        raise ValueError(msg)
    return repaired


def to_json_value(node_name: str, value: Any) -> Any:
    """Return ``value`` as a JSON object, list or scalar.

    Text is parsed with ``parse_or_repair_json``. Values that are already JSON pass through
    unchanged, and anything else is parsed from its string form.

    Raises:
        ValueError: If ``value`` can't be read as JSON.
    """
    if isinstance(value, str):
        return parse_or_repair_json(node_name, value)
    if isinstance(value, JSON_NATIVE_TYPES):
        return value
    return parse_or_repair_json(node_name, str(value))
