import json
import re
from typing import Any

from json_repair import repair_json

JSON_NATIVE_TYPES = (dict, list, str, int, float, bool, type(None))

# Text that repair_json is allowed to fix starts like a JSON value, so prose that only contains a
# bracket ("I am [busy]") is rejected rather than repaired into a fragment.
JSON_TEXT_START = re.compile(r'^(?:[\[{"\-\d]|true\b|false\b|null\b)')
FENCED_BLOCK = re.compile(r"```(?:json)?[^\S\n]*\n(.*?)(?:\n```|$)", re.DOTALL)


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


def repair_json_text(node_name: str, text: str) -> Any:
    """Parse ``text`` as JSON, repairing malformed JSON such as unclosed brackets or trailing commas.

    Returns ``None`` for blank text.

    Raises:
        ValueError: If ``text`` holds no JSON that can be repaired, such as plain prose.
    """
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = FENCED_BLOCK.search(text)
    stripped = (fenced.group(1) if fenced else text).strip()
    # repair_json returns "" when it finds nothing JSON-like in the text.
    repaired = repair_json(stripped, return_objects=True) if JSON_TEXT_START.match(stripped) else ""
    if repaired == "":
        msg = f"{node_name}: Input is not JSON and could not be repaired into JSON. Input was: {text[:200]!r}"
        raise ValueError(msg)
    return repaired


def to_json_value(node_name: str, value: Any) -> Any:
    """Return ``value`` as a JSON value: strings are parsed with repair, JSON-native values pass through.

    Other objects are converted with ``str()`` and then parsed.

    Raises:
        ValueError: If the text holds no JSON that can be repaired.
    """
    if isinstance(value, str):
        return repair_json_text(node_name, value)
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, JSON_NATIVE_TYPES):
        return value
    return repair_json_text(node_name, str(value))
