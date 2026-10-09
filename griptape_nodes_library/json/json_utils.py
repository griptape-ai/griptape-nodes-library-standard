import json
from typing import Any


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
