"""Set a value inside nested mappings and lists by a dot notation path such as `items[0].title`."""

import copy
import re
from typing import Any

_ARRAY_PART = re.compile(r"^(.+)\[(\d+)\]$")
_PATH_SEPARATOR = re.compile(r"\.(?![^\[]*\])")


def set_value_at_path(data: Any, path: str, new_value: Any) -> Any:
    """Return a copy of data with new_value at path. An empty path replaces data entirely.

    Missing keys are created, and a list shorter than an index is padded with nulls. A missing or
    null value in the middle of the path becomes an empty mapping. Raises ValueError when a part of
    the path lands on any other value that is not a mapping or list.
    """
    if not path:
        return new_value

    result = copy.deepcopy(data)
    *parent_parts, final_part = _PATH_SEPARATOR.split(path)

    container = result
    for part in parent_parts:
        container = _step_into(container, part, path)

    array_part = _parse_array_part(final_part, path)
    if array_part is None:
        _require_mapping(container, final_part, path)[final_part] = new_value
        return result

    key, index = array_part
    items = _child_list(container, key, final_part, path)
    _pad_to(items, index)
    if index == len(items):
        items.append(new_value)
    else:
        items[index] = new_value
    return result


def _step_into(container: Any, part: str, path: str) -> Any:
    """Return the child at part, making it an empty mapping for the next part if missing or null."""
    array_part = _parse_array_part(part, path)
    if array_part is None:
        mapping = _require_mapping(container, part, path)
        if mapping.get(part) is None:
            mapping[part] = {}
        return mapping[part]

    key, index = array_part
    items = _child_list(container, key, part, path)
    _pad_to(items, index)
    if index == len(items):
        items.append({})
    elif items[index] is None:
        items[index] = {}
    return items[index]


def _parse_array_part(part: str, path: str) -> tuple[str, int] | None:
    match = _ARRAY_PART.match(part)
    if not match:
        return None
    key = match.group(1)
    if "[" in key or "]" in key:
        msg = f"Cannot follow path '{path}' at '{part}': only one list index per part is supported."
        raise ValueError(msg)
    return key, int(match.group(2))


def _require_mapping(container: Any, part: str, path: str) -> dict:
    if not isinstance(container, dict):
        msg = f"Cannot follow path '{path}' at '{part}': expected a mapping, found {_describe(container)}."
        raise ValueError(msg)  # noqa: TRY004
    return container


def _child_list(container: Any, key: str, part: str, path: str) -> list:
    mapping = _require_mapping(container, part, path)
    if mapping.get(key) is None:
        mapping[key] = []
    items = mapping[key]
    if not isinstance(items, list):
        msg = f"Cannot follow path '{path}' at '{part}': '{key}' is {_describe(items)}, not a list."
        raise ValueError(msg)  # noqa: TRY004
    return items


def _pad_to(items: list, index: int) -> None:
    while len(items) < index:
        items.append(None)


def _describe(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, list):
        return "a list"
    return f"a {type(value).__name__} value"
