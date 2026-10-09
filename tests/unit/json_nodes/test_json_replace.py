"""Tests for the JsonReplace node."""

from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.json.json_replace import JsonReplace


def _run(json_input: Any, path: str, replacement: Any) -> Any:
    node = JsonReplace("json_replace")
    node.set_parameter_value("json", json_input)
    node.set_parameter_value("path", path)
    node.set_parameter_value("replacement_value", replacement)
    node.process()
    return node.get_parameter_value("output")


@pytest.mark.parametrize("replacement", [3, 1.5, True, False, 0, "text", {"x": 1}])
def test_replacement_keeps_its_type(replacement: Any) -> None:
    result = _run({"a": 1, "keep": [1]}, "a", replacement)

    assert result == {"a": replacement, "keep": [1]}
    assert type(result["a"]) is type(replacement)


def test_string_json_input_is_parsed() -> None:
    assert _run('{"a": 1, "b": {"c": 2}}', "b.c", 5) == {"a": 1, "b": {"c": 5}}


def test_invalid_json_string_raises_on_process() -> None:
    node = JsonReplace("json_replace")
    node.set_parameter_value("json", "{not json")
    node.set_parameter_value("path", "a")

    with pytest.raises(ValueError, match="Invalid JSON string"):
        node.process()


def test_invalid_json_string_does_not_raise_while_editing() -> None:
    node = JsonReplace("json_replace")

    node.set_parameter_value("json", "{not json")
