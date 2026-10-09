"""Tests for the shared JSON input parser."""

from __future__ import annotations

import pytest

from griptape_nodes_library.json.json_extract_value import JsonExtractValue
from griptape_nodes_library.json.json_find import JsonFind
from griptape_nodes_library.json.json_utils import parse_json_input


def test_parses_json_string() -> None:
    assert parse_json_input("node", '{"a": [1, 2]}') == {"a": [1, 2]}


@pytest.mark.parametrize("value", [{"a": 1}, [1, 2], 3, None])
def test_non_string_passes_through(value: object) -> None:
    assert parse_json_input("node", value) is value


def test_invalid_string_raises_with_node_name_and_input() -> None:
    with pytest.raises(ValueError, match=r"my_node: Invalid JSON string.*\{bad"):
        parse_json_input("my_node", "{bad")


def test_extract_value_parses_string_input(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []
    monkeypatch.setattr(
        "griptape_nodes_library.json.json_extract_value.GriptapeNodes.handle_request",
        lambda request: captured.append(request.value),
    )
    node = JsonExtractValue("extract")
    node.set_parameter_value("json", '{"a": {"b": 7}}')
    node.set_parameter_value("path", "a.b")

    node.process()

    assert captured[-1] == 7


def test_find_raises_on_invalid_string_with_node_name() -> None:
    node = JsonFind("finder")

    with pytest.raises(ValueError, match="finder: Invalid JSON string"):
        node.set_parameter_value("json", "{bad")
