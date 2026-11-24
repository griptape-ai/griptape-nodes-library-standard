"""Tests for the shared JSON input parser."""

from __future__ import annotations

import pytest

from griptape_nodes_library.json.json_extract_value import JsonExtractValue
from griptape_nodes_library.json.json_find import JsonFind
from griptape_nodes_library.json.json_utils import parse_json_input, parse_or_repair_json, to_json_value


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
    node.set_parameter_value("json", "{bad")

    with pytest.raises(ValueError, match="finder: Invalid JSON string"):
        node.process()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("[3,1,2,", [3, 1, 2]),
        ('""', ""),
        ("null", None),
        ("  ", None),
    ],
)
def test_parse_or_repair_json_returns_objects(text: str, expected: object) -> None:
    assert parse_or_repair_json("node", text) == expected


@pytest.mark.parametrize("text", ["hello world", "abc"])
def test_parse_or_repair_json_raises_on_text_that_is_not_json(text: str) -> None:
    with pytest.raises(ValueError, match="node: Could not read the input as JSON"):
        parse_or_repair_json("node", text)


@pytest.mark.parametrize("value", [{"a": None}, [1, None], 3, 2.5, True, None])
def test_to_json_value_passes_json_values_through(value: object) -> None:
    assert to_json_value("node", value) is value


def test_to_json_value_parses_text() -> None:
    assert to_json_value("node", "[1, 2,") == [1, 2]
