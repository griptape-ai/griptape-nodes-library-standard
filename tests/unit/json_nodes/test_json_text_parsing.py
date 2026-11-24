"""Tests for JSON text parsing with repair in To JSON, JSON Input, Display JSON, and Sort List."""

from __future__ import annotations

import pytest

from griptape_nodes_library.convert.to_json import ToJson
from griptape_nodes_library.json.display_json import DisplayJson
from griptape_nodes_library.json.json_input import JsonInput
from griptape_nodes_library.json.json_utils import repair_json_text, to_json_value
from griptape_nodes_library.lists.sort_list import SortList


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ("[1,2", [1, 2]),
        ('{"a": 1,}', {"a": 1}),
        ("5", 5),
        ('""', ""),
        ("null", None),
        ("   ", None),
    ],
)
def test_repair_json_text_returns_objects(text: str, expected: object) -> None:
    assert repair_json_text("node", text) == expected


@pytest.mark.parametrize("text", ['```json\n{"a": 1\n```', 'Here is the JSON:\n```json\n{"a": 1}\n```'])
def test_repair_json_text_repairs_fenced_block(text: str) -> None:
    assert repair_json_text("node", text) == {"a": 1}


@pytest.mark.parametrize("text", ["hello world", "abc: 1", "I am [busy]", 'hello {"a": 1}'])
def test_repair_json_text_raises_on_plain_text(text: str) -> None:
    with pytest.raises(ValueError, match=r"my_node: Input is not JSON"):
        repair_json_text("my_node", text)


@pytest.mark.parametrize("value", [{"a": 1}, [1, 2], 5, 1.5, True, None])
def test_to_json_value_passes_json_native_values_through(value: object) -> None:
    assert to_json_value("node", value) == value


def test_to_json_value_converts_tuple_to_list() -> None:
    assert to_json_value("node", (1, 2)) == [1, 2]


def test_to_json_parses_string_to_object() -> None:
    node = ToJson("to_json")
    node.set_parameter_value("from", '{"a": 1}')
    node.process()
    assert node.parameter_output_values["output"] == {"a": 1}


def test_to_json_keeps_number_as_number() -> None:
    node = ToJson("to_json")
    node.set_parameter_value("from", 5)
    node.process()
    assert node.parameter_output_values["output"] == 5


def test_to_json_keeps_list_as_list() -> None:
    node = ToJson("to_json")
    node.set_parameter_value("from", [1, "a"])
    node.process()
    assert node.parameter_output_values["output"] == [1, "a"]


def test_to_json_blank_input_outputs_none() -> None:
    node = ToJson("to_json")
    node.set_parameter_value("from", "")
    node.process()
    assert node.parameter_output_values["output"] is None


def test_to_json_raises_on_plain_text_when_run() -> None:
    node = ToJson("to_json")
    node.set_parameter_value("from", "hello world")
    assert node.parameter_output_values["output"] is None
    with pytest.raises(ValueError, match=r"to_json: Input is not JSON"):
        node.process()


@pytest.mark.parametrize(("text", "expected"), [('{"a": 1}', {"a": 1}), ("[1,2", [1, 2])])
def test_json_input_parses_string_to_object(text: str, expected: object) -> None:
    node = JsonInput("json_input")
    node.set_parameter_value("json", text)
    node.process()
    assert node.parameter_output_values["json"] == expected


def test_json_input_raises_on_plain_text() -> None:
    node = JsonInput("json_input")
    node.set_parameter_value("json", "hello world")
    with pytest.raises(ValueError, match=r"json_input: Input is not JSON"):
        node.process()


def test_display_json_repairs_to_object() -> None:
    node = DisplayJson("display_json")
    node.set_parameter_value("json", '{"a": 1,}')
    assert node.parameter_output_values["json"] == {"a": 1}


@pytest.mark.parametrize(("text", "expected"), [("hello world", "hello world"), ("abc: 1", {"abc": 1})])
def test_display_json_falls_back_to_yaml_for_non_json_text(text: str, expected: object) -> None:
    node = DisplayJson("display_json")
    node.set_parameter_value("json", text)
    assert node.parameter_output_values["json"] == expected


@pytest.mark.parametrize(("text", "expected"), [("[3,1,2,", [3, 1, 2]), ("[3, 1, 2", [3, 1, 2])])
def test_sort_list_repairs_malformed_list(text: str, expected: list[int]) -> None:
    assert SortList("sort")._parse_list_input(text) == expected


def test_sort_list_plain_text_is_not_a_list() -> None:
    assert SortList("sort")._parse_list_input("hello world") is None
