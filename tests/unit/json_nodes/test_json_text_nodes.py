"""Tests for the nodes that turn text into JSON: JSON Input, To JSON and Display JSON."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.convert.to_json import ToJson
from griptape_nodes_library.json.display_json import DisplayJson
from griptape_nodes_library.json.json_input import JsonInput

PARSED_CASES = [
    ('{"a": 1}', {"a": 1}),
    ("[1,2", [1, 2]),
    ("5", 5),
]


class TestJsonInput:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> JsonInput:  # noqa: ARG002
        return JsonInput(name="test_json_input")

    @pytest.mark.parametrize(("raw", "expected"), PARSED_CASES)
    def test_parses_text(self, node: JsonInput, raw: str, expected: object) -> None:
        node.set_parameter_value("json", raw)
        node.process()

        assert node.parameter_output_values["json"] == expected

    def test_plain_text_raises(self, node: JsonInput) -> None:
        node.set_parameter_value("json", "hello world")

        with pytest.raises(ValueError, match="Could not read the input as JSON"):
            node.process()


class TestToJson:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> ToJson:  # noqa: ARG002
        return ToJson(name="test_to_json")

    @pytest.mark.parametrize(("raw", "expected"), PARSED_CASES)
    def test_parses_text(self, node: ToJson, raw: str, expected: object) -> None:
        node.set_parameter_value("from", raw)
        node.process()

        assert node.parameter_output_values["output"] == expected

    @pytest.mark.parametrize("value", [5, True, None, [1, None]])
    def test_json_values_pass_through(self, node: ToJson, value: object) -> None:
        node.set_parameter_value("from", value)
        node.process()

        assert node.parameter_output_values["output"] == value

    def test_plain_text_clears_output_while_editing_and_raises_on_run(self, node: ToJson) -> None:
        node.set_parameter_value("from", '{"a": 1}')
        node.set_parameter_value("from", "hello world")

        assert node.parameter_output_values["output"] is None
        with pytest.raises(ValueError, match="Could not read the input as JSON"):
            node.process()


class TestDisplayJson:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> DisplayJson:  # noqa: ARG002
        return DisplayJson(name="test_display_json")

    def test_repairs_malformed_json_into_an_object(self, node: DisplayJson) -> None:
        node.set_parameter_value("json", '{"a": [1, 2,]}')
        node.process()

        assert node.parameter_output_values["json"] == {"a": [1, 2]}

    @pytest.mark.parametrize("raw", ["@foo", "`x`"])
    def test_text_that_is_not_json_or_yaml_raises(self, node: DisplayJson, raw: str) -> None:
        with pytest.raises(ValueError, match="Failed to parse input as JSON or YAML"):
            node.set_parameter_value("json", raw)
