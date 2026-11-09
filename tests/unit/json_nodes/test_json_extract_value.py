"""Tests for the JsonExtractValue node."""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.json.json_extract_value import JsonExtractValue


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> JsonExtractValue:
    node = JsonExtractValue("json_extract_value")

    # The node is not registered with a flow, so apply output requests to it directly.
    def handle_request(request: Any) -> None:
        if isinstance(request, SetParameterValueRequest):
            node.set_parameter_value(request.parameter_name, request.value)

    monkeypatch.setattr(GriptapeNodes, "handle_request", handle_request)
    return node


def test_string_json_input_is_extracted(node: JsonExtractValue) -> None:
    node.set_parameter_value("json", '{"a": {"b": 2}}')
    node.set_parameter_value("path", "a.b")
    node.process()

    assert node.get_parameter_value("output") == 2


def test_invalid_json_string_raises_on_process(node: JsonExtractValue) -> None:
    node.set_parameter_value("json", "{not json")

    with pytest.raises(ValueError, match="Invalid JSON string"):
        node.process()


def test_invalid_path_raises_on_process(node: JsonExtractValue) -> None:
    node.set_parameter_value("path", "a[")

    with pytest.raises(ValueError, match="Invalid JMESPath expression"):
        node.process()


@pytest.mark.parametrize(("parameter", "value"), [("json", "{not json"), ("path", "a[")])
def test_unparseable_input_does_not_raise_while_editing(node: JsonExtractValue, parameter: str, value: str) -> None:
    node.set_parameter_value(parameter, value)


def test_unparseable_edit_clears_previous_output(node: JsonExtractValue) -> None:
    node.set_parameter_value("path", "a")
    node.set_parameter_value("json", '{"a": 1}')
    assert node.get_parameter_value("output") == 1

    node.set_parameter_value("json", '{"a": 1')

    assert node.get_parameter_value("output") is None
