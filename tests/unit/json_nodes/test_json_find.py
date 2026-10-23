"""Tests for the JsonFind node."""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.json.json_find import JsonFind


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> JsonFind:
    node = JsonFind("json_find")

    # The node is not registered with a flow, so apply output requests to it directly.
    def handle_request(request: Any) -> None:
        if isinstance(request, SetParameterValueRequest):
            node.set_parameter_value(request.parameter_name, request.value)

    monkeypatch.setattr(GriptapeNodes, "handle_request", handle_request)
    node.set_parameter_value("search_field", "name")
    node.set_parameter_value("search_value", "b")
    return node


def test_string_json_input_is_searched(node: JsonFind) -> None:
    node.set_parameter_value("json", '[{"name": "a"}, {"name": "b"}]')
    node.process()

    assert node.get_parameter_value("found_item") == {"name": "b"}
    assert node.get_parameter_value("found_count") == 1
    assert node.get_parameter_value("found_index") == 1


def test_invalid_json_string_raises_on_process(node: JsonFind) -> None:
    node.set_parameter_value("json", "{not json")

    with pytest.raises(ValueError, match="Invalid JSON string"):
        node.process()


def test_invalid_json_string_does_not_raise_while_editing(node: JsonFind) -> None:
    node.set_parameter_value("json", "{not json")


def test_unparseable_edit_clears_previous_outputs(node: JsonFind) -> None:
    node.set_parameter_value("json", '[{"name": "a"}, {"name": "b"}]')
    assert node.get_parameter_value("found_count") == 1

    node.set_parameter_value("json", '[{"name": "a"}')

    assert node.get_parameter_value("found_item") is None
    assert node.get_parameter_value("found_count") == 0
    assert node.get_parameter_value("found_index") == -1
