"""Switch adapts its options to the type wired into 'Value', through the real engine."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    DeleteConnectionRequest,
    DeleteConnectionResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.execution.switch import Switch

CANVAS = "switch_canvas"


@pytest.fixture
def canvas(griptape_nodes: Any) -> Generator[str, None, None]:  # noqa: ARG001
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="switch_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=CANVAS))
        assert isinstance(result, CreateFlowResultSuccess)
        yield CANVAS
    finally:
        context_manager.pop_workflow()


def _create(node_type: str, flow: str) -> Any:
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess), result.result_details
    return GriptapeNodes.NodeManager().get_node_by_name(result.node_name)


def _hidden(node: Switch) -> set[str]:
    return {p.name for p in (node.path_param, node.match_mode_param, node.case_sensitive_param) if p.hide}


def test_a_bool_wired_in_hides_every_option_and_removing_it_brings_them_back(canvas: str) -> None:
    switch = _create("Switch", canvas)
    source = _create("ToBool", canvas)
    assert _hidden(switch) == set()

    connected = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source.name,
            source_parameter_name="output",
            target_node_name=switch.name,
            target_parameter_name="value",
        )
    )
    assert isinstance(connected, CreateConnectionResultSuccess), connected.result_details
    assert _hidden(switch) == {"path", "match_mode", "case_sensitive"}

    removed = GriptapeNodes.handle_request(
        DeleteConnectionRequest(
            source_node_name=source.name,
            source_parameter_name="output",
            target_node_name=switch.name,
            target_parameter_name="value",
        )
    )
    assert isinstance(removed, DeleteConnectionResultSuccess)
    assert _hidden(switch) == set()


@pytest.mark.parametrize(
    ("value_type", "hidden"),
    [
        ("str", {"path"}),
        ("int", {"path", "match_mode", "case_sensitive"}),
        ("float", {"path", "match_mode", "case_sensitive"}),
        ("json", set()),
        ("yaml", set()),
        ("dict", set()),
        ("any", set()),
    ],
)
def test_each_type_shows_only_the_options_that_mean_something(canvas: str, value_type: str, hidden: set[str]) -> None:
    switch = _create("Switch", canvas)
    switch._show_options_for(value_type)
    assert _hidden(switch) == hidden
