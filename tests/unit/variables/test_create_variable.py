"""Tests for CreateVariable's handling of connections when variable_type changes."""

from collections.abc import Generator
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.base_events import RequestPayload
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import (
    SetParameterValueRequest,
    SetParameterValueResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_create_variable_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


def _create_node(node_type: str, flow_name: str) -> str:
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow_name))
    assert isinstance(result, CreateNodeResultSuccess)
    return result.node_name


def _connect_value_to_display_integer(create_variable: str, display_integer: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=create_variable,
            source_parameter_name="value",
            target_node_name=display_integer,
            target_parameter_name="integer",
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess)


def _outgoing_value_connections(node_name: str) -> list[Any]:
    result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node_name))
    assert isinstance(result, ListConnectionsForNodeResultSuccess)
    return [c for c in result.outgoing_connections if c.source_parameter_name == "value"]


def _set_variable_type(node_name: str, value: str, *, initial_setup: bool) -> None:
    result = GriptapeNodes.handle_request(
        SetParameterValueRequest(
            node_name=node_name, parameter_name="variable_type", value=value, initial_setup=initial_setup
        )
    )
    assert isinstance(result, SetParameterValueResultSuccess)


def test_editing_variable_type_removes_incompatible_outgoing_connection(flow: str) -> None:
    create_variable = _create_node("CreateVariable", flow)
    display_integer = _create_node("DisplayInteger", flow)
    _connect_value_to_display_integer(create_variable, display_integer)

    _set_variable_type(create_variable, "str", initial_setup=False)

    assert _outgoing_value_connections(create_variable) == []


def test_loading_variable_type_leaves_connections_alone(flow: str, monkeypatch: pytest.MonkeyPatch) -> None:
    create_variable = _create_node("CreateVariable", flow)
    display_integer = _create_node("DisplayInteger", flow)
    _connect_value_to_display_integer(create_variable, display_integer)

    sent: list[type[RequestPayload]] = []
    handle_request = GriptapeNodes.handle_request

    def recording_handle_request(request: RequestPayload, *args: Any, **kwargs: Any) -> Any:
        sent.append(type(request))
        return handle_request(request, *args, **kwargs)

    monkeypatch.setattr(GriptapeNodes, "handle_request", recording_handle_request)
    _set_variable_type(create_variable, "str", initial_setup=True)
    monkeypatch.undo()

    assert ListConnectionsForNodeRequest not in sent[1:]
    assert len(_outgoing_value_connections(create_variable)) == 1
