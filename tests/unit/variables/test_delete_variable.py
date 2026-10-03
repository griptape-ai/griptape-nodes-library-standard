"""Tests for DeleteVariable node."""

from collections.abc import Generator

import pytest
from griptape_nodes.exe_types.node_types import NodeResolutionState
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

from griptape_nodes_library.variables.delete_variable import DeleteVariable

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_delete_variable_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


@pytest.fixture
def delete_variable_node(flow: str) -> DeleteVariable:
    """Create a DeleteVariable node inside the test flow and return the instance."""
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="DeleteVariable", override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess)
    node = GriptapeNodes.NodeManager().get_node_by_name(result.node_name)
    assert type(node).__name__ == "DeleteVariable"
    return node  # type: ignore[return-value]


def _create_variable(name: str, flow_name: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateVariableRequest(name=name, type="str", is_global=False, value="", owning_flow=flow_name)
    )
    assert isinstance(result, CreateVariableResultSuccess)


def _has_variable(name: str, flow_name: str) -> bool:
    result = GriptapeNodes.handle_request(
        HasVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow_name)
    )
    assert isinstance(result, HasVariableResultSuccess)
    return result.exists


class TestDeleteVariableProcess:
    def test_deletes_single_variable(self, delete_variable_node: DeleteVariable, flow: str) -> None:
        _create_variable("a", flow)
        _create_variable("keep", flow)
        delete_variable_node.set_parameter_value("variable_names", "a")

        delete_variable_node.process()

        assert not _has_variable("a", flow)
        assert _has_variable("keep", flow)
        assert delete_variable_node.parameter_output_values["deleted_names"] == ["a"]

    def test_deletes_list_of_variables(self, delete_variable_node: DeleteVariable, flow: str) -> None:
        _create_variable("a", flow)
        _create_variable("b", flow)
        delete_variable_node.set_parameter_value("variable_names", ["a", "b", "a"])

        delete_variable_node.process()

        assert not _has_variable("a", flow)
        assert not _has_variable("b", flow)
        assert delete_variable_node.parameter_output_values["deleted_names"] == ["a", "b"]

    def test_missing_variable_is_skipped_by_default(self, delete_variable_node: DeleteVariable, flow: str) -> None:
        _create_variable("a", flow)
        delete_variable_node.set_parameter_value("variable_names", ["missing", "a"])

        delete_variable_node.process()

        assert not _has_variable("a", flow)
        assert delete_variable_node.parameter_output_values["deleted_names"] == ["a"]

    def test_missing_variable_fails_when_requested(self, delete_variable_node: DeleteVariable, flow: str) -> None:
        _create_variable("a", flow)
        delete_variable_node.set_parameter_value("variable_names", ["missing", "a"])
        delete_variable_node.set_parameter_value("fail_if_missing", True)

        with pytest.raises(RuntimeError, match="'missing' does not exist"):
            delete_variable_node.process()

        assert not _has_variable("a", flow)
        assert delete_variable_node.parameter_output_values["deleted_names"] == ["a"]

    def test_non_string_entry_raises(self, delete_variable_node: DeleteVariable) -> None:
        delete_variable_node.set_parameter_value("variable_names", ["a", 3])

        with pytest.raises(TypeError, match="must be a string"):
            delete_variable_node.process()


class TestDeleteVariableState:
    def test_unresolved_when_named_variable_exists(self, delete_variable_node: DeleteVariable, flow: str) -> None:
        _create_variable("a", flow)
        delete_variable_node.set_parameter_value("variable_names", "a")
        delete_variable_node.process()
        delete_variable_node.state = NodeResolutionState.RESOLVED
        assert delete_variable_node.state == NodeResolutionState.RESOLVED

        _create_variable("a", flow)
        assert delete_variable_node.state == NodeResolutionState.UNRESOLVED
