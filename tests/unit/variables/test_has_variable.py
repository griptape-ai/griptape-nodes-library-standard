"""Tests for HasVariable: variable_name accepts arbitrary names (Options with allow_custom) and process() reports whether the variable exists."""

from collections.abc import Generator

import pytest
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    DeleteVariableRequest,
    DeleteVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.options import Options

from griptape_nodes_library.variables.has_variable import HasVariable

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_has_variable_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


@pytest.fixture
def has_variable_node(flow: str) -> HasVariable:
    """Create a HasVariable node inside the test flow and return the instance."""
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="HasVariable", override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess)
    node = GriptapeNodes.NodeManager().get_node_by_name(result.node_name)
    assert type(node).__name__ == "HasVariable"
    return node  # type: ignore[return-value]


class TestHasVariableFreeText:
    """The variable_name field is a dropdown of existing names that also accepts arbitrary names."""

    def test_variable_name_is_connectable_and_settable(self, has_variable_node: HasVariable) -> None:
        """variable_name is a str usable as INPUT, OUTPUT, and PROPERTY."""
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        assert param.type == "str"
        assert ParameterMode.INPUT in param.allowed_modes
        assert ParameterMode.OUTPUT in param.allowed_modes
        assert ParameterMode.PROPERTY in param.allowed_modes

    def test_variable_name_allows_custom_options(self, has_variable_node: HasVariable) -> None:
        """variable_name's Options trait allows custom values."""
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        options = param.find_elements_by_type(Options)
        assert len(options) == 1
        assert options[0].allow_custom is True

    def test_unknown_name_is_kept(self, has_variable_node: HasVariable) -> None:
        """A name that isn't among the choices is kept as set."""
        has_variable_node.set_parameter_value("variable_name", "not_yet")

        assert has_variable_node.get_parameter_value("variable_name") == "not_yet"

    def test_name_survives_variable_disappearing(self, has_variable_node: HasVariable, flow: str) -> None:
        """A name picked from the dropdown is kept after the variable is deleted and the choices refreshed."""
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="foo", type="str", is_global=False, value="v", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)
        has_variable_node._refresh_variable_names(button=None, button_details=None)  # type: ignore[arg-type]
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        options = param.find_elements_by_type(Options)
        assert "foo" in options[0].choices
        has_variable_node.set_parameter_value("variable_name", "foo")

        delete_result = GriptapeNodes.handle_request(
            DeleteVariableRequest(name="foo", lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow)
        )
        assert isinstance(delete_result, DeleteVariableResultSuccess)
        has_variable_node._refresh_variable_names(button=None, button_details=None)  # type: ignore[arg-type]

        assert has_variable_node.get_parameter_value("variable_name") == "foo"
        assert "foo" not in options[0].choices

        has_variable_node.process()

        assert has_variable_node.parameter_output_values["exists"] is False


class TestHasVariableProcess:
    """process() reports existence correctly for both present and absent names."""

    def test_returns_false_for_nonexistent_variable(self, has_variable_node: HasVariable) -> None:
        """The core use case: checking a name that does not exist must be possible and return False."""
        has_variable_node.set_parameter_value("variable_name", "does_not_exist")

        has_variable_node.process()

        assert has_variable_node.parameter_output_values["exists"] is False

    def test_returns_true_for_existing_variable(self, has_variable_node: HasVariable, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="present", type="str", is_global=False, value="v", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)
        has_variable_node._refresh_variable_names(button=None, button_details=None)  # type: ignore[arg-type]

        has_variable_node.set_parameter_value("variable_name", "present")

        has_variable_node.process()

        assert has_variable_node.parameter_output_values["exists"] is True
        assert has_variable_node.parameter_output_values["variable_name"] == "present"
