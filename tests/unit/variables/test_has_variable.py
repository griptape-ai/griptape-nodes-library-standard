"""Tests for HasVariable node.

Covers the fix for issue #469: HasVariable must let you check an arbitrary variable
name — including one that does not exist — rather than constraining ``variable_name`` to
an ``Options`` dropdown of only-existing variables (which defeats the node's purpose).
"""

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
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button
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
    """The variable_name field must accept arbitrary names, not a fixed dropdown of existing ones."""

    def test_variable_name_has_no_options_dropdown(self, has_variable_node: HasVariable) -> None:
        """variable_name must not carry an Options trait — that would limit it to existing names."""
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        assert param.find_elements_by_type(Options) == []

    def test_variable_name_has_no_refresh_button(self, has_variable_node: HasVariable) -> None:
        """The refresh button only makes sense alongside a dropdown; it should be gone too."""
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        assert param.find_elements_by_type(Button) == []

    def test_variable_name_still_connectable_and_settable(self, has_variable_node: HasVariable) -> None:
        """variable_name stays a str usable as INPUT/OUTPUT/PROPERTY so it can be typed or wired."""
        param = has_variable_node.get_parameter_by_name("variable_name")
        assert param is not None
        assert param.type == "str"
        assert ParameterMode.INPUT in param.allowed_modes
        assert ParameterMode.PROPERTY in param.allowed_modes


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

        has_variable_node.set_parameter_value("variable_name", "present")

        has_variable_node.process()

        assert has_variable_node.parameter_output_values["exists"] is True
        assert has_variable_node.parameter_output_values["variable_name"] == "present"
