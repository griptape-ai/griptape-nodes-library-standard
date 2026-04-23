"""Tests for SetVariable node.

Covers the create-if-missing behavior (issue #4411): SetVariable no longer requires a
pre-existing variable. It eagerly registers the variable with the engine when the user
sets ``variable_name`` and falls back to create-or-update during ``process()``.
"""

from collections.abc import Generator

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    GetVariableRequest,
    GetVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_set_variable_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


@pytest.fixture
def set_variable_node(flow: str) -> BaseNode:
    """Create a SetVariable node inside the test flow and return the instance.

    Returns the node as ``BaseNode`` because library nodes are loaded via a dynamic module,
    so the imported ``SetVariable`` class and the instantiated class are different class
    objects even though they share a name.
    """
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="SetVariable", override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess)
    node = GriptapeNodes.NodeManager().get_node_by_name(result.node_name)
    assert type(node).__name__ == "SetVariable"
    return node


def _has_variable(name: str, flow_name: str) -> bool:
    result = GriptapeNodes.handle_request(
        HasVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow_name)
    )
    assert isinstance(result, HasVariableResultSuccess)
    return result.exists


def _get_variable_value(name: str, flow_name: str) -> object:
    result = GriptapeNodes.handle_request(
        GetVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow_name)
    )
    assert isinstance(result, GetVariableResultSuccess)
    return result.variable.value


class TestSetVariableProcess:
    """Exercises the rewritten ``aprocess()``: create-if-missing, else update."""

    @pytest.mark.asyncio
    async def test_creates_variable_when_missing(self, set_variable_node: BaseNode, flow: str) -> None:
        set_variable_node.set_parameter_value("variable_name", "my_var")
        set_variable_node.set_parameter_value("value", "hello")

        await set_variable_node.aprocess()

        assert _has_variable("my_var", flow)
        assert _get_variable_value("my_var", flow) == "hello"

    @pytest.mark.asyncio
    async def test_updates_existing_variable(self, set_variable_node: BaseNode, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="existing", type="str", is_global=False, value="old", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)

        set_variable_node.set_parameter_value("variable_name", "existing")
        set_variable_node.set_parameter_value("value", "new")

        await set_variable_node.aprocess()

        assert _get_variable_value("existing", flow) == "new"

    @pytest.mark.asyncio
    async def test_empty_variable_name_raises(self, set_variable_node: BaseNode) -> None:
        set_variable_node.set_parameter_value("variable_name", "")
        set_variable_node.set_parameter_value("value", "anything")

        with pytest.raises(ValueError, match="requires a non-empty variable_name"):
            await set_variable_node.aprocess()


class TestSetVariableEagerRegistration:
    """Exercises ``before_value_set``: variables register the moment the user enters a name."""

    def test_entering_name_registers_variable(self, set_variable_node: BaseNode, flow: str) -> None:
        assert not _has_variable("eager", flow)

        set_variable_node.set_parameter_value("variable_name", "eager")

        assert _has_variable("eager", flow)

    def test_changing_name_renames_variable(self, set_variable_node: BaseNode, flow: str) -> None:
        set_variable_node.set_parameter_value("variable_name", "first")
        assert _has_variable("first", flow)

        set_variable_node.set_parameter_value("variable_name", "second")

        assert not _has_variable("first", flow)
        assert _has_variable("second", flow)

    def test_clearing_name_leaves_variable_alone(self, set_variable_node: BaseNode, flow: str) -> None:
        set_variable_node.set_parameter_value("variable_name", "sticky")
        assert _has_variable("sticky", flow)

        set_variable_node.set_parameter_value("variable_name", "")

        # Clearing is a no-op: the engine-side variable persists.
        assert _has_variable("sticky", flow)

    def test_rename_collision_is_silent(self, set_variable_node: BaseNode, flow: str) -> None:
        collision_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="taken", type="str", is_global=False, value="reserved", owning_flow=flow)
        )
        assert isinstance(collision_result, CreateVariableResultSuccess)

        set_variable_node.set_parameter_value("variable_name", "mine")
        assert _has_variable("mine", flow)

        # Attempting to rename into the taken name must not raise, and must not clobber
        # the existing variable's value.
        set_variable_node.set_parameter_value("variable_name", "taken")

        assert _has_variable("taken", flow)
        assert _get_variable_value("taken", flow) == "reserved"

    def test_duplicate_name_adopts_existing(self, set_variable_node: BaseNode, flow: str) -> None:
        pre_existing_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="shared", type="str", is_global=False, value="keep", owning_flow=flow)
        )
        assert isinstance(pre_existing_result, CreateVariableResultSuccess)

        set_variable_node.set_parameter_value("variable_name", "shared")

        # Entering an existing name must not clobber its value.
        assert _get_variable_value("shared", flow) == "keep"
