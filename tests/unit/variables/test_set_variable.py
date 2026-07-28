"""Tests for SetVariable node.

Covers the create-if-missing behavior (issue #4411): SetVariable no longer requires a
pre-existing variable. It eagerly registers the variable with the engine when the user
sets ``variable_name`` and falls back to create-or-update during ``process()``.
"""

from collections.abc import Generator
from typing import cast

import pytest
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

from griptape_nodes_library.variables.set_variable import CREATE_NEW_SENTINEL, SetVariable

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
def set_variable_node(flow: str) -> SetVariable:
    """Create a SetVariable node inside the test flow and return the instance."""
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="SetVariable", override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess)
    node = GriptapeNodes.NodeManager().get_node_by_name(result.node_name)
    assert type(node).__name__ == "SetVariable"
    return node  # type: ignore[return-value]


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


def _add_to_dropdown(node: SetVariable, name: str, flow_name: str) -> None:
    """Ensure *name* exists as an engine variable and appears in the dropdown.

    Creates the variable via ``CreateVariableRequest`` (a no-op when it already exists)
    then invokes the node's own ``_refresh_variable_names`` — the same code path the
    real UI refresh button traverses.
    """
    if not _has_variable(name, flow_name):
        result = GriptapeNodes.handle_request(
            CreateVariableRequest(name=name, type="str", is_global=False, value="", owning_flow=flow_name)
        )
        assert isinstance(result, CreateVariableResultSuccess)
    node._refresh_variable_names(button=None, button_details=None)  # type: ignore[arg-type]


class TestSetVariableProcess:
    """Exercises the rewritten ``aprocess()``: create-if-missing, else update."""

    @pytest.mark.asyncio
    async def test_creates_variable_when_missing(self, set_variable_node: SetVariable, flow: str) -> None:
        set_variable_node.set_parameter_value("variable_name", CREATE_NEW_SENTINEL)
        set_variable_node.set_parameter_value("new_variable_name", "my_var")
        set_variable_node.set_parameter_value("value", "hello")

        await set_variable_node.aprocess()

        assert _has_variable("my_var", flow)
        assert _get_variable_value("my_var", flow) == "hello"

    @pytest.mark.asyncio
    async def test_updates_existing_variable(self, set_variable_node: SetVariable, flow: str) -> None:
        create_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="existing", type="str", is_global=False, value="old", owning_flow=flow)
        )
        assert isinstance(create_result, CreateVariableResultSuccess)

        _add_to_dropdown(set_variable_node, "existing", flow)
        set_variable_node.set_parameter_value("variable_name", "existing")
        set_variable_node.set_parameter_value("value", "new")

        await set_variable_node.aprocess()

        assert _get_variable_value("existing", flow) == "new"

    @pytest.mark.asyncio
    async def test_empty_variable_name_raises(self, set_variable_node: SetVariable) -> None:
        set_variable_node.set_parameter_value("variable_name", CREATE_NEW_SENTINEL)
        set_variable_node.set_parameter_value("new_variable_name", "")
        set_variable_node.set_parameter_value("value", "anything")

        with pytest.raises(ValueError, match="requires a non-empty variable_name"):
            await set_variable_node.aprocess()


class TestSetVariableEagerRegistration:
    """Exercises ``before_value_set``: variables register the moment the user selects a name."""

    def test_entering_name_registers_variable(self, set_variable_node: SetVariable, flow: str) -> None:
        assert not _has_variable("eager", flow)

        _add_to_dropdown(set_variable_node, "eager", flow)
        set_variable_node.set_parameter_value("variable_name", "eager")

        assert _has_variable("eager", flow)

    def test_changing_name_registers_new_and_leaves_old_alone(self, set_variable_node: SetVariable, flow: str) -> None:
        """Editing ``variable_name`` always means 'point this node at a different variable'.

        We intentionally do not rename the prior variable — there is no UI signal to
        distinguish 'rename' from 'switch', and renaming would orphan any other node that
        was pointing at the old name. Leftover variables get filtered out at save time
        by ``NodeDependencies``-driven serialization.
        """
        _add_to_dropdown(set_variable_node, "first", flow)
        set_variable_node.set_parameter_value("variable_name", "first")
        assert _has_variable("first", flow)

        _add_to_dropdown(set_variable_node, "second", flow)
        set_variable_node.set_parameter_value("variable_name", "second")

        assert _has_variable("first", flow)
        assert _has_variable("second", flow)

    def test_clearing_name_leaves_variable_alone(self, set_variable_node: SetVariable, flow: str) -> None:
        _add_to_dropdown(set_variable_node, "sticky", flow)
        set_variable_node.set_parameter_value("variable_name", "sticky")
        assert _has_variable("sticky", flow)

        set_variable_node.set_parameter_value("variable_name", CREATE_NEW_SENTINEL)

        # Switching to the sentinel is a no-op for existing variables.
        assert _has_variable("sticky", flow)

    def test_repointing_to_existing_name_does_not_clobber(self, set_variable_node: SetVariable, flow: str) -> None:
        collision_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="taken", type="str", is_global=False, value="reserved", owning_flow=flow)
        )
        assert isinstance(collision_result, CreateVariableResultSuccess)

        _add_to_dropdown(set_variable_node, "mine", flow)
        set_variable_node.set_parameter_value("variable_name", "mine")
        assert _has_variable("mine", flow)

        # Pointing this node at an existing name must not raise, and must not clobber
        # the existing variable's value.
        _add_to_dropdown(set_variable_node, "taken", flow)
        set_variable_node.set_parameter_value("variable_name", "taken")

        assert _has_variable("taken", flow)
        assert _get_variable_value("taken", flow) == "reserved"

    def test_duplicate_name_adopts_existing(self, set_variable_node: SetVariable, flow: str) -> None:
        pre_existing_result = GriptapeNodes.handle_request(
            CreateVariableRequest(name="shared", type="str", is_global=False, value="keep", owning_flow=flow)
        )
        assert isinstance(pre_existing_result, CreateVariableResultSuccess)

        _add_to_dropdown(set_variable_node, "shared", flow)
        set_variable_node.set_parameter_value("variable_name", "shared")

        # Entering an existing name must not clobber its value.
        assert _get_variable_value("shared", flow) == "keep"

    def test_repointing_one_node_does_not_orphan_other_aliases(self, flow: str) -> None:
        """Repointing one node's ``variable_name`` must never mutate another node's variable.

        Scenario: node A owns ``X``, node B owns ``Y``, node C also points at ``X`` (adopted
        silently). Re-point C from ``X`` to a brand-new name ``Z``. Node A must still see ``X``.
        This is the aliasing bug that motivated dropping the eager-rename path.
        """
        create_a = GriptapeNodes.handle_request(
            CreateNodeRequest(node_type="SetVariable", override_parent_flow_name=flow)
        )
        create_b = GriptapeNodes.handle_request(
            CreateNodeRequest(node_type="SetVariable", override_parent_flow_name=flow)
        )
        create_c = GriptapeNodes.handle_request(
            CreateNodeRequest(node_type="SetVariable", override_parent_flow_name=flow)
        )
        assert isinstance(create_a, CreateNodeResultSuccess)
        assert isinstance(create_b, CreateNodeResultSuccess)
        assert isinstance(create_c, CreateNodeResultSuccess)

        node_a = cast(SetVariable, GriptapeNodes.NodeManager().get_node_by_name(create_a.node_name))
        node_b = cast(SetVariable, GriptapeNodes.NodeManager().get_node_by_name(create_b.node_name))
        node_c = cast(SetVariable, GriptapeNodes.NodeManager().get_node_by_name(create_c.node_name))

        _add_to_dropdown(node_a, "X", flow)
        node_a.set_parameter_value("variable_name", "X")
        _add_to_dropdown(node_b, "Y", flow)
        node_b.set_parameter_value("variable_name", "Y")
        _add_to_dropdown(node_c, "X", flow)
        node_c.set_parameter_value("variable_name", "X")  # silently adopts existing X

        assert _has_variable("X", flow)
        assert _has_variable("Y", flow)

        # Re-point C to a brand-new name. X must remain so that A still resolves.
        _add_to_dropdown(node_c, "Z", flow)
        node_c.set_parameter_value("variable_name", "Z")

        assert _has_variable("X", flow), "Node A's variable was orphaned by Node C's edit"
        assert _has_variable("Y", flow)
        assert _has_variable("Z", flow)
