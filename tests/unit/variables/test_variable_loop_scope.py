"""Regression tests: Get/Set Variable inside a loop body must share the user's variable.

Covers the bug where a loop body's packaged (transient) flow or a node group's structural
subflow shadowed the real workflow variable with an empty local copy the moment the user
typed a ``variable_name``. ``SetVariable._eager_create_variable`` decided "does this
variable already exist" by checking only the node's immediate flow
(``VariableScope.CURRENT_FLOW_ONLY``), instead of the node's own declared scope
(hierarchical by default) -- so it never saw the outer variable and created a shadow that
local reads/writes could never escape.

These tests exercise a real ``ForLoopGroupNode`` run through ``StartFlowRequest`` so they
reproduce the actual packaging path, not just the manager-level lookup.
"""

from collections.abc import Generator

import pytest
from griptape_nodes.retained_mode.engine import Engine
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, CreateConnectionResultSuccess
from griptape_nodes.retained_mode.events.execution_events import StartFlowRequest, StartFlowResultSuccess
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import GetParameterValueRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.events.variable_events import CreateVariableRequest, CreateVariableResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

FLOW_NAME = "canvas"
VARIABLE_NAME = "fb"


@pytest.fixture
def flow(griptape_nodes: Engine) -> Generator[str, None, None]:  # noqa: ARG001
    """Create a fresh top-level flow (under an ambient test workflow) for each test."""
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_variable_loop_scope_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


def _connect(source: str, source_param: str, target: str, target_param: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source,
            source_parameter_name=source_param,
            target_node_name=target,
            target_parameter_name=target_param,
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess)


def _build_loop(flow: str, *, end: int) -> None:
    """Build: SetVariable(fb=seed) [outside]; For Loop(0..end): GetFb -> SetFb(fb='written')."""
    create_result = GriptapeNodes.handle_request(
        CreateVariableRequest(name=VARIABLE_NAME, type="str", is_global=False, value="seed", owning_flow=flow)
    )
    assert isinstance(create_result, CreateVariableResultSuccess)

    loop_result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="ForLoopGroupNode",
            node_name="Loop",
            override_parent_flow_name=flow,
            metadata={"position": {"x": 0, "y": 0}},
        )
    )
    assert isinstance(loop_result, CreateNodeResultSuccess)
    GriptapeNodes.handle_request(SetParameterValueRequest(node_name="Loop", parameter_name="start", value=0))
    GriptapeNodes.handle_request(SetParameterValueRequest(node_name="Loop", parameter_name="end", value=end))
    GriptapeNodes.handle_request(SetParameterValueRequest(node_name="Loop", parameter_name="end_inclusive", value=True))

    get_result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="GetVariable",
            node_name="GetFb",
            parent_group_name="Loop",
            metadata={"position": {"x": 0, "y": 0}},
        )
    )
    assert isinstance(get_result, CreateNodeResultSuccess)
    GriptapeNodes.handle_request(
        SetParameterValueRequest(node_name="GetFb", parameter_name="variable_name", value=VARIABLE_NAME)
    )

    set_result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="SetVariable",
            node_name="SetFb",
            parent_group_name="Loop",
            metadata={"position": {"x": 50, "y": 0}},
        )
    )
    assert isinstance(set_result, CreateNodeResultSuccess)
    GriptapeNodes.handle_request(
        SetParameterValueRequest(node_name="SetFb", parameter_name="variable_name", value=VARIABLE_NAME)
    )
    GriptapeNodes.handle_request(SetParameterValueRequest(node_name="SetFb", parameter_name="value", value="written"))

    _connect("Loop", "on_each", "GetFb", "exec_in")
    _connect("GetFb", "exec_out", "SetFb", "exec_in")
    _connect("SetFb", "exec_out", "Loop", "loop_complete")


def _get_param_value(node_name: str, parameter_name: str) -> object:
    result = GriptapeNodes.handle_request(GetParameterValueRequest(node_name=node_name, parameter_name=parameter_name))
    return result.value  # type: ignore[attr-defined]


class TestVariableScopeInsideForLoop:
    """Get/Set Variable inside a For Loop body must read and write the workflow's variable."""

    @pytest.mark.asyncio
    async def test_outer_seed_is_visible_to_first_in_loop_get(self, flow: str) -> None:
        """A Set Variable outside the loop seeds a value the first in-loop Get must see."""
        _build_loop(flow, end=0)  # single iteration

        start_result = await GriptapeNodes.ahandle_request(StartFlowRequest(flow_name=flow))
        assert isinstance(start_result, StartFlowResultSuccess)

        assert _get_param_value("GetFb", "value") == "seed"

    @pytest.mark.asyncio
    async def test_in_loop_write_is_visible_to_the_next_iterations_get(self, flow: str) -> None:
        """A write in one iteration must be visible to the Get at the start of the next."""
        _build_loop(flow, end=1)  # two iterations

        start_result = await GriptapeNodes.ahandle_request(StartFlowRequest(flow_name=flow))
        assert isinstance(start_result, StartFlowResultSuccess)

        # Iteration 2's Get must see iteration 1's write, not the original seed.
        assert _get_param_value("GetFb", "value") == "written"

    @pytest.mark.asyncio
    async def test_write_inside_loop_is_visible_after_the_loop(self, flow: str) -> None:
        """A Get Variable placed after the loop must see the loop body's last write."""
        _build_loop(flow, end=0)  # single iteration is enough to prove the write escapes

        after_result = GriptapeNodes.handle_request(
            CreateNodeRequest(
                node_type="GetVariable",
                node_name="GetFbAfter",
                override_parent_flow_name=flow,
                metadata={"position": {"x": 200, "y": 0}},
            )
        )
        assert isinstance(after_result, CreateNodeResultSuccess)
        GriptapeNodes.handle_request(
            SetParameterValueRequest(node_name="GetFbAfter", parameter_name="variable_name", value=VARIABLE_NAME)
        )
        _connect("Loop", "exec_out", "GetFbAfter", "exec_in")

        start_result = await GriptapeNodes.ahandle_request(StartFlowRequest(flow_name=flow))
        assert isinstance(start_result, StartFlowResultSuccess)

        assert _get_param_value("GetFbAfter", "value") == "written"
