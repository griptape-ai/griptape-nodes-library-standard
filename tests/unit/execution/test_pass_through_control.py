"""Control wires routed through Dot and Reroute nodes."""

from __future__ import annotations

import uuid

import pytest
from griptape_nodes.exe_types.core_types import ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import NodeResolutionState
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    DeleteConnectionRequest,
    DeleteConnectionResultSuccess,
)
from griptape_nodes.retained_mode.events.execution_events import StartFlowRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.execution.base_pass_through import BasePassThroughNode

LIBRARY_NAME = "Griptape Nodes Library"
PASS_THROUGH_TYPES = ["DotNode", "Reroute"]


@pytest.fixture()
def flow_name() -> str:
    suffix = uuid.uuid4().hex[:8]
    GriptapeNodes.ContextManager().push_workflow(workflow_name=f"pass_through_{suffix}")
    result = GriptapeNodes.handle_request(
        CreateFlowRequest(parent_flow_name=None, flow_name=f"PassThrough_{suffix}", set_as_new_context=True)
    )
    assert isinstance(result, CreateFlowResultSuccess), result
    return result.flow_name


def _create(node_type: str, flow_name: str) -> str:
    result = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type=node_type,
            specific_library_name=LIBRARY_NAME,
            node_name=f"{node_type}_{uuid.uuid4().hex[:8]}",
            override_parent_flow_name=flow_name,
        )
    )
    assert isinstance(result, CreateNodeResultSuccess), result
    return result.node_name


def _connect(source_node: str, source_param: str, target_node: str, target_param: str) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source_node,
            source_parameter_name=source_param,
            target_node_name=target_node,
            target_parameter_name=target_param,
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess), result


def _disconnect(source_node: str, source_param: str, target_node: str, target_param: str) -> None:
    result = GriptapeNodes.handle_request(
        DeleteConnectionRequest(
            source_node_name=source_node,
            source_parameter_name=source_param,
            target_node_name=target_node,
            target_parameter_name=target_param,
        )
    )
    assert isinstance(result, DeleteConnectionResultSuccess), result


def _node(name: str) -> BasePassThroughNode:
    node = GriptapeNodes.NodeManager().get_node_by_name(name)
    assert isinstance(node, BasePassThroughNode)
    return node


def _state(name: str) -> NodeResolutionState:
    return GriptapeNodes.NodeManager().get_node_by_name(name).state


def _route_then_through(flow_name: str, pass_through_types: list[str], *, outgoing_first: bool) -> tuple[str, str, str]:
    """Build IfElse whose Then branch runs through the given pass-through nodes and whose Else branch is direct."""
    if_else = _create("IfElse", flow_name)
    then_target = _create("ToText", flow_name)
    else_target = _create("ToText", flow_name)
    hops = [(if_else, "Then")]
    for node_type in pass_through_types:
        name = _create(node_type, flow_name)
        hops.append((name, _node(name).get_pass_thru_parameter().name))
    hops.append((then_target, "exec_in"))

    wires = [(*hops[i], *hops[i + 1]) for i in range(len(hops) - 1)]
    for wire in reversed(wires) if outgoing_first else wires:
        _connect(*wire)
    _connect(if_else, "Else", else_target, "exec_in")
    return if_else, then_target, else_target


@pytest.mark.asyncio
@pytest.mark.parametrize("outgoing_first", [False, True])
@pytest.mark.parametrize("evaluate", [True, False])
@pytest.mark.parametrize(
    "pass_through_types",
    [["DotNode"], ["Reroute"], ["DotNode", "Reroute"]],
    ids=["dot", "reroute", "dot_then_reroute"],
)
async def test_if_else_branch_routes_through_pass_through(
    flow_name: str,
    pass_through_types: list[str],
    evaluate: bool,  # noqa: FBT001
    outgoing_first: bool,  # noqa: FBT001
) -> None:
    if_else, then_target, else_target = _route_then_through(
        flow_name, pass_through_types, outgoing_first=outgoing_first
    )
    GriptapeNodes.NodeManager().get_node_by_name(if_else).set_parameter_value("evaluate", evaluate)

    await GriptapeNodes.ahandle_request(StartFlowRequest(flow_name=flow_name))

    taken, skipped = (then_target, else_target) if evaluate else (else_target, then_target)
    assert _state(taken) == NodeResolutionState.RESOLVED
    assert _state(skipped) == NodeResolutionState.UNRESOLVED


@pytest.mark.parametrize("node_type", PASS_THROUGH_TYPES)
def test_control_wire_retypes_pass_through_to_control(flow_name: str, node_type: str) -> None:
    if_else = _create("IfElse", flow_name)
    name = _create(node_type, flow_name)
    node = _node(name)
    param = node.get_pass_thru_parameter()

    _connect(if_else, "Then", name, param.name)

    assert param.type == ParameterTypeBuiltin.CONTROL_TYPE.value
    assert param.output_type == ParameterTypeBuiltin.CONTROL_TYPE.value
    assert node.get_next_control_output() is param

    _disconnect(if_else, "Then", name, param.name)

    assert param.type == ParameterTypeBuiltin.ANY.value
    assert param.output_type == ParameterTypeBuiltin.ALL.value
    assert node.get_next_control_output() is not param


@pytest.mark.parametrize("node_type", PASS_THROUGH_TYPES)
def test_data_wire_keeps_pass_through_out_of_control_flow(flow_name: str, node_type: str) -> None:
    source = _create("ToText", flow_name)
    name = _create(node_type, flow_name)
    node = _node(name)
    param = node.get_pass_thru_parameter()

    _connect(source, "output", name, param.name)

    assert param.type == "str"
    assert node.get_next_control_output() is not param


@pytest.mark.parametrize("node_type", PASS_THROUGH_TYPES)
def test_control_wire_rejected_into_fan_out(flow_name: str, node_type: str) -> None:
    if_else = _create("IfElse", flow_name)
    first = _create(node_type, flow_name)
    second = _create(node_type, flow_name)
    first_param = _node(first).get_pass_thru_parameter().name
    second_param = _node(second).get_pass_thru_parameter().name
    _connect(first, first_param, second, second_param)
    for _ in range(2):
        _connect(second, second_param, _create("ToText", flow_name), "from")

    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=if_else,
            source_parameter_name="Then",
            target_node_name=first,
            target_parameter_name=first_param,
        )
    )

    assert not isinstance(result, CreateConnectionResultSuccess)
    assert _node(first).get_pass_thru_parameter().type != ParameterTypeBuiltin.CONTROL_TYPE.value


@pytest.mark.parametrize("node_type", PASS_THROUGH_TYPES)
def test_removing_one_merged_control_wire_keeps_the_other(flow_name: str, node_type: str) -> None:
    first_if_else = _create("IfElse", flow_name)
    second_if_else = _create("IfElse", flow_name)
    target = _create("ToText", flow_name)
    name = _create(node_type, flow_name)
    node = _node(name)
    param = node.get_pass_thru_parameter()
    _connect(name, param.name, target, "exec_in")
    _connect(first_if_else, "Then", name, param.name)
    _connect(second_if_else, "Then", name, param.name)

    _disconnect(second_if_else, "Then", name, param.name)
    _disconnect(name, param.name, target, "exec_in")

    assert node.incoming_source_parameter is not None
    source_node = node.incoming_source_parameter.get_node()
    assert source_node is not None
    assert source_node.name == first_if_else
    assert node.get_next_control_output() is param


@pytest.mark.parametrize("node_type", PASS_THROUGH_TYPES)
def test_control_wires_merge_before_an_outgoing_wire_exists(flow_name: str, node_type: str) -> None:
    """Saved workflows replay incoming wires before outgoing ones, so a merge must hold without an outgoing wire."""
    first_if_else = _create("IfElse", flow_name)
    second_if_else = _create("IfElse", flow_name)
    name = _create(node_type, flow_name)
    param = _node(name).get_pass_thru_parameter()

    _connect(first_if_else, "Then", name, param.name)
    _connect(second_if_else, "Then", name, param.name)

    connections = GriptapeNodes.FlowManager().get_connections()
    incoming_ids = connections.incoming_index.get(name, {}).get(param.name, [])
    sources = {connections.connections[connection_id].source_node.name for connection_id in incoming_ids}
    assert sources == {first_if_else, second_if_else}
