"""Build a flow from a spec inside a workflow file's `build_workflow()`."""

from typing import Any

from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

LIB = "Griptape Nodes Library"


async def _req(request: Any) -> Any:
    result = await GriptapeNodes.ahandle_request(request)
    if result.failed():
        msg = f"{type(request).__name__} failed: {request} -> {result.result_details}"
        raise RuntimeError(msg)
    return result


async def build(spec: dict, file_path: str) -> None:
    ctx = GriptapeNodes.ContextManager()
    if not ctx.has_current_workflow():
        ctx.push_workflow(file_path=file_path)
    flow = (
        await _req(
            CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
        )
    ).flow_name
    with ctx.flow(flow):
        for i, (name, (node_type, _values)) in enumerate(spec["nodes"].items()):
            await _req(
                CreateNodeRequest(
                    node_type=node_type,
                    specific_library_name=LIB,
                    node_name=name,
                    metadata={"position": {"x": 400 * (i % 6), "y": 400 * (i // 6)}},
                    initial_setup=True,
                )
            )
        for name, (_node_type, values) in spec["nodes"].items():
            with ctx.node(name):
                for param, value in values.items():
                    await _req(SetParameterValueRequest(parameter_name=param, node_name=name, value=value))
        for src, src_param, dst, dst_param in spec.get("conns", []):
            node = GriptapeNodes.NodeManager().get_node_by_name(dst)
            if isinstance(node.get_parameter_by_name(dst_param), ParameterList):
                with ctx.node(dst):
                    added = await _req(AddParameterToNodeRequest(node_name=dst, parent_container_name=dst_param))
                dst_param = added.parameter_name  # noqa: PLW2901
            await _req(
                CreateConnectionRequest(
                    source_node_name=src,
                    source_parameter_name=src_param,
                    target_node_name=dst,
                    target_parameter_name=dst_param,
                )
            )
        for name, values in spec.get("after", {}).items():
            with ctx.node(name):
                for param, value in values.items():
                    await _req(SetParameterValueRequest(parameter_name=param, node_name=name, value=value))
