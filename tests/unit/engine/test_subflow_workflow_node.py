"""Tests for ``SubflowWorkflowNode`` shape handling under Start/End node renames.

Regression coverage for issue #5134 (same root cause as the earlier, closed-in-error #4993):
when a child workflow is imported and its ``Start Flow`` / ``End Flow`` nodes collide with
existing node names, they are renamed (``Start Flow_1`` / ``End Flow_1``). The child's saved
``workflow_shape`` still holds the ORIGINAL names, so routing by those names silently drops the
values.

The fix re-derives the shape from the freshly-imported subflow via the same discovery logic
used at save time (``WorkflowManager.extract_workflow_shape``), so the shape's node-name keys
match the live nodes. These tests drive ``aprocess`` with the surrounding engine calls stubbed
and assert that (a) inputs/outputs are routed to the live (renamed) nodes, and (b) a subflow
without a shape fails with a clear message.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import EndNode, StartNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry, WorkflowShape
from griptape_nodes.retained_mode.events.execution_events import StartLocalSubflowResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.events.workflow_events import ListCallableWorkflowsResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.flow_manager import FlowManager
from griptape_nodes.retained_mode.managers.object_manager import ObjectManager
from griptape_nodes.retained_mode.managers.workflow_manager import WorkflowManager

from griptape_nodes_library.engine.subflow_workflow_node import SubflowWorkflowNode


@pytest.fixture
def shape_param() -> dict[str, Any]:
    """Minimal parameter-shape dict as stored in a ``workflow_shape`` entry."""
    return {"type": "str", "input_types": ["str"], "output_type": "str", "default_value": ""}


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> SubflowWorkflowNode:
    """A ``SubflowWorkflowNode`` primed to run against a stubbed, already-loaded subflow.

    ``__init__`` issues a ``ListCallableWorkflowsRequest`` to populate the workflow dropdown; we
    return "child" as the sole choice so it becomes the selected ``workflow_file`` default.
    """
    list_result = Mock(spec=ListCallableWorkflowsResultSuccess)
    list_result.workflow_names = ["child"]
    monkeypatch.setattr(GriptapeNodes, "handle_request", lambda _request: list_result)
    node = SubflowWorkflowNode(name="Workflow Node")
    # Pretend the child's subflow has already been imported.
    node.metadata["subflow_name"] = "sub"
    return node


@pytest.fixture
def workflow_manager(monkeypatch: pytest.MonkeyPatch, shape_param: dict[str, Any]) -> Mock:
    """Stub the engine calls ``aprocess`` makes up to the routing step and return the
    ``WorkflowManager`` mock so each test can configure ``extract_workflow_shape``.
    """
    monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)

    # The saved shape keeps the ORIGINAL (pre-rename) node names. If the fix regressed to using
    # this instead of re-deriving from the live subflow, the routing assertions below would fail.
    stale_shape = WorkflowShape(
        inputs={"Start Flow": {"text_in": shape_param}},
        outputs={"End Flow": {"text_out": shape_param}},
    )
    monkeypatch.setattr(
        WorkflowRegistry,
        "get_workflow_by_name",
        lambda _name: SimpleNamespace(metadata=SimpleNamespace(workflow_shape=stale_shape)),
    )

    object_manager = Mock(spec=ObjectManager)
    object_manager.get_filtered_subset.return_value = {"sub": object()}  # subflow already loaded
    monkeypatch.setattr(GriptapeNodes, "ObjectManager", lambda: object_manager)

    manager = Mock(spec=WorkflowManager)
    monkeypatch.setattr(GriptapeNodes, "WorkflowManager", lambda: manager)

    async def _ok(_request: Any) -> Mock:
        return Mock(spec=StartLocalSubflowResultSuccess)  # not a failure -> aprocess proceeds

    monkeypatch.setattr(GriptapeNodes, "ahandle_request", _ok)
    return manager


class TestAprocessReDerivesShapeFromSubflow:
    """aprocess must route using the shape extracted from the live subflow, not the stale
    saved shape whose node names may have been renamed on import.
    """

    @pytest.mark.asyncio
    async def test_routes_io_to_renamed_subflow_nodes(
        self,
        node: SubflowWorkflowNode,
        monkeypatch: pytest.MonkeyPatch,
        shape_param: dict[str, Any],
        workflow_manager: Mock,
    ) -> None:
        # The live subflow's Start/End were renamed on import; extract reports the live names.
        workflow_manager.extract_workflow_shape.return_value = {
            "input": {"Start Flow_1": {"text_in": shape_param}},
            "output": {"End Flow_1": {"text_out": shape_param}},
        }

        # The node carries the input value to forward and an output slot to receive.
        node.add_parameter(
            Parameter(
                name="text_in",
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
            )
        )
        node.set_parameter_value("text_in", "hello")
        node.add_parameter(
            Parameter(name="text_out", type="str", allowed_modes={ParameterMode.OUTPUT}, default_value="")
        )

        # The live subflow with the renamed nodes the routing must target.
        end_node = Mock(spec=EndNode)
        end_node.parameter_output_values = {"text_out": "world"}
        flow = SimpleNamespace(nodes={"Start Flow_1": Mock(spec=StartNode), "End Flow_1": end_node})
        flow_manager = Mock(spec=FlowManager)
        flow_manager.get_flow_by_name.return_value = flow
        monkeypatch.setattr(GriptapeNodes, "FlowManager", lambda: flow_manager)

        # Let _set_workflow_inputs run and capture the SetParameterValueRequests it issues.
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        await node.aprocess()

        # Input routed to the renamed start node...
        set_requests = [r for r in requests if isinstance(r, SetParameterValueRequest)]
        assert any(
            r.node_name == "Start Flow_1" and r.parameter_name == "text_in" and r.value == "hello" for r in set_requests
        ), f"input was not routed to the renamed start node; requests={set_requests}"
        # ...and output collected from the renamed end node.
        assert node.parameter_output_values["text_out"] == "world"

    @pytest.mark.asyncio
    async def test_missing_shape_reports_failure(self, node: SubflowWorkflowNode, workflow_manager: Mock) -> None:
        # extract_workflow_shape raises ValueError when the subflow has no Start/End nodes.
        workflow_manager.extract_workflow_shape.side_effect = ValueError("no start/end nodes")

        # The failure output is unconnected on a bare node, so the failure re-raises.
        with pytest.raises(RuntimeError, match="has no shape defined"):
            await node.aprocess()
