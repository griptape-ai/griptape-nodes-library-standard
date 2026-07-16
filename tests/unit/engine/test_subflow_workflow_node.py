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

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import EndNode, StartNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry, WorkflowShape
from griptape_nodes.retained_mode.events.execution_events import StartLocalSubflowResultSuccess
from griptape_nodes.retained_mode.events.flow_events import DeleteFlowRequest, SetFlowMetadataRequest
from griptape_nodes.retained_mode.events.node_events import GetFlowForNodeRequest, GetFlowForNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.events.workflow_events import (
    ImportWorkflowAsReferencedSubFlowRequest,
    ImportWorkflowAsReferencedSubFlowResultSuccess,
    ListCallableWorkflowsResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.flow_manager import FlowManager
from griptape_nodes.retained_mode.managers.object_manager import ObjectManager
from griptape_nodes.retained_mode.managers.workflow_manager import WorkflowManager

from griptape_nodes_library.engine.subflow_workflow_node import (
    SUBFLOW_NAME_KEY,
    SUBFLOW_OWNER_KEY,
    SUBFLOW_WORKFLOW_KEY,
    SubflowWorkflowNode,
)


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
    # Pretend the child's subflow has already been imported and this node owns it.
    node.metadata[SUBFLOW_NAME_KEY] = "sub"
    node.metadata[SUBFLOW_OWNER_KEY] = "owner-sub"
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
    # The loaded subflow carries the node's owner UUID so the quick-path resolver binds to it
    # without any legacy-claim side effects.
    loaded_flows = {"sub": SimpleNamespace(metadata={SUBFLOW_OWNER_KEY: "owner-sub"})}
    object_manager.get_filtered_subset.return_value = loaded_flows
    object_manager.attempt_get_object_by_name_as_type.side_effect = lambda name, _type: loaded_flows.get(name)
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


def _flow(owner_uuid: str | None = None) -> SimpleNamespace:
    """A stand-in ``ControlFlow`` exposing only the ``.metadata`` dict the resolver reads."""
    metadata: dict[str, Any] = {}
    if owner_uuid is not None:
        metadata[SUBFLOW_OWNER_KEY] = owner_uuid
    return SimpleNamespace(metadata=metadata)


def _stub_flows(monkeypatch: pytest.MonkeyPatch, flows: dict[str, Any]) -> None:
    """Back both the O(1) name lookup and the full scan with a fixed name -> flow mapping."""
    object_manager = Mock(spec=ObjectManager)
    object_manager.get_filtered_subset.return_value = flows
    object_manager.attempt_get_object_by_name_as_type.side_effect = lambda name, _type: flows.get(name)
    monkeypatch.setattr(GriptapeNodes, "ObjectManager", lambda: object_manager)


def _stub_referenced_workflow(monkeypatch: pytest.MonkeyPatch, workflow_name: str | None) -> None:
    """Make ``FlowManager.get_referenced_workflow_name`` report the workflow any flow was imported from."""
    flow_manager = Mock(spec=FlowManager)
    flow_manager.get_referenced_workflow_name.return_value = workflow_name
    monkeypatch.setattr(GriptapeNodes, "FlowManager", lambda: flow_manager)


class TestResolveOwnedSubflow:
    """``_resolve_subflow_name`` binds a node to the subflow it owns via a stable owner UUID.

    Regression coverage for issue #5116: two Workflow nodes referencing the same child workflow
    (or a nested Workflow node whose ``subflow_name`` was de-duplicated / shifted on import) must
    each bind to their OWN imported subflow rather than colliding on a shared/stale ``subflow_name``.
    """

    def test_quick_path_returns_recorded_name_when_uuid_matches(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("U1")})

        assert node._resolve_subflow_name("child") == "ControlFlow_2"

    def test_slow_path_finds_flow_by_uuid_when_recorded_name_is_stale(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # ``subflow_name`` points at a flow owned by a DIFFERENT node (names shifted on nested
        # import); the resolver must scan by UUID and return the flow this node actually owns.
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("OTHER"), "ControlFlow_3": _flow("U1")})

        assert node._resolve_subflow_name("child") == "ControlFlow_3"

    def test_two_nodes_same_workflow_bind_to_distinct_flows(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Same recorded name, different owner UUIDs -> each node resolves to its own flow.
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("U1"), "ControlFlow_3": _flow("U2")})
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"

        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        assert node._resolve_subflow_name("child") == "ControlFlow_2"

        node.metadata[SUBFLOW_OWNER_KEY] = "U2"
        assert node._resolve_subflow_name("child") == "ControlFlow_3"

    def test_returns_none_when_no_owned_flow_exists(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("OTHER")})

        assert node._resolve_subflow_name("child") is None

    def test_legacy_flow_without_uuid_is_claimed_and_warns(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Pre-UUID workflow: the node carries an owner UUID (established in __init__) but the recorded
        # flow, recreated from pre-UUID saved data, has none. The recorded flow genuinely references
        # the workflow this node expects, so the node claims it by stamping its own UUID, and warns.
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata[SUBFLOW_WORKFLOW_KEY] = "child"
        owner_uuid = node.metadata[SUBFLOW_OWNER_KEY]
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow(None)})
        _stub_referenced_workflow(monkeypatch, "child")  # recorded flow references the expected workflow

        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        with caplog.at_level(logging.WARNING, logger="griptape_nodes"):
            result = node._resolve_subflow_name("child")

        assert result == "ControlFlow_2"
        # The node's owner UUID was stamped onto the claimed flow.
        stamp_requests = [r for r in requests if isinstance(r, SetFlowMetadataRequest)]
        assert any(
            r.flow_name == "ControlFlow_2" and r.metadata.get(SUBFLOW_OWNER_KEY) == owner_uuid for r in stamp_requests
        ), f"legacy flow was not stamped with the node's owner UUID; requests={stamp_requests}"
        # ...and a loud warning was emitted.
        assert any(rec.levelno == logging.WARNING for rec in caplog.records)

    def test_legacy_flow_referencing_other_workflow_is_not_adopted(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The recorded name resolves to an unclaimed flow, but that flow was imported from a DIFFERENT
        # referenced workflow (e.g. a nested node whose stale name now points at a sibling/parent
        # subflow). Without a UUID to disambiguate, the workflow-reference check must reject it so we
        # never adopt the wrong subflow — regardless of execution order.
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata[SUBFLOW_WORKFLOW_KEY] = "child"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow(None)})
        _stub_referenced_workflow(monkeypatch, "some_other_workflow")

        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        assert node._resolve_subflow_name("child") is None
        # Nothing was claimed/stamped.
        assert not [r for r in requests if isinstance(r, SetFlowMetadataRequest)]

    def test_recorded_flow_without_expected_workflow_is_not_adopted(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Defensive: SUBFLOW_NAME_KEY and SUBFLOW_WORKFLOW_KEY are always written together, so a
        # recorded name with no expected workflow is inconsistent state -> never adopt it.
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata.pop(SUBFLOW_WORKFLOW_KEY, None)
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow(None)})

        assert node._resolve_subflow_name("child") is None


class TestReloadStampsOwnerUuid:
    """``_reload_subflow`` mirrors the node's (already-established) owner UUID onto the imported flow."""

    def test_import_stamps_owner_uuid_on_created_flow(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        # No subflow owned yet: the resolver called at the top of _reload_subflow finds nothing. The
        # node's owner UUID was established in __init__ and must be reused (not regenerated).
        node.metadata.pop(SUBFLOW_NAME_KEY, None)
        owner_uuid = node.metadata[SUBFLOW_OWNER_KEY]
        _stub_flows(monkeypatch, {})

        requests: list[Any] = []

        def _handle(request: Any) -> Any:
            requests.append(request)
            if isinstance(request, GetFlowForNodeRequest):
                return GetFlowForNodeResultSuccess(flow_name="ParentFlow", result_details="stubbed")
            if isinstance(request, ImportWorkflowAsReferencedSubFlowRequest):
                return ImportWorkflowAsReferencedSubFlowResultSuccess(
                    created_flow_name="ControlFlow_2", result_details="stubbed"
                )
            return Mock()

        monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)

        node._reload_subflow("child")

        # The owner UUID is unchanged (reused from __init__), and the created flow was stamped with it.
        assert node.metadata[SUBFLOW_OWNER_KEY] == owner_uuid
        assert node.metadata.get(SUBFLOW_NAME_KEY) == "ControlFlow_2"
        stamp_requests = [r for r in requests if isinstance(r, SetFlowMetadataRequest)]
        assert any(
            r.flow_name == "ControlFlow_2" and r.metadata.get(SUBFLOW_OWNER_KEY) == owner_uuid for r in stamp_requests
        ), f"created flow was not stamped with the node's owner UUID; requests={stamp_requests}"


class TestInitEstablishesOwnerUuid:
    """``__init__`` gives every node a stable owner UUID up front (generated once, preserved on load)."""

    @staticmethod
    def _make_node(monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any] | None = None) -> SubflowWorkflowNode:
        list_result = Mock(spec=ListCallableWorkflowsResultSuccess)
        list_result.workflow_names = ["child"]
        monkeypatch.setattr(GriptapeNodes, "handle_request", lambda _request: list_result)
        return SubflowWorkflowNode(name="Workflow Node", metadata=metadata)

    def test_generates_owner_uuid_for_new_node(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._make_node(monkeypatch)
        assert node.metadata.get(SUBFLOW_OWNER_KEY)

    def test_preserves_owner_uuid_from_saved_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = self._make_node(monkeypatch, {SUBFLOW_OWNER_KEY: "baked-uuid"})
        assert node.metadata[SUBFLOW_OWNER_KEY] == "baked-uuid"


class TestAfterNodeDeleted:
    """Deleting the node cleans up the flow it OWNS (by UUID), never a stale-named flow."""

    def test_deletes_owned_flow_not_stale_recorded_name(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # subflow_name is stale (points at another node's flow); the owned flow is elsewhere by UUID.
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("OTHER"), "ControlFlow_5": _flow("U1")})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node.after_node_deleted()

        deleted = [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)]
        assert deleted == ["ControlFlow_5"]

    def test_does_not_delete_flow_owned_by_another_node(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"  # stale: owned by U2, and no U1 flow exists
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("U2")})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node.after_node_deleted()

        assert not [r for r in requests if isinstance(r, DeleteFlowRequest)]

    def test_does_not_delete_unowned_recorded_flow(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A recorded flow with no owner UUID is NOT deleted on node deletion: without a UUID match we
        # can't prove it's ours, and a None owned-lookup can mean "our subflow was legitimately
        # deleted" (its name possibly reused) rather than "we're a legacy node". Deleting by name
        # would risk destroying an unrelated flow, so we leave it (parent-flow teardown cleans it up).
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata[SUBFLOW_WORKFLOW_KEY] = "child"
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow(None)})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node.after_node_deleted()

        assert not [r for r in requests if isinstance(r, DeleteFlowRequest)]

    def test_no_delete_when_owned_flow_already_gone(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"  # already deleted (parent-flow teardown)
        _stub_flows(monkeypatch, {})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node.after_node_deleted()

        assert not [r for r in requests if isinstance(r, DeleteFlowRequest)]


class TestOnRefreshWorkflow:
    """The Refresh button re-imports the child workflow even when the selection hasn't changed."""

    def test_forces_reload_of_already_loaded_workflow(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange the exact state in which _reload_subflow would normally SKIP: the node already owns a
        # live subflow for the selected workflow (SUBFLOW_WORKFLOW_KEY == workflow_file, which the
        # fixture defaults to "child"). Refresh must still tear it down and re-import, proving it
        # forced the reload.
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata[SUBFLOW_OWNER_KEY] = "owner-sub"
        node.metadata[SUBFLOW_WORKFLOW_KEY] = "child"
        # Shape-parameter maintenance is orthogonal to the reload; isolate it.
        monkeypatch.setattr(node, "_update_workflow_shape_parameters", lambda *args, **kwargs: None)
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow("owner-sub")})

        requests: list[Any] = []

        def _handle(request: Any) -> Any:
            requests.append(request)
            if isinstance(request, GetFlowForNodeRequest):
                return GetFlowForNodeResultSuccess(flow_name="ParentFlow", result_details="stubbed")
            if isinstance(request, ImportWorkflowAsReferencedSubFlowRequest):
                return ImportWorkflowAsReferencedSubFlowResultSuccess(
                    created_flow_name="ControlFlow_9", result_details="stubbed"
                )
            return Mock()

        monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)

        node._on_refresh_workflow(Mock(), Mock())

        # The previously-owned subflow was deleted and a fresh one imported (neither would happen if
        # the reload had been skipped for the unchanged workflow).
        assert "ControlFlow_2" in [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)]
        assert any(isinstance(r, ImportWorkflowAsReferencedSubFlowRequest) for r in requests)
        assert node.metadata[SUBFLOW_NAME_KEY] == "ControlFlow_9"

    def test_forces_reload_of_unclaimed_legacy_workflow_deletes_old_flow(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Legacy node: the recorded flow exists but was never UUID-stamped (pre-UUID save, never
        # executed), so it resolves only via the legacy path, which needs SUBFLOW_WORKFLOW_KEY.
        # Refresh must adopt-and-delete the old flow, then re-import — not orphan it. Regression for
        # the bug where refresh popped SUBFLOW_WORKFLOW_KEY, blinding the legacy resolver so the old
        # flow leaked (a fresh flow imported while the original was left behind).
        node.metadata[SUBFLOW_NAME_KEY] = "ControlFlow_2"
        node.metadata[SUBFLOW_OWNER_KEY] = "U1"  # the node has a UUID, but its flow does NOT (legacy)
        node.metadata[SUBFLOW_WORKFLOW_KEY] = "child"
        monkeypatch.setattr(node, "_update_workflow_shape_parameters", lambda *args, **kwargs: None)
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        _stub_flows(monkeypatch, {"ControlFlow_2": _flow(None)})  # unclaimed legacy flow
        _stub_referenced_workflow(monkeypatch, "child")  # it does reference the expected workflow

        requests: list[Any] = []

        def _handle(request: Any) -> Any:
            requests.append(request)
            if isinstance(request, GetFlowForNodeRequest):
                return GetFlowForNodeResultSuccess(flow_name="ParentFlow", result_details="stubbed")
            if isinstance(request, ImportWorkflowAsReferencedSubFlowRequest):
                return ImportWorkflowAsReferencedSubFlowResultSuccess(
                    created_flow_name="ControlFlow_9", result_details="stubbed"
                )
            return Mock()

        monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)

        node._on_refresh_workflow(Mock(), Mock())

        # The old legacy flow was torn down (not orphaned) and a fresh one imported.
        assert "ControlFlow_2" in [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)]
        assert any(isinstance(r, ImportWorkflowAsReferencedSubFlowRequest) for r in requests)
        assert node.metadata[SUBFLOW_NAME_KEY] == "ControlFlow_9"

    def test_noop_when_no_workflow_selected(self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch) -> None:
        # No workflow selected -> refresh is a no-op (stub the getter to avoid after_value_set churn).
        monkeypatch.setattr(node, "get_parameter_value", lambda name: "" if name == "workflow_file" else None)
        reload_calls: list[str] = []
        monkeypatch.setattr(node, "_reload_subflow", lambda workflow_name: reload_calls.append(workflow_name))

        node._on_refresh_workflow(Mock(), Mock())

        assert reload_calls == []
