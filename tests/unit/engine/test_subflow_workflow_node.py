"""Tests for ``SubflowWorkflowNode``: transient-subflow lifecycle + shape handling under renames.

The node imports its selected workflow as a **transient** child subflow — a runtime-only artifact
that is never serialized (the engine skips flows flagged ``transient``). The live subflow name is
tracked as a runtime ``subflow_name`` metadata hint that is popped on load (so it never round-trips
to go stale, collide across copy-paste, or mis-bind under nesting) and re-established when the subflow
is (re)imported at execution time; it is reused while it stays live. The editor reads that hint (via a
fresh ``GetNodeMetadataRequest`` when the preview opens) to show the in-memory subflow.

Also retains regression coverage for issue #5134: when a child workflow is imported and its
``Start Flow`` / ``End Flow`` nodes collide with existing names, they are renamed
(``Start Flow_1`` / ``End Flow_1``). The child's saved ``workflow_shape`` still holds the ORIGINAL
names, so ``aprocess`` re-derives the shape from the freshly-imported subflow via
``WorkflowManager.extract_workflow_shape`` instead of trusting the saved shape.
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
from griptape_nodes.retained_mode.events.flow_events import (
    DeleteFlowRequest,
    GetFlowDetailsRequest,
    GetFlowDetailsResultSuccess,
    GetFlowMetadataRequest,
    GetFlowMetadataResultFailure,
    GetFlowMetadataResultSuccess,
    SetFlowMetadataRequest,
    SetFlowMetadataResultSuccess,
)
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

from griptape_nodes_library.engine.subflow_workflow_node import SubflowWorkflowNode


@pytest.fixture
def shape_param() -> dict[str, Any]:
    """Minimal parameter-shape dict as stored in a ``workflow_shape`` entry."""
    return {"type": "str", "input_types": ["str"], "output_type": "str", "default_value": ""}


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> SubflowWorkflowNode:
    """A ``SubflowWorkflowNode`` with the workflow dropdown stubbed.

    ``__init__`` issues a ``ListCallableWorkflowsRequest`` to populate the dropdown; we return
    "child" as the sole choice so it becomes the selected ``workflow_file`` default.
    """
    list_result = Mock(spec=ListCallableWorkflowsResultSuccess)
    list_result.workflow_names = ["child"]
    monkeypatch.setattr(GriptapeNodes, "handle_request", lambda _request: list_result)
    return SubflowWorkflowNode(name="Workflow Node")


def _live_flow(name: str) -> SimpleNamespace:
    """A stand-in ``ControlFlow`` exposing only the ``.metadata`` dict."""
    return SimpleNamespace(name=name, metadata={})


def _stub_object_manager(monkeypatch: pytest.MonkeyPatch, live_flows: dict[str, Any]) -> None:
    """Back ``ObjectManager.attempt_get_object_by_name_as_type`` with a fixed name -> flow map."""
    object_manager = Mock(spec=ObjectManager)
    object_manager.attempt_get_object_by_name_as_type.side_effect = lambda name, _type: live_flows.get(name)
    monkeypatch.setattr(GriptapeNodes, "ObjectManager", lambda: object_manager)


def _install_import_handler(
    monkeypatch: pytest.MonkeyPatch, *, parent_flow: str = "ParentFlow", created_flow: str = "sub"
) -> list[Any]:
    """Install a ``handle_request`` that answers the import path and captures every request."""
    requests: list[Any] = []

    def _handle(request: Any) -> Any:
        requests.append(request)
        if isinstance(request, GetFlowForNodeRequest):
            return GetFlowForNodeResultSuccess(flow_name=parent_flow, result_details="stubbed")
        if isinstance(request, ImportWorkflowAsReferencedSubFlowRequest):
            return ImportWorkflowAsReferencedSubFlowResultSuccess(
                created_flow_name=created_flow, result_details="stubbed"
            )
        return None

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)
    return requests


@pytest.fixture
def workflow_manager(monkeypatch: pytest.MonkeyPatch, shape_param: dict[str, Any]) -> Mock:
    """Stub the engine calls ``aprocess`` makes up to the routing step, with the subflow already live.

    Returns the ``WorkflowManager`` mock so each test can configure ``extract_workflow_shape``.
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

    # The subflow "sub" is already imported and live -> _load_subflow reuses it (no import).
    _stub_object_manager(monkeypatch, {"sub": _live_flow("sub")})

    manager = Mock(spec=WorkflowManager)
    monkeypatch.setattr(GriptapeNodes, "WorkflowManager", lambda: manager)

    async def _ok(_request: Any) -> Mock:
        return Mock(spec=StartLocalSubflowResultSuccess)  # not a failure -> aprocess proceeds

    monkeypatch.setattr(GriptapeNodes, "ahandle_request", _ok)
    return manager


class TestLoadSubflowImportsTransient:
    """``_load_subflow`` imports the selected workflow as a transient, execution-time subflow."""

    def test_imports_and_flags_created_flow_transient(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        _stub_object_manager(monkeypatch, {})  # nothing live yet
        requests = _install_import_handler(monkeypatch, created_flow="sub")

        result = node._load_subflow("child")

        assert result == "sub"
        # The live subflow name is tracked in metadata (a runtime hint the editor reads to
        # preview the in-memory subflow); it is popped on load so it never goes stale.
        assert node.metadata.get("subflow_name") == "sub"
        # The import flagged the created flow transient so it is never serialized.
        import_requests = [r for r in requests if isinstance(r, ImportWorkflowAsReferencedSubFlowRequest)]
        assert len(import_requests) == 1
        assert import_requests[0].imported_flow_metadata == {"transient": True}
        # No legacy association marker is written by a new-scheme node.
        assert "_subflow_workflow" not in node.metadata

    def test_reuses_live_subflow_without_reimporting(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        node.metadata["subflow_name"] = "sub"
        _stub_object_manager(monkeypatch, {"sub": _live_flow("sub")})
        requests = _install_import_handler(monkeypatch)

        result = node._load_subflow("child")

        assert result == "sub"
        # Already live -> no new import.
        assert not [r for r in requests if isinstance(r, ImportWorkflowAsReferencedSubFlowRequest)]

    def test_reimports_when_tracked_subflow_no_longer_exists(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Tracked name points at a flow that was torn down (e.g. parent flow deleted) -> re-import.
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        node.metadata["subflow_name"] = "gone"
        _stub_object_manager(monkeypatch, {})  # "gone" is not live
        requests = _install_import_handler(monkeypatch, created_flow="sub_2")

        result = node._load_subflow("child")

        assert result == "sub_2"
        assert node.metadata.get("subflow_name") == "sub_2"
        assert len([r for r in requests if isinstance(r, ImportWorkflowAsReferencedSubFlowRequest)]) == 1

    def test_returns_none_when_workflow_not_registered(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: False)
        _stub_object_manager(monkeypatch, {})
        requests = _install_import_handler(monkeypatch)

        assert node._load_subflow("child") is None
        assert not [r for r in requests if isinstance(r, ImportWorkflowAsReferencedSubFlowRequest)]


class TestDiscardSubflow:
    """The node tears down the live subflow it imported (on delete / workflow change / refresh)."""

    def test_discard_deletes_live_subflow(self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch) -> None:
        node.metadata["subflow_name"] = "sub"
        _stub_object_manager(monkeypatch, {"sub": _live_flow("sub")})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node._discard_subflow()

        assert [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)] == ["sub"]
        assert "subflow_name" not in node.metadata

    def test_discard_is_noop_when_nothing_tracked(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _stub_object_manager(monkeypatch, {})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node._discard_subflow()

        assert not [r for r in requests if isinstance(r, DeleteFlowRequest)]

    def test_discard_clears_tracking_when_flow_already_gone(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The subflow was already deleted (e.g. parent-flow teardown) -> no delete request, but the
        # stale metadata hint is cleared.
        node.metadata["subflow_name"] = "sub"
        _stub_object_manager(monkeypatch, {})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node._discard_subflow()

        assert not [r for r in requests if isinstance(r, DeleteFlowRequest)]
        assert "subflow_name" not in node.metadata

    def test_after_node_deleted_discards_subflow(
        self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node.metadata["subflow_name"] = "sub"
        _stub_object_manager(monkeypatch, {"sub": _live_flow("sub")})
        requests: list[Any] = []
        monkeypatch.setattr(GriptapeNodes, "handle_request", requests.append)

        node.after_node_deleted()

        assert [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)] == ["sub"]
        assert "subflow_name" not in node.metadata


class TestAprocessReDerivesShapeFromSubflow:
    """aprocess routes using the shape extracted from the live subflow, not the stale saved shape
    whose node names may have been renamed on import.
    """

    @pytest.mark.asyncio
    async def test_routes_io_to_renamed_subflow_nodes(
        self,
        node: SubflowWorkflowNode,
        monkeypatch: pytest.MonkeyPatch,
        shape_param: dict[str, Any],
        workflow_manager: Mock,
    ) -> None:
        # The subflow is already live; aprocess reuses it.
        node.metadata["subflow_name"] = "sub"

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
        node.metadata["subflow_name"] = "sub"
        # extract_workflow_shape raises ValueError when the subflow has no Start/End nodes.
        workflow_manager.extract_workflow_shape.side_effect = ValueError("no start/end nodes")

        # The failure output is unconnected on a bare node, so the failure re-raises.
        with pytest.raises(RuntimeError, match="has no shape defined"):
            await node.aprocess()


def _legacy_flow(name: str, referenced: str, *, transient: bool = False) -> SimpleNamespace:
    """A stand-in referenced ``ControlFlow``: carries its referenced-workflow name and transient flag.

    ``_referenced`` is the workflow the flow was imported from — what ``GetFlowDetailsRequest`` reports
    as ``referenced_workflow_name``.
    """
    metadata: dict[str, Any] = {}
    if transient:
        metadata["transient"] = True
    return SimpleNamespace(name=name, metadata=metadata, _referenced=referenced)


def _install_legacy_env(
    monkeypatch: pytest.MonkeyPatch,
    *,
    flows: dict[str, SimpleNamespace],
    parents: dict[str, str],
    node_flow: str = "ParentFlow",
) -> list[Any]:
    """Stub the engine surface ``_claim_legacy_serialised_flow`` walks and capture every request.

    ``flows`` maps flow name -> stub flow (see ``_legacy_flow``); ``parents`` maps flow name -> its
    parent flow name; ``node_flow`` is the node's containing flow (answer to GetFlowForNodeRequest).
    ``GetFlowMetadataRequest`` reports each flow's metadata (or failure when absent);
    ``GetFlowDetailsRequest`` reports each flow's parent + referenced workflow; ``SetFlowMetadataRequest``
    is mirrored onto the stub flow's metadata so a claim's transient flag is visible to a subsequent
    scan (as the real merge handler would make it).
    """
    object_manager = Mock(spec=ObjectManager)
    object_manager.attempt_get_object_by_name_as_type.side_effect = lambda name, _type: flows.get(name)
    object_manager.get_filtered_subset.return_value = flows
    monkeypatch.setattr(GriptapeNodes, "ObjectManager", lambda: object_manager)

    requests: list[Any] = []

    def _handle(request: Any) -> Any:
        requests.append(request)
        if isinstance(request, GetFlowForNodeRequest):
            return GetFlowForNodeResultSuccess(flow_name=node_flow, result_details="stubbed")
        if isinstance(request, GetFlowMetadataRequest):
            flow = flows.get(request.flow_name) if request.flow_name is not None else None
            if flow is None:
                return GetFlowMetadataResultFailure(result_details="not found")
            return GetFlowMetadataResultSuccess(metadata=flow.metadata, result_details="stubbed")
        if isinstance(request, GetFlowDetailsRequest):
            flow = flows.get(request.flow_name) if request.flow_name is not None else None
            return GetFlowDetailsResultSuccess(
                referenced_workflow_name=getattr(flow, "_referenced", None) if flow is not None else None,
                parent_flow_name=parents.get(request.flow_name) if request.flow_name is not None else None,
                flow_type=None,
                result_details="stubbed",
            )
        if isinstance(request, SetFlowMetadataRequest):
            flow = flows.get(request.flow_name) if request.flow_name is not None else None
            if flow is not None:
                flow.metadata.update(request.metadata)
            return SetFlowMetadataResultSuccess(result_details="stubbed")
        return None

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle)
    return requests


def _make_legacy_node(
    monkeypatch: pytest.MonkeyPatch, name: str, *, recorded: str, workflow: str
) -> SubflowWorkflowNode:
    """Construct a node from pre-transient legacy metadata (recorded subflow name + source workflow).

    The metadata is passed to ``__init__`` (as it is on load) so the node extracts the legacy pair into
    instance vars and pops the keys — matching how a legacy workflow file deserializes.
    """
    list_result = Mock(spec=ListCallableWorkflowsResultSuccess)
    list_result.workflow_names = [workflow]
    monkeypatch.setattr(GriptapeNodes, "handle_request", lambda _request: list_result)
    return SubflowWorkflowNode(
        name=name,
        metadata={"subflow_name": recorded, "_subflow_workflow": workflow},
    )


class TestClaimLegacySerialisedFlow:
    """Migrating pre-transient baked subflows: adopt once, flag ``transient``, drop legacy metadata.

    Legacy workflows serialize the subflow as a baked ``ImportWorkflowAsReferencedSubFlowRequest``
    that recreates a persistent child flow at load and record its (possibly renamed) name in
    ``metadata["subflow_name"]``. The node adopts that flow as its live subflow and flags it
    ``transient`` so a re-save is clean new-scheme.
    """

    def test_adopts_recorded_flow_and_marks_transient(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        requests = _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        assert node._claim_legacy_serialised_flow() == "ControlFlow_2"
        # Flagged transient so the engine never serializes it again.
        assert flows["ControlFlow_2"].metadata.get("transient") is True
        assert any(
            isinstance(r, SetFlowMetadataRequest)
            and r.flow_name == "ControlFlow_2"
            and r.metadata.get("transient") is True
            for r in requests
        )
        # Legacy metadata dropped so a re-save is clean new-scheme.
        assert "subflow_name" not in node.metadata
        assert "_subflow_workflow" not in node.metadata

    def test_ignores_flow_already_migrated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child", transient=True)}
        _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        assert node._claim_legacy_serialised_flow() is None

    def test_rejects_flow_referencing_other_workflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "some_other_workflow")}
        _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        assert node._claim_legacy_serialised_flow() is None

    def test_new_scheme_node_is_noop(self, node: SubflowWorkflowNode, monkeypatch: pytest.MonkeyPatch) -> None:
        # A native new-scheme node carries no _subflow_workflow metadata -> never a legacy candidate.
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        requests = _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        assert node._claim_legacy_serialised_flow() is None
        assert not [r for r in requests if isinstance(r, SetFlowMetadataRequest)]

    def test_scan_recovers_from_stale_recorded_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # subflow_name points at a name deduped away on import; the real flow (ControlFlow_5) is a
        # child of the node's own flow that references the expected workflow.
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_5": _legacy_flow("ControlFlow_5", "child")}
        _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_5": "ParentFlow"})

        assert node._claim_legacy_serialised_flow() == "ControlFlow_5"
        assert flows["ControlFlow_5"].metadata.get("transient") is True

    def test_recorded_name_in_wrong_parent_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The recorded name still resolves and references the expected workflow, but the flow is a
        # child of a DIFFERENT parent (e.g. a sibling's subflow after a de-dup rename). The
        # containing-flow check must reject it on the fast path (not just the scan).
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "SomeOtherParent"})

        assert node._claim_legacy_serialised_flow() is None

    def test_scan_skips_flow_in_another_parent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Same referenced workflow, but a child of a DIFFERENT parent -> not this node's subflow.
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="stale", workflow="child")
        flows = {"OtherChild": _legacy_flow("OtherChild", "child")}
        _install_legacy_env(monkeypatch, flows=flows, parents={"OtherChild": "SomeOtherParent"})

        assert node._claim_legacy_serialised_flow() is None

    def test_scan_distributes_duplicate_siblings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two legacy sibling nodes, two baked flows referencing the same workflow under the same
        # parent, both recorded names stale. Each must claim a DIFFERENT flow: claiming marks the
        # flow transient and the scan skips transient flows.
        node_a = _make_legacy_node(monkeypatch, "A", recorded="stale", workflow="child")
        node_b = _make_legacy_node(monkeypatch, "B", recorded="stale", workflow="child")
        flows = {"F1": _legacy_flow("F1", "child"), "F2": _legacy_flow("F2", "child")}
        _install_legacy_env(monkeypatch, flows=flows, parents={"F1": "ParentFlow", "F2": "ParentFlow"})

        claimed_a = node_a._claim_legacy_serialised_flow()
        claimed_b = node_b._claim_legacy_serialised_flow()

        assert claimed_a is not None
        assert claimed_b is not None
        assert claimed_a != claimed_b
        assert flows["F1"].metadata.get("transient") is True
        assert flows["F2"].metadata.get("transient") is True


class TestLoadSubflowAdoptsLegacy:
    """``_load_subflow`` adopts a node's legacy baked subflow instead of importing a duplicate."""

    def test_adopts_legacy_instead_of_importing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        requests = _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        assert node._load_subflow("child") == "ControlFlow_2"
        assert node.metadata.get("subflow_name") == "ControlFlow_2"
        # No fresh import — the pre-existing baked flow was adopted.
        assert not [r for r in requests if isinstance(r, ImportWorkflowAsReferencedSubFlowRequest)]

    def test_does_not_adopt_legacy_when_selection_changed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The baked flow was imported from "child", but the selection is now "other" (changed since
        # the legacy save). Adopting + running it under "other"'s shape would be wrong, so the node
        # must ignore the baked flow and import a fresh transient subflow for the new selection.
        monkeypatch.setattr(WorkflowRegistry, "has_workflow_with_name", lambda _name: True)
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        requests = _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        node._load_subflow("other")

        # The baked "child" flow was neither adopted nor claimed (no transient stamp)...
        assert node.metadata.get("subflow_name") != "ControlFlow_2"
        assert not [r for r in requests if isinstance(r, SetFlowMetadataRequest)]
        # ...instead a fresh transient import was attempted for the new selection.
        assert any(
            isinstance(r, ImportWorkflowAsReferencedSubFlowRequest) and r.workflow_name == "other" for r in requests
        )


class TestAfterNodeDeletedCleansLegacy:
    """Deleting a legacy node that was never run still tears down its baked subflow."""

    def test_deletes_legacy_baked_flow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_legacy_node(monkeypatch, "Workflow Node", recorded="ControlFlow_2", workflow="child")
        flows = {"ControlFlow_2": _legacy_flow("ControlFlow_2", "child")}
        requests = _install_legacy_env(monkeypatch, flows=flows, parents={"ControlFlow_2": "ParentFlow"})

        node.after_node_deleted()

        assert [r.flow_name for r in requests if isinstance(r, DeleteFlowRequest)] == ["ControlFlow_2"]


def _make_node_with_metadata(monkeypatch: pytest.MonkeyPatch, metadata: dict[str, Any]) -> SubflowWorkflowNode:
    """Construct a node with arbitrary initial metadata passed to ``__init__`` (as on load)."""
    list_result = Mock(spec=ListCallableWorkflowsResultSuccess)
    list_result.workflow_names = ["child"]
    monkeypatch.setattr(GriptapeNodes, "handle_request", lambda _request: list_result)
    return SubflowWorkflowNode(name="Workflow Node", metadata=dict(metadata))


class TestInitExtractsLegacyMetadata:
    """``__init__`` clears the runtime ``subflow_name`` hint from loaded metadata and retains the
    legacy migration pair only when the ``_subflow_workflow`` marker identifies a pre-transient save.

    The subflow is never serialized, so any ``subflow_name`` present at load time is stale: either a
    runtime hint saved while a subflow was live (new scheme) or a baked legacy reference. Popping it
    stops the editor previewing a dead/stale flow; the legacy pair is kept (in instance vars) so the
    first execution can adopt the baked flow.
    """

    def test_legacy_metadata_is_extracted_and_popped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_node_with_metadata(monkeypatch, {"subflow_name": "ControlFlow_2", "_subflow_workflow": "child"})

        # Popped from serializable metadata so the editor never previews the baked flow directly.
        assert "subflow_name" not in node.metadata
        assert "_subflow_workflow" not in node.metadata
        # Retained as instance state for one-time migration on first use.
        assert node._legacy_workflow == "child"
        assert node._legacy_flow_name == "ControlFlow_2"

    def test_stale_runtime_hint_is_popped_without_legacy_migration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # New-scheme node saved while a subflow was live: subflow_name present, no _subflow_workflow.
        # The referenced transient flow was not serialized, so the hint is stale -> drop it, and it is
        # NOT a legacy candidate.
        node = _make_node_with_metadata(monkeypatch, {"subflow_name": "ControlFlow_9"})

        assert "subflow_name" not in node.metadata
        assert node._legacy_workflow is None
        assert node._legacy_flow_name is None

    def test_native_new_scheme_node_has_no_legacy_state(self, node: SubflowWorkflowNode) -> None:
        assert node._legacy_workflow is None
        assert node._legacy_flow_name is None
