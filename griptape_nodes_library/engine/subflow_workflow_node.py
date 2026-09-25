from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.flow import ControlFlow
from griptape_nodes.exe_types.node_types import NodeDependencies, SuccessFailureNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry, WorkflowShape
from griptape_nodes.retained_mode.events.execution_events import (
    StartLocalSubflowRequest,
    StartLocalSubflowResultFailure,
)
from griptape_nodes.retained_mode.events.flow_events import (
    DeleteFlowRequest,
    GetFlowDetailsRequest,
    GetFlowDetailsResultSuccess,
    GetFlowMetadataRequest,
    GetFlowMetadataResultSuccess,
    SetFlowMetadataRequest,
)
from griptape_nodes.retained_mode.events.node_events import GetFlowForNodeRequest, GetFlowForNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import (
    RemoveParameterFromNodeRequest,
    SetParameterValueRequest,
)

# Add pyright ignore because this is a new event - we know it does exist
from griptape_nodes.retained_mode.events.workflow_events import (
    ImportWorkflowAsReferencedSubFlowRequest,
    ImportWorkflowAsReferencedSubFlowResultSuccess,
    ListCallableWorkflowsRequest,  # pyright: ignore[reportAttributeAccessIssue]
    ListCallableWorkflowsResultSuccess,  # pyright: ignore[reportAttributeAccessIssue]
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

# Flow-metadata flag marking a subflow as a runtime-only artifact the engine
# must never serialize. The node (re)imports its subflow at execution time, so
# it should never round-trip through a save.
TRANSIENT_KEY = "transient"

# Legacy (pre-transient) node metadata: the recorded name of the baked child
# subflow (may be stale after a de-dup rename on import) and the workflow it was
# imported from. Only present on workflows saved before the transient scheme;
# used solely to migrate them (see _claim_legacy_serialised_flow).
SUBFLOW_NAME_KEY = "subflow_name"
SUBFLOW_WORKFLOW_KEY = "_subflow_workflow"


class SubflowWorkflowNode(SuccessFailureNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # The imported subflow is a transient, execution-time artifact tracked
        # via a runtime ``subflow_name`` metadata hint (see _load_subflow). The
        # subflow itself is never serialized, so any ``subflow_name`` present at
        # load time is stale; or worse, in the case of copy-paste it belongs to
        # another node.
        #
        # Use the defunct ``_subflow_workflow`` metadata as a signal that the
        # node was restored from a legacy saved workflow. Keep track of both keys
        # to help us claim the legacy baked subflow and migrate it.
        self._legacy_workflow: str | None = self.metadata.pop(SUBFLOW_WORKFLOW_KEY, None)
        recorded_subflow_name = self.metadata.pop(SUBFLOW_NAME_KEY, None)
        self._legacy_flow_name: str | None = recorded_subflow_name if self._legacy_workflow else None

        result = GriptapeNodes.handle_request(ListCallableWorkflowsRequest())
        if isinstance(result, ListCallableWorkflowsResultSuccess):
            workflow_names = result.workflow_names
            choices = workflow_names
        else:
            workflow_names = []
            choices = [""]

        # If a workflow was previously selected, restore it as the default.
        saved_workflow = self.metadata.get("_workflow_file_value")
        default = saved_workflow if saved_workflow and saved_workflow in workflow_names else choices[0]

        self.workflow_file = Parameter(
            name="workflow_file",
            tooltip="Select a workflow to execute",
            type=ParameterTypeBuiltin.STR,
            allowed_modes={ParameterMode.PROPERTY},
            default_value=default,
            traits={
                Options(choices=choices),
                Button(
                    label="Refresh Workflow",
                    icon="refresh-cw",
                    variant="secondary",
                    size="sm",
                    on_click=self._on_refresh_workflow,
                ),
            },
        )
        self.add_parameter(self.workflow_file)
        self.metadata["workflow_node"] = True

        # Pre-populate dynamic parameters so they exist before connections and
        # set-value commands are applied during deserialization.
        if saved_workflow and saved_workflow in workflow_names:
            # Ensure workflow_shape_params is empty so _remove_workflow_shape_parameters
            # is a no-op (the node isn't registered yet, so the request would fail).
            self.metadata["workflow_shape_params"] = []
            self._update_workflow_shape_parameters(saved_workflow)

        self._create_status_parameters()

    def after_node_deleted(self) -> None:
        # Tear down the transient subflow we imported (if it is still live).
        self._discard_subflow()

    def _on_refresh_workflow(self, button: Button, button_details: ButtonDetailsMessagePayload) -> None:  # noqa: ARG002
        workflow_name = self.get_parameter_value("workflow_file")
        if not workflow_name:
            return
        self._update_workflow_shape_parameters(workflow_name, preserve_matching=True)
        # Drop the live subflow so the next execution re-imports the latest
        # workflow definition.
        self._discard_subflow()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter == self.workflow_file:
            # Only update if the workflow actually changed to avoid tearing down
            # parameters (and their connections) when the saved value is restored
            # during deserialization.
            current = self.metadata.get("_workflow_file_value")
            if value != current:
                self._update_workflow_shape_parameters(value)
                self.metadata["_workflow_file_value"] = value
                # The selection changed; drop the old subflow so the next
                # execution imports the new one.
                self._discard_subflow()
        super().after_value_set(parameter, value)

    def get_node_dependencies(self) -> NodeDependencies | None:
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()
        workflow_name = self.get_parameter_value("workflow_file")
        if workflow_name and WorkflowRegistry.has_workflow_with_name(workflow_name):
            deps.referenced_workflows.add(workflow_name)
        else:
            msg = f"Node {self.name} has a dependency on workflow {workflow_name} but it is not in your WorkflowRegistry. Register this workflow for proper dependency bundling."
            logger.warning(msg)
        return deps

    def _load_subflow(self, workflow_name: str) -> str | None:
        """Import ``workflow_name`` as a transient child subflow and return its
        name (``None`` on failure).

        Reuses the subflow already imported by this node if it is still live.
        The created flow is flagged ``transient`` so the engine never serializes
        it — the node re-imports it at execution time, which keeps the
        node<->subflow link out of the saved workflow, preventing a whole class
        of serialization issues with flow name de-duplication.

        The live subflow name is stored in ``metadata[SUBFLOW_NAME_KEY]``. It
        serves purely as a runtime hint the editor reads to show the in-memory
        subflow ("View Subflow" button); it is popped again on the next load.
        """
        # Reuse the subflow we already imported, if it still exists.
        tracked = self.metadata.get(SUBFLOW_NAME_KEY)
        if tracked is not None:
            existing = GriptapeNodes.ObjectManager().attempt_get_object_by_name_as_type(tracked, ControlFlow)
            if existing is not None:
                return tracked
            # It was torn down out from under us (e.g. parent-flow deletion);
            # drop the stale hint.
            self.metadata.pop(SUBFLOW_NAME_KEY, None)

        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return None

        # Migrate a legacy baked subflow on first use (adopt it instead of
        # importing a duplicate).
        # Defensively check that we're loading the workflow_name corresponding
        # to the legacy workflow, but technically not necessary since changing
        # the dropdown will trigger after_value_set(), which will adopt and
        # discard the legacy flow before we get here.
        if self._legacy_workflow == workflow_name:
            legacy_subflow = self._claim_legacy_serialised_flow()
            if legacy_subflow is not None:
                self.metadata[SUBFLOW_NAME_KEY] = legacy_subflow
                return legacy_subflow

        # Resolve the parent flow for this node so the import knows where to
        # create the subflow.
        flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=self.name))
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            return None

        result = GriptapeNodes.handle_request(
            ImportWorkflowAsReferencedSubFlowRequest(
                workflow_name=workflow_name,
                flow_name=flow_result.flow_name,
                # Flag as transient so the flow is not serialized on save.
                imported_flow_metadata={TRANSIENT_KEY: True},
            )
        )
        if not isinstance(result, ImportWorkflowAsReferencedSubFlowResultSuccess):
            return None
        self.metadata[SUBFLOW_NAME_KEY] = result.created_flow_name
        return result.created_flow_name

    def _discard_subflow(self) -> None:
        """Delete the subflow this node owns, if it is still live, and stop
        tracking it.

        A legacy node loaded but never executed has no tracked subflow yet;
        adopt its baked flow first so it is torn down rather than leaked.
        """
        tracked = self.metadata.get(SUBFLOW_NAME_KEY)
        if tracked is None:
            tracked = self._claim_legacy_serialised_flow()
        if tracked is None:
            return
        subflow = GriptapeNodes.ObjectManager().attempt_get_object_by_name_as_type(tracked, ControlFlow)
        if subflow is not None:
            GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=tracked))
        self.metadata.pop(SUBFLOW_NAME_KEY, None)

    def _claim_legacy_serialised_flow(self) -> str | None:
        """Adopt this node's legacy baked subflow and flag it ``transient``.

        Workflows saved before the `"transient"` metadata was added serialize
        the subflow as a baked ``ImportWorkflowAsReferencedSubFlowRequest`` that
        recreates a persistent child flow at load, recording its (possibly
        incorrect due to de-duping) name and source workflow in metadata. That
        pair is lifted into ``self._legacy_flow_name`` /
        ``self._legacy_workflow`` at load (see ``__init__``). Migration adopts
        that flow as this node's live subflow and marks it ``transient`` so it
        is never persisted again — after which a re-save is clean new-scheme.

        Returns the adopted flow name, or ``None`` when this node has no
        un-migrated legacy subflow — including every native new-scheme node,
        which carries no legacy workflow marker.
        """
        if not self._legacy_workflow:
            return None

        # A node's subflow is always a direct child of the node's own flow.
        # Resolve that flow once so both the recorded-name and scan paths can
        # require the candidate to be parented to it.
        flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=self.name))
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            return None
        containing_flow = flow_result.flow_name

        # Fast path: the recorded name, when it was not renamed on import.
        claimed = self._claim_flow_if_legacy(self._legacy_flow_name, self._legacy_workflow, containing_flow)
        # Fallback: the recorded name went stale (de-dup rename) -> scan this
        # node's own child flows.
        if claimed is None:
            claimed = self._scan_for_and_claim_legacy_flow(self._legacy_workflow, containing_flow)

        if claimed is not None:
            logger.warning(
                "%s (%s): claiming legacy serialised sub-flow '%s'. This claim might be incorrect in complex graphs"
                " with nested workflows where flow name de-duplication has occurred. Save and reload the parent"
                " workflow so that child workflows load just-in-time instead, avoiding this problem.",
                self.name,
                self._legacy_workflow,
                claimed,
            )
            # Migrated: forget the legacy markers so a re-claim is a no-op.
            self._legacy_workflow = None
            self._legacy_flow_name = None

        return claimed

    def _claim_flow_if_legacy(self, flow_name: str | None, expected_workflow: str, containing_flow: str) -> str | None:
        """Return ``flow_name`` (after flagging it ``transient``) iff it is an
        un-migrated legacy import of ``expected_workflow`` parented to
        ``containing_flow`` (this node's own flow)."""
        if not flow_name:
            return None
        metadata_result = GriptapeNodes.handle_request(GetFlowMetadataRequest(flow_name=flow_name))
        if not isinstance(metadata_result, GetFlowMetadataResultSuccess):
            # Flow not found.
            return None
        if metadata_result.metadata.get(TRANSIENT_KEY):
            # Already claimed.
            return None
        details = GriptapeNodes.handle_request(GetFlowDetailsRequest(flow_name=flow_name))
        if not isinstance(details, GetFlowDetailsResultSuccess):
            return None
        if details.parent_flow_name != containing_flow:
            # Wrong parent. Likely due to flow name de-duping.
            return None
        if details.referenced_workflow_name != expected_workflow:
            # Wrong workflow file. Again, likely due to flow name de-duping.
            return None
        # Tag as transient so that (a) the flow will no longer be serialised;
        # and (b) another SubflowWorkflowNode will not try to claim it.
        self._mark_flow_transient(flow_name)
        return flow_name

    def _scan_for_and_claim_legacy_flow(self, expected_workflow: str, containing_flow: str) -> str | None:
        """Find and claim an un-migrated legacy import of ``expected_workflow``
        among this node's own child flows."""
        for candidate_name in GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow):
            if self._claim_flow_if_legacy(candidate_name, expected_workflow, containing_flow) is not None:
                return candidate_name
        return None

    def _mark_flow_transient(self, flow_name: str) -> None:
        GriptapeNodes.handle_request(SetFlowMetadataRequest(flow_name=flow_name, metadata={TRANSIENT_KEY: True}))

    def _create_shape_parameter(
        self, param_name: str, param_dict: dict, allowed_modes: set[ParameterMode]
    ) -> Parameter:
        return Parameter(
            name=param_name,
            tooltip=param_dict.get("tooltip", ""),
            type=param_dict.get("type"),
            input_types=param_dict.get("input_types"),
            output_type=param_dict.get("output_type"),
            default_value=param_dict.get("default_value"),
            allowed_modes=allowed_modes,
            ui_options=param_dict.get("ui_options"),
        )

    def _update_workflow_shape_parameters(self, workflow_name: str, preserve_matching: bool = False) -> None:
        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            self._remove_workflow_shape_parameters()
            return

        workflow = WorkflowRegistry.get_workflow_by_name(workflow_name)
        workflow_shape = workflow.metadata.workflow_shape
        if workflow_shape is None:
            self._remove_workflow_shape_parameters()
            return

        # Collect desired params: name -> (param_dict, allowed_modes)
        desired: dict[str, tuple[dict, set[ParameterMode]]] = {}
        for _node_name, params in workflow_shape.inputs.items():
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                input_modes: set[ParameterMode] = {ParameterMode.INPUT}
                if param_dict.get("mode_allowed_property"):
                    input_modes.add(ParameterMode.PROPERTY)
                desired[param_name] = (param_dict, input_modes)

        for _node_name, params in workflow_shape.outputs.items():
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                if param_name in desired:
                    # Same name in inputs and outputs — merge modes so one parameter serves both roles.
                    existing_dict, existing_modes = desired[param_name]
                    desired[param_name] = (existing_dict, existing_modes | {ParameterMode.OUTPUT})
                else:
                    desired[param_name] = (param_dict, {ParameterMode.OUTPUT})

        current_names = set(self.metadata.get("workflow_shape_params", []))

        # Drop any desired param that collides with an existing built-in parameter
        # (one that isn't a previously-added shape param), e.g. 'was_successful'.
        for name in list(desired.keys()):
            if name not in current_names and self.get_parameter_by_name(name) is not None:
                desired.pop(name)

        desired_names = set(desired.keys())

        # When preserve_matching=True, keep params whose names appear in both sets
        # (their values and connections survive untouched). Otherwise replace everything.
        to_remove = current_names - desired_names if preserve_matching else current_names
        to_add = desired_names - current_names if preserve_matching else desired_names

        for param_name in to_remove:
            self.parameter_output_values.pop(param_name, None)

        self._remove_shape_params(to_remove)

        for param_name in to_add:
            param_dict, allowed_modes = desired[param_name]
            self.add_parameter(self._create_shape_parameter(param_name, param_dict, allowed_modes))

        self.metadata["workflow_shape_params"] = list(desired_names)

        # Rebuild side lists from scratch so ordering is correct for the full desired set,
        # including any preserved params. A param with both INPUT and OUTPUT modes appears
        # on both sides.
        self.metadata["left_parameters"] = []
        self.metadata["right_parameters"] = []
        for param_name, (_, allowed_modes) in desired.items():
            if ParameterMode.INPUT in allowed_modes:
                self.metadata["left_parameters"].append(param_name)
            if ParameterMode.OUTPUT in allowed_modes:
                self.metadata["right_parameters"].append(param_name)

        # Enforce display order: left params (inputs) at top, right params (outputs) below.
        for param_name in self.metadata["left_parameters"] + self.metadata["right_parameters"]:
            self.move_element_to_position(param_name, "last")

    def _remove_workflow_shape_parameters(self) -> None:
        self._remove_shape_params(set(self.metadata.get("workflow_shape_params", [])))
        self.metadata["workflow_shape_params"] = []
        self.metadata["left_parameters"] = []
        self.metadata["right_parameters"] = []

    def _remove_shape_params(self, param_names: set[str]) -> None:
        for param_name in param_names:
            param = self.get_parameter_by_name(param_name)
            if param is not None:
                # RemoveParameterFromNodeRequest rejects non-user-defined parameters to protect
                # built-in node params. Shape params are intentionally user_defined=False so they
                # are NOT serialized as AddParameterToNodeRequest commands (which would cause _1
                # duplicates on load since __init__ pre-creates them). We temporarily flip the flag
                # here so the removal handler allows the delete while still cleaning up connections.
                param.user_defined = True
            GriptapeNodes.handle_request(RemoveParameterFromNodeRequest(node_name=self.name, parameter_name=param_name))

    async def aprocess(self) -> None:
        workflow_name = self.get_parameter_value("workflow_file")
        if not workflow_name:
            msg = f"Node '{self.name}' has no workflow selected."
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return
        if not WorkflowRegistry.has_workflow_with_name(workflow_name):
            msg = f"Node '{self.name}' references workflow '{workflow_name}' which is not registered."
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return

        # Import the subflow now (execution time), reusing it if this node already has one live.
        # Load it before deriving the shape so the shape reflects the live (imported) nodes.
        subflow_name = self._load_subflow(workflow_name)
        if subflow_name is None:
            msg = f"Failed to load subflow for workflow '{workflow_name}'."
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return

        # Re-derive the shape from the freshly-imported subflow instead of
        # trusting the shape saved with the workflow. Importing renames any node
        # whose name collides with an existing one (e.g. 'Start Flow' -> 'Start
        # Flow_1'), which leaves the saved shape's node-name keys pointing at
        # nodes that no longer exist.
        try:
            shape = GriptapeNodes.WorkflowManager().extract_workflow_shape(workflow_name, flow_name=subflow_name)
        except ValueError:
            msg = f"Workflow '{workflow_name}' has no shape defined."
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return
        workflow_shape = WorkflowShape(inputs=shape["input"], outputs=shape["output"])

        self._set_workflow_inputs(subflow_name, workflow_shape)
        result = await GriptapeNodes.ahandle_request(StartLocalSubflowRequest(flow_name=subflow_name))
        if isinstance(result, StartLocalSubflowResultFailure):
            msg = f"Workflow '{workflow_name}' execution failed: {result.result_details}"
            self._set_status_results(was_successful=False, result_details=msg)
            self._handle_failure_exception(RuntimeError(msg))
            return
        self._collect_workflow_outputs(subflow_name, workflow_shape)
        self._set_status_results(
            was_successful=True, result_details=f"Workflow '{workflow_name}' completed successfully."
        )

    def _set_workflow_inputs(self, flow_name: str, workflow_shape: Any) -> None:
        flow = GriptapeNodes.FlowManager().get_flow_by_name(flow_name)
        for start_node_name, params in workflow_shape.inputs.items():
            start_node = flow.nodes.get(start_node_name)
            if start_node is None:
                continue
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                value = self.get_parameter_value(param_name)
                GriptapeNodes.handle_request(
                    SetParameterValueRequest(
                        parameter_name=param_name,
                        node_name=start_node_name,
                        value=value,
                    )
                )

    def _collect_workflow_outputs(self, flow_name: str, workflow_shape: Any) -> None:
        flow = GriptapeNodes.FlowManager().get_flow_by_name(flow_name)
        for end_node_name, params in workflow_shape.outputs.items():
            end_node = flow.nodes.get(end_node_name)
            if end_node is None:
                continue
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                if param_name in end_node.parameter_output_values:
                    value = end_node.parameter_output_values[param_name]
                else:
                    value = end_node.get_parameter_value(param_name)
                if value is not None:
                    self.parameter_output_values[param_name] = value
