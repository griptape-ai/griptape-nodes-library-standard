from __future__ import annotations

from typing import Any
from uuid import uuid4

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.flow import ControlFlow
from griptape_nodes.exe_types.node_types import NodeDependencies, SuccessFailureNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry, WorkflowShape
from griptape_nodes.retained_mode.events.execution_events import (
    StartLocalSubflowRequest,
    StartLocalSubflowResultFailure,
)
from griptape_nodes.retained_mode.events.flow_events import DeleteFlowRequest, SetFlowMetadataRequest
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

# Node-metadata key holding the name of the imported child flow this node last
# bound to. It is a fast hint only; ``SUBFLOW_OWNER_KEY`` is the authoritative
# link (see below).
SUBFLOW_NAME_KEY = "subflow_name"

# Node-metadata key recording which registered workflow the current subflow was
# imported from, so a reload can tell "already loaded" from "workflow changed".
SUBFLOW_WORKFLOW_KEY = "_subflow_workflow"

# Metadata key stored on BOTH the Workflow node and its imported child flow. The
# shared UUID binds a specific node to the specific subflow it imported, so the
# two stay linked across saves, flow-name de-duplication, and nesting where the
# recorded ``subflow_name`` may no longer match.
SUBFLOW_OWNER_KEY = "subflow_owner_uuid"


class SubflowWorkflowNode(SuccessFailureNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        # Establish this node's stable owner UUID once, up front, so it exists
        # before any subflow import needs it. Preserved from saved metadata on
        # load; generated for brand-new nodes. It binds the node to the specific
        # child flow it imports.
        self.metadata.setdefault(SUBFLOW_OWNER_KEY, str(uuid4()))
        result = GriptapeNodes.handle_request(ListCallableWorkflowsRequest())
        if isinstance(result, ListCallableWorkflowsResultSuccess):
            # Add pyright ignore here becuase we know ListCallableWOrkflowsResultSuccess reutrns workflow_names
            workflow_names = result.workflow_names  # pyright: ignore[reportAttributeAccessIssue]
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
        # Delete the child flow we own.
        #
        # A None result can mean either:
        #
        # - our subflow was already legitimately deleted (flows are cleaned
        #   before nodes when the parent flow is deleted); or
        # - we point to a legacy subflow (i.e. without UUID) AND this node has
        #   never been executed (so a matching subflow has not been claimed).
        #
        # In the second case, we may leak the subflow. This seems preferable to
        # potentially deleting an unrelated flow (since in this case we only have
        # subflow_name to match with, which is an unstable reference).
        owned_subflow = self._find_owned_subflow_name()
        if owned_subflow is None:
            return
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=owned_subflow))

    def _on_refresh_workflow(self, button: Button, button_details: ButtonDetailsMessagePayload) -> None:  # noqa: ARG002
        workflow_name = self.get_parameter_value("workflow_file")
        if not workflow_name:
            return
        self._update_workflow_shape_parameters(workflow_name, preserve_matching=True)
        # Clear so _reload_subflow doesn't skip the reload for the same workflow.
        self.metadata.pop(SUBFLOW_WORKFLOW_KEY, None)
        self._reload_subflow(workflow_name)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter == self.workflow_file:
            # Only update if the workflow actually changed to avoid tearing down
            # parameters (and their connections) when the saved value is restored
            # during deserialization.
            current = self.metadata.get("_workflow_file_value")
            if value != current:
                self._update_workflow_shape_parameters(value)
                self.metadata["_workflow_file_value"] = value
                self._reload_subflow(value)
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

    def _reload_subflow(self, workflow_name: str) -> None:
        # Find the subflow this node actually owns (by UUID), never one it
        # merely shares a name with.
        owned_subflow = self._resolve_subflow_name(workflow_name)
        if owned_subflow is not None:
            # Already own a live subflow for THIS workflow -> keep it.
            if self.metadata.get(SUBFLOW_WORKFLOW_KEY) == workflow_name:
                self.metadata[SUBFLOW_NAME_KEY] = owned_subflow
                return
            # Own a subflow for a DIFFERENT workflow -> tear it down before
            # importing the new one.
            GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=owned_subflow))
        self.metadata.pop(SUBFLOW_NAME_KEY, None)
        self.metadata.pop(SUBFLOW_WORKFLOW_KEY, None)

        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return

        # Resolve the parent flow for this node so the import knows where to create the subflow.
        flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=self.name))
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            return

        result = GriptapeNodes.handle_request(
            ImportWorkflowAsReferencedSubFlowRequest(workflow_name=workflow_name, flow_name=flow_result.flow_name)
        )
        if isinstance(result, ImportWorkflowAsReferencedSubFlowResultSuccess):
            self.metadata[SUBFLOW_NAME_KEY] = result.created_flow_name
            self.metadata[SUBFLOW_WORKFLOW_KEY] = workflow_name
            # Mirror this node's owner UUID (established in __init__) onto the
            # imported flow so the pairing survives serialization and any later
            # flow-name changes.
            self._stamp_subflow_uuid(result.created_flow_name, self.metadata[SUBFLOW_OWNER_KEY])

    def _resolve_subflow_name(self, workflow_name: str) -> str | None:
        """Return the name of the subflow this node owns, or ``None`` if it isn't loaded.

        Prefers the flow stamped with this node's owner UUID, so two nodes
        referencing the same workflow - or a nested node whose recorded name
        shifted on import - each bind to their OWN subflow rather than
        colliding. Falls back to adopting a legacy (pre-UUID) recorded subflow,
        stamping it so future lookups match by UUID.
        """
        owned = self._find_owned_subflow_name()
        if owned is not None:
            return owned

        # Fallback for legacy workflows saved without UUID metadata.
        recorded = self._recorded_subflow_if_unclaimed()
        if recorded is None:
            return None

        # Legacy (pre-UUID) subflow: adopt the recorded flow by stamping this
        # node's owner UUID.
        self._stamp_subflow_uuid(recorded, self.metadata[SUBFLOW_OWNER_KEY])
        logger.warning(
            "Node '%s' adopted legacy referenced subflow '%s' (workflow '%s') that carried no owner UUID. Stamped"
            " owner UUID '%s' onto it; re-save the workflow to persist the link.",
            self.name,
            recorded,
            workflow_name,
            self.metadata[SUBFLOW_OWNER_KEY],
        )
        return recorded

    def _find_owned_subflow_name(self) -> str | None:
        """Return the flow stamped with this node's owner UUID, or ``None``.

        The authoritative node<->flow link, with no side effects. Tries the
        recorded ``subflow_name`` first as an O(1) lookup (the common case) and
        only scans every flow when that hint is missing or no longer points at a
        flow we own (e.g. after de-dup / nesting renames).
        """
        owner_uuid = self.metadata[SUBFLOW_OWNER_KEY]

        # Quick path: the recorded name usually still points at the flow we own.
        recorded = self.metadata.get(SUBFLOW_NAME_KEY)
        if recorded:
            recorded_flow = GriptapeNodes.ObjectManager().attempt_get_object_by_name_as_type(recorded, ControlFlow)
            if recorded_flow is not None and recorded_flow.metadata.get(SUBFLOW_OWNER_KEY) == owner_uuid:
                return recorded

        # Slow path: scan every flow for this node's owner UUID.
        flows = GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow)
        for candidate_name, flow in flows.items():
            if flow.metadata.get(SUBFLOW_OWNER_KEY) == owner_uuid:
                return candidate_name
        return None

    def _recorded_subflow_if_unclaimed(self) -> str | None:
        """Return the recorded ``subflow_name`` only if it is safe to treat as ours.

        .. deprecated::
            Legacy-compatibility only. Covers pre-UUID subflows that were never
            stamped with an owner UUID (so ``_find_owned_subflow_name`` misses
            them), binding by recorded name instead. Once every saved workflow
            carries owner UUIDs, nothing reaches this path and it (with its
            callers' fallbacks) can be removed.

        We only claim the recorded flow if it still exists, carries no owner
        UUID (or ours) — never one already owned by a different node — AND was
        imported from the workflow this node expects. O(1) name lookup.
        """
        recorded = self.metadata.get(SUBFLOW_NAME_KEY)
        expected_workflow = self.metadata.get(SUBFLOW_WORKFLOW_KEY)
        # SUBFLOW_NAME_KEY and SUBFLOW_WORKFLOW_KEY are always written together;
        # require both so we never adopt a flow without knowing which workflow
        # it should reference.
        if not recorded or not expected_workflow:
            return None
        flow = GriptapeNodes.ObjectManager().attempt_get_object_by_name_as_type(recorded, ControlFlow)
        if flow is None:
            return None
        if flow.metadata.get(SUBFLOW_OWNER_KEY) not in (None, self.metadata[SUBFLOW_OWNER_KEY]):
            # Already claimed by a different owner.
            return None
        # Confidence check: the recorded flow must actually be an import of the
        # workflow we expect.
        if GriptapeNodes.FlowManager().get_referenced_workflow_name(flow) != expected_workflow:
            return None

        return recorded

    def _stamp_subflow_uuid(self, flow_name: str, owner_uuid: str) -> None:
        GriptapeNodes.handle_request(
            SetFlowMetadataRequest(flow_name=flow_name, metadata={SUBFLOW_OWNER_KEY: owner_uuid})
        )

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

        # Bind to the subflow this node owns (by UUID); import it if it isn't
        # loaded yet. Bind before deriving the shape so the shape reflects the
        # live (imported) nodes.
        subflow_name = self._resolve_subflow_name(workflow_name)
        if subflow_name is None:
            self._reload_subflow(workflow_name)
            subflow_name = self.metadata.get(SUBFLOW_NAME_KEY)
            if not subflow_name:
                msg = f"Failed to load subflow for workflow '{workflow_name}'."
                self._set_status_results(was_successful=False, result_details=msg)
                self._handle_failure_exception(RuntimeError(msg))
                return
        self.metadata[SUBFLOW_NAME_KEY] = subflow_name

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
