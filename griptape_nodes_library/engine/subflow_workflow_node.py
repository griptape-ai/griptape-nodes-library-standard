from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.flow import ControlFlow
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.events.execution_events import (
    StartLocalSubflowRequest,
    StartLocalSubflowResultFailure,
)
from griptape_nodes.retained_mode.events.flow_events import DeleteFlowRequest
from griptape_nodes.retained_mode.events.node_events import GetFlowForNodeRequest, GetFlowForNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import (
    RemoveParameterFromNodeRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.events.workflow_events import (
    ImportWorkflowAsReferencedSubFlowRequest,
    ImportWorkflowAsReferencedSubFlowResultSuccess,
    ListCallableWorkflowsRequest,
    ListCallableWorkflowsResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options


class SubflowWorkflowNode(BaseNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        result = GriptapeNodes.handle_request(ListCallableWorkflowsRequest())
        workflow_names = result.workflow_names if isinstance(result, ListCallableWorkflowsResultSuccess) else []
        choices = workflow_names if workflow_names else [""]

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

    def after_node_deleted(self) -> None:
        subflow_name = self.metadata.get("subflow_name")
        if subflow_name is not None:
            # The subflow may have already been deleted if the parent flow was deleted first
            # (parent flow deletion deletes child flows before nodes).
            subflow = GriptapeNodes.ObjectManager().attempt_get_object_by_name_as_type(subflow_name, ControlFlow)
            if subflow is not None:
                GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=subflow_name))

    def _on_refresh_workflow(self, button: Button, button_details: ButtonDetailsMessagePayload) -> None:  # noqa: ARG002
        workflow_name = self.get_parameter_value("workflow_file")
        if not workflow_name:
            return
        self._update_workflow_shape_parameters(workflow_name, preserve_matching=True)
        # Clear so _reload_subflow doesn't skip the reload for the same workflow.
        self.metadata.pop("_subflow_workflow", None)
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

    def _reload_subflow(self, workflow_name: str) -> None:
        existing_subflow = self.metadata.get("subflow_name")
        if existing_subflow:
            existing_flow_names = set(GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow).keys())
            # Skip if this workflow is already loaded and the flow still exists.
            if self.metadata.get("_subflow_workflow") == workflow_name and existing_subflow in existing_flow_names:
                return
            # Only delete if the flow still exists (it may be stale from a previous session).
            if existing_subflow in existing_flow_names:
                GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=existing_subflow))
            del self.metadata["subflow_name"]
            self.metadata.pop("_subflow_workflow", None)

        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return

        # Resolve the parent flow for this node so the import knows where to create the subflow.
        flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=self.name))
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            return

        result = GriptapeNodes.handle_request(
            ImportWorkflowAsReferencedSubFlowRequest(
                workflow_name=workflow_name, flow_name=flow_result.flow_name, track_as_referenced=False
            )
        )
        if isinstance(result, ImportWorkflowAsReferencedSubFlowResultSuccess):
            self.metadata["subflow_name"] = result.created_flow_name
            self.metadata["_subflow_workflow"] = workflow_name

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
                desired[param_name] = (param_dict, {ParameterMode.OUTPUT})

        current_names = set(self.metadata.get("workflow_shape_params", []))
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
        # including any preserved params.
        self.metadata["left_parameters"] = []
        self.metadata["right_parameters"] = []
        for param_name, (_, allowed_modes) in desired.items():
            if ParameterMode.OUTPUT in allowed_modes:
                self.metadata["right_parameters"].append(param_name)
            else:
                self.metadata["left_parameters"].append(param_name)

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
        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return

        workflow = WorkflowRegistry.get_workflow_by_name(workflow_name)
        workflow_shape = workflow.metadata.workflow_shape
        if workflow_shape is None:
            return

        # Use the pre-loaded subflow if available; otherwise load it now.
        existing_flow_names = set(GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow).keys())
        subflow_name = self.metadata.get("subflow_name")
        if not subflow_name or subflow_name not in existing_flow_names:
            self._reload_subflow(workflow_name)
            subflow_name = self.metadata.get("subflow_name")
            if not subflow_name:
                return

        self._set_workflow_inputs(subflow_name, workflow_shape)
        result = await GriptapeNodes.FlowManager().on_start_local_subflow_request(
            StartLocalSubflowRequest(flow_name=subflow_name)
        )
        if isinstance(result, StartLocalSubflowResultFailure):
            msg = f"Workflow '{workflow_name}' execution failed: {result.result_details}"
            raise RuntimeError(msg)
        self._collect_workflow_outputs(subflow_name, workflow_shape)

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
