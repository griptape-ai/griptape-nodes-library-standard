from __future__ import annotations

from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.flow import ControlFlow
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.events.execution_events import StartLocalSubflowRequest, StartLocalSubflowResultFailure
from griptape_nodes.retained_mode.events.flow_events import DeleteFlowRequest
from griptape_nodes.retained_mode.events.parameter_events import RemoveParameterFromNodeRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options


class SubflowWorkflowNode(BaseNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        workflow_names = list(WorkflowRegistry.list_valid_workflows().keys())
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
            traits={Options(choices=choices)},
        )
        self.add_parameter(self.workflow_file)

        # Pre-populate dynamic parameters so they exist before connections and
        # set-value commands are applied during deserialization.
        if saved_workflow and saved_workflow in workflow_names:
            # Ensure workflow_shape_params is empty so _remove_workflow_shape_parameters
            # is a no-op (the node isn't registered yet, so the request would fail).
            self.metadata["workflow_shape_params"] = []
            self._update_workflow_shape_parameters(saved_workflow)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "workflow_file":
            # Only update if the workflow actually changed to avoid tearing down
            # parameters (and their connections) when the saved value is restored
            # during deserialization.
            current = self.metadata.get("_workflow_file_value")
            if value != current:
                self._update_workflow_shape_parameters(value)
                self.metadata["_workflow_file_value"] = value
        super().after_value_set(parameter, value)

    def _update_workflow_shape_parameters(self, workflow_name: str) -> None:
        self._remove_workflow_shape_parameters()

        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return

        workflow = WorkflowRegistry.get_workflow_by_name(workflow_name)
        workflow_shape = workflow.metadata.workflow_shape
        if workflow_shape is None:
            return

        shape_param_names = []

        for _node_name, params in workflow_shape.inputs.items():
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                param = Parameter(
                    name=param_name,
                    tooltip=param_dict.get("tooltip", ""),
                    type=param_dict.get("type"),
                    input_types=param_dict.get("input_types"),
                    output_type=param_dict.get("output_type"),
                    default_value=param_dict.get("default_value"),
                    allowed_modes={ParameterMode.INPUT},
                    ui_options=param_dict.get("ui_options"),
                )
                self.add_parameter(param)
                shape_param_names.append(param_name)
                if "left_parameters" not in self.metadata:
                    self.metadata["left_parameters"] = []
                self.metadata["left_parameters"].append(param_name)

        for _node_name, params in workflow_shape.outputs.items():
            for param_name, param_dict in params.items():
                if param_dict.get("type") == ParameterTypeBuiltin.CONTROL_TYPE:
                    continue
                param = Parameter(
                    name=param_name,
                    tooltip=param_dict.get("tooltip", ""),
                    type=param_dict.get("type"),
                    input_types=param_dict.get("input_types"),
                    output_type=param_dict.get("output_type"),
                    default_value=param_dict.get("default_value"),
                    allowed_modes={ParameterMode.OUTPUT},
                    ui_options=param_dict.get("ui_options"),
                )
                self.add_parameter(param)
                shape_param_names.append(param_name)
                if "right_parameters" not in self.metadata:
                    self.metadata["right_parameters"] = []
                self.metadata["right_parameters"].append(param_name)

        self.metadata["workflow_shape_params"] = shape_param_names

    def _remove_workflow_shape_parameters(self) -> None:
        shape_param_names = self.metadata.get("workflow_shape_params", [])
        for param_name in shape_param_names:
            GriptapeNodes.handle_request(
                RemoveParameterFromNodeRequest(node_name=self.name, parameter_name=param_name)
            )
            for side in ("left_parameters", "right_parameters"):
                if side in self.metadata and param_name in self.metadata[side]:
                    self.metadata[side].remove(param_name)
        self.metadata["workflow_shape_params"] = []

    async def aprocess(self) -> None:
        workflow_name = self.get_parameter_value("workflow_file")
        if not workflow_name or not WorkflowRegistry.has_workflow_with_name(workflow_name):
            return

        workflow = WorkflowRegistry.get_workflow_by_name(workflow_name)
        workflow_shape = workflow.metadata.workflow_shape
        if workflow_shape is None:
            return

        file_path = WorkflowRegistry.get_complete_file_path(workflow.file_path)
        content = Path(file_path).read_text(encoding="utf-8")

        existing_flows = set(GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow).keys())
        exec(content, {"__file__": file_path, "__name__": "workflow_module", "__builtins__": __builtins__})  # noqa: S102
        new_flows = set(GriptapeNodes.ObjectManager().get_filtered_subset(type=ControlFlow).keys()) - existing_flows

        if not new_flows:
            return

        workflow_flow_name = next(iter(new_flows))

        try:
            self._set_workflow_inputs(workflow_flow_name, workflow_shape)
            result = await GriptapeNodes.FlowManager().on_start_local_subflow_request(
                StartLocalSubflowRequest(flow_name=workflow_flow_name)
            )
            if isinstance(result, StartLocalSubflowResultFailure):
                msg = f"Workflow '{workflow_name}' execution failed: {result.result_details}"
                raise RuntimeError(msg)
            self._collect_workflow_outputs(workflow_flow_name, workflow_shape)
        finally:
            GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=workflow_flow_name))

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

    def process(self) -> Any:
        raise NotImplementedError
