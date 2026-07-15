# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "subflow_middle_echo"
# schema_version = "0.20.0"
# engine_version_created_with = "0.93.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.82.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "StartFlow"], ["Griptape Nodes Library", "SubflowWorkflowNode"]]
# workflows_referenced = ["subflow_inner_echo"]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-07-15T11:39:02.660858Z
# last_modified_date = 2026-07-15T11:39:02.665086Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":false,\"mode_allowed_output\":true,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"a\":{\"name\":\"a\",\"tooltip\":\"workflow input\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null},\"b\":{\"name\":\"b\",\"tooltip\":\"workflow input\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":true,\"mode_allowed_output\":false,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"out_a\":{\"name\":\"out_a\",\"tooltip\":\"workflow output\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null},\"out_b\":{\"name\":\"out_b\",\"tooltip\":\"workflow output\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
import pickle

# --- ADDED FOR THE INTEGRATION TEST (not emitted by the serialiser) ---
# Used by the inner-workflow registration block inside build_workflow() below.
from pathlib import Path
from typing import Any

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
from griptape_nodes.node_library.workflow_registry import WorkflowRegistry
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    GetTopLevelFlowRequest,
    GetTopLevelFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.events.workflow_events import (
    ImportWorkflowAsReferencedSubFlowRequest,
    LoadWorkflowMetadata,
    LoadWorkflowMetadataResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

# --- END ADDED IMPORTS ---


# === ADDED FOR THE INTEGRATION TEST — NOT PART OF THE SERIALISED WORKFLOW ===
async def _ensure_inner_workflow_registered() -> None:
    """Register the co-located inner echo workflow under its bare registry key.

    This middle workflow references ``subflow_inner_echo`` by its bare key TWICE (the two
    ``ImportWorkflowAsReferencedSubFlowRequest`` calls below). When the middle is itself imported
    as a referenced sub-flow, ``WorkflowManager.run_workflow`` execs this module and awaits
    ``build_workflow()``, so the inner workflow must be registered here before those imports run.
    It lives next to this file rather than in the engine workspace, so we register the bare key
    (the stable stem the serialisation expects) against the absolute path resolved from __file__.
    """
    if WorkflowRegistry.has_workflow_with_name("subflow_inner_echo"):
        return
    inner_path = Path(__file__).parent / "subflow_inner_echo.py"
    meta = await GriptapeNodes.ahandle_request(LoadWorkflowMetadata(file_name=str(inner_path)))
    if not isinstance(meta, LoadWorkflowMetadataResultSuccess):
        msg = f"Failed to load inner workflow metadata from '{inner_path}': {meta.result_details}"
        raise RuntimeError(msg)
    WorkflowRegistry.generate_new_workflow(
        registry_key="subflow_inner_echo", metadata=meta.metadata, file_path=str(inner_path)
    )


# === END ADDED SECTION ===


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # === ADDED FOR THE INTEGRATION TEST — register the co-located inner workflow before the
    # two ImportWorkflowAsReferencedSubFlowRequest calls (and the nodes) that reference it. ===
    await _ensure_inner_workflow_registered()
    # === END ADDED SECTION ===
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {
        "a5ba7020-4296-4d1a-80c3-7944857b18c1": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "280817bc-b91e-486c-9c8f-1506be65cbc1": pickle.loads(b"\x80\x04\x89."),
        "c3aa9af3-06c1-4f1b-b618-df8c83f283d4": pickle.loads(
            b"\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00\x8c\x12subflow_inner_echo\x94."
        ),
    }
    # Create the Flow, then do work within it as context.
    flow0_name = (
        await GriptapeNodes.ahandle_request(
            CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
        )
    ).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="StartFlow",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Start Flow",
                    metadata={
                        "library_node_metadata": {
                            "category": "workflows",
                            "description": "Define the start of a workflow and pass parameters into the flow",
                            "display_name": "Start Flow",
                            "tags": ["workflow", "execution"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "StartFlow",
                        "showaddparameter": True,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="a",
                    default_value="",
                    tooltip="workflow input",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"is_custom": True, "is_user_added": True},
                    parent_container_name="",
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="b",
                    default_value="",
                    tooltip="workflow input",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"is_custom": True, "is_user_added": True},
                    parent_container_name="",
                    initial_setup=True,
                )
            )
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="EndFlow",
                    specific_library_name="Griptape Nodes Library",
                    node_name="End Flow",
                    metadata={
                        "library_node_metadata": {
                            "category": "workflows",
                            "description": "Define the end of a workflow and return parameters from the flow",
                            "display_name": "End Flow",
                            "tags": ["workflow", "execution"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "EndFlow",
                        "showaddparameter": True,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="out_a",
                    default_value="",
                    tooltip="workflow output",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"is_custom": True, "is_user_added": True},
                    parent_container_name="",
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="out_b",
                    default_value="",
                    tooltip="workflow output",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"is_custom": True, "is_user_added": True},
                    parent_container_name="",
                    initial_setup=True,
                )
            )
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="SubflowWorkflowNode",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Workflow A",
                    metadata={
                        "library_node_metadata": {
                            "category": "execution_flow",
                            "description": "Selects and executes a registered workflow, routing inputs and outputs via the workflow shape",
                            "display_name": "Workflow Node",
                            "tags": ["workflow", "execution", "subflow"],
                            "icon": "Layers",
                            "color": None,
                            "group": None,
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "SubflowWorkflowNode",
                        "workflow_node": True,
                        "workflow_shape_params": ["text_out", "text_in"],
                        "left_parameters": ["text_in"],
                        "right_parameters": ["text_out"],
                        "_workflow_file_value": "subflow_inner_echo",
                        "subflow_name": "ControlFlow_2",
                        "_subflow_workflow": "subflow_inner_echo",
                        "subflow_owner_uuid": "94ee3e95-6ad3-4488-9e03-68a6d52a6ec1",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="text_in", default_value="", tooltip="workflow input", initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="text_out", default_value="", tooltip="workflow output", initial_setup=True
                )
            )
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="SubflowWorkflowNode",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Workflow B",
                    metadata={
                        "library_node_metadata": {
                            "category": "execution_flow",
                            "description": "Selects and executes a registered workflow, routing inputs and outputs via the workflow shape",
                            "display_name": "Workflow Node",
                            "tags": ["workflow", "execution", "subflow"],
                            "icon": "Layers",
                            "color": None,
                            "group": None,
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "SubflowWorkflowNode",
                        "workflow_node": True,
                        "workflow_shape_params": ["text_out", "text_in"],
                        "left_parameters": ["text_in"],
                        "right_parameters": ["text_out"],
                        "_workflow_file_value": "subflow_inner_echo",
                        "subflow_name": "ControlFlow_3",
                        "_subflow_workflow": "subflow_inner_echo",
                        "subflow_owner_uuid": "020a4d8f-1a36-426f-a71b-f0e147f06b9f",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="text_in", default_value="", tooltip="workflow input", initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="text_out", default_value="", tooltip="workflow output", initial_setup=True
                )
            )
        # Serialiser emits `flowN_name = (await ...).created_flow_name`; the returned names are
        # unused here (each Workflow node rebinds its own subflow by owner UUID at run time), so we
        # keep only the side-effecting imports to satisfy ruff (F841). imported_flow_metadata carries
        # each node's owner UUID so the recreated inner flows are stamped and can be rebound.
        await GriptapeNodes.ahandle_request(
            ImportWorkflowAsReferencedSubFlowRequest(
                workflow_name="subflow_inner_echo",
                imported_flow_metadata={"subflow_owner_uuid": "94ee3e95-6ad3-4488-9e03-68a6d52a6ec1"},
            )
        )
        await GriptapeNodes.ahandle_request(
            ImportWorkflowAsReferencedSubFlowRequest(
                workflow_name="subflow_inner_echo",
                imported_flow_metadata={"subflow_owner_uuid": "020a4d8f-1a36-426f-a71b-f0e147f06b9f"},
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node2_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="exec_out",
                target_node_name=node3_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="exec_out",
                target_node_name=node1_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="a",
                target_node_name=node2_name,
                target_parameter_name="text_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="text_out",
                target_node_name=node1_name,
                target_parameter_name="out_a",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="b",
                target_node_name=node3_name,
                target_parameter_name="text_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="text_out",
                target_node_name=node1_name,
                target_parameter_name="out_b",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="a",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a5ba7020-4296-4d1a-80c3-7944857b18c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="b",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a5ba7020-4296-4d1a-80c3-7944857b18c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["280817bc-b91e-486c-9c8f-1506be65cbc1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="workflow_file",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["c3aa9af3-06c1-4f1b-b618-df8c83f283d4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["280817bc-b91e-486c-9c8f-1506be65cbc1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text_out",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a5ba7020-4296-4d1a-80c3-7944857b18c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="workflow_file",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["c3aa9af3-06c1-4f1b-b618-df8c83f283d4"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["280817bc-b91e-486c-9c8f-1506be65cbc1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text_out",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["a5ba7020-4296-4d1a-80c3-7944857b18c1"],
                    initial_setup=True,
                    is_output=False,
                )
            )


async def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_request = GetTopLevelFlowRequest()
        top_level_flow_result = await GriptapeNodes.ahandle_request(top_level_flow_request)
        if (
            isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess)
            and top_level_flow_result.flow_name is not None
        ):
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(input: dict, *, workflow_executor: WorkflowExecutor | None = None, **kwargs: Any) -> dict | None:
    return asyncio.run(aexecute_workflow(input=input, workflow_executor=workflow_executor, **kwargs))


async def aexecute_workflow(
    input: dict, *, workflow_executor: WorkflowExecutor | None = None, **kwargs: Any
) -> dict | None:
    await build_workflow()
    await _ensure_workflow_context()
    if workflow_executor is None:
        kwargs.setdefault("pickle_control_flow_result", False)
        workflow_executor = LocalWorkflowExecutor(skip_library_loading=True, workflows_to_register=[__file__], **kwargs)
    async with workflow_executor as executor:
        await executor.arun(flow_input=input, **kwargs)
    return executor.output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    LocalWorkflowExecutor.add_cli_arguments(parser, pickle_control_flow_result_default=False)
    parser.add_argument(
        "--json-input",
        default=None,
        help="JSON string containing parameter values. Takes precedence over individual parameter arguments if provided.",
    )
    parser.add_argument(
        "--exec_out", dest="exec_out", default=None, help="Connection to the next node in the execution chain"
    )
    parser.add_argument("--a", dest="a", default=None, help="workflow input")
    parser.add_argument("--b", dest="b", default=None, help="workflow input")
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.exec_out is not None:
            flow_input["Start Flow"]["exec_out"] = args.exec_out
        if args.a is not None:
            flow_input["Start Flow"]["a"] = args.a
        if args.b is not None:
            flow_input["Start Flow"]["b"] = args.b
    executor = LocalWorkflowExecutor.from_cli_args(args, skip_library_loading=True, workflows_to_register=[__file__])
    workflow_output = execute_workflow(input=flow_input, workflow_executor=executor)
    print(workflow_output)
