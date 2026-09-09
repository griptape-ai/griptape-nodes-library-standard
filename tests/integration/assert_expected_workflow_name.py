# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "assert_expected_workflow_name"
# schema_version = "0.20.0"
# engine_version_created_with = "0.100.0"
# node_libraries_referenced = [["Griptape Nodes Testing Library", "0.1.0"], ["Griptape Nodes Library", "0.85.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "ResolveMacroPath"], ["Griptape Nodes Library", "StartFlow"], ["Griptape Nodes Testing Library", "AssertStrings"]]
# is_griptape_provided = false
# is_template = false
# is_internal = false
# creation_date = 2026-09-01T14:32:48.988423Z
# last_modified_date = 2026-09-03T12:30:29.294883Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":false,\"mode_allowed_output\":true,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":true,\"mode_allowed_output\":false,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"was_successful_1\":{\"name\":\"was_successful_1\",\"tooltip\":\"New parameter\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null},\"result_details_1\":{\"name\":\"result_details_1\",\"tooltip\":\"New parameter\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details on the assertion will appear here.\",\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
import pickle
from typing import Any

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
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
    AlterParameterGroupDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Nodes Testing Library", perform_discovery_if_not_found=True
        )
    )
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {
        "d10e3820-ca44-4163-b82a-f13494c79bc5": pickle.loads(
            b"\x80\x04\x95\x13\x00\x00\x00\x00\x00\x00\x00\x8c\x0f{workflow_name}\x94."
        ),
        "4ef19411-23fb-4874-9e0a-64081cda6c18": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "101512df-c730-43c9-a11d-9aedc47cde50": pickle.loads(b"\x80\x04\x89."),
        "2897a660-25d8-4c3a-bc6f-69d8b8d8426f": pickle.loads(
            b"\x80\x04\x95\x1a\x00\x00\x00\x00\x00\x00\x00\x8c\x16expected_workflow_name\x94."
        ),
        "05d187b6-653e-4a92-aa5e-a11636df84d3": pickle.loads(
            b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tends_with\x94."
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
                        "position": {"x": 19.999999999999943, "y": 448.3333333333334},
                        "tempId": "placing-1788273177445-5aoeu",
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
                        "size": {"width": 600, "height": 190},
                        "publish_config": {"publish_output_directory": "/tmp"},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="ResolveMacroPath",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Resolve Macro Path",
                    metadata={
                        "library_node_metadata": {
                            "category": "files",
                            "description": "Resolve a macro path to an absolute filesystem path (e.g. {inputs}/file.txt → /home/user/project/inputs/file.txt).",
                            "display_name": "Resolve Macro Path",
                            "tags": ["file", "macro", "path"],
                            "icon": "FolderSearch",
                            "color": None,
                            "group": None,
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "ResolveMacroPath",
                        "position": {"x": 775.0000000000001, "y": 436.6666666666668},
                        "size": {"width": 600, "height": 388},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="AssertStrings",
                    specific_library_name="Griptape Nodes Testing Library",
                    node_name="Assert Strings",
                    metadata={
                        "library_node_metadata": {
                            "category": "assert",
                            "description": "Asserts a string comparison using a selected operator.",
                            "display_name": "Assert Strings",
                            "tags": None,
                            "icon": "ShieldCheck",
                            "color": None,
                            "group": "assert",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Testing Library",
                        "node_type": "AssertStrings",
                        "position": {"x": 1654.5911072668646, "y": 448.3333333333334},
                        "size": {"width": 600, "height": 476},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterGroupDetailsRequest(
                    group_name="Status", ui_options={"collapsed": False}, initial_setup=True
                )
            )
        node3_name = (
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "EndFlow",
                        "showaddparameter": True,
                        "position": {"x": 2408.5399726944356, "y": 448.3333333333334},
                        "size": {"width": 600, "height": 366},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="was_successful_1",
                    default_value="",
                    tooltip="New parameter",
                    type="bool",
                    input_types=["bool"],
                    output_type="bool",
                    ui_options={"is_custom": True, "is_user_added": True},
                    parent_container_name="",
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="result_details_1",
                    default_value="",
                    tooltip="New parameter",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "multiline": True,
                        "placeholder_text": "Details on the assertion will appear here.",
                        "is_custom": True,
                        "is_user_added": True,
                    },
                    parent_container_name="",
                    initial_setup=True,
                )
            )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node1_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="exec_out",
                target_node_name=node2_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="resolved_path",
                target_node_name=node2_name,
                target_parameter_name="actual",
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
                source_node_name=node2_name,
                source_parameter_name="was_successful",
                target_node_name=node3_name,
                target_parameter_name="was_successful_1",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="result_details",
                target_node_name=node3_name,
                target_parameter_name="result_details_1",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["d10e3820-ca44-4163-b82a-f13494c79bc5"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolved_path",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["4ef19411-23fb-4874-9e0a-64081cda6c18"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["101512df-c730-43c9-a11d-9aedc47cde50"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="expected",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["2897a660-25d8-4c3a-bc6f-69d8b8d8426f"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="operator",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["05d187b6-653e-4a92-aa5e-a11636df84d3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="message",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["4ef19411-23fb-4874-9e0a-64081cda6c18"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["101512df-c730-43c9-a11d-9aedc47cde50"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["101512df-c730-43c9-a11d-9aedc47cde50"],
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
    if workflow_executor is None:
        kwargs.setdefault("pickle_control_flow_result", False)
        workflow_executor = LocalWorkflowExecutor(skip_library_loading=True, workflows_to_register=[__file__], **kwargs)
    async with workflow_executor as executor:
        await build_workflow()
        await _ensure_workflow_context()
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
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.exec_out is not None:
            flow_input["Start Flow"]["exec_out"] = args.exec_out
    executor = LocalWorkflowExecutor.from_cli_args(args, skip_library_loading=True, workflows_to_register=[__file__])
    workflow_output = execute_workflow(input=flow_input, workflow_executor=executor)
    print(workflow_output)
