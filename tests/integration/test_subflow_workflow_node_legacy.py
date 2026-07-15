# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "test_subflow_workflow_node_legacy"
# schema_version = "0.20.0"
# engine_version_created_with = "0.93.0"
# node_libraries_referenced = [["Griptape Nodes Testing Library", "0.1.0"], ["Griptape Nodes Library", "0.82.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "StartFlow"], ["Griptape Nodes Library", "SubflowWorkflowNode"], ["Griptape Nodes Testing Library", "AssertStrings"]]
# workflows_referenced = ["subflow_middle_echo_legacy"]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-07-15T16:26:41.850456Z
# last_modified_date = 2026-07-15T16:28:06.782714Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":false,\"mode_allowed_output\":true,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":true,\"mode_allowed_output\":false,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
import pickle

# --- ADDED FOR THE INTEGRATION TEST (not emitted by the serialiser) ---
# Used by the referenced-workflow registration block inside build_workflow() below.
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
from griptape_nodes.retained_mode.events.parameter_events import AlterParameterDetailsRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.events.workflow_events import (
    ImportWorkflowAsReferencedSubFlowRequest,
    LoadWorkflowMetadata,
    LoadWorkflowMetadataResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

# --- END ADDED IMPORTS ---


# === ADDED FOR THE INTEGRATION TEST — NOT PART OF THE SERIALISED WORKFLOW ===
# LEGACY variant of test_subflow_workflow_node: the same three-level stack (outer ->
# subflow_middle_echo_legacy -> subflow_inner_echo x2) but serialised on a PRE-UUID engine, so no
# node/flow carries a ``subflow_owner_uuid``. It proves the fix still binds nested subflows correctly
# for workflows saved before the owner-UUID mechanism existed: on the fix branch each node establishes
# a fresh UUID and adopts its recorded (unstamped) subflow via the legacy-claim path — no infinite
# recursion, and the per-branch asserts (a="alpha" -> out_a, b="bravo" -> out_b) still hold.
_REFERENCED_WORKFLOWS = ("subflow_inner_echo", "subflow_middle_echo_legacy")


async def _ensure_referenced_workflows_registered() -> None:
    """Register the co-located referenced workflows under their bare registry keys.

    The serialised outer refers to ``subflow_middle_echo_legacy`` by its bare key, which in turn
    refers to ``subflow_inner_echo`` by its bare key. At test time neither is in the engine workspace
    (they live next to this file), so we register both here. Called from build_workflow() (so the
    build-time import resolves) and again after the executor is entered (entry broadcasts
    AppInitializationComplete -> refresh_workflow_registry -> clear_user_workflows(), which would
    otherwise drop these registrations before the flow runs in the standalone __main__ path).
    """
    for key in _REFERENCED_WORKFLOWS:
        if WorkflowRegistry.has_workflow_with_name(key):
            continue
        path = Path(__file__).parent / f"{key}.py"
        meta = await GriptapeNodes.ahandle_request(LoadWorkflowMetadata(file_name=str(path)))
        if not isinstance(meta, LoadWorkflowMetadataResultSuccess):
            msg = f"Failed to load referenced workflow metadata from '{path}': {meta.result_details}"
            raise RuntimeError(msg)
        WorkflowRegistry.generate_new_workflow(registry_key=key, metadata=meta.metadata, file_path=str(path))


# === END ADDED SECTION ===


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
    # === ADDED FOR THE INTEGRATION TEST — register the co-located referenced workflows before any
    # node or ImportWorkflowAsReferencedSubFlowRequest that resolves them by bare key. ===
    await _ensure_referenced_workflows_registered()
    # === END ADDED SECTION ===
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {
        "df2be1af-e3af-4c24-85ea-6284477c3d6b": pickle.loads(b"\x80\x04\x89."),
        "10a91064-91b5-4977-aa19-dba0d63c7333": pickle.loads(
            b"\x80\x04\x95\x1e\x00\x00\x00\x00\x00\x00\x00\x8c\x1asubflow_middle_echo_legacy\x94."
        ),
        "5856d55d-72f0-42d4-8b20-ea741ff27a05": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05alpha\x94."
        ),
        "1b1146b1-de22-43df-a9d6-c68c86903b10": pickle.loads(
            b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05bravo\x94."
        ),
        "51750017-4791-4916-b2d5-433e606dcfae": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "b3df0b06-0237-4202-8c45-657fe7980756": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00\x8c\x02==\x94."
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
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="SubflowWorkflowNode",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Workflow Node",
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
                        "workflow_shape_params": ["out_b", "b", "a", "out_a"],
                        "left_parameters": ["a", "b"],
                        "right_parameters": ["out_a", "out_b"],
                        "_workflow_file_value": "subflow_middle_echo_legacy",
                        "subflow_name": "ControlFlow_2",
                        "_subflow_workflow": "subflow_middle_echo_legacy",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="a", default_value="", tooltip="workflow input", initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="b", default_value="", tooltip="workflow input", initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="out_a", default_value="", tooltip="workflow output", initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="out_b", default_value="", tooltip="workflow output", initial_setup=True
                )
            )
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="AssertStrings",
                    specific_library_name="Griptape Nodes Testing Library",
                    node_name="Assert A",
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
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="AssertStrings",
                    specific_library_name="Griptape Nodes Testing Library",
                    node_name="Assert B",
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
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        # Serialiser emits `flow1_name = (await ...).created_flow_name`; the returned name is unused,
        # so we keep only the side-effecting import to satisfy ruff (F841). LEGACY: imported_flow_metadata
        # is empty (pre-UUID), so the recreated middle carries no owner UUID.
        await GriptapeNodes.ahandle_request(
            ImportWorkflowAsReferencedSubFlowRequest(
                workflow_name="subflow_middle_echo_legacy", imported_flow_metadata={}
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
                target_node_name=node4_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="exec_out",
                target_node_name=node1_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="out_a",
                target_node_name=node3_name,
                target_parameter_name="actual",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="out_b",
                target_node_name=node4_name,
                target_parameter_name="actual",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["df2be1af-e3af-4c24-85ea-6284477c3d6b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="workflow_file",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["10a91064-91b5-4977-aa19-dba0d63c7333"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["df2be1af-e3af-4c24-85ea-6284477c3d6b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="a",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["5856d55d-72f0-42d4-8b20-ea741ff27a05"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="b",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["1b1146b1-de22-43df-a9d6-c68c86903b10"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="out_a",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["51750017-4791-4916-b2d5-433e606dcfae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="out_b",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["51750017-4791-4916-b2d5-433e606dcfae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="expected",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["5856d55d-72f0-42d4-8b20-ea741ff27a05"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="operator",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["b3df0b06-0237-4202-8c45-657fe7980756"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="message",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["51750017-4791-4916-b2d5-433e606dcfae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["df2be1af-e3af-4c24-85ea-6284477c3d6b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="expected",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["1b1146b1-de22-43df-a9d6-c68c86903b10"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="operator",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["b3df0b06-0237-4202-8c45-657fe7980756"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="message",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["51750017-4791-4916-b2d5-433e606dcfae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["df2be1af-e3af-4c24-85ea-6284477c3d6b"],
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
        # === ADDED FOR THE INTEGRATION TEST — NOT PART OF THE SERIALISED WORKFLOW ===
        # Entering the executor broadcasts AppInitializationComplete -> refresh_workflow_registry
        # -> clear_user_workflows(), which drops the referenced workflows registered during
        # build_workflow() above. Re-register them here so the Workflow node (and the nested import
        # inside subflow_middle_echo_legacy) can resolve them when the flow runs.
        await _ensure_referenced_workflows_registered()
        # === END ADDED SECTION ===
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
