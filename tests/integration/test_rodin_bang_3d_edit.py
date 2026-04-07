# /// script
# dependencies = []
# [tool.griptape-nodes]
# name = "test_rodin_bang_3d_edit"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "RodinBang3DEdit"], ["Griptape Nodes Library", "StartFlow"]]
# is_griptape_provided = false
# is_template = false
# ///
import argparse
import asyncio
import json
import logging
from pathlib import Path

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.drivers.storage.storage_backend import StorageBackend
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    GetTopLevelFlowRequest,
    GetTopLevelFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

GriptapeNodes.handle_request(
    RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
)

context_manager = GriptapeNodes.ContextManager()
if not context_manager.has_current_workflow():
    context_manager.push_workflow(file_path=__file__)

flow_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
).flow_name

with GriptapeNodes.ContextManager().flow(flow_name):
    start_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="StartFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="Start Flow",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(start_node):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="asset_id",
                default_value="",
                tooltip="Asset ID",
                type="str",
                input_types=["any"],
                output_type="str",
                ui_options={"display_name": "Asset ID"},
                parent_container_name="",
                initial_setup=True,
            )
        )
    gen_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="RodinBang3DEdit",
            specific_library_name="Griptape Nodes Library",
            node_name="RodinBang3DEdit",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    end_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="EndFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="End Flow",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(end_node):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="str",
                default_value="",
                tooltip="Result",
                type="str",
                input_types=["str"],
                output_type="str",
                ui_options={},
                parent_container_name="",
                initial_setup=True,
            )
        )

    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=start_node,
            source_parameter_name="asset_id",
            target_node_name=gen_node,
            target_parameter_name="asset_id",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=gen_node,
            source_parameter_name="result_details",
            target_node_name=end_node,
            target_parameter_name="str",
            initial_setup=True,
        )
    )


def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if (
            isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess)
            and top_level_flow_result.flow_name is not None
        ):
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(
    input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False
):
    return asyncio.run(
        aexecute_workflow(
            input=input,
            storage_backend=storage_backend,
            project_file_path=project_file_path,
            workflow_executor=workflow_executor,
            pickle_control_flow_result=pickle_control_flow_result,
        )
    )


async def aexecute_workflow(
    input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False
):
    _ensure_workflow_context()
    storage_backend_enum = StorageBackend(storage_backend)
    project_file_path_resolved = Path(project_file_path) if project_file_path is not None else None
    workflow_executor = workflow_executor or LocalWorkflowExecutor(
        storage_backend=storage_backend_enum,
        project_file_path=project_file_path_resolved,
        skip_library_loading=True,
        workflows_to_register=[__file__],
    )
    async with workflow_executor as executor:
        await executor.arun(flow_input=input, pickle_control_flow_result=pickle_control_flow_result)
    return executor.output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-backend", choices=["local", "gtc"], default="local")
    parser.add_argument("--project-file-path", default=None)
    parser.add_argument("--json-input", default=None)
    parser.add_argument("--asset-id", default=None)
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.asset_id is not None:
            flow_input["Start Flow"]["asset_id"] = args.asset_id
    workflow_output = execute_workflow(
        input=flow_input, storage_backend=args.storage_backend, project_file_path=args.project_file_path
    )
    print(workflow_output)
