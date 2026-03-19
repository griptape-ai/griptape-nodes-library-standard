# /// script
# dependencies = []
# [tool.griptape-nodes]
# name = "test_invert_mask"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"], ["Griptape Nodes Testing Library", "0.1.0"]]
# node_types_used = [["Griptape Nodes Testing Library", "AssertFileExists"], ["Griptape Nodes Library", "CreateColorBars"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "InvertMask"], ["Griptape Nodes Library", "ToText"]]
# is_griptape_provided = false
# is_template = false
# ///
import asyncio
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
    source_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="CreateColorBars",
            specific_library_name="Griptape Nodes Library",
            node_name="Create Color Bars",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    gen_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="InvertMask",
            specific_library_name="Griptape Nodes Library",
            node_name="InvertMask",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    to_text_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="ToText",
            specific_library_name="Griptape Nodes Library",
            node_name="To Text",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    assert_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="AssertFileExists",
            specific_library_name="Griptape Nodes Testing Library",
            node_name="Assert File Exists",
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
            source_node_name=source_node,
            source_parameter_name="image",
            target_node_name=gen_node,
            target_parameter_name="input_mask",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=gen_node,
            source_parameter_name="output_mask",
            target_node_name=to_text_node,
            target_parameter_name="from",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=to_text_node,
            source_parameter_name="output",
            target_node_name=assert_node,
            target_parameter_name="file_path",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=assert_node,
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
    workflow_output = execute_workflow(input={})
    print(workflow_output)
