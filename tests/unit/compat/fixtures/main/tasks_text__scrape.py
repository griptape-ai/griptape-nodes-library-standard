# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__scrape"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "ScrapeWeb"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:27.987315Z
# last_modified_date = 2026-10-07T00:30:27.988109Z
# 
# ///

import pickle
from griptape_nodes.node_library.library_registry import IconVariant, NodeDeprecationMetadata, NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, AlterParameterDetailsRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(RegisterLibraryFromFileRequest(library_name='Griptape Nodes Library', perform_discovery_if_not_found=True))
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {'ba3f3f49-6587-400d-8784-84578b5171fe': pickle.loads(b'\x80\x04\x95.\x00\x00\x00\x00\x00\x00\x00\x8c*What is the title of https://example.com ?\x94.'), '48dc3fd0-e66a-4a56-9400-ef1a0a56941d': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94.'), '1edeada9-2f2f-479f-a896-e5c41d3326a3': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), 'de349c46-957e-4e14-92aa-d9fc222c6668': pickle.loads(b'\x80\x04\x95\xa0\x00\x00\x00\x00\x00\x00\x00\x8c\x9cThis domain is for use in documentation examples without needing permission. This is not a service; avoid relying on it for testing and monitoring purposes.\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='ScrapeWeb', specific_library_name='Griptape Nodes Library', node_name='scrape', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Scrape the web for information', 'display_name': 'Scrape Web', 'tags': ['text', 'web', 'api', 'scrape'], 'icon': 'globe', 'color': None, 'group': 'Input/Output', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'ScrapeWeb'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['ba3f3f49-6587-400d-8784-84578b5171fe'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['48dc3fd0-e66a-4a56-9400-ef1a0a56941d'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['1edeada9-2f2f-479f-a896-e5c41d3326a3'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['de349c46-957e-4e14-92aa-d9fc222c6668'], initial_setup=True, is_output=True))
