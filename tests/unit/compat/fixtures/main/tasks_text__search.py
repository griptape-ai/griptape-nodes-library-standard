# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__search"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "SearchWeb"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:35.061098Z
# last_modified_date = 2026-10-07T00:30:35.062022Z
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
    top_level_unique_values_dict = {'0f339873-ab34-4b49-a170-b47021f5337d': pickle.loads(b'\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eGriptape Nodes\x94.'), 'efa74e86-14b4-4407-a161-363414d37525': pickle.loads(b'\x80\x04\x88.'), 'd9d795aa-8121-4556-9578-24c893d81d4d': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94.'), 'a2e518f7-b76b-47de-84bd-a6a2e7010453': pickle.loads(b'\x80\x04\x95\x0e\x00\x00\x00\x00\x00\x00\x00\x8c\nDuckDuckGo\x94.'), 'ef1e526f-2d99-4298-bb53-f059e2a15ac1': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), '20566585-eb0a-4d37-a791-461647ac4792': pickle.loads(b'\x80\x04\x95\x9c\x02\x00\x00\x00\x00\x00\x00X\x95\x02\x00\x00Griptape Nodes is an intuitive, drag-and-drop interface that allows users to create advanced creative pipelines using graphs, nodes, and flowcharts. It helps streamline workflows by generating tailored prompts from custom image descriptions and supports greater customization by utilizing multiple image generation models. It is designed for real-world creative challenges, enabling users to build workflows that scale from quick experiments to full production pipelines. Griptape Nodes is available as a browser interface or a desktop application for Windows, Mac, or Linux.\n\nYou can find more information on their official site: https://www.griptapenodes.com/\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='SearchWeb', specific_library_name='Griptape Nodes Library', node_name='search', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Search the web for information', 'display_name': 'Search Web', 'tags': ['text', 'web', 'api', 'search'], 'icon': 'binoculars', 'color': None, 'group': 'Input/Output', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'SearchWeb'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['0f339873-ab34-4b49-a170-b47021f5337d'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='summarize', node_name=node0_name, value=top_level_unique_values_dict['efa74e86-14b4-4407-a161-363414d37525'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['d9d795aa-8121-4556-9578-24c893d81d4d'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='search_engine', node_name=node0_name, value=top_level_unique_values_dict['a2e518f7-b76b-47de-84bd-a6a2e7010453'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['ef1e526f-2d99-4298-bb53-f059e2a15ac1'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['20566585-eb0a-4d37-a791-461647ac4792'], initial_setup=True, is_output=True))
