# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__date_custom"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "DateAndTime"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:20.353072Z
# last_modified_date = 2026-10-07T00:30:20.354019Z
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
    top_level_unique_values_dict = {'1807b968-cfd3-4f72-9e9a-ebe17ace4e97': pickle.loads(b'\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0echristmas 2031\x94.'), '161dd0dd-6055-48fc-9db5-a7cf0c79c1fc': pickle.loads(b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00\x8c\rCustom format\x94.'), 'a261e089-b6f1-4d24-9c53-fe7683ea1346': pickle.loads(b'\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08%d/%m/%Y\x94.'), '55019409-af37-4164-93e3-94082a6b347d': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94.'), '6a33a351-dba8-499c-8a1b-84821d090a91': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), '7bea664d-7993-402c-bff3-1577155c5d7a': pickle.loads(b'\x80\x04\x95\x0e\x00\x00\x00\x00\x00\x00\x00\x8c\n25/12/2031\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='DateAndTime', specific_library_name='Griptape Nodes Library', node_name='date_custom', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Get the date and time', 'display_name': 'Date and Time', 'tags': ['text', 'utility', 'datetime'], 'icon': 'calendar', 'color': None, 'group': 'tasks', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'DateAndTime'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='custom_format', ui_options={'hide': False, 'placeholder_text': 'any custom format'}, initial_setup=True))
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['1807b968-cfd3-4f72-9e9a-ebe17ace4e97'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='format', node_name=node0_name, value=top_level_unique_values_dict['161dd0dd-6055-48fc-9db5-a7cf0c79c1fc'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='custom_format', node_name=node0_name, value=top_level_unique_values_dict['a261e089-b6f1-4d24-9c53-fe7683ea1346'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['55019409-af37-4164-93e3-94082a6b347d'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['6a33a351-dba8-499c-8a1b-84821d090a91'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['7bea664d-7993-402c-bff3-1577155c5d7a'], initial_setup=True, is_output=True))
