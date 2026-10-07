# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__random"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "RandomText"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:41.352523Z
# last_modified_date = 2026-10-07T00:30:41.353486Z
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
    top_level_unique_values_dict = {'7a603132-8d22-426a-b47a-0381368c1281': pickle.loads(b'\x80\x04\x95\x1a\x00\x00\x00\x00\x00\x00\x00\x8c\x16One. Two. Three. Four.\x94.'), '5465e798-40f8-4346-be12-01f754e95a7f': pickle.loads(b'\x80\x04\x89.'), 'ef77af26-64ba-4b94-a445-04802e498300': pickle.loads(b'\x80\x04K\x07.'), 'a9de2e13-65a2-4384-b473-71c9586efd94': pickle.loads(b'\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08sentence\x94.'), '08e671b3-0148-4764-9328-835ca701dce5': pickle.loads(b'\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05Three\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='RandomText', specific_library_name='Griptape Nodes Library', node_name='random', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Selects a random character, word, sentence, or paragraph from input text, or generates random content if no input is provided.', 'display_name': 'Random Text', 'tags': ['text', 'random'], 'icon': 'dices', 'color': None, 'group': 'tasks', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1_nano']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'RandomText'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='input_text', node_name=node0_name, value=top_level_unique_values_dict['7a603132-8d22-426a-b47a-0381368c1281'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='randomize_seed', node_name=node0_name, value=top_level_unique_values_dict['5465e798-40f8-4346-be12-01f754e95a7f'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='seed', node_name=node0_name, value=top_level_unique_values_dict['ef77af26-64ba-4b94-a445-04802e498300'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='selection_type', node_name=node0_name, value=top_level_unique_values_dict['a9de2e13-65a2-4384-b473-71c9586efd94'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['08e671b3-0148-4764-9328-835ca701dce5'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['08e671b3-0148-4764-9328-835ca701dce5'], initial_setup=True, is_output=True))
