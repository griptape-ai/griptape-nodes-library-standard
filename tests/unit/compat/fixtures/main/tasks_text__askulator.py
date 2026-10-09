# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__askulator"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "Askulator"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:11.069289Z
# last_modified_date = 2026-10-07T00:30:11.070111Z
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
    top_level_unique_values_dict = {'1007ce3c-c018-422e-8766-e44ef186e667': pickle.loads(b'\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00\x8c\x12What is 15% of 80?\x94.'), '37b29d1e-004c-49e1-b128-dc201c6558f0': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-nano\x94.'), '1bd34356-a045-479d-834f-33e8407c8cc4': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), 'a727641b-d812-48d3-a4ff-2740ef0c814e': pickle.loads(b'\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x17Using a CalculatorTool\n\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='Askulator', specific_library_name='Griptape Nodes Library', node_name='askulator', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'number', 'description': 'Askulator was once a humble desk calculator in a university math lab. One day, lightning struck the building during a particularly spicy differential equations final. Now imbued with the power of language... and attitude... it nodes among us. Solving math, interpreting riddles, and offering just a hint of sarcasm.', 'display_name': 'Askulator', 'tags': ['text', 'agent', 'math', 'fun'], 'icon': 'message-circle-question-mark', 'color': None, 'group': 'tasks', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'Askulator'}, initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='instruction', node_name=node0_name, value=top_level_unique_values_dict['1007ce3c-c018-422e-8766-e44ef186e667'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['37b29d1e-004c-49e1-b128-dc201c6558f0'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='result', node_name=node0_name, value=top_level_unique_values_dict['1bd34356-a045-479d-834f-33e8407c8cc4'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['a727641b-d812-48d3-a4ff-2740ef0c814e'], initial_setup=True, is_output=True))
