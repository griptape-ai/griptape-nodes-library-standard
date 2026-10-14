# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_tasks_text__evaluate"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "EvaluateTextResult"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:30:26.103716Z
# last_modified_date = 2026-10-07T00:30:26.104807Z
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
    top_level_unique_values_dict = {'782de862-138f-4ef4-943f-3d33cf93c3e3': pickle.loads(b'\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11Choose a preset..\x94.'), 'bc7c4e28-201e-4b5b-95ba-af14ea409e90': pickle.loads(b'\x80\x04\x95"\x00\x00\x00\x00\x00\x00\x00\x8c\x1eWhat is the capital of France?\x94.'), 'd94f9224-d2b8-4923-91d8-18cae61283aa': pickle.loads(b'\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05Paris\x94.'), '1f89f89e-378c-4b56-8b7e-5ccd2803f476': pickle.loads(b'\x80\x04\x95#\x00\x00\x00\x00\x00\x00\x00\x8c\x1fThe capital of France is Paris.\x94.'), '72ef1d5a-3698-4a5b-ba90-110c595024c6': pickle.loads(b'\x80\x04\x95E\x00\x00\x00\x00\x00\x00\x00\x8cAIs the actual output factually equivalent to the expected output?\x94.'), '9773d9e3-7ec2-46c6-81ff-5639e9d2d7e5': pickle.loads(b'\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94.'), 'f0bfc01b-547a-4b09-8336-0cd6563b09c4': pickle.loads(b'\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00.'), 'cbf4a0af-e7ab-4036-a94b-50fa18b8c8b5': pickle.loads(b'\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe9\x99\x99\x99\x99\x99\x9a.'), 'efe08bf6-2cf8-4f3c-a8a8-12c58e120443': pickle.loads(b'\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94.'), '0cf071d0-c8b5-476f-9fc3-229f37cab5e3': pickle.loads(b"\x80\x04\x95?\x01\x00\x00\x00\x00\x00\x00X8\x01\x00\x00The actual output correctly identifies Paris as the capital of France, matching the expected output's key information. However, the actual output includes additional wording ('The capital of France is'), which is not present in the expected output, leading to a slight deviation from the expected concise format.\x94.")}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='EvaluateTextResult', specific_library_name='Griptape Nodes Library', node_name='evaluate', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'text', 'description': 'Evaluate the results of some text against some criteria', 'display_name': 'Evaluate Text Result', 'tags': ['text', 'agent', 'ai', 'evaluation'], 'icon': None, 'color': None, 'group': 'describe', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_4_1', 'gtc_gpt_4_1_mini', 'gtc_gpt_4_1_nano', 'gtc_gpt_5']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'EvaluateTextResult'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='Examples', node_name=node0_name, value=top_level_unique_values_dict['782de862-138f-4ef4-943f-3d33cf93c3e3'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='input', node_name=node0_name, value=top_level_unique_values_dict['bc7c4e28-201e-4b5b-95ba-af14ea409e90'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='expected_output', node_name=node0_name, value=top_level_unique_values_dict['d94f9224-d2b8-4923-91d8-18cae61283aa'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='actual_output', node_name=node0_name, value=top_level_unique_values_dict['1f89f89e-378c-4b56-8b7e-5ccd2803f476'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='criteria', node_name=node0_name, value=top_level_unique_values_dict['72ef1d5a-3698-4a5b-ba90-110c595024c6'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['9773d9e3-7ec2-46c6-81ff-5639e9d2d7e5'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='score', node_name=node0_name, value=top_level_unique_values_dict['f0bfc01b-547a-4b09-8336-0cd6563b09c4'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='score', node_name=node0_name, value=top_level_unique_values_dict['cbf4a0af-e7ab-4036-a94b-50fa18b8c8b5'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='reason', node_name=node0_name, value=top_level_unique_values_dict['efe08bf6-2cf8-4f3c-a8a8-12c58e120443'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='reason', node_name=node0_name, value=top_level_unique_values_dict['0cf071d0-c8b5-476f-9fc3-229f37cab5e3'], initial_setup=True, is_output=True))
