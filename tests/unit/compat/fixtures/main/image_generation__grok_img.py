# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_image_generation__grok_img"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "GenerateImage"], ["Griptape Nodes Library", "GrokImage"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:31:20.323700Z
# last_modified_date = 2026-10-07T00:31:20.325231Z
# 
# ///

import pickle
from griptape.artifacts.image_url_artifact import ImageUrlArtifact
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
    top_level_unique_values_dict = {'fcacb927-f793-47ce-981f-751dbdae6b78': pickle.loads(b'\x80\x04\x955\x00\x00\x00\x00\x00\x00\x00\x8c1\xe2\x9a\xa0\xef\xb8\x8f This node requires an API key to function.\x94.'), '401bf002-b3bd-4163-93f1-1745736467fe': pickle.loads(b'\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11grok-2-image-1212\x94.'), '4fe38d79-9c1f-4f12-8b98-e9ef04295a9b': pickle.loads(b'\x80\x04\x95\x1e\x00\x00\x00\x00\x00\x00\x00\x8c\x1aA flat green triangle icon\x94.'), '520c83da-cb6b-4d8c-bfb6-aeebe0ebb734': pickle.loads(b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\t1024x1024\x94.'), '684dc5a4-d86e-4fab-9532-e27fe73eda75': pickle.loads(b'\x80\x04\x89.'), '08b70c46-c0bf-40cd-9b73-347d6e431444': pickle.loads(b"\x80\x04\x95\r\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c 38d04ff24c0a4bf8b263659e7f2e0689\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c'{outputs}/images/grok_triangle_v001.png\x94ub."), 'f6c913ee-8167-40e8-8b6c-8e891a82ca81': pickle.loads(b'\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11grok_triangle.png\x94.'), '9631b9c3-88f6-4339-89e7-1a64ad00110a': pickle.loads(b'\x80\x04\x95=\x00\x00\x00\x00\x00\x00\x00\x8c9Prompt enhancement disabled.\nStarting processing image..\n\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='GrokImage', specific_library_name='Griptape Nodes Library', node_name='grok_img', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'image/image_models', 'description': 'Image Generation Models available from xAI Grok', 'display_name': 'Grok Image', 'tags': ['image', 'api', 'driver', 'config', 'grok', 'xai'], 'icon': {'light': 'logos/grok_dark.svg', 'dark': 'logos/grok.svg'}, 'color': None, 'group': 'create', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['xai_grok_2_image_1212']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'GrokImage'}, initial_setup=True))).node_name
        node1_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='GenerateImage', specific_library_name='Griptape Nodes Library', node_name='gen_grok', metadata={'position': {'x': 400, 'y': 0}, 'library_node_metadata': {'category': 'image', 'description': 'Generates an image using Griptape Cloud, or other provided image generation models', 'display_name': 'Generate Image', 'tags': ['image', 'generation', 'ai'], 'icon': None, 'color': None, 'group': 'create', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_image_1_mini', 'gtc_gpt_image_1_5', 'gtc_gpt_4o']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'GenerateImage'}, initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='model', type='Image Generation Driver', output_type='Image Generation Driver', mode_allowed_property=False, ui_options={'display_name': 'image_model_config', 'data': [{'name': 'gpt-image-1-mini', 'label': 'GPT Image 1 Mini'}, {'name': 'gpt-image-1.5', 'label': 'GPT Image 1.5'}], 'dropdown_row_icons': True, 'dropdown_row_subtitles': True}, traits=[{'trait_name': 'Button', 'trait_module': 'griptape_nodes.traits.button', 'trait_state': {'icon_position': 'left'}}], initial_setup=True))
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='image_size', ui_options={'hide_label': False, 'hide_property': False, 'hide': True}, initial_setup=True))
        await GriptapeNodes.ahandle_request(CreateConnectionRequest(source_node_name=node0_name, source_parameter_name='image_model_config', target_node_name=node1_name, target_parameter_name='model', initial_setup=True))
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='message', node_name=node0_name, value=top_level_unique_values_dict['fcacb927-f793-47ce-981f-751dbdae6b78'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['401bf002-b3bd-4163-93f1-1745736467fe'], initial_setup=True, is_output=False))
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node1_name, value=top_level_unique_values_dict['4fe38d79-9c1f-4f12-8b98-e9ef04295a9b'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='image_size', node_name=node1_name, value=top_level_unique_values_dict['520c83da-cb6b-4d8c-bfb6-aeebe0ebb734'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='enhance_prompt', node_name=node1_name, value=top_level_unique_values_dict['684dc5a4-d86e-4fab-9532-e27fe73eda75'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node1_name, value=top_level_unique_values_dict['08b70c46-c0bf-40cd-9b73-347d6e431444'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_file', node_name=node1_name, value=top_level_unique_values_dict['f6c913ee-8167-40e8-8b6c-8e891a82ca81'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='include_details', node_name=node1_name, value=top_level_unique_values_dict['684dc5a4-d86e-4fab-9532-e27fe73eda75'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='logs', node_name=node1_name, value=top_level_unique_values_dict['9631b9c3-88f6-4339-89e7-1a64ad00110a'], initial_setup=True, is_output=True))
