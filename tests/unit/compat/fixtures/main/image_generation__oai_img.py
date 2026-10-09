# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_image_generation__oai_img"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "GenerateImage"], ["Griptape Nodes Library", "OpenAiImage"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:31:20.113608Z
# last_modified_date = 2026-10-07T00:31:20.115838Z
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
    top_level_unique_values_dict = {'f49d97de-5561-4a16-835b-b410b1c304f1': pickle.loads(b'\x80\x04\x95\x96\x00\x00\x00\x00\x00\x00\x00\x8c\x92\xe2\x9a\xa0\xef\xb8\x8f This node requires an API key from OpenAI\nPlease visit https://platform.openai.com/api-keys to obtain a valid key and update your settings.\x94.'), 'b7caa0d9-2b7c-4990-8747-dc69d835c5bf': pickle.loads(b'\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0bgpt-image-1\x94.'), '86bdda36-8dc5-488d-b647-2b2083239f0c': pickle.loads(b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\t1024x1024\x94.'), '3e80e3f5-1c3d-4b04-910d-649791ef79bc': pickle.loads(b'\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00\x8c\x05vivid\x94.'), '606c0149-6f16-4012-a6b1-149e82a37bd7': pickle.loads(b'\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03low\x94.'), '25b87d38-3bc4-45f7-9de0-bfc4263398c7': pickle.loads(b'\x80\x04\x95\x0f\x00\x00\x00\x00\x00\x00\x00\x8c\x0btransparent\x94.'), 'cb8333c5-45f1-4a26-ac8b-57a24cc453b7': pickle.loads(b'\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03png\x94.'), '617d8899-899c-4f74-b351-0542ee668494': pickle.loads(b'\x80\x04KP.'), '32b15bb5-bb77-46d5-af21-d5a1e5c31ea3': pickle.loads(b'\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x17A flat blue circle icon\x94.'), 'cbf19e6c-da54-4e6d-a2cb-ef1d778fddf9': pickle.loads(b'\x80\x04\x89.'), '320e9bf5-39c9-4eef-a96a-a140bef0b677': pickle.loads(b'\x80\x04\x95\n\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c b2804e47d25c44d0b77efbfd74c2d414\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c${outputs}/images/oai_circle_v001.png\x94ub.'), '9edc4b32-6170-4019-b4b3-2bdccce23f4f': pickle.loads(b'\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eoai_circle.png\x94.'), '906ad26b-477d-476e-986a-641b807f97ef': pickle.loads(b'\x80\x04\x95=\x00\x00\x00\x00\x00\x00\x00\x8c9Prompt enhancement disabled.\nStarting processing image..\n\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='OpenAiImage', specific_library_name='Griptape Nodes Library', node_name='oai_img', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'image/image_models', 'description': 'Image Generation Models available from OpenAI', 'display_name': 'Open Ai Image', 'tags': ['image', 'api', 'driver', 'config', 'openai'], 'icon': {'light': 'logos/openai_dark.svg', 'dark': 'logos/openai.svg'}, 'color': None, 'group': 'create', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_image_1', 'openai_dall_e_3', 'openai_dall_e_2']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'OpenAiImage'}, initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='message', default_value='⚠️ This node requires an API key from OpenAI\nPlease visit https://platform.openai.com/api-keys to obtain a valid key and update your settings.', initial_setup=True))
        node1_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='GenerateImage', specific_library_name='Griptape Nodes Library', node_name='gen_oai', metadata={'position': {'x': 400, 'y': 0}, 'library_node_metadata': {'category': 'image', 'description': 'Generates an image using Griptape Cloud, or other provided image generation models', 'display_name': 'Generate Image', 'tags': ['image', 'generation', 'ai'], 'icon': None, 'color': None, 'group': 'create', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_image_1_mini', 'gtc_gpt_image_1_5', 'gtc_gpt_4o']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'GenerateImage'}, initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='model', type='Image Generation Driver', output_type='Image Generation Driver', mode_allowed_property=False, ui_options={'display_name': 'image_model_config', 'data': [{'name': 'gpt-image-1-mini', 'label': 'GPT Image 1 Mini'}, {'name': 'gpt-image-1.5', 'label': 'GPT Image 1.5'}], 'dropdown_row_icons': True, 'dropdown_row_subtitles': True}, traits=[{'trait_name': 'Button', 'trait_module': 'griptape_nodes.traits.button', 'trait_state': {'icon_position': 'left'}}], initial_setup=True))
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='image_size', ui_options={'hide_label': False, 'hide_property': False, 'hide': True}, initial_setup=True))
        await GriptapeNodes.ahandle_request(CreateConnectionRequest(source_node_name=node0_name, source_parameter_name='image_model_config', target_node_name=node1_name, target_parameter_name='model', initial_setup=True))
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='message', node_name=node0_name, value=top_level_unique_values_dict['f49d97de-5561-4a16-835b-b410b1c304f1'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['b7caa0d9-2b7c-4990-8747-dc69d835c5bf'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='image_size', node_name=node0_name, value=top_level_unique_values_dict['86bdda36-8dc5-488d-b647-2b2083239f0c'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='style', node_name=node0_name, value=top_level_unique_values_dict['3e80e3f5-1c3d-4b04-910d-649791ef79bc'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='quality', node_name=node0_name, value=top_level_unique_values_dict['606c0149-6f16-4012-a6b1-149e82a37bd7'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='background', node_name=node0_name, value=top_level_unique_values_dict['25b87d38-3bc4-45f7-9de0-bfc4263398c7'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='moderation', node_name=node0_name, value=top_level_unique_values_dict['606c0149-6f16-4012-a6b1-149e82a37bd7'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_format', node_name=node0_name, value=top_level_unique_values_dict['cb8333c5-45f1-4a26-ac8b-57a24cc453b7'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_compression', node_name=node0_name, value=top_level_unique_values_dict['617d8899-899c-4f74-b351-0542ee668494'], initial_setup=True, is_output=False))
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node1_name, value=top_level_unique_values_dict['32b15bb5-bb77-46d5-af21-d5a1e5c31ea3'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='image_size', node_name=node1_name, value=top_level_unique_values_dict['86bdda36-8dc5-488d-b647-2b2083239f0c'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='enhance_prompt', node_name=node1_name, value=top_level_unique_values_dict['cbf19e6c-da54-4e6d-a2cb-ef1d778fddf9'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node1_name, value=top_level_unique_values_dict['320e9bf5-39c9-4eef-a96a-a140bef0b677'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_file', node_name=node1_name, value=top_level_unique_values_dict['9edc4b32-6170-4019-b4b3-2bdccce23f4f'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='include_details', node_name=node1_name, value=top_level_unique_values_dict['cbf19e6c-da54-4e6d-a2cb-ef1d778fddf9'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='logs', node_name=node1_name, value=top_level_unique_values_dict['906ad26b-477d-476e-986a-641b807f97ef'], initial_setup=True, is_output=True))
