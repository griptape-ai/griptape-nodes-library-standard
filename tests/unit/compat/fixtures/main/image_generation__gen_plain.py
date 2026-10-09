# /// script
# dependencies = []
# 
# [tool.griptape-nodes]
# name = "compat_image_generation__gen_plain"
# schema_version = "0.20.0"
# engine_version_created_with = "0.103.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "GenerateImage"]]
# is_griptape_provided = false
# is_internal = false
# creation_date = 2026-10-07T00:32:04.993359Z
# last_modified_date = 2026-10-07T00:32:04.994599Z
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
    top_level_unique_values_dict = {'f82e2468-b59a-4f05-9db4-5cb6ff9a3eaf': pickle.loads(b'\x80\x04\x95&\n\x00\x00\x00\x00\x00\x00}\x94(\x8c\x05agent\x94}\x94(\x8c\x04type\x94\x8c\x12GriptapeNodesAgent\x94\x8c\x02id\x94\x8c a6f551872cd04e40b373bd716c9ea8dc\x94\x8c\x13conversation_memory\x94}\x94(h\x03\x8c\x12ConversationMemory\x94\x8c\x04runs\x94]\x94(}\x94(h\x03\x8c\x03Run\x94h\x05\x8c 611656980c2f43c499152ca315df2728\x94\x8c\x04meta\x94N\x8c\x05input\x94}\x94(h\x03\x8c\x0cListArtifact\x94h\x05\x8c 38c903b0fb72477b894369ba2ed7b1fb\x94\x8c\treference\x94Nh\x0f}\x94\x8c\x04name\x94h\x13\x8c\x05value\x94]\x94(}\x94(h\x03\x8c\x0cTextArtifact\x94h\x05\x8c 6537787f5c464d7d8ffafea939fa8301\x94h\x14Nh\x0f}\x94h\x16h\x1bh\x17Xc\x02\x00\x00\nEnhance the following prompt for an image generation engine. Return only the image generation prompt.\nInclude unique details that make the subject stand out.\nSpecify a specific depth of field, and time of day.\nUse dust in the air to create a sense of depth.\nUse a slight vignetting on the edges of the image.\nUse a color palette that is complementary to the subject.\nFocus on qualities that will make this the most professional looking photo in the world.\nIMPORTANT: Output must be a single, raw prompt string for an image generation model. Do not include any preamble, explanation, or conversational language.\x94u}\x94(h\x03h\x1ah\x05\x8c eb8723e5e6ef4d79a7a52f28f28be716\x94h\x14Nh\x0f}\x94h\x16h\x1fh\x17\x8c\x14\nUser:\nA yellow star\x94ue\x8c\x0eitem_separator\x94\x8c\x02\n\n\x94\x8c\x16validate_uniform_types\x94\x89u\x8c\x06output\x94}\x94(h\x03h\x1ah\x05\x8c 18454b509ff44345a842e6678aa6b57d\x94h\x14Nh\x0f}\x94\x8c\x0fis_react_prompt\x94\x89sh\x16h\'h\x17X\x83\x02\x00\x00A radiant, glowing yellow star with intricate surface textures and solar flares, suspended in the vastness of space, surrounded by faint cosmic dust particles that create a sense of depth and scale; the scene is captured with a shallow depth of field, softly blurring the distant stars in the background. The time of day is represented as a celestial twilight, with a complementary color palette of deep indigos and soft purples contrasting against the warm golden hues of the star. Subtle dust in the air enhances the dimensionality, while a gentle vignette frames the composition, drawing the viewer\'s eye to the luminous star at the center.\x94uu}\x94(h\x03h\rh\x05\x8c 4acf83bad20c4c12bbf5de60572b11a7\x94h\x0fNh\x10}\x94(h\x03h\x1ah\x05\x8c f855c7e9bd53470b887a3ea539cfd394\x94h\x14Nh\x0f}\x94h\x16h.h\x17\x8c\rA yellow star\x94uh%}\x94(h\x03h\x1ah\x05\x8c d7dd26fe72824267b6265bb7004a8f9d\x94h\x14Nh\x0f}\x94h\x16h2h\x17\x8csI created an image based on your prompt.\n<THOUGHT>\nmeta={"used_tool": True, "tool": "GenerateImageTool"}\n</THOUGHT>\x94uueh\x0f}\x94\x8c\x08max_runs\x94Nu\x8c\x1cconversation_memory_strategy\x94\x8c\rper_structure\x94\x8c\x05tasks\x94]\x94}\x94(h\x03\x8c\nPromptTask\x94h\x05\x8c 05b246dde9814e388973d068786f6a36\x94\x8c\x05state\x94\x8c\x0eState.FINISHED\x94\x8c\nparent_ids\x94]\x94\x8c\tchild_ids\x94]\x94\x8c\x17max_meta_memory_entries\x94K\x14\x8c\x07context\x94}\x94\x8c\rprompt_driver\x94}\x94(h\x03\x8c\x19GriptapeCloudPromptDriver\x94\x8c\x0btemperature\x94G?\xb9\x99\x99\x99\x99\x99\x9a\x8c\nmax_tokens\x94N\x8c\x06stream\x94\x88\x8c\x0cextra_params\x94}\x94\x8c\x05model\x94\x8c\x06gpt-4o\x94\x8c\x1astructured_output_strategy\x94\x8c\x06native\x94u\x8c\x05tools\x94]\x94\x8c\x0cmax_subtasks\x94K\x14uauhS]\x94\x8c\x08rulesets\x94]\x94u.'), 'a69e2327-f33b-47b8-ab91-ac99d95bcff5': pickle.loads(b'\x80\x04\x95\x14\x00\x00\x00\x00\x00\x00\x00\x8c\x10gpt-image-1-mini\x94.'), '09e23f1f-095c-43cb-8a1f-f0411f00c745': pickle.loads(b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00\x8c\rA yellow star\x94.'), 'bc7b272f-e2f8-4cfc-a7d7-07e3c428452a': pickle.loads(b'\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\t1536x1024\x94.'), '22ae1e2d-fd98-4811-9905-12feaa0a571e': pickle.loads(b'\x80\x04\x88.'), '9e66fdb5-d792-453f-b3d2-742de8ecb062': pickle.loads(b'\x80\x04\x95\t\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94h\x01\x8c\x0bmodule_name\x94h\x00\x8c\x02id\x94\x8c 6a47369075614a8395996f32d47d04f4\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x08\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c#{outputs}/images/generated_v001.png\x94ub.'), '5de6332f-3982-4ada-96c1-034cbda558ac': pickle.loads(b'\x80\x04\x95\x11\x00\x00\x00\x00\x00\x00\x00\x8c\rgenerated.png\x94.'), '4f4a196c-cb57-43e8-b2df-7a9b542e4394': pickle.loads(b'\x80\x04\x95l\x00\x00\x00\x00\x00\x00\x00\x8chEnhancing prompt...\nFinished enhancing prompt...\nStarting processing image..\nFinished processing image.\n\x94.')}
    # Create the Flow, then do work within it as context.
    flow0_name = (await GriptapeNodes.ahandle_request(CreateFlowRequest(parent_flow_name=None, flow_name='ControlFlow_1', set_as_new_context=False, metadata={}))).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (await GriptapeNodes.ahandle_request(CreateNodeRequest(node_type='GenerateImage', specific_library_name='Griptape Nodes Library', node_name='gen_plain', metadata={'position': {'x': 0, 'y': 0}, 'library_node_metadata': {'category': 'image', 'description': 'Generates an image using Griptape Cloud, or other provided image generation models', 'display_name': 'Generate Image', 'tags': ['image', 'generation', 'ai'], 'icon': None, 'color': None, 'group': 'create', 'deprecation': None, 'is_node_group': None, 'declarations': [{'type': 'model_usage', 'model_ids': ['gtc_gpt_image_1_mini', 'gtc_gpt_image_1_5', 'gtc_gpt_4o']}]}, 'library': 'Griptape Nodes Library', 'node_type': 'GenerateImage'}, resolution='resolved', initial_setup=True))).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(AlterParameterDetailsRequest(parameter_name='output_file', traits=[{'trait_name': 'FileSystemPicker', 'trait_module': 'griptape_nodes.traits.file_system_picker', 'trait_state': {}}, {'trait_name': 'Button', 'trait_module': 'griptape_nodes.traits.button', 'trait_state': {}}], initial_setup=True))
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='agent', node_name=node0_name, value=top_level_unique_values_dict['f82e2468-b59a-4f05-9db4-5cb6ff9a3eaf'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='model', node_name=node0_name, value=top_level_unique_values_dict['a69e2327-f33b-47b8-ab91-ac99d95bcff5'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='prompt', node_name=node0_name, value=top_level_unique_values_dict['09e23f1f-095c-43cb-8a1f-f0411f00c745'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='image_size', node_name=node0_name, value=top_level_unique_values_dict['bc7b272f-e2f8-4cfc-a7d7-07e3c428452a'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='enhance_prompt', node_name=node0_name, value=top_level_unique_values_dict['22ae1e2d-fd98-4811-9905-12feaa0a571e'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output', node_name=node0_name, value=top_level_unique_values_dict['9e66fdb5-d792-453f-b3d2-742de8ecb062'], initial_setup=True, is_output=True))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='output_file', node_name=node0_name, value=top_level_unique_values_dict['5de6332f-3982-4ada-96c1-034cbda558ac'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='include_details', node_name=node0_name, value=top_level_unique_values_dict['22ae1e2d-fd98-4811-9905-12feaa0a571e'], initial_setup=True, is_output=False))
            await GriptapeNodes.ahandle_request(SetParameterValueRequest(parameter_name='logs', node_name=node0_name, value=top_level_unique_values_dict['4f4a196c-cb57-43e8-b2df-7a9b542e4394'], initial_setup=True, is_output=True))
