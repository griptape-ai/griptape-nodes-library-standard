# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "/Users/collindutter/Projects/griptape/griptape-nodes-libraries/griptape-nodes-library-standard/workflows/templates/rodin_3d_-_3d_model_from_image_7"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "DescribeImage"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "Flux2ImageGeneration"], ["Griptape Nodes Library", "LoadImage"], ["Griptape Nodes Library", "MergeTexts"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "Rodin23DGeneration"], ["Griptape Nodes Library", "StartFlow"]]
# description = "Generate a 3D model from an image using Hyper3D Rodin, Flux-2 Klein, and OpenAI gpt-4.1-mini."
# image = "https://github.com/griptape-ai/griptape-nodes-library-standard/blob/main/workflows/templates/thumbnail_rodin_3d_-_3d_model_from_image.webp?raw=true"
# is_griptape_provided = false
# is_template = false
# creation_date = 2026-03-12T23:17:40.656985Z
# last_modified_date = 2026-03-12T23:17:48.689231Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"image_url\":{\"name\":\"image_url\",\"tooltip\":\"Enter text/string for image_url.\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"https://images.pdimagearchive.org/collections/posed-portraits-of-19th-century-baseball-stars/4051199982_718521f59f_z.jpg?zz=1?width=438&height=800\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"image\":{\"name\":\"image\",\"tooltip\":\"New parameter\",\"type\":\"ImageUrlArtifact\",\"input_types\":[\"ImageUrlArtifact\",\"ImageArtifact\",\"str\"],\"output_type\":\"ImageUrlArtifact\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"clickable_file_browser\":true,\"expander\":true,\"edit_mask\":true,\"display_name\":\"image\",\"is_custom\":true,\"is_user_added\":true,\"hide\":false},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"model_url\":{\"name\":\"model_url\",\"tooltip\":\"New parameter\",\"type\":\"ThreeDUrlArtifact\",\"input_types\":[\"ThreeDUrlArtifact\"],\"output_type\":\"ThreeDUrlArtifact\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_full_width\":true,\"pulse_on_run\":true,\"display_name\":\"3D Model\",\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null},\"image\":{\"name\":\"image\",\"tooltip\":\"New parameter\",\"type\":\"ImageUrlArtifact\",\"input_types\":[\"ImageUrlArtifact\"],\"output_type\":\"ImageUrlArtifact\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_full_width\":true,\"pulse_on_run\":true,\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null},\"image_1\":{\"name\":\"image_1\",\"tooltip\":\"New parameter\",\"type\":\"ImageUrlArtifact\",\"input_types\":[\"ImageUrlArtifact\"],\"output_type\":\"ImageUrlArtifact\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_full_width\":true,\"pulse_on_run\":true,\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
import pickle
from pathlib import Path

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
from griptape_nodes.drivers.storage.storage_backend import StorageBackend
from griptape_nodes.node_library.library_registry import NodeMetadata
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    GetTopLevelFlowRequest,
    GetTopLevelFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

GriptapeNodes.handle_request(
    RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
)

context_manager = GriptapeNodes.ContextManager()

if not context_manager.has_current_workflow():
    context_manager.push_workflow(file_path=__file__)

"""
1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
   This minimizes the size of the code, especially for large objects like serialized image files.
2. We're using a prefix so that it's clear which Flow these values are associated with.
3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
   them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
   would be difficult to serialize.
"""
top_level_unique_values_dict = {
    "0883576b-1af6-42fa-aaf0-7dedbf9e3c81": pickle.loads(
        b"\x80\x04\x95\xaa\x00\x00\x00\x00\x00\x00\x00\x8c\xa6https://images.pdimagearchive.org/collections/joseph-ducreux-self-portraits/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?width=755&height=800\x94."
    ),
    "cf863f46-9dcb-416c-9f12-ede9bac037c5": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."),
    "5b402711-f794-4900-b296-96c1b24080e1": pickle.loads(
        b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08mask.png\x94."
    ),
    "fc02b9fb-7b1c-488a-b6de-7fdb184b2f81": pickle.loads(
        b"\x80\x04\x95\x9c\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 0e4a94c3cb434c9daa6b8c7c08eeaa6a\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94ub."
    ),
    "b5e60772-ac9b-4469-a32b-90f4c4fcc634": pickle.loads(
        b"\x80\x04\x95\x83\x00\x00\x00\x00\x00\x00\x00\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94."
    ),
    "124dff02-725c-4cf7-88c4-b7550672ab1b": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04none\x94."
    ),
    "e6fb1acd-1638-4708-ba7a-997c835a0801": pickle.loads(b"\x80\x04\x89."),
    "23e3386d-279e-4dcb-a4c1-1715147f37f9": pickle.loads(
        b"\x80\x04\x950\x00\x00\x00\x00\x00\x00\x00\x8c,<Results will appear when the node executes>\x94."
    ),
    "7ed9d558-4fe7-4afd-8bcd-8dd31ce6b1f4": pickle.loads(
        b"\x80\x04\x956\x03\x00\x00\x00\x00\x00\x00X/\x03\x00\x00# Rodin 3D - Create a 3D Model from an Image\n\nThis workflow demonstrates the use of various models to generate a 3D model.\n\nIt first grabs an image from the [Public Domain Image Archive](https://pdimagearchive.org/galleries/all) - a fantastic resource for inspirational and unique images.\n\nThen, uses [OpenAI gpt-4.1-mini](https://platform.openai.com/docs/models/gpt-4.1-mini) to describe the hat in that image.\n\nNext, it uses [Flux-2 Klein](https://bfl.ai/models/flux-2-klein) to render the a nice image of the hat in perspective view, then renders another view of the bottom of the hat. \n\nThese images are pasted to [Hyper3D Rodin](https://hyper3d.ai/) to generate a 3d model.\n\n_Note_: You can provide your own image by simply disconnecting the `image_url` parameter and instead connecting the `image` parameter. \x94."
    ),
    "39526a24-733a-4455-88be-a479b2ce66e1": pickle.loads(
        b"\x80\x04\x95\xad\x00\x00\x00\x00\x00\x00\x00\x8c\xa9# Inputs\n\nProvide the url of a person from the [Public Domain Image Archive](https://pdimagearchive.org/galleries/all), or provide your own image and connect it instead.\x94."
    ),
    "e0b1835c-c802-4bae-984a-9ca175d8b275": pickle.loads(
        b"\x80\x04\x95~\x00\x00\x00\x00\x00\x00\x00\x8czUse OpenAI gpt-4.1-mini to describe the hat from the image so that it can be generated cleanly with Google Nano Banana Pro\x94."
    ),
    "8be46441-f361-4d89-b739-d06ed9f314d7": pickle.loads(
        b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x00\x8c#Merge the text to generate a prompt\x94."
    ),
    "aa43d21e-8d1d-4187-8059-32ef1853afe9": pickle.loads(
        b"\x80\x04\x95}\x00\x00\x00\x00\x00\x00\x00\x8cyCreate a 3d high resolution render of the hat from this image, using information from the description as supporting data.\x94."
    ),
    "d245c88d-2f25-4376-8bf2-bc16c83be627": pickle.loads(
        b"\x80\x04\x95_\x02\x00\x00\x00\x00\x00\x00XX\x02\x00\x00The hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\x94."
    ),
    "1216491f-c966-4819-8b9d-b7851147a452": pickle.loads(
        b"\x80\x04\x95#\x00\x00\x00\x00\x00\x00\x00\x8c\x1flight background, soft shadows.\x94."
    ),
    "5c9ee93f-259a-441e-8a77-fd5320d620ac": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04\\n\\n\x94."
    ),
    "f02f4500-5929-4384-aa19-6f249bb5b275": pickle.loads(
        b"\x80\x04\x95\xfb\x02\x00\x00\x00\x00\x00\x00X\xf4\x02\x00\x00Create a 3d high resolution render of the hat from this image, using information from the description as supporting data.\n\nThe hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\n\nlight background, soft shadows.\x94."
    ),
    "769d542e-6b06-4ed3-ab2c-32a6decce5aa": pickle.loads(
        b"\x80\x04\x95]\x00\x00\x00\x00\x00\x00\x00\x8cYRun the workflow to view the results. You'll get the 3d model, and two images of the hat.\x94."
    ),
    "d4329e31-0185-44ca-9d29-dd3f1a3bab1a": pickle.loads(
        b"\x80\x04\x95\xa9\x00\x00\x00\x00\x00\x00\x00\x8c\xa5Use Flux.2 Klein to generate a 3d perspective view and a 3/4 underside image of the hat. These two images will provide enough detail to Ronin to create the 3d model.\x94."
    ),
    "4162f45e-14eb-47c9-b0a0-8fb64a07996c": pickle.loads(
        b"\x80\x04\x95\xa3\x00\x00\x00\x00\x00\x00\x00\x8c\x9fGenerate a 3d Model with [Hyper 3D Ronin](https://hyper3d.ai/).\n\nThis currently outputs **glb** format, but you can make changes to any parameters you require.\x94."
    ),
    "b765ceb5-5f19-42c7-8d9a-e9fee8c5e396": pickle.loads(
        b"\x80\x04\x95\x84\x06\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04type\x94\x8c\x12GriptapeNodesAgent\x94\x8c\x08rulesets\x94]\x94\x8c\x05rules\x94]\x94\x8c\x02id\x94\x8c 187014a4c2d94dc18e20cba7a539b15a\x94\x8c\x13conversation_memory\x94}\x94(h\x01\x8c\x12ConversationMemory\x94\x8c\x04runs\x94]\x94}\x94(h\x01\x8c\x03Run\x94h\x07\x8c c459b3301d6b4466a21bd59d87edd4e8\x94\x8c\x04meta\x94N\x8c\x05input\x94}\x94(h\x01\x8c\x0cTextArtifact\x94h\x07\x8c 1f09e3f8361e44d5924f0d1180a62bbd\x94\x8c\treference\x94Nh\x11}\x94\x8c\x04name\x94h\x15\x8c\x05value\x94\x8c\xf6Describe the hat from this image. Be specific about the shape and design. Include detailed reference to height, width, and creasing. Include specific details on brim shape, and any dents or textures in the surface.\n\nOutput image description only.\x94u\x8c\x06output\x94}\x94(h\x01h\x14h\x07\x8c 7b2164ded78844b8b7f8de1022d8d6d5\x94h\x16Nh\x11}\x94h\x18h\x1dh\x19XX\x02\x00\x00The hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\x94uuah\x11}\x94\x8c\x08max_runs\x94Nu\x8c\x1cconversation_memory_strategy\x94\x8c\rper_structure\x94\x8c\x05tasks\x94]\x94}\x94(h\x01\x8c\nPromptTask\x94h\x03]\x94h\x05]\x94h\x07\x8c e6e9e656090e42e7be331bbb4be223fa\x94\x8c\x05state\x94\x8c\x0eState.FINISHED\x94\x8c\nparent_ids\x94]\x94\x8c\tchild_ids\x94]\x94\x8c\x17max_meta_memory_entries\x94K\x14\x8c\x07context\x94}\x94\x8c\rprompt_driver\x94}\x94(h\x01\x8c\x19GriptapeCloudPromptDriver\x94\x8c\x0btemperature\x94G?\xb9\x99\x99\x99\x99\x99\x9a\x8c\nmax_tokens\x94N\x8c\x06stream\x94\x89\x8c\x0cextra_params\x94}\x94\x8c\x05model\x94\x8c\x0cgpt-4.1-mini\x94\x8c\x1astructured_output_strategy\x94\x8c\x06native\x94u\x8c\x05tools\x94]\x94\x8c\x0cmax_subtasks\x94K\x14uau."
    ),
    "bc6692c4-4933-4a9f-9880-ce56e82a8ad3": pickle.loads(
        b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94."
    ),
    "ce0c98b1-c1d8-424f-80b9-001857842b3d": pickle.loads(
        b"\x80\x04\x95\xda\x00\x00\x00\x00\x00\x00\x00\x8c\xd6Describe the hat from this image. Be specific about the shape and design. Include detailed reference to height, width, and creasing. Include specific details on brim shape, and any dents or textures in the surface.\x94."
    ),
    "7345ff53-1dd7-41bd-87b5-75cf644e7998": pickle.loads(b"\x80\x04\x88."),
    "19c39e2d-fe35-43a9-a9ad-2f9197b556c1": pickle.loads(
        b"\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11Flux.2 [klein] 9B\x94."
    ),
    "5ed2cc5c-acb5-4ac7-b346-0e9a1ae9e9ea": pickle.loads(
        b"\x80\x04\x95\x9f\x01\x00\x00\x00\x00\x00\x00]\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 0e4a94c3cb434c9daa6b8c7c08eeaa6a\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0b\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94uba."
    ),
    "7d6130f2-7a13-4c0b-b1ef-472287f3ffa6": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M`\x05."),
    "7505748e-d58e-46c5-b5c3-be0fda163ae4": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x03."),
    "9751a66a-cf87-4e3d-be08-03a636b6ff42": pickle.loads(b"\x80\x04K*."),
    "30b21d96-7705-43f2-9b04-9ba959f788f8": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04jpeg\x94."
    ),
    "1ebb3947-30b2-4edd-90bd-283a315e797a": pickle.loads(
        b"\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11least restrictive\x94."
    ),
    "9fd6669e-ed93-424a-9327-5935968aad34": pickle.loads(b"\x80\x04K2."),
    "be42758d-3989-4de2-a71f-5416df342093": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x12\x00\x00\x00\x00\x00\x00."
    ),
    "68158fbd-9d19-46c1-9598-6055af2f88bc": pickle.loads(
        b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eflux_image.jpg\x94."
    ),
    "12d6e668-c76c-4162-8da7-b47253cf0a9c": pickle.loads(
        b"\x80\x04\x95g\x00\x00\x00\x00\x00\x00\x00\x8ccsame hat viewed from a worm's eye 3/4 angle revealing the underside. light background, soft shadows\x94."
    ),
    "6c9d998c-20e3-4456-8729-bbc51ac61991": pickle.loads(b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00]\x94]\x94a."),
    "1a5950a4-235f-48fe-b444-28984695f6a3": pickle.loads(b"\x80\x04]\x94."),
    "e4590a94-f430-478d-9224-0d61f670704a": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03hat\x94."
    ),
    "f96208af-4eeb-4242-9d15-0e48b6143e20": pickle.loads(
        b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00]\x94(]\x94]\x94e."
    ),
    "f9099acc-55de-4acb-9c83-c4d65d6f2c2e": pickle.loads(b"\x80\x04]\x94."),
    "57ba6597-02eb-47da-8a3f-c2831c966484": pickle.loads(b"\x80\x04]\x94."),
    "d344884c-0863-4609-8a52-7da3ab3eeb5b": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06concat\x94."
    ),
    "557e7953-b45e-45aa-b9c2-68b710674bb9": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03glb\x94."
    ),
    "f3310dd6-e2a7-4683-89d2-2045994a5eee": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03PBR\x94."
    ),
    "1427657f-7c48-4e90-9b0a-c578567a47e5": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06medium\x94."
    ),
    "d5f7f7af-20e8-4733-8d84-d9ec70880cf2": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04Quad\x94."
    ),
    "875778ea-bb94-41bd-8d90-5ca0d1600199": pickle.loads(
        b"\x80\x04\x95\r\x00\x00\x00\x00\x00\x00\x00\x8c\tmodel.glb\x94."
    ),
}

"# Create the Flow, then do work within it as context."

flow0_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
).flow_name

with GriptapeNodes.ContextManager().flow(flow0_name):
    node0_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="StartFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="Start Flow",
            metadata={
                "position": {"x": -1079.3184088273833, "y": 166.3418546447166},
                "tempId": "placing-1765996350930-qa8fpl",
                "library_node_metadata": NodeMetadata(
                    category="workflows",
                    description="Define the start of a workflow and pass parameters into the flow",
                    display_name="Start Flow",
                    tags=None,
                    icon=None,
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "StartFlow",
                "showaddparameter": True,
                "size": {"width": 603, "height": 433},
                "category": "workflows",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node0_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="image_url",
                default_value="https://images.pdimagearchive.org/collections/posed-portraits-of-19th-century-baseball-stars/4051199982_718521f59f_z.jpg?zz=1?width=438&height=800",
                tooltip="Enter text/string for image_url.",
                type="str",
                input_types=["str"],
                output_type="str",
                ui_options={"is_custom": True, "is_user_added": True},
                mode_allowed_input=False,
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="image",
                default_value="",
                tooltip="New parameter",
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                output_type="ImageUrlArtifact",
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                    "edit_mask": True,
                    "display_name": "image",
                    "is_custom": True,
                    "is_user_added": True,
                    "hide": False,
                },
                parent_container_name="",
                initial_setup=True,
            )
        )
    node1_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="LoadImage",
            specific_library_name="Griptape Nodes Library",
            node_name="Load Image",
            metadata={
                "position": {"x": -314.31840882738334, "y": 166.3418546447166},
                "tempId": "placing-1766027122379-tyn6li",
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Loads an image from disk",
                    display_name="Load Image",
                    tags=None,
                    icon="image-up",
                    color=None,
                    group="Input/Output",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "LoadImage",
                "showaddparameter": False,
                "size": {"width": 600, "height": 540},
                "category": "image",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(parameter_name="image", settable=False, initial_setup=True)
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(parameter_name="path", settable=False, initial_setup=True)
        )
    node2_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Workflow Description",
            metadata={
                "position": {"x": -1079.3184088273833, "y": -637.7329772284488},
                "tempId": "placing-1766077471138-5mp4lm",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 743, "height": 424},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node3_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 1",
            metadata={
                "position": {"x": -1079.3184088273833, "y": -108.15814535528341},
                "tempId": "placing-1766003334530-pnkgh",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 1365, "height": 242},
                "category": "misc",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node4_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 2",
            metadata={
                "position": {"x": 511.1197906682836, "y": -109.15814535528341},
                "tempId": "placing-1766029587785-4avcqn",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 192},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node5_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 3",
            metadata={
                "position": {"x": 1309.040155645493, "y": -109.15814535528341},
                "tempId": "placing-1766003370124-fo9hq",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 196},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node6_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="MergeTexts",
            specific_library_name="Griptape Nodes Library",
            node_name="Merge Texts",
            metadata={
                "position": {"x": 1309.040155645493, "y": 166.3418546447166},
                "tempId": "placing-1766082110482-wb2nal",
                "library_node_metadata": NodeMetadata(
                    category="text",
                    description="MergeTexts node",
                    display_name="Merge Texts",
                    tags=None,
                    icon="merge",
                    color=None,
                    group="merge",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "MergeTexts",
                "showaddparameter": False,
                "size": {"width": 600, "height": 500},
                "category": "text",
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node7_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="EndFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="End Flow",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="workflows",
                    description="Define the end of a workflow and return parameters from the flow",
                    display_name="End Flow",
                    tags=None,
                    icon=None,
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "EndFlow",
                "showaddparameter": True,
                "position": {"x": 4424.484610936857, "y": 166.3418546447166},
                "size": {"width": 945, "height": 1244},
                "category": "workflows",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node7_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="model_url",
                default_value="",
                tooltip="New parameter",
                type="ThreeDUrlArtifact",
                input_types=["ThreeDUrlArtifact"],
                output_type="ThreeDUrlArtifact",
                ui_options={
                    "is_full_width": True,
                    "pulse_on_run": True,
                    "display_name": "3D Model",
                    "is_custom": True,
                    "is_user_added": True,
                },
                parent_container_name="",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="image",
                default_value="",
                tooltip="New parameter",
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact"],
                output_type="ImageUrlArtifact",
                ui_options={"is_full_width": True, "pulse_on_run": True, "is_custom": True, "is_user_added": True},
                parent_container_name="",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="image_1",
                default_value="",
                tooltip="New parameter",
                type="ImageUrlArtifact",
                input_types=["ImageUrlArtifact"],
                output_type="ImageUrlArtifact",
                ui_options={"is_full_width": True, "pulse_on_run": True, "is_custom": True, "is_user_added": True},
                parent_container_name="",
                initial_setup=True,
            )
        )
    node8_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 6",
            metadata={
                "position": {"x": 4424.484610936857, "y": -109.15814535528341},
                "tempId": "placing-1766003370124-fo9hq",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 600, "height": 163},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node9_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 4",
            metadata={
                "position": {"x": 2015.6453451374286, "y": -109.15814535528341},
                "tempId": "placing-1766003370124-fo9hq",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 1219, "height": 208},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node10_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 5",
            metadata={
                "position": {"x": 3550.4141023772922, "y": -109.15814535528341},
                "tempId": "placing-1766003370124-fo9hq",
                "library_node_metadata": NodeMetadata(
                    category="misc",
                    description="Create a note node to provide helpful context in your workflow",
                    display_name="Note",
                    tags=None,
                    icon="notepad-text",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Note",
                "showaddparameter": False,
                "size": {"width": 607, "height": 207},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node11_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="DescribeImage",
            specific_library_name="Griptape Nodes Library",
            node_name="Describe Image",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Can be used to describe an image",
                    display_name="Describe Image",
                    tags=None,
                    icon=None,
                    color=None,
                    group="describe",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "DescribeImage",
                "position": {"x": 511.1197906682836, "y": 166.3418546447166},
                "size": {"width": 600, "height": 990},
            },
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    node12_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Flux2ImageGeneration",
            specific_library_name="Griptape Nodes Library",
            node_name="FLUX.2 Image Generation_1",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Generate images using FLUX.2 models via Griptape model proxy",
                    display_name="FLUX.2 Image Generation",
                    tags=None,
                    icon="Zap",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Flux2ImageGeneration",
                "position": {"x": 2015.6453451374286, "y": 166.3418546447166},
                "size": {"width": 699, "height": 1475},
                "showaddparameter": False,
                "category": "image",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node12_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_0bcd26854c754ea9b6a45983dca8c1b2",
                default_value=[],
                tooltip="Optional input images for image-to-image generation (supports up to 20MB or 20 megapixels)",
                type="ImageArtifact",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                output_type="ImageArtifact",
                ui_options={"expander": True, "display_name": "Input Images"},
                mode_allowed_output=False,
                parent_container_name="input_images",
                initial_setup=True,
            )
        )
    node13_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Flux2ImageGeneration",
            specific_library_name="Griptape Nodes Library",
            node_name="FLUX.2 Image Generation",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Generate images using FLUX.2 models via Griptape model proxy",
                    display_name="FLUX.2 Image Generation",
                    tags=None,
                    icon="Zap",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Flux2ImageGeneration",
                "position": {"x": 2785.261814043148, "y": 166.3418546447166},
                "size": {"width": 699, "height": 1475},
                "showaddparameter": False,
                "category": "image",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node13_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_f2d7fc9dc388472782115353eb172892",
                default_value=[],
                tooltip="Optional input images for image-to-image generation (supports up to 20MB or 20 megapixels)",
                type="ImageArtifact",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                output_type="ImageArtifact",
                ui_options={"expander": True, "display_name": "Input Images"},
                mode_allowed_output=False,
                parent_container_name="input_images",
                initial_setup=True,
            )
        )
    node14_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Rodin23DGeneration",
            specific_library_name="Griptape Nodes Library",
            node_name="Rodin 3D Generation",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="3D",
                    description="Generate 3D models using Rodin Gen-2 via Griptape model proxy",
                    display_name="Rodin 3D Generation",
                    tags=None,
                    icon=None,
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Rodin23DGeneration",
                "position": {"x": 3550.4141023772922, "y": 166.3418546447166},
                "size": {"width": 600, "height": 1422},
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node14_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_b958b2e2df4644bbbef1e06dd0f7d5d7",
                default_value=[],
                tooltip="Optional input images for Image-to-3D generation (up to 5 images)",
                type="ImageArtifact",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                output_type="ImageArtifact",
                ui_options={"expander": True, "display_name": "Input Images"},
                mode_allowed_output=False,
                parent_container_name="input_images",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_aebd848451ae4a7e9d5bcd7bba2c3867",
                default_value=[],
                tooltip="Optional input images for Image-to-3D generation (up to 5 images)",
                type="ImageArtifact",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                output_type="ImageArtifact",
                ui_options={"expander": True, "display_name": "Input Images"},
                mode_allowed_output=False,
                parent_container_name="input_images",
                initial_setup=True,
            )
        )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node0_name,
            source_parameter_name="image_url",
            target_node_name=node1_name,
            target_parameter_name="image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node1_name,
            source_parameter_name="image",
            target_node_name=node11_name,
            target_parameter_name="image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node11_name,
            source_parameter_name="output",
            target_node_name=node6_name,
            target_parameter_name="input_2",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node6_name,
            source_parameter_name="output",
            target_node_name=node12_name,
            target_parameter_name="prompt",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node11_name,
            source_parameter_name="image",
            target_node_name=node12_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_0bcd26854c754ea9b6a45983dca8c1b2",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node12_name,
            source_parameter_name="image_url",
            target_node_name=node13_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_f2d7fc9dc388472782115353eb172892",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node14_name,
            source_parameter_name="model_url",
            target_node_name=node7_name,
            target_parameter_name="model_url",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node12_name,
            source_parameter_name="image_url",
            target_node_name=node14_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_b958b2e2df4644bbbef1e06dd0f7d5d7",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node13_name,
            source_parameter_name="image_url",
            target_node_name=node14_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_aebd848451ae4a7e9d5bcd7bba2c3867",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node12_name,
            source_parameter_name="image_url",
            target_node_name=node7_name,
            target_parameter_name="image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node13_name,
            source_parameter_name="image_url",
            target_node_name=node7_name,
            target_parameter_name="image_1",
            initial_setup=True,
        )
    )
    with GriptapeNodes.ContextManager().node(node0_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image_url",
                node_name=node0_name,
                value=top_level_unique_values_dict["0883576b-1af6-42fa-aaf0-7dedbf9e3c81"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node0_name,
                value=top_level_unique_values_dict["cf863f46-9dcb-416c-9f12-ede9bac037c5"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node1_name,
                value=top_level_unique_values_dict["5b402711-f794-4900-b296-96c1b24080e1"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node1_name,
                value=top_level_unique_values_dict["fc02b9fb-7b1c-488a-b6de-7fdb184b2f81"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node1_name,
                value=top_level_unique_values_dict["fc02b9fb-7b1c-488a-b6de-7fdb184b2f81"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node1_name,
                value=top_level_unique_values_dict["b5e60772-ac9b-4469-a32b-90f4c4fcc634"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node1_name,
                value=top_level_unique_values_dict["b5e60772-ac9b-4469-a32b-90f4c4fcc634"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="mask_channel",
                node_name=node1_name,
                value=top_level_unique_values_dict["124dff02-725c-4cf7-88c4-b7550672ab1b"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node1_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node1_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result_details",
                node_name=node1_name,
                value=top_level_unique_values_dict["23e3386d-279e-4dcb-a4c1-1715147f37f9"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result_details",
                node_name=node1_name,
                value=top_level_unique_values_dict["23e3386d-279e-4dcb-a4c1-1715147f37f9"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node2_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node2_name,
                value=top_level_unique_values_dict["7ed9d558-4fe7-4afd-8bcd-8dd31ce6b1f4"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node3_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node3_name,
                value=top_level_unique_values_dict["39526a24-733a-4455-88be-a479b2ce66e1"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node4_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node4_name,
                value=top_level_unique_values_dict["e0b1835c-c802-4bae-984a-9ca175d8b275"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node5_name,
                value=top_level_unique_values_dict["8be46441-f361-4d89-b739-d06ed9f314d7"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node6_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_1",
                node_name=node6_name,
                value=top_level_unique_values_dict["aa43d21e-8d1d-4187-8059-32ef1853afe9"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_2",
                node_name=node6_name,
                value=top_level_unique_values_dict["d245c88d-2f25-4376-8bf2-bc16c83be627"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_3",
                node_name=node6_name,
                value=top_level_unique_values_dict["1216491f-c966-4819-8b9d-b7851147a452"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="merge_string",
                node_name=node6_name,
                value=top_level_unique_values_dict["5c9ee93f-259a-441e-8a77-fd5320d620ac"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="whitespace",
                node_name=node6_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node6_name,
                value=top_level_unique_values_dict["f02f4500-5929-4384-aa19-6f249bb5b275"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node6_name,
                value=top_level_unique_values_dict["f02f4500-5929-4384-aa19-6f249bb5b275"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node7_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node7_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model_url",
                node_name=node7_name,
                value=top_level_unique_values_dict["cf863f46-9dcb-416c-9f12-ede9bac037c5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node7_name,
                value=top_level_unique_values_dict["cf863f46-9dcb-416c-9f12-ede9bac037c5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image_1",
                node_name=node7_name,
                value=top_level_unique_values_dict["cf863f46-9dcb-416c-9f12-ede9bac037c5"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node8_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node8_name,
                value=top_level_unique_values_dict["769d542e-6b06-4ed3-ab2c-32a6decce5aa"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node9_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node9_name,
                value=top_level_unique_values_dict["d4329e31-0185-44ca-9d29-dd3f1a3bab1a"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node10_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node10_name,
                value=top_level_unique_values_dict["4162f45e-14eb-47c9-b0a0-8fb64a07996c"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node11_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="agent",
                node_name=node11_name,
                value=top_level_unique_values_dict["b765ceb5-5f19-42c7-8d9a-e9fee8c5e396"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node11_name,
                value=top_level_unique_values_dict["bc6692c4-4933-4a9f-9880-ce56e82a8ad3"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node11_name,
                value=top_level_unique_values_dict["fc02b9fb-7b1c-488a-b6de-7fdb184b2f81"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node11_name,
                value=top_level_unique_values_dict["ce0c98b1-c1d8-424f-80b9-001857842b3d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="description_only",
                node_name=node11_name,
                value=top_level_unique_values_dict["7345ff53-1dd7-41bd-87b5-75cf644e7998"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node11_name,
                value=top_level_unique_values_dict["d245c88d-2f25-4376-8bf2-bc16c83be627"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node12_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node12_name,
                value=top_level_unique_values_dict["19c39e2d-fe35-43a9-a9ad-2f9197b556c1"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node12_name,
                value=top_level_unique_values_dict["f02f4500-5929-4384-aa19-6f249bb5b275"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node12_name,
                value=top_level_unique_values_dict["5ed2cc5c-acb5-4ac7-b346-0e9a1ae9e9ea"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_0bcd26854c754ea9b6a45983dca8c1b2",
                node_name=node12_name,
                value=top_level_unique_values_dict["fc02b9fb-7b1c-488a-b6de-7fdb184b2f81"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node12_name,
                value=top_level_unique_values_dict["7d6130f2-7a13-4c0b-b1ef-472287f3ffa6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node12_name,
                value=top_level_unique_values_dict["7505748e-d58e-46c5-b5c3-be0fda163ae4"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="force_output_dimension",
                node_name=node12_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node12_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node12_name,
                value=top_level_unique_values_dict["9751a66a-cf87-4e3d-be08-03a636b6ff42"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node12_name,
                value=top_level_unique_values_dict["30b21d96-7705-43f2-9b04-9ba959f788f8"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="safety_tolerance",
                node_name=node12_name,
                value=top_level_unique_values_dict["1ebb3947-30b2-4edd-90bd-283a315e797a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="steps",
                node_name=node12_name,
                value=top_level_unique_values_dict["9fd6669e-ed93-424a-9327-5935968aad34"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="guidance",
                node_name=node12_name,
                value=top_level_unique_values_dict["be42758d-3989-4de2-a71f-5416df342093"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node12_name,
                value=top_level_unique_values_dict["68158fbd-9d19-46c1-9598-6055af2f88bc"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node12_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node13_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node13_name,
                value=top_level_unique_values_dict["19c39e2d-fe35-43a9-a9ad-2f9197b556c1"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node13_name,
                value=top_level_unique_values_dict["12d6e668-c76c-4162-8da7-b47253cf0a9c"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node13_name,
                value=top_level_unique_values_dict["6c9d998c-20e3-4456-8729-bbc51ac61991"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_f2d7fc9dc388472782115353eb172892",
                node_name=node13_name,
                value=top_level_unique_values_dict["1a5950a4-235f-48fe-b444-28984695f6a3"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node13_name,
                value=top_level_unique_values_dict["7d6130f2-7a13-4c0b-b1ef-472287f3ffa6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node13_name,
                value=top_level_unique_values_dict["7d6130f2-7a13-4c0b-b1ef-472287f3ffa6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="force_output_dimension",
                node_name=node13_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node13_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node13_name,
                value=top_level_unique_values_dict["9751a66a-cf87-4e3d-be08-03a636b6ff42"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node13_name,
                value=top_level_unique_values_dict["30b21d96-7705-43f2-9b04-9ba959f788f8"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="safety_tolerance",
                node_name=node13_name,
                value=top_level_unique_values_dict["1ebb3947-30b2-4edd-90bd-283a315e797a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="steps",
                node_name=node13_name,
                value=top_level_unique_values_dict["9fd6669e-ed93-424a-9327-5935968aad34"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="guidance",
                node_name=node13_name,
                value=top_level_unique_values_dict["be42758d-3989-4de2-a71f-5416df342093"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node13_name,
                value=top_level_unique_values_dict["68158fbd-9d19-46c1-9598-6055af2f88bc"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node13_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node14_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node14_name,
                value=top_level_unique_values_dict["e4590a94-f430-478d-9224-0d61f670704a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node14_name,
                value=top_level_unique_values_dict["f96208af-4eeb-4242-9d15-0e48b6143e20"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_b958b2e2df4644bbbef1e06dd0f7d5d7",
                node_name=node14_name,
                value=top_level_unique_values_dict["f9099acc-55de-4acb-9c83-c4d65d6f2c2e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_aebd848451ae4a7e9d5bcd7bba2c3867",
                node_name=node14_name,
                value=top_level_unique_values_dict["57ba6597-02eb-47da-8a3f-c2831c966484"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="condition_mode",
                node_name=node14_name,
                value=top_level_unique_values_dict["d344884c-0863-4609-8a52-7da3ab3eeb5b"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="geometry_file_format",
                node_name=node14_name,
                value=top_level_unique_values_dict["557e7953-b45e-45aa-b9c2-68b710674bb9"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="material",
                node_name=node14_name,
                value=top_level_unique_values_dict["f3310dd6-e2a7-4683-89d2-2045994a5eee"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="quality",
                node_name=node14_name,
                value=top_level_unique_values_dict["1427657f-7c48-4e90-9b0a-c578567a47e5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="mesh_mode",
                node_name=node14_name,
                value=top_level_unique_values_dict["d5f7f7af-20e8-4733-8d84-d9ec70880cf2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="use_original_alpha",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ta_pose",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="bbox_condition",
                node_name=node14_name,
                value=top_level_unique_values_dict["cf863f46-9dcb-416c-9f12-ede9bac037c5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="high_pack",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="preview_render",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node14_name,
                value=top_level_unique_values_dict["9751a66a-cf87-4e3d-be08-03a636b6ff42"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node14_name,
                value=top_level_unique_values_dict["875778ea-bb94-41bd-8d90-5ca0d1600199"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node14_name,
                value=top_level_unique_values_dict["e6fb1acd-1638-4708-ba7a-997c835a0801"],
                initial_setup=True,
                is_output=False,
            )
        )


def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_request = GetTopLevelFlowRequest()
        top_level_flow_result = GriptapeNodes.handle_request(top_level_flow_request)
        if (
            isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess)
            and top_level_flow_result.flow_name is not None
        ):
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(
    input: dict,
    storage_backend: str = "local",
    project_file_path: str | None = None,
    workflow_executor: WorkflowExecutor | None = None,
    pickle_control_flow_result: bool = False,
) -> dict | None:
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
    input: dict,
    storage_backend: str = "local",
    project_file_path: str | None = None,
    workflow_executor: WorkflowExecutor | None = None,
    pickle_control_flow_result: bool = False,
) -> dict | None:
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--storage-backend",
        choices=["local", "gtc"],
        default="local",
        help="Storage backend to use: 'local' for local filesystem or 'gtc' for Griptape Cloud",
    )
    parser.add_argument(
        "--project-file-path", default=None, help="Path to a project file to load for the workflow execution"
    )
    parser.add_argument(
        "--json-input",
        default=None,
        help="JSON string containing parameter values. Takes precedence over individual parameter arguments if provided.",
    )
    parser.add_argument("--exec_out", default=None, help="Connection to the next node in the execution chain")
    parser.add_argument("--image_url", default=None, help="Enter text/string for image_url.")
    parser.add_argument("--image", default=None, help="New parameter")
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.exec_out is not None:
            flow_input["Start Flow"]["exec_out"] = args.exec_out
        if args.image_url is not None:
            flow_input["Start Flow"]["image_url"] = args.image_url
        if args.image is not None:
            flow_input["Start Flow"]["image"] = args.image
    workflow_output = execute_workflow(
        input=flow_input, storage_backend=args.storage_backend, project_file_path=args.project_file_path
    )
    print(workflow_output)
