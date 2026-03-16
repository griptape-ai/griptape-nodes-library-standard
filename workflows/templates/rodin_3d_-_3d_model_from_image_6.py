# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "/Users/collindutter/Projects/griptape/griptape-nodes-libraries/griptape-nodes-library-standard/workflows/templates/rodin_3d_-_3d_model_from_image_6"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "DescribeImage"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "Flux2ImageGeneration"], ["Griptape Nodes Library", "LoadImage"], ["Griptape Nodes Library", "MergeTexts"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "Rodin23DGeneration"], ["Griptape Nodes Library", "StartFlow"]]
# description = "Generate a 3D model from an image using Hyper3D Rodin, Flux-2 Klein, and OpenAI gpt-4.1-mini."
# image = "https://github.com/griptape-ai/griptape-nodes-library-standard/blob/main/workflows/templates/thumbnail_rodin_3d_-_3d_model_from_image.webp?raw=true"
# is_griptape_provided = false
# is_template = false
# creation_date = 2026-03-12T23:17:20.609598Z
# last_modified_date = 2026-03-12T23:17:24.899862Z
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
    "628ab4cc-dfeb-4681-a950-41e81aacea00": pickle.loads(
        b"\x80\x04\x95\xaa\x00\x00\x00\x00\x00\x00\x00\x8c\xa6https://images.pdimagearchive.org/collections/joseph-ducreux-self-portraits/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?width=755&height=800\x94."
    ),
    "7184ad4c-86b9-41f9-b82c-2ad108825204": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."),
    "6f1edaba-4bb3-4e99-9bb0-2788bf670eeb": pickle.loads(
        b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08mask.png\x94."
    ),
    "e64cddf0-8586-4f2d-a90a-eac34c361166": pickle.loads(
        b"\x80\x04\x95\x9c\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 0e4a94c3cb434c9daa6b8c7c08eeaa6a\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94ub."
    ),
    "177e9188-2763-404e-a325-473157aad587": pickle.loads(
        b"\x80\x04\x95\x83\x00\x00\x00\x00\x00\x00\x00\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94."
    ),
    "dc7afcff-c9b8-400b-8497-8652167f23ae": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04none\x94."
    ),
    "225700eb-59b4-42bd-ad03-f30f6c3d9d7d": pickle.loads(b"\x80\x04\x89."),
    "8e627a11-f6eb-46c1-8ae9-5407e2ac8a66": pickle.loads(
        b"\x80\x04\x950\x00\x00\x00\x00\x00\x00\x00\x8c,<Results will appear when the node executes>\x94."
    ),
    "068fe3b7-908b-473e-a547-02806890483d": pickle.loads(
        b"\x80\x04\x956\x03\x00\x00\x00\x00\x00\x00X/\x03\x00\x00# Rodin 3D - Create a 3D Model from an Image\n\nThis workflow demonstrates the use of various models to generate a 3D model.\n\nIt first grabs an image from the [Public Domain Image Archive](https://pdimagearchive.org/galleries/all) - a fantastic resource for inspirational and unique images.\n\nThen, uses [OpenAI gpt-4.1-mini](https://platform.openai.com/docs/models/gpt-4.1-mini) to describe the hat in that image.\n\nNext, it uses [Flux-2 Klein](https://bfl.ai/models/flux-2-klein) to render the a nice image of the hat in perspective view, then renders another view of the bottom of the hat. \n\nThese images are pasted to [Hyper3D Rodin](https://hyper3d.ai/) to generate a 3d model.\n\n_Note_: You can provide your own image by simply disconnecting the `image_url` parameter and instead connecting the `image` parameter. \x94."
    ),
    "543602dd-4e7f-48bc-80a3-527dfddb3335": pickle.loads(
        b"\x80\x04\x95\xad\x00\x00\x00\x00\x00\x00\x00\x8c\xa9# Inputs\n\nProvide the url of a person from the [Public Domain Image Archive](https://pdimagearchive.org/galleries/all), or provide your own image and connect it instead.\x94."
    ),
    "33e0ec12-391c-41cb-ba2a-ac42f50a90c3": pickle.loads(
        b"\x80\x04\x95~\x00\x00\x00\x00\x00\x00\x00\x8czUse OpenAI gpt-4.1-mini to describe the hat from the image so that it can be generated cleanly with Google Nano Banana Pro\x94."
    ),
    "cfd037b7-fe6a-4cba-bde1-22ad2faa7125": pickle.loads(
        b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x00\x8c#Merge the text to generate a prompt\x94."
    ),
    "dc4b6d3d-935e-4090-b64e-532e68723584": pickle.loads(
        b"\x80\x04\x95}\x00\x00\x00\x00\x00\x00\x00\x8cyCreate a 3d high resolution render of the hat from this image, using information from the description as supporting data.\x94."
    ),
    "77ad3b75-93b9-4f66-9b98-755a4a3ec37d": pickle.loads(
        b"\x80\x04\x95_\x02\x00\x00\x00\x00\x00\x00XX\x02\x00\x00The hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\x94."
    ),
    "ad47bd2d-55b6-4037-b550-172833bbd166": pickle.loads(
        b"\x80\x04\x95#\x00\x00\x00\x00\x00\x00\x00\x8c\x1flight background, soft shadows.\x94."
    ),
    "cb81a175-0410-4a06-91a0-3abd727f04b8": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04\\n\\n\x94."
    ),
    "5740f1b4-f06a-4f68-a2c0-a6c941f5e1d4": pickle.loads(
        b"\x80\x04\x95\xfb\x02\x00\x00\x00\x00\x00\x00X\xf4\x02\x00\x00Create a 3d high resolution render of the hat from this image, using information from the description as supporting data.\n\nThe hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\n\nlight background, soft shadows.\x94."
    ),
    "08b8646b-e192-416e-a077-728b89e51961": pickle.loads(
        b"\x80\x04\x95]\x00\x00\x00\x00\x00\x00\x00\x8cYRun the workflow to view the results. You'll get the 3d model, and two images of the hat.\x94."
    ),
    "9a7c6b76-158a-47a3-a810-e372f336ca55": pickle.loads(
        b"\x80\x04\x95\xa9\x00\x00\x00\x00\x00\x00\x00\x8c\xa5Use Flux.2 Klein to generate a 3d perspective view and a 3/4 underside image of the hat. These two images will provide enough detail to Ronin to create the 3d model.\x94."
    ),
    "795f0fae-72df-4ac3-9b2b-9365715f786a": pickle.loads(
        b"\x80\x04\x95\xa3\x00\x00\x00\x00\x00\x00\x00\x8c\x9fGenerate a 3d Model with [Hyper 3D Ronin](https://hyper3d.ai/).\n\nThis currently outputs **glb** format, but you can make changes to any parameters you require.\x94."
    ),
    "91a4cbce-0d3b-4bd2-bc71-27a443aaf1ad": pickle.loads(
        b"\x80\x04\x95\x84\x06\x00\x00\x00\x00\x00\x00}\x94(\x8c\x04type\x94\x8c\x12GriptapeNodesAgent\x94\x8c\x08rulesets\x94]\x94\x8c\x05rules\x94]\x94\x8c\x02id\x94\x8c 187014a4c2d94dc18e20cba7a539b15a\x94\x8c\x13conversation_memory\x94}\x94(h\x01\x8c\x12ConversationMemory\x94\x8c\x04runs\x94]\x94}\x94(h\x01\x8c\x03Run\x94h\x07\x8c c459b3301d6b4466a21bd59d87edd4e8\x94\x8c\x04meta\x94N\x8c\x05input\x94}\x94(h\x01\x8c\x0cTextArtifact\x94h\x07\x8c 1f09e3f8361e44d5924f0d1180a62bbd\x94\x8c\treference\x94Nh\x11}\x94\x8c\x04name\x94h\x15\x8c\x05value\x94\x8c\xf6Describe the hat from this image. Be specific about the shape and design. Include detailed reference to height, width, and creasing. Include specific details on brim shape, and any dents or textures in the surface.\n\nOutput image description only.\x94u\x8c\x06output\x94}\x94(h\x01h\x14h\x07\x8c 7b2164ded78844b8b7f8de1022d8d6d5\x94h\x16Nh\x11}\x94h\x18h\x1dh\x19XX\x02\x00\x00The hat in the image is a tall, black top hat with a cylindrical crown that is relatively high, approximately 6 to 7 inches in height. The crown is smooth and rounded at the top without any visible dents or creases, giving it a firm and structured appearance. The brim is wide and slightly curved upwards around the edges, extending outward about 2 to 3 inches from the base of the crown. The surface of the hat appears to have a matte texture with a subtle sheen, suggesting it is made of felt or a similar material. The overall design is classic and formal, typical of 18th or 19th-century fashion.\x94uuah\x11}\x94\x8c\x08max_runs\x94Nu\x8c\x1cconversation_memory_strategy\x94\x8c\rper_structure\x94\x8c\x05tasks\x94]\x94}\x94(h\x01\x8c\nPromptTask\x94h\x03]\x94h\x05]\x94h\x07\x8c e6e9e656090e42e7be331bbb4be223fa\x94\x8c\x05state\x94\x8c\x0eState.FINISHED\x94\x8c\nparent_ids\x94]\x94\x8c\tchild_ids\x94]\x94\x8c\x17max_meta_memory_entries\x94K\x14\x8c\x07context\x94}\x94\x8c\rprompt_driver\x94}\x94(h\x01\x8c\x19GriptapeCloudPromptDriver\x94\x8c\x0btemperature\x94G?\xb9\x99\x99\x99\x99\x99\x9a\x8c\nmax_tokens\x94N\x8c\x06stream\x94\x89\x8c\x0cextra_params\x94}\x94\x8c\x05model\x94\x8c\x0cgpt-4.1-mini\x94\x8c\x1astructured_output_strategy\x94\x8c\x06native\x94u\x8c\x05tools\x94]\x94\x8c\x0cmax_subtasks\x94K\x14uau."
    ),
    "c891320b-f85f-49f1-aa71-2a3c242fe8ed": pickle.loads(
        b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cgpt-4.1-mini\x94."
    ),
    "72c9c7a4-a179-4aaa-b76e-69c7100b2575": pickle.loads(
        b"\x80\x04\x95\xda\x00\x00\x00\x00\x00\x00\x00\x8c\xd6Describe the hat from this image. Be specific about the shape and design. Include detailed reference to height, width, and creasing. Include specific details on brim shape, and any dents or textures in the surface.\x94."
    ),
    "45c622fa-894b-4464-a642-178b57bb1c9d": pickle.loads(b"\x80\x04\x88."),
    "4e7a5ef3-a12f-4599-a274-98c9a70eca62": pickle.loads(
        b"\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11Flux.2 [klein] 9B\x94."
    ),
    "0ece129e-b257-4aa7-b322-cfdd422e3940": pickle.loads(
        b"\x80\x04\x95\x9f\x01\x00\x00\x00\x00\x00\x00]\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c 0e4a94c3cb434c9daa6b8c7c08eeaa6a\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\x0b\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8c\x7fhttp://localhost:8124/workspace/static_files/6a7dc663cb2b4ad4e42b05584699e7003971c66d247f3301e6ff42125321-edit.jpg?t=1766088547\x94uba."
    ),
    "4097acad-620f-4ab6-aa28-197d7667c2d2": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M`\x05."),
    "4496658f-1878-4da3-9d21-ed0bacf5b799": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x03."),
    "61b7b744-9a1e-4a18-8422-91e30e8d85de": pickle.loads(b"\x80\x04K*."),
    "3c86d193-8e76-4948-9cd7-651a8c8a6f4e": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04jpeg\x94."
    ),
    "26032d33-ec39-465e-8390-9753efed51ec": pickle.loads(
        b"\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11least restrictive\x94."
    ),
    "7e2960bd-b8a8-4bab-9da8-b757bab4d9a5": pickle.loads(b"\x80\x04K2."),
    "ad6d86d2-6c2d-431c-a5ef-66a8cf5a304a": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x12\x00\x00\x00\x00\x00\x00."
    ),
    "15db9810-9e01-4f42-ad16-1a94d0b4225f": pickle.loads(
        b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eflux_image.jpg\x94."
    ),
    "04c28265-4522-405c-958e-2d967fccd300": pickle.loads(
        b"\x80\x04\x95g\x00\x00\x00\x00\x00\x00\x00\x8ccsame hat viewed from a worm's eye 3/4 angle revealing the underside. light background, soft shadows\x94."
    ),
    "b11c6936-bf70-47ab-bb37-da1e0a844001": pickle.loads(b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00]\x94]\x94a."),
    "f60ef8e6-79c9-435b-9435-e926051643fa": pickle.loads(b"\x80\x04]\x94."),
    "232eb821-efab-471c-9971-9abbab83c2fc": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03hat\x94."
    ),
    "69609c13-213b-4e7b-9d00-ff397d64610a": pickle.loads(
        b"\x80\x04\x95\t\x00\x00\x00\x00\x00\x00\x00]\x94(]\x94]\x94e."
    ),
    "7c1b0b37-6a85-4ec6-b54f-754468165176": pickle.loads(b"\x80\x04]\x94."),
    "a5083174-eb3e-4c99-9083-37b5c484ce88": pickle.loads(b"\x80\x04]\x94."),
    "3cd4cb75-4a27-4e1e-9ca7-373060b58051": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06concat\x94."
    ),
    "52c2bc7d-0af0-45f2-ae2a-618d4bdbbb10": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03glb\x94."
    ),
    "a4f50238-17c4-4d9a-8b54-4c4fa2ab82d4": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03PBR\x94."
    ),
    "553851c6-7751-46e6-b7c2-62fcc198f42e": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06medium\x94."
    ),
    "0fa4f018-67c3-423f-b573-044a335ba9b2": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04Quad\x94."
    ),
    "020c4a3b-8852-4488-aada-70f49e7f3e04": pickle.loads(
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
                value=top_level_unique_values_dict["628ab4cc-dfeb-4681-a950-41e81aacea00"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node0_name,
                value=top_level_unique_values_dict["7184ad4c-86b9-41f9-b82c-2ad108825204"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node1_name,
                value=top_level_unique_values_dict["6f1edaba-4bb3-4e99-9bb0-2788bf670eeb"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node1_name,
                value=top_level_unique_values_dict["e64cddf0-8586-4f2d-a90a-eac34c361166"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node1_name,
                value=top_level_unique_values_dict["e64cddf0-8586-4f2d-a90a-eac34c361166"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node1_name,
                value=top_level_unique_values_dict["177e9188-2763-404e-a325-473157aad587"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node1_name,
                value=top_level_unique_values_dict["177e9188-2763-404e-a325-473157aad587"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="mask_channel",
                node_name=node1_name,
                value=top_level_unique_values_dict["dc7afcff-c9b8-400b-8497-8652167f23ae"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node1_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node1_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result_details",
                node_name=node1_name,
                value=top_level_unique_values_dict["8e627a11-f6eb-46c1-8ae9-5407e2ac8a66"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result_details",
                node_name=node1_name,
                value=top_level_unique_values_dict["8e627a11-f6eb-46c1-8ae9-5407e2ac8a66"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node2_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node2_name,
                value=top_level_unique_values_dict["068fe3b7-908b-473e-a547-02806890483d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node3_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node3_name,
                value=top_level_unique_values_dict["543602dd-4e7f-48bc-80a3-527dfddb3335"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node4_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node4_name,
                value=top_level_unique_values_dict["33e0ec12-391c-41cb-ba2a-ac42f50a90c3"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node5_name,
                value=top_level_unique_values_dict["cfd037b7-fe6a-4cba-bde1-22ad2faa7125"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node6_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_1",
                node_name=node6_name,
                value=top_level_unique_values_dict["dc4b6d3d-935e-4090-b64e-532e68723584"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_2",
                node_name=node6_name,
                value=top_level_unique_values_dict["77ad3b75-93b9-4f66-9b98-755a4a3ec37d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_3",
                node_name=node6_name,
                value=top_level_unique_values_dict["ad47bd2d-55b6-4037-b550-172833bbd166"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="merge_string",
                node_name=node6_name,
                value=top_level_unique_values_dict["cb81a175-0410-4a06-91a0-3abd727f04b8"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="whitespace",
                node_name=node6_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node6_name,
                value=top_level_unique_values_dict["5740f1b4-f06a-4f68-a2c0-a6c941f5e1d4"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node6_name,
                value=top_level_unique_values_dict["5740f1b4-f06a-4f68-a2c0-a6c941f5e1d4"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node7_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node7_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model_url",
                node_name=node7_name,
                value=top_level_unique_values_dict["7184ad4c-86b9-41f9-b82c-2ad108825204"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node7_name,
                value=top_level_unique_values_dict["7184ad4c-86b9-41f9-b82c-2ad108825204"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image_1",
                node_name=node7_name,
                value=top_level_unique_values_dict["7184ad4c-86b9-41f9-b82c-2ad108825204"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node8_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node8_name,
                value=top_level_unique_values_dict["08b8646b-e192-416e-a077-728b89e51961"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node9_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node9_name,
                value=top_level_unique_values_dict["9a7c6b76-158a-47a3-a810-e372f336ca55"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node10_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node10_name,
                value=top_level_unique_values_dict["795f0fae-72df-4ac3-9b2b-9365715f786a"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node11_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="agent",
                node_name=node11_name,
                value=top_level_unique_values_dict["91a4cbce-0d3b-4bd2-bc71-27a443aaf1ad"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node11_name,
                value=top_level_unique_values_dict["c891320b-f85f-49f1-aa71-2a3c242fe8ed"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node11_name,
                value=top_level_unique_values_dict["e64cddf0-8586-4f2d-a90a-eac34c361166"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node11_name,
                value=top_level_unique_values_dict["72c9c7a4-a179-4aaa-b76e-69c7100b2575"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="description_only",
                node_name=node11_name,
                value=top_level_unique_values_dict["45c622fa-894b-4464-a642-178b57bb1c9d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output",
                node_name=node11_name,
                value=top_level_unique_values_dict["77ad3b75-93b9-4f66-9b98-755a4a3ec37d"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node12_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node12_name,
                value=top_level_unique_values_dict["4e7a5ef3-a12f-4599-a274-98c9a70eca62"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node12_name,
                value=top_level_unique_values_dict["5740f1b4-f06a-4f68-a2c0-a6c941f5e1d4"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node12_name,
                value=top_level_unique_values_dict["0ece129e-b257-4aa7-b322-cfdd422e3940"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_0bcd26854c754ea9b6a45983dca8c1b2",
                node_name=node12_name,
                value=top_level_unique_values_dict["e64cddf0-8586-4f2d-a90a-eac34c361166"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node12_name,
                value=top_level_unique_values_dict["4097acad-620f-4ab6-aa28-197d7667c2d2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node12_name,
                value=top_level_unique_values_dict["4496658f-1878-4da3-9d21-ed0bacf5b799"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="force_output_dimension",
                node_name=node12_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node12_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node12_name,
                value=top_level_unique_values_dict["61b7b744-9a1e-4a18-8422-91e30e8d85de"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node12_name,
                value=top_level_unique_values_dict["3c86d193-8e76-4948-9cd7-651a8c8a6f4e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="safety_tolerance",
                node_name=node12_name,
                value=top_level_unique_values_dict["26032d33-ec39-465e-8390-9753efed51ec"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="steps",
                node_name=node12_name,
                value=top_level_unique_values_dict["7e2960bd-b8a8-4bab-9da8-b757bab4d9a5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="guidance",
                node_name=node12_name,
                value=top_level_unique_values_dict["ad6d86d2-6c2d-431c-a5ef-66a8cf5a304a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node12_name,
                value=top_level_unique_values_dict["15db9810-9e01-4f42-ad16-1a94d0b4225f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node12_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node13_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node13_name,
                value=top_level_unique_values_dict["4e7a5ef3-a12f-4599-a274-98c9a70eca62"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node13_name,
                value=top_level_unique_values_dict["04c28265-4522-405c-958e-2d967fccd300"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node13_name,
                value=top_level_unique_values_dict["b11c6936-bf70-47ab-bb37-da1e0a844001"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_f2d7fc9dc388472782115353eb172892",
                node_name=node13_name,
                value=top_level_unique_values_dict["f60ef8e6-79c9-435b-9435-e926051643fa"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node13_name,
                value=top_level_unique_values_dict["4097acad-620f-4ab6-aa28-197d7667c2d2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node13_name,
                value=top_level_unique_values_dict["4097acad-620f-4ab6-aa28-197d7667c2d2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="force_output_dimension",
                node_name=node13_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node13_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node13_name,
                value=top_level_unique_values_dict["61b7b744-9a1e-4a18-8422-91e30e8d85de"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node13_name,
                value=top_level_unique_values_dict["3c86d193-8e76-4948-9cd7-651a8c8a6f4e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="safety_tolerance",
                node_name=node13_name,
                value=top_level_unique_values_dict["26032d33-ec39-465e-8390-9753efed51ec"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="steps",
                node_name=node13_name,
                value=top_level_unique_values_dict["7e2960bd-b8a8-4bab-9da8-b757bab4d9a5"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="guidance",
                node_name=node13_name,
                value=top_level_unique_values_dict["ad6d86d2-6c2d-431c-a5ef-66a8cf5a304a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node13_name,
                value=top_level_unique_values_dict["15db9810-9e01-4f42-ad16-1a94d0b4225f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node13_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node14_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node14_name,
                value=top_level_unique_values_dict["232eb821-efab-471c-9971-9abbab83c2fc"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node14_name,
                value=top_level_unique_values_dict["69609c13-213b-4e7b-9d00-ff397d64610a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_b958b2e2df4644bbbef1e06dd0f7d5d7",
                node_name=node14_name,
                value=top_level_unique_values_dict["7c1b0b37-6a85-4ec6-b54f-754468165176"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images_ParameterListUniqueParamID_aebd848451ae4a7e9d5bcd7bba2c3867",
                node_name=node14_name,
                value=top_level_unique_values_dict["a5083174-eb3e-4c99-9083-37b5c484ce88"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="condition_mode",
                node_name=node14_name,
                value=top_level_unique_values_dict["3cd4cb75-4a27-4e1e-9ca7-373060b58051"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="geometry_file_format",
                node_name=node14_name,
                value=top_level_unique_values_dict["52c2bc7d-0af0-45f2-ae2a-618d4bdbbb10"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="material",
                node_name=node14_name,
                value=top_level_unique_values_dict["a4f50238-17c4-4d9a-8b54-4c4fa2ab82d4"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="quality",
                node_name=node14_name,
                value=top_level_unique_values_dict["553851c6-7751-46e6-b7c2-62fcc198f42e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="mesh_mode",
                node_name=node14_name,
                value=top_level_unique_values_dict["0fa4f018-67c3-423f-b573-044a335ba9b2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="use_original_alpha",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ta_pose",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="bbox_condition",
                node_name=node14_name,
                value=top_level_unique_values_dict["7184ad4c-86b9-41f9-b82c-2ad108825204"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="high_pack",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="preview_render",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node14_name,
                value=top_level_unique_values_dict["61b7b744-9a1e-4a18-8422-91e30e8d85de"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_file",
                node_name=node14_name,
                value=top_level_unique_values_dict["020c4a3b-8852-4488-aada-70f49e7f3e04"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node14_name,
                value=top_level_unique_values_dict["225700eb-59b4-42bd-ad03-f30f6c3d9d7d"],
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
