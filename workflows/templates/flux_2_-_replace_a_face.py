# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "flux_2_-_replace_a_face"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.4"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "Flux2ImageGeneration"], ["Griptape Nodes Library", "ImageDetails"], ["Griptape Nodes Library", "IntegerInput"], ["Griptape Nodes Library", "InvertImage"], ["Griptape Nodes Library", "LoadImage"], ["Griptape Nodes Library", "MathExpression"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "PaintMask"], ["Griptape Nodes Library", "RescaleImage"], ["Griptape Nodes Library", "StartFlow"], ["Griptape Nodes Library", "TextInput"], ["Griptape Nodes Library", "Webcam"]]
# description = "Replace a face in an image using inpainting with Flux 2, the Paint Mask node, and a Webcam photo of yourself."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_flux_2_-_replace_a_face.webp"
# is_griptape_provided = true
# is_template = true
# creation_date = 2026-03-16T20:17:25.244014Z
# last_modified_date = 2026-03-17T17:53:58.407040Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"input_url\":{\"name\":\"input_url\",\"tooltip\":\"Enter text/string for input_url.\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"https://images.pdimagearchive.org/collections/rogues-a-study-of-characters-samuel-g-szabo/rogues-samuel-g-szabo-00017.jpeg\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"image_url\":{\"name\":\"image_url\",\"tooltip\":\"New parameter\",\"type\":\"ImageUrlArtifact\",\"input_types\":[\"ImageUrlArtifact\"],\"output_type\":\"ImageUrlArtifact\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"ui_options\":{\"is_full_width\":true,\"pulse_on_run\":true,\"is_custom\":true,\"is_user_added\":true},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
import pickle

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
from pathlib import Path

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
    "b2abd5f6-6df7-4a32-8141-c0056de9a195": pickle.loads(
        b'\x80\x04\x95\x0e\x03\x00\x00\x00\x00\x00\x00X\x07\x03\x00\x00# Flux 2 - Replace part of an Image\n\nThis workflow demonstrates using a mask to replace part of an image with Black Forest Labs Flux 2 model.\n\nThis technique is called "inpainting" and is very useful for providing a specific location to a model for updating.\n\nTo demonstrate the technique, we\'ll get an image from the Public Digital Image Archive, paint a mask over a particular area, and replace that area with another image.\n\n## Video Tutorial\n\n<iframe width="560" height="315" src="https://www.youtube.com/embed/rZI9rPsAPsA?si=Fo6TtKLhhnoZelcq" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>\x94.'
    ),
    "3f5fee0d-d328-4d1c-a4cf-49996e2ba7d2": pickle.loads(
        b"\x80\x04\x95~\x00\x00\x00\x00\x00\x00\x00\x8czhttps://images.pdimagearchive.org/collections/rogues-a-study-of-characters-samuel-g-szabo/rogues-samuel-g-szabo-00017.jpeg\x94."
    ),
    "afb17b7d-2911-485b-8027-d15a0c13dc46": pickle.loads(
        b"\x80\x04\x95\x80\x00\x00\x00\x00\x00\x00\x00\x8c|Get an image from the [Public Domain Image Archive](https://pdimagearchive.org/images/83f00ea0-3c8f-4a3d-83a7-d89324d44160/)\x94."
    ),
    "a793b129-f868-445b-8393-f16ba6aee5b4": pickle.loads(
        b"\x80\x04\x95\x88\x00\x00\x00\x00\x00\x00\x00\x8c\x84Rescale the image using the new width and height.\n\n_Note: we're using a `fit_mode` of `fit` to make sure we don't distort the image_\x94."
    ),
    "7ea7485f-f416-4e4b-97b4-e9db1d3aa5d7": pickle.loads(
        b"\x80\x04\x95\xfd\x00\x00\x00\x00\x00\x00\x00\x8c\xf9## Paint Mask Node\n\nNow we'll paint a mask in the image where we want to **inpaint** a new face.\n\n1. Hover over the input image\n2. Click the _mask_ icon\n3. In the **Paint Mask Editor** choose the **Eraser** and erase part of the image.\n4. Click Save\x94."
    ),
    "499c3401-5983-40fb-91b7-f45455fd364d": pickle.loads(b"\x80\x04\x89."),
    "b0e68914-24a5-4d5f-a7b0-9c8c02b14a6e": pickle.loads(
        b"\x80\x04\x95\x97\x01\x00\x00\x00\x00\x00\x00\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x10ImageUrlArtifact\x94\x93\x94)\x81\x94}\x94(\x8c\x04type\x94\x8c\x10ImageUrlArtifact\x94\x8c\x0bmodule_name\x94\x8c%griptape.artifacts.image_url_artifact\x94\x8c\x02id\x94\x8c e5624b5ae3dd4840aa1c031d356bbfc7\x94\x8c\treference\x94N\x8c\x04meta\x94}\x94\x8c\x04name\x94h\n\x8c\x16encoding_error_handler\x94\x8c\x06strict\x94\x8c\x08encoding\x94\x8c\x05utf-8\x94\x8c\x05value\x94\x8czhttps://images.pdimagearchive.org/collections/rogues-a-study-of-characters-samuel-g-szabo/rogues-samuel-g-szabo-00017.jpeg\x94ub."
    ),
    "fd381e5b-e313-49e6-9132-41b12b90557f": pickle.loads(b"\x80\x04K\x00."),
    "9f166067-ddaf-407d-9a9b-79f657ef02a6": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M'\x03."),
    "bb20875e-c70e-4af2-b9ca-88cdc813d554": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x80\x04."),
    "0e64f9a8-3c25-41ab-82c2-6027cc95018e": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x030:0\x94."
    ),
    "dae4113f-92ef-4b46-8304-05b513a2f0dd": pickle.loads(
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07269:384\x94."
    ),
    "54b6e1bc-adcd-47b6-8f8e-bee658ef837d": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G\x00\x00\x00\x00\x00\x00\x00\x00."
    ),
    "d240bcbb-9df0-4d34-9dc2-eb09519e8b84": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe6j\xaa\xaa\xaa\xaa\xab."
    ),
    "d454ff7b-be8b-4127-8457-e7272afeee8a": pickle.loads(
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07UNKNOWN\x94."
    ),
    "bab906ee-eba5-4432-a346-505f0ffe3554": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03RGB\x94."
    ),
    "47140898-9c0f-4a0c-be19-87da7bad1fc0": pickle.loads(b"\x80\x04K\x03."),
    "1849ab0a-b503-40df-9c93-758746f165d1": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04JPEG\x94."
    ),
    "9ee8238b-fb40-4e6a-ac2a-6f2553d564ab": pickle.loads(
        b"\x80\x04\x95\xd5\x00\x00\x00\x00\x00\x00\x00\x8c\xd1The Flux 2 Image Generation node requires width and height to be divisible by `16`. \n\nSo we'll use the **Image Details** node, some math, and a **Rescale Image** node to generate the image at the correct size.\x94."
    ),
    "bb6c3b78-f231-46d9-91bd-40056bc6c3c0": pickle.loads(
        b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0eround(a/b) * b\x94."
    ),
    "c4f44639-fbb9-4c55-9768-9e9f05ec1719": pickle.loads(b"\x80\x04K\x10."),
    "00e13a5b-8404-43b6-a557-dc38c1411afa": pickle.loads(b"\x80\x04K\x02."),
    "046d0fac-d8bd-47fd-8804-8aa4aa51cf02": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03int\x94."
    ),
    "9709dc09-19e2-4b41-90d4-ecff202805ed": pickle.loads(b"\x80\x04K\x06."),
    "cea13423-d42a-4b32-8c45-4bb0500dfa55": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M \x03."),
    "b35aa816-e555-4283-9a22-e83beae95e2a": pickle.loads(
        b"\x80\x04\x95w\x00\x00\x00\x00\x00\x00\x00\x8csThe model will require the area you want to mask to be white. To do this we can just use the **Invert Image** node.\x94."
    ),
    "61ce78e9-6de4-4571-bfa5-4a92cb36f423": pickle.loads(
        b"\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00\x8c\x12Review the results\x94."
    ),
    "7cffc52b-16d0-41e3-a9da-96d55fda6db2": pickle.loads(
        b"\x80\x04\x95\x93\x00\x00\x00\x00\x00\x00\x00\x8c\x8fGenerate the image using **flux-2-max**.\n\nThe prompt should include information about what input to use as the mask, and what image to replace.\x94."
    ),
    "3273f645-17af-47ee-a32d-aea08da6db6b": pickle.loads(
        b"\x80\x04\x95\x93\x00\x00\x00\x00\x00\x00\x00\x8c\x8fTo provide an image to replace, use the **Webcam** node to take a screenshot of your face, or use a **Load Image** node and connect it instead.\x94."
    ),
    "b93751b1-6320-4446-a107-d4f7a5b07f71": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04none\x94."
    ),
    "414f5eaa-0409-467c-8cb6-c879f6ae4d98": pickle.loads(
        b"\x80\x04\x95\x14\x00\x00\x00\x00\x00\x00\x00\x8c\x10width and height\x94."
    ),
    "3159257b-d093-4b28-bef9-1e7b9b4f7dfe": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\xe8\x03."),
    "55bf7d11-c058-4a29-9fa5-cb48da0c0313": pickle.loads(b"\x80\x04Kd."),
    "2fb7b6e0-5268-43a6-9833-3b403bc3528f": pickle.loads(
        b"\x80\x04\x95\x07\x00\x00\x00\x00\x00\x00\x00\x8c\x03fit\x94."
    ),
    "f36216f1-f763-4bcd-880b-c32ee3313080": pickle.loads(
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07#000000\x94."
    ),
    "a372e78b-8d9e-41d0-84d4-515a131c4720": pickle.loads(
        b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07lanczos\x94."
    ),
    "587b4b4e-61c9-4c69-9c1e-cd9bc5899ce9": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04auto\x94."
    ),
    "f44aee52-becf-4aae-898b-8a4b05709c58": pickle.loads(
        b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00\x8c\x0295\x94."
    ),
    "038377c4-70c8-408c-9a53-61c24e07f7a2": pickle.loads(
        b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cFlux.2 [pro]\x94."
    ),
    "ba1b0f64-59d9-498a-b390-01d0aabca793": pickle.loads(
        b"\x80\x04\x95\xb2\x00\x00\x00\x00\x00\x00\x00\x8c\xaeInpaint the person from Image 3 into Image 1 using the mask from Image 2\nMatch the texture, art style, color, tone, hue, contrast, brightness, levels, and clothing of Image 1\x94."
    ),
    "71f493df-228c-4931-91f4-6bd7288138b3": pickle.loads(b"\x80\x04]\x94."),
    "0e7950a1-9b05-4505-a960-77505a138172": pickle.loads(b"\x80\x04K*."),
    "c3c8ce16-fe9c-4938-b76e-9b6aa2b691e2": pickle.loads(
        b"\x80\x04\x95\x08\x00\x00\x00\x00\x00\x00\x00\x8c\x04jpeg\x94."
    ),
    "f93a8399-a24e-4ba0-85f2-ed62afee7f34": pickle.loads(
        b"\x80\x04\x95\x15\x00\x00\x00\x00\x00\x00\x00\x8c\x11least restrictive\x94."
    ),
    "8aa955bd-1671-492f-8138-bff903d93319": pickle.loads(b"\x80\x04K2."),
    "825b2280-9334-461a-b2d4-7dfd92e7e38d": pickle.loads(
        b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G@\x12\x00\x00\x00\x00\x00\x00."
    ),
}

"# Create the Flow, then do work within it as context."

flow0_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
).flow_name

with GriptapeNodes.ContextManager().flow(flow0_name):
    node0_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Workflow Description",
            metadata={
                "position": {"x": -617.6213590658149, "y": -773.552661767276},
                "tempId": "placing-1766172442198-lnzeb9",
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
                "size": {"width": 686, "height": 755},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node1_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="StartFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="Start Flow",
            metadata={
                "position": {"x": -617.6213590658149, "y": 259.5337121055135},
                "tempId": "placing-1766172731607-uuljzo",
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
                "size": {"width": 600, "height": 314},
                "category": "workflows",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_url",
                default_value="https://images.pdimagearchive.org/collections/rogues-a-study-of-characters-samuel-g-szabo/rogues-samuel-g-szabo-00017.jpeg",
                tooltip="Enter text/string for input_url.",
                type="str",
                input_types=["str"],
                output_type="str",
                ui_options={"is_custom": True, "is_user_added": True},
                mode_allowed_input=False,
                initial_setup=True,
            )
        )
    node2_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 1",
            metadata={
                "position": {"x": -617.6213590658149, "y": 20.308852128403174},
                "tempId": "placing-1766172797527-apyqbo",
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
    node3_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 3",
            metadata={
                "position": {"x": 2714.859505294859, "y": 588.0337121055135},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 600, "height": 157},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node4_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 4",
            metadata={
                "position": {"x": 3597.63776993708, "y": 47.8421783721635},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 600, "height": 278},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node5_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="EndFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="End Flow",
            metadata={
                "position": {"x": 6732.806736277839, "y": 325.8421783721635},
                "tempId": "placing-1766174695668-giuybt",
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
                "size": {"width": 1053, "height": 1587},
                "category": "workflows",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="image_url",
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
    node6_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="ImageDetails",
            specific_library_name="Griptape Nodes Library",
            node_name="Image Details",
            metadata={
                "position": {"x": 970.45085383902, "y": 834.6207556461201},
                "tempId": "placing-1766190560199-slwyv8",
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Extract detailed information from an image including dimensions, aspect ratio, color space, and format",
                    display_name="Image Details",
                    tags=None,
                    icon="info",
                    color=None,
                    group="describe",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "ImageDetails",
                "showaddparameter": False,
                "size": {"width": 600, "height": 581},
                "category": "image",
            },
            initial_setup=True,
        )
    ).node_name
    node7_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 2",
            metadata={
                "position": {"x": 970.45085383902, "y": 573.5337121055135},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 600, "height": 186},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node8_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="TextInput",
            specific_library_name="Griptape Nodes Library",
            node_name="Rounding function",
            metadata={
                "position": {"x": 970.45085383902, "y": 1517.185056131189},
                "tempId": "placing-1766190587182-pcl8b4",
                "library_node_metadata": NodeMetadata(
                    category="text",
                    description="TextInput node",
                    display_name="Text Input",
                    tags=None,
                    icon="text-cursor",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "TextInput",
                "showaddparameter": False,
                "size": {"width": 600, "height": 243},
                "category": "text",
            },
            initial_setup=True,
        )
    ).node_name
    node9_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="IntegerInput",
            specific_library_name="Griptape Nodes Library",
            node_name="Divisible by..",
            metadata={
                "position": {"x": 970.45085383902, "y": 1789.8931460401946},
                "tempId": "placing-1766178986516-f4m0db",
                "library_node_metadata": NodeMetadata(
                    category="number",
                    description="Create an integer value",
                    display_name="Integer Input",
                    tags=None,
                    icon=None,
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "IntegerInput",
                "showaddparameter": False,
                "size": {"width": 600, "height": 286},
                "category": "number",
            },
            initial_setup=True,
        )
    ).node_name
    node10_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="MathExpression",
            specific_library_name="Griptape Nodes Library",
            node_name="Calculate Width",
            metadata={
                "position": {"x": 1976.3859264792384, "y": 1391.6890123127607},
                "tempId": "placing-1766178967820-3ofyyqn",
                "library_node_metadata": NodeMetadata(
                    category="number",
                    description="Evaluate mathematical expressions with variable inputs (a-h) and math functions",
                    display_name="Math Expression",
                    tags=None,
                    icon="function-square",
                    color=None,
                    group="tasks",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "MathExpression",
                "showaddparameter": False,
                "size": {"width": 600, "height": 302},
                "category": "number",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node10_name):
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="output_type",
                ui_options={
                    "simple_dropdown": ["float", "int"],
                    "show_search": True,
                    "search_filter": "",
                    "hide": True,
                },
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="precision",
                ui_options={"hide_label": False, "hide_property": False, "hide": True},
                initial_setup=True,
            )
        )
    node11_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="MathExpression",
            specific_library_name="Griptape Nodes Library",
            node_name="Calculate Height",
            metadata={
                "position": {"x": 1976.3859264792384, "y": 1760.185056131189},
                "tempId": "placing-1766178967820-3ofyyqn",
                "library_node_metadata": NodeMetadata(
                    category="number",
                    description="Evaluate mathematical expressions with variable inputs (a-h) and math functions",
                    display_name="Math Expression",
                    tags=None,
                    icon="function-square",
                    color=None,
                    group="tasks",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "MathExpression",
                "showaddparameter": False,
                "size": {"width": 601, "height": 374},
                "category": "number",
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node11_name):
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="output_type",
                ui_options={
                    "simple_dropdown": ["float", "int"],
                    "show_search": True,
                    "search_filter": "",
                    "hide": True,
                },
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="precision",
                ui_options={"hide_label": False, "hide_property": False, "hide": True},
                initial_setup=True,
            )
        )
    node12_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 5",
            metadata={
                "position": {"x": 4312.196985799474, "y": 47.8421783721635},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 602, "height": 242},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node13_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 9",
            metadata={
                "position": {"x": 6732.806736277839, "y": 47.8421783721635},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 600, "height": 185},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node14_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 7",
            metadata={
                "position": {"x": 5828.9707855127435, "y": 47.8421783721635},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 601, "height": 235},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node15_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Note",
            specific_library_name="Griptape Nodes Library",
            node_name="Step 6",
            metadata={
                "position": {"x": 5047.32947017138, "y": 925.8959362470698},
                "tempId": "placing-1766172797527-apyqbo",
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
                "size": {"width": 600, "height": 156},
                "category": "misc",
            },
            initial_setup=True,
        )
    ).node_name
    node16_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="Webcam",
            specific_library_name="Griptape Nodes Library",
            node_name="Webcam",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Capture an image using the device's camera",
                    display_name="Webcam",
                    tags=None,
                    icon="webcam",
                    color=None,
                    group="create",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "Webcam",
                "position": {"x": 5047.32947017138, "y": 1097.4687236788404},
                "size": {"width": 600, "height": 348},
            },
            initial_setup=True,
        )
    ).node_name
    node17_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="LoadImage",
            specific_library_name="Griptape Nodes Library",
            node_name="Load Image",
            metadata={
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
                "position": {"x": 132.1129308185267, "y": 259.5337121055135},
                "size": {"width": 604, "height": 640},
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node17_name):
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(parameter_name="image", settable=False, initial_setup=True)
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(parameter_name="path", settable=False, initial_setup=True)
        )
    node18_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="RescaleImage",
            specific_library_name="Griptape Nodes Library",
            node_name="Rescale Image_1",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description='Resize images with separate parameters for target size (pixels) and percentage scale, plus resample filter options. Previously named "Rescale Image".',
                    display_name="Resize Image",
                    tags=None,
                    icon="image-upscale",
                    color=None,
                    group="edit",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "RescaleImage",
                "position": {"x": 2714.859505294859, "y": 834.6207556461201},
                "size": {"width": 600, "height": 900},
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node18_name):
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="resize_mode",
                ui_options={
                    "simple_dropdown": ["width", "height", "width and height", "percentage"],
                    "show_search": True,
                    "search_filter": "",
                    "hide_label": False,
                    "hide_property": False,
                    "hide": False,
                },
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="percentage_scale",
                ui_options={
                    "slider": {"min_val": 1, "max_val": 500},
                    "hide_label": False,
                    "hide_property": False,
                    "hide": True,
                },
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="target_width",
                ui_options={
                    "slider": {"min_val": 1, "max_val": 8000},
                    "hide_label": False,
                    "hide_property": False,
                    "hide": False,
                },
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            AlterParameterDetailsRequest(
                parameter_name="target_height",
                ui_options={
                    "slider": {"min_val": 1, "max_val": 8000},
                    "hide_label": False,
                    "hide_property": False,
                    "hide": False,
                },
                initial_setup=True,
            )
        )
    node19_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="PaintMask",
            specific_library_name="Griptape Nodes Library",
            node_name="Paint Mask",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Paint a mask on an image.",
                    display_name="Paint Mask",
                    tags=None,
                    icon=None,
                    color=None,
                    group="mask",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "PaintMask",
                "position": {"x": 3597.63776993708, "y": 337.3959362470698},
                "size": {"width": 600, "height": 537},
            },
            initial_setup=True,
        )
    ).node_name
    node20_name = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="InvertImage",
            specific_library_name="Griptape Nodes Library",
            node_name="Invert Image",
            metadata={
                "library_node_metadata": NodeMetadata(
                    category="image",
                    description="Invert a full image (creates a negative).",
                    display_name="Invert Image",
                    tags=None,
                    icon=None,
                    color=None,
                    group="edit",
                    deprecation=None,
                    is_node_group=None,
                ),
                "library": "Griptape Nodes Library",
                "node_type": "InvertImage",
                "position": {"x": 4312.196985799474, "y": 337.3959362470698},
                "size": {"width": 600, "height": 544},
            },
            initial_setup=True,
        )
    ).node_name
    node21_name = GriptapeNodes.handle_request(
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
                "position": {"x": 5828.9707855127435, "y": 337.3959362470698},
                "size": {"width": 607, "height": 1333},
            },
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(node21_name):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_479a5aa638df48a08dc1f13f99fdcd7e",
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
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_d8cdf84f8ce3485298b8469f534e0504",
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
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="input_images_ParameterListUniqueParamID_60fed342ec35430d9de278cecc78c0c5",
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
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node9_name,
            source_parameter_name="integer",
            target_node_name=node10_name,
            target_parameter_name="b",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node9_name,
            source_parameter_name="integer",
            target_node_name=node11_name,
            target_parameter_name="b",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node6_name,
            source_parameter_name="width",
            target_node_name=node10_name,
            target_parameter_name="a",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node6_name,
            source_parameter_name="height",
            target_node_name=node11_name,
            target_parameter_name="a",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node8_name,
            source_parameter_name="text",
            target_node_name=node10_name,
            target_parameter_name="expression",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node8_name,
            source_parameter_name="text",
            target_node_name=node11_name,
            target_parameter_name="expression",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node1_name,
            source_parameter_name="input_url",
            target_node_name=node17_name,
            target_parameter_name="image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node17_name,
            source_parameter_name="image",
            target_node_name=node6_name,
            target_parameter_name="image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node10_name,
            source_parameter_name="result",
            target_node_name=node18_name,
            target_parameter_name="target_width",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node11_name,
            source_parameter_name="result",
            target_node_name=node18_name,
            target_parameter_name="target_height",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node17_name,
            source_parameter_name="image",
            target_node_name=node18_name,
            target_parameter_name="input_image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node18_name,
            source_parameter_name="output",
            target_node_name=node19_name,
            target_parameter_name="input_image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node19_name,
            source_parameter_name="output_mask",
            target_node_name=node20_name,
            target_parameter_name="input_image",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node18_name,
            source_parameter_name="target_width",
            target_node_name=node21_name,
            target_parameter_name="width",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node18_name,
            source_parameter_name="target_height",
            target_node_name=node21_name,
            target_parameter_name="height",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node21_name,
            source_parameter_name="image_url",
            target_node_name=node5_name,
            target_parameter_name="image_url",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node18_name,
            source_parameter_name="output",
            target_node_name=node21_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_479a5aa638df48a08dc1f13f99fdcd7e",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node20_name,
            source_parameter_name="output",
            target_node_name=node21_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_d8cdf84f8ce3485298b8469f534e0504",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=node16_name,
            source_parameter_name="image",
            target_node_name=node21_name,
            target_parameter_name="input_images_ParameterListUniqueParamID_60fed342ec35430d9de278cecc78c0c5",
            initial_setup=True,
        )
    )
    with GriptapeNodes.ContextManager().node(node0_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node0_name,
                value=top_level_unique_values_dict["b2abd5f6-6df7-4a32-8141-c0056de9a195"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node1_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_url",
                node_name=node1_name,
                value=top_level_unique_values_dict["3f5fee0d-d328-4d1c-a4cf-49996e2ba7d2"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node2_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node2_name,
                value=top_level_unique_values_dict["afb17b7d-2911-485b-8027-d15a0c13dc46"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node3_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node3_name,
                value=top_level_unique_values_dict["a793b129-f868-445b-8393-f16ba6aee5b4"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node4_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node4_name,
                value=top_level_unique_values_dict["7ea7485f-f416-4e4b-97b4-e9db1d3aa5d7"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node5_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node5_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node6_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node6_name,
                value=top_level_unique_values_dict["b0e68914-24a5-4d5f-a7b0-9c8c02b14a6e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node6_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node6_name,
                value=top_level_unique_values_dict["9f166067-ddaf-407d-9a9b-79f657ef02a6"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node6_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node6_name,
                value=top_level_unique_values_dict["bb20875e-c70e-4af2-b9ca-88cdc813d554"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ratio_str",
                node_name=node6_name,
                value=top_level_unique_values_dict["0e64f9a8-3c25-41ab-82c2-6027cc95018e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ratio_str",
                node_name=node6_name,
                value=top_level_unique_values_dict["dae4113f-92ef-4b46-8304-05b513a2f0dd"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ratio_decimal",
                node_name=node6_name,
                value=top_level_unique_values_dict["54b6e1bc-adcd-47b6-8f8e-bee658ef837d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="ratio_decimal",
                node_name=node6_name,
                value=top_level_unique_values_dict["d240bcbb-9df0-4d34-9dc2-eb09519e8b84"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="color_space",
                node_name=node6_name,
                value=top_level_unique_values_dict["d454ff7b-be8b-4127-8457-e7272afeee8a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="color_space",
                node_name=node6_name,
                value=top_level_unique_values_dict["bab906ee-eba5-4432-a346-505f0ffe3554"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="channels",
                node_name=node6_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="channels",
                node_name=node6_name,
                value=top_level_unique_values_dict["47140898-9c0f-4a0c-be19-87da7bad1fc0"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="format",
                node_name=node6_name,
                value=top_level_unique_values_dict["d454ff7b-be8b-4127-8457-e7272afeee8a"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="format",
                node_name=node6_name,
                value=top_level_unique_values_dict["1849ab0a-b503-40df-9c93-758746f165d1"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node7_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node7_name,
                value=top_level_unique_values_dict["9ee8238b-fb40-4e6a-ac2a-6f2553d564ab"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node8_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="text",
                node_name=node8_name,
                value=top_level_unique_values_dict["bb6c3b78-f231-46d9-91bd-40056bc6c3c0"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="text",
                node_name=node8_name,
                value=top_level_unique_values_dict["bb6c3b78-f231-46d9-91bd-40056bc6c3c0"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node9_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="integer",
                node_name=node9_name,
                value=top_level_unique_values_dict["c4f44639-fbb9-4c55-9768-9e9f05ec1719"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="integer",
                node_name=node9_name,
                value=top_level_unique_values_dict["c4f44639-fbb9-4c55-9768-9e9f05ec1719"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node10_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="expression",
                node_name=node10_name,
                value=top_level_unique_values_dict["bb6c3b78-f231-46d9-91bd-40056bc6c3c0"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="num_variables",
                node_name=node10_name,
                value=top_level_unique_values_dict["00e13a5b-8404-43b6-a557-dc38c1411afa"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="a",
                node_name=node10_name,
                value=top_level_unique_values_dict["9f166067-ddaf-407d-9a9b-79f657ef02a6"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="b",
                node_name=node10_name,
                value=top_level_unique_values_dict["c4f44639-fbb9-4c55-9768-9e9f05ec1719"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="c",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="d",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="e",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="f",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="g",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="h",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="i",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="j",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="k",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="l",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="m",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="n",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="o",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="p",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="q",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="r",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="s",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="t",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="u",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="v",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="w",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="x",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="y",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="z",
                node_name=node10_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_type",
                node_name=node10_name,
                value=top_level_unique_values_dict["046d0fac-d8bd-47fd-8804-8aa4aa51cf02"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="precision",
                node_name=node10_name,
                value=top_level_unique_values_dict["9709dc09-19e2-4b41-90d4-ecff202805ed"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result",
                node_name=node10_name,
                value=top_level_unique_values_dict["54b6e1bc-adcd-47b6-8f8e-bee658ef837d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result",
                node_name=node10_name,
                value=top_level_unique_values_dict["cea13423-d42a-4b32-8c45-4bb0500dfa55"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node11_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="expression",
                node_name=node11_name,
                value=top_level_unique_values_dict["bb6c3b78-f231-46d9-91bd-40056bc6c3c0"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="num_variables",
                node_name=node11_name,
                value=top_level_unique_values_dict["00e13a5b-8404-43b6-a557-dc38c1411afa"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="a",
                node_name=node11_name,
                value=top_level_unique_values_dict["bb20875e-c70e-4af2-b9ca-88cdc813d554"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="b",
                node_name=node11_name,
                value=top_level_unique_values_dict["c4f44639-fbb9-4c55-9768-9e9f05ec1719"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="c",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="d",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="e",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="f",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="g",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="h",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="i",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="j",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="k",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="l",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="m",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="n",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="o",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="p",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="q",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="r",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="s",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="t",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="u",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="v",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="w",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="x",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="y",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="z",
                node_name=node11_name,
                value=top_level_unique_values_dict["fd381e5b-e313-49e6-9132-41b12b90557f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_type",
                node_name=node11_name,
                value=top_level_unique_values_dict["046d0fac-d8bd-47fd-8804-8aa4aa51cf02"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="precision",
                node_name=node11_name,
                value=top_level_unique_values_dict["9709dc09-19e2-4b41-90d4-ecff202805ed"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result",
                node_name=node11_name,
                value=top_level_unique_values_dict["54b6e1bc-adcd-47b6-8f8e-bee658ef837d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="result",
                node_name=node11_name,
                value=top_level_unique_values_dict["bb20875e-c70e-4af2-b9ca-88cdc813d554"],
                initial_setup=True,
                is_output=True,
            )
        )
    with GriptapeNodes.ContextManager().node(node12_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node12_name,
                value=top_level_unique_values_dict["b35aa816-e555-4283-9a22-e83beae95e2a"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node13_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node13_name,
                value=top_level_unique_values_dict["61ce78e9-6de4-4571-bfa5-4a92cb36f423"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node14_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node14_name,
                value=top_level_unique_values_dict["7cffc52b-16d0-41e3-a9da-96d55fda6db2"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node15_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="note",
                node_name=node15_name,
                value=top_level_unique_values_dict["3273f645-17af-47ee-a32d-aea08da6db6b"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node17_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node17_name,
                value=top_level_unique_values_dict["b0e68914-24a5-4d5f-a7b0-9c8c02b14a6e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="image",
                node_name=node17_name,
                value=top_level_unique_values_dict["b0e68914-24a5-4d5f-a7b0-9c8c02b14a6e"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node17_name,
                value=top_level_unique_values_dict["3f5fee0d-d328-4d1c-a4cf-49996e2ba7d2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="path",
                node_name=node17_name,
                value=top_level_unique_values_dict["3f5fee0d-d328-4d1c-a4cf-49996e2ba7d2"],
                initial_setup=True,
                is_output=True,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="mask_channel",
                node_name=node17_name,
                value=top_level_unique_values_dict["b93751b1-6320-4446-a107-d4f7a5b07f71"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node17_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node18_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_image",
                node_name=node18_name,
                value=top_level_unique_values_dict["b0e68914-24a5-4d5f-a7b0-9c8c02b14a6e"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="resize_mode",
                node_name=node18_name,
                value=top_level_unique_values_dict["414f5eaa-0409-467c-8cb6-c879f6ae4d98"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="target_size",
                node_name=node18_name,
                value=top_level_unique_values_dict["3159257b-d093-4b28-bef9-1e7b9b4f7dfe"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="percentage_scale",
                node_name=node18_name,
                value=top_level_unique_values_dict["55bf7d11-c058-4a29-9fa5-cb48da0c0313"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="target_width",
                node_name=node18_name,
                value=top_level_unique_values_dict["cea13423-d42a-4b32-8c45-4bb0500dfa55"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="target_height",
                node_name=node18_name,
                value=top_level_unique_values_dict["bb20875e-c70e-4af2-b9ca-88cdc813d554"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="fit_mode",
                node_name=node18_name,
                value=top_level_unique_values_dict["2fb7b6e0-5268-43a6-9833-3b403bc3528f"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="background_color",
                node_name=node18_name,
                value=top_level_unique_values_dict["f36216f1-f763-4bcd-880b-c32ee3313080"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="resample_filter",
                node_name=node18_name,
                value=top_level_unique_values_dict["a372e78b-8d9e-41d0-84d4-515a131c4720"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node18_name,
                value=top_level_unique_values_dict["587b4b4e-61c9-4c69-9c1e-cd9bc5899ce9"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="quality",
                node_name=node18_name,
                value=top_level_unique_values_dict["f44aee52-becf-4aae-898b-8a4b05709c58"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node18_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node19_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="invert_mask",
                node_name=node19_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="grow_shrink",
                node_name=node19_name,
                value=top_level_unique_values_dict["54b6e1bc-adcd-47b6-8f8e-bee658ef837d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="blur_mask",
                node_name=node19_name,
                value=top_level_unique_values_dict["54b6e1bc-adcd-47b6-8f8e-bee658ef837d"],
                initial_setup=True,
                is_output=False,
            )
        )
    with GriptapeNodes.ContextManager().node(node21_name):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="model",
                node_name=node21_name,
                value=top_level_unique_values_dict["038377c4-70c8-408c-9a53-61c24e07f7a2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="prompt",
                node_name=node21_name,
                value=top_level_unique_values_dict["ba1b0f64-59d9-498a-b390-01d0aabca793"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="input_images",
                node_name=node21_name,
                value=top_level_unique_values_dict["71f493df-228c-4931-91f4-6bd7288138b3"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="width",
                node_name=node21_name,
                value=top_level_unique_values_dict["cea13423-d42a-4b32-8c45-4bb0500dfa55"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="height",
                node_name=node21_name,
                value=top_level_unique_values_dict["bb20875e-c70e-4af2-b9ca-88cdc813d554"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="force_output_dimension",
                node_name=node21_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="randomize_seed",
                node_name=node21_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="seed",
                node_name=node21_name,
                value=top_level_unique_values_dict["0e7950a1-9b05-4505-a960-77505a138172"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="output_format",
                node_name=node21_name,
                value=top_level_unique_values_dict["c3c8ce16-fe9c-4938-b76e-9b6aa2b691e2"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="safety_tolerance",
                node_name=node21_name,
                value=top_level_unique_values_dict["f93a8399-a24e-4ba0-85f2-ed62afee7f34"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="steps",
                node_name=node21_name,
                value=top_level_unique_values_dict["8aa955bd-1671-492f-8138-bff903d93319"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="guidance",
                node_name=node21_name,
                value=top_level_unique_values_dict["825b2280-9334-461a-b2d4-7dfd92e7e38d"],
                initial_setup=True,
                is_output=False,
            )
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="was_successful",
                node_name=node21_name,
                value=top_level_unique_values_dict["499c3401-5983-40fb-91b7-f45455fd364d"],
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
    parser.add_argument("--input_url", default=None, help="Enter text/string for input_url.")
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.exec_out is not None:
            flow_input["Start Flow"]["exec_out"] = args.exec_out
        if args.input_url is not None:
            flow_input["Start Flow"]["input_url"] = args.input_url
    workflow_output = execute_workflow(
        input=flow_input, storage_backend=args.storage_backend, project_file_path=args.project_file_path
    )
    print(workflow_output)
