# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "test_topaz_video_upscale_astra"
# schema_version = "0.20.0"
# engine_version_created_with = "0.101.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.86.0"], ["Griptape Nodes Testing Library", "0.1.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "LTXTextToVideoGeneration"], ["Griptape Nodes Library", "ResolveMacroPath"], ["Griptape Nodes Library", "TopazVideoUpscale"], ["Griptape Nodes Testing Library", "AssertFileExists"]]
# is_griptape_provided = false
# is_template = false
# is_internal = true
# creation_date = 2026-09-10T08:43:20.265255Z
# last_modified_date = 2026-09-10T10:33:27.588472Z
#
# ///

import pickle

from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    AlterParameterGroupDetailsRequest,
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(
            library_name="Griptape Nodes Testing Library", perform_discovery_if_not_found=True
        )
    )
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)

    # 1. We've collated all of the unique parameter values into a dictionary so that we do not have to duplicate them.
    #    This minimizes the size of the code, especially for large objects like serialized image files.
    # 2. We're using a prefix so that it's clear which Flow these values are associated with.
    # 3. The values are serialized using pickle, which is a binary format. This makes them harder to read, but makes
    #    them consistently save and load. It allows us to serialize complex objects like custom classes, which otherwise
    #    would be difficult to serialize.
    top_level_unique_values_dict = {
        "59f51e04-196c-457b-bb49-cb76ea845e8b": pickle.loads(b"\x80\x04\x89."),
        "9f6db6d9-0ec6-48d8-814b-d78446435f81": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\xb0\x04."),
        "2c47a793-6843-4f7f-b84d-4c3a360f0c1d": pickle.loads(
            b"\x80\x04\x95\x10\x00\x00\x00\x00\x00\x00\x00\x8c\x0cltx-2-5-fast\x94."
        ),
        "9a9014d9-002e-4864-b6bf-dc84403408c8": pickle.loads(
            b"\x80\x04\x95\x13\x00\x00\x00\x00\x00\x00\x00\x8c\x0fA ball bouncing\x94."
        ),
        "462e5481-0b4a-46c1-93d5-83a8cf4bcf36": pickle.loads(
            b"\x80\x04\x95\x0c\x00\x00\x00\x00\x00\x00\x00\x8c\x08720x1280\x94."
        ),
        "c292dcaa-a4d5-4a36-9500-60fbfa5ccdf0": pickle.loads(b"\x80\x04K\x06."),
        "6f6a2abd-50b2-4b95-8812-5884b8b3fbf1": pickle.loads(b"\x80\x04K\x18."),
        "022cfd25-ce47-49cf-bfb4-24c50338f667": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06static\x94."
        ),
        "791aecc4-b679-460c-ac81-46a7f5cc2be3": pickle.loads(b"\x80\x04\x88."),
        "a0ce2d66-bf9c-43a9-b3a7-9a4166bad54a": pickle.loads(
            b"\x80\x04\x95\x16\x00\x00\x00\x00\x00\x00\x00\x8c\x12ltx_text_video.mp4\x94."
        ),
        "da388b5f-30a9-48fe-aafc-2781b7207f73": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "322a0fe0-a8f8-4c0c-bdb8-9629cc2890ae": pickle.loads(
            b"\x80\x04\x95%\x00\x00\x00\x00\x00\x00\x00\x8c!{outputs}/topaz_video_upscale.mp4\x94."
        ),
        "7887189e-b73f-458d-a4dc-ea5c2b6dbb0e": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M \x03."),
        "5c27bbcf-5e7b-4184-8571-234d89b2c18e": pickle.loads(
            b"\x80\x04\x95\x0b\x00\x00\x00\x00\x00\x00\x00\x8c\x07Astra 2\x94."
        ),
        "604e16c6-e39c-4eb6-b49e-6bc591b83ac3": pickle.loads(
            b"\x80\x04\x95)\x00\x00\x00\x00\x00\x00\x00\x8c%A ball bouncing, crisp rubber texture\x94."
        ),
        "6416a526-e1a9-4fb1-b789-57c99361eacb": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00G?\xe0\x00\x00\x00\x00\x00\x00."
        ),
        "a2116863-f0eb-4dad-bb44-d3b2fc64c388": pickle.loads(
            b"\x80\x04\x95l\x00\x00\x00\x00\x00\x00\x00\x8cHgriptape_nodes.node_libraries.griptape_nodes_library.topaz_video_upscale\x94\x8c\nResizeMode\x94\x93\x94\x8c\npercentage\x94\x85\x94R\x94."
        ),
        "33a4012a-d6e4-44db-9f4f-42b1d0e6d997": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x80\x07."),
        "54734eee-568e-4dac-accc-677192143e63": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00M\x00\x0f."),
        "99643af8-b0eb-496b-a556-a730e1a32418": pickle.loads(b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00Mp\x08."),
        "cedb9638-bf44-49c9-87e5-735eb9eae429": pickle.loads(b"\x80\x04Kn."),
        "a53e76d5-7461-4ae8-b2f9-34ec5a65b8a6": pickle.loads(
            b"\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x17topaz_video_upscale.mp4\x94."
        ),
    }
    # Create the Flow, then do work within it as context.
    flow0_name = (
        await GriptapeNodes.ahandle_request(
            CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
        )
    ).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="LTXTextToVideoGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="LTX Text To Video Generation",
                    metadata={
                        "library_node_metadata": {
                            "category": "video",
                            "description": "Generate a video from text using LTX AI models via Griptape Cloud model proxy.",
                            "display_name": "LTX Text to Video Generation",
                            "tags": ["video", "text-to-video", "ai", "api", "ltx"],
                            "icon": "Sparkles",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": [
                                        "gtc_ltx_2_pro",
                                        "gtc_ltx_2_fast",
                                        "gtc_ltx_2_3_pro",
                                        "gtc_ltx_2_3_fast",
                                        "gtc_ltx_2_5_pro",
                                        "gtc_ltx_2_5_fast",
                                    ],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "LTXTextToVideoGeneration",
                        "position": {"x": -1539.348686441772, "y": -205.6155909712729},
                        "size": {"width": 600, "height": 1577},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="ResolveMacroPath",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Resolve Macro Path",
                    metadata={
                        "library_node_metadata": {
                            "category": "files",
                            "description": "Resolve a macro path to an absolute filesystem path (e.g. {inputs}/file.txt → /home/user/project/inputs/file.txt).",
                            "display_name": "Resolve Macro Path",
                            "tags": ["file", "macro", "path"],
                            "icon": "FolderSearch",
                            "color": None,
                            "group": None,
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "ResolveMacroPath",
                        "position": {"x": -840.4956381925438, "y": 1002.7266045955726},
                        "size": {"width": 600, "height": 388},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TopazVideoUpscale",
                    specific_library_name="Griptape Nodes Library",
                    node_name="TopazVideoUpscale",
                    metadata={
                        "library_node_metadata": {
                            "category": "video",
                            "description": "Upscale a video using Topaz Starlight Precise or Astra 2 generative upscaling via the Griptape model proxy.",
                            "display_name": "Topaz Video Upscale",
                            "tags": [
                                "video",
                                "upscale",
                                "enhance",
                                "generative",
                                "ai",
                                "api",
                                "topaz",
                                "starlight",
                                "astra",
                            ],
                            "icon": "Sparkles",
                            "color": None,
                            "group": "edit",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": [
                                        "gtc_topaz_video_slp_2_5",
                                        "gtc_topaz_video_slp_2_6",
                                        "gtc_topaz_video_ast_2",
                                    ],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "TopazVideoUpscale",
                        "size": {"width": 604, "height": 981},
                        "position": {"x": -144.91154341217586, "y": 107.56433235834346},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="prompt",
                    ui_options={
                        "multiline": True,
                        "placeholder_text": "Optional: describe the detail Astra should generate...",
                        "hide": False,
                        "hide_label": False,
                        "hide_property": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="creativity",
                    ui_options={
                        "slider": {"min_val": 0.0, "max_val": 1.0},
                        "hide": False,
                        "hide_label": False,
                        "hide_property": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="sharp",
                    ui_options={
                        "slider": {"min_val": 0.0, "max_val": 1.0},
                        "hide": False,
                        "hide_label": False,
                        "hide_property": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="realism",
                    ui_options={
                        "slider": {"min_val": 0.0, "max_val": 1.0},
                        "hide": False,
                        "hide_label": False,
                        "hide_property": False,
                    },
                    initial_setup=True,
                )
            )
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="AssertFileExists",
                    specific_library_name="Griptape Nodes Testing Library",
                    node_name="Assert File Exists",
                    metadata={
                        "library_node_metadata": {
                            "category": "assert",
                            "description": "Asserts that a file exists at the given path.",
                            "display_name": "Assert File Exists",
                            "tags": None,
                            "icon": "ShieldCheck",
                            "color": None,
                            "group": "assert",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Testing Library",
                        "node_type": "AssertFileExists",
                        "position": {"x": 564.812087625447, "y": 1042.9949009588793},
                        "size": {"width": 602, "height": 426},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterGroupDetailsRequest(
                    group_name="Status", ui_options={"collapsed": False}, initial_setup=True
                )
            )
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="EndFlow",
                    specific_library_name="Griptape Nodes Library",
                    node_name="End Flow",
                    metadata={
                        "library_node_metadata": {
                            "category": "workflows",
                            "description": "Define the end of a workflow and return parameters from the flow",
                            "display_name": "End Flow",
                            "tags": ["workflow", "execution"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "EndFlow",
                        "showaddparameter": True,
                        "position": {"x": 1386.7059489369328, "y": 1046.7266045955726},
                        "size": {"width": 600, "height": 300},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="result_details_1",
                    default_value="",
                    tooltip="New parameter",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "multiline": True,
                        "placeholder_text": "Details on the assertion will appear here.",
                        "is_custom": True,
                        "is_user_added": True,
                    },
                    parent_container_name="",
                    initial_setup=True,
                )
            )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node2_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="video_url",
                target_node_name=node2_name,
                target_parameter_name="video",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="resolved_path",
                target_node_name=node2_name,
                target_parameter_name="output_file",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="exec_out",
                target_node_name=node3_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="resolved_path",
                target_node_name=node3_name,
                target_parameter_name="file_path",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="exec_out",
                target_node_name=node4_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="result_details",
                target_node_name=node4_name,
                target_parameter_name="result_details_1",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["9f6db6d9-0ec6-48d8-814b-d78446435f81"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["2c47a793-6843-4f7f-b84d-4c3a360f0c1d"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["9a9014d9-002e-4864-b6bf-dc84403408c8"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolution",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["462e5481-0b4a-46c1-93d5-83a8cf4bcf36"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="duration",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["c292dcaa-a4d5-4a36-9500-60fbfa5ccdf0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="fps",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["6f6a2abd-50b2-4b95-8812-5884b8b3fbf1"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="camera_motion",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["022cfd25-ce47-49cf-bfb4-24c50338f667"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generate_audio",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["791aecc4-b679-460c-ac81-46a7f5cc2be3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["a0ce2d66-bf9c-43a9-b3a7-9a4166bad54a"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="path",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["322a0fe0-a8f8-4c0c-bdb8-9629cc2890ae"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolved_path",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["7887189e-b73f-458d-a4dc-ea5c2b6dbb0e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["5c27bbcf-5e7b-4184-8571-234d89b2c18e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            # The prompt is what this workflow exists to exercise: `filters` passthrough is
            # the one part of the Astra design that unit tests cannot verify, because it rests
            # on the proxy forwarding our filter dict verbatim and stamping the model code onto
            # it itself.
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["604e16c6-e39c-4eb6-b49e-6bc591b83ac3"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="creativity",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["6416a526-e1a9-4fb1-b789-57c99361eacb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="sharp",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["6416a526-e1a9-4fb1-b789-57c99361eacb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="realism",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["6416a526-e1a9-4fb1-b789-57c99361eacb"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resize_mode",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a2116863-f0eb-4dad-bb44-d3b2fc64c388"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_size",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["33a4012a-d6e4-44db-9f4f-42b1d0e6d997"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_width",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["54734eee-568e-4dac-accc-677192143e63"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="target_height",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["99643af8-b0eb-496b-a556-a730e1a32418"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="percentage",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["cedb9638-bf44-49c9-87e5-735eb9eae429"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a53e76d5-7461-4ae8-b2f9-34ec5a65b8a6"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="file_path",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="message",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["59f51e04-196c-457b-bb49-cb76ea845e8b"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details_1",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["da388b5f-30a9-48fe-aafc-2781b7207f73"],
                    initial_setup=True,
                    is_output=False,
                )
            )
