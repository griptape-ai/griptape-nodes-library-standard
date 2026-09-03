# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "coordinating_agents"
# schema_version = "0.20.0"
# engine_version_created_with = "0.100.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.85.0"]]
# node_types_used = [["Griptape Nodes Library", "Agent"], ["Griptape Nodes Library", "DisplayText"], ["Griptape Nodes Library", "MergeTexts"], ["Griptape Nodes Library", "Note"]]
# description = "Multiple agents with different jobs."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_coordinating_agents.webp"
# is_griptape_provided = true
# is_template = true
# creation_date = 2025-10-22T19:03:54.190207Z
# last_modified_date = 2026-09-03T14:55:37.079195Z
#
# ///

import pickle

from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    SetParameterValueRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
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
        "15d872a2-3381-47a6-952c-602c425316fe": pickle.loads(
            b'\x80\x04\x95\x98\x01\x00\x00\x00\x00\x00\x00X\x91\x01\x00\x00This workflow serves as the lesson material for the tutorial located at:\n\nhttps://docs.griptapenodes.com/en/stable/ftue/02_coordinating_agents/FTUE_02_coordinating_agents/\n\nThe concepts covered are:\n\n- Multi-agent workflows where agents have different "jobs"\n- How to use Merge Text nodes to better pass information between agents\n- Understanding execution chains to control the order things happen in\x94.'
        ),
        "77c89870-326b-4e8d-8d55-ef20e1ca8867": pickle.loads(
            b"\x80\x04\x95\xf6\x00\x00\x00\x00\x00\x00\x00\x8c\xf2If you're following along with our Getting Started tutorials, check out the next suggested template: Compare_Prompts.\n\nLoad the next tutorial page here:\nhttps://docs.griptapenodes.com/en/stable/ftue/03_compare_prompts/FTUE_03_compare_prompts/\x94."
        ),
        "926aa940-fdc8-4fa1-b2c8-61530d6f4b0e": pickle.loads(
            b"\x80\x04\x95\x12\x00\x00\x00\x00\x00\x00\x00\x8c\x0egriptape_cloud\x94."
        ),
        "87e6664d-9ec8-4192-a21b-b4db2e435860": pickle.loads(
            b"\x80\x04\x95\n\x00\x00\x00\x00\x00\x00\x00\x8c\x06gpt-4o\x94."
        ),
        "a6cbe774-374f-48f8-ab3e-1dab344e6590": pickle.loads(b"\x80\x04}\x94."),
        "f6e85633-8f7c-4301-8e03-17260ba81330": pickle.loads(
            b'\x80\x04\x95&\x00\x00\x00\x00\x00\x00\x00\x8c"Write me a 4-line story in Spanish\x94.'
        ),
        "577dedcb-eb49-4094-a9e4-1caca4e84063": pickle.loads(
            b"\x80\x04\x95\x04\x00\x00\x00\x00\x00\x00\x00\x8c\x00\x94."
        ),
        "ece96dc8-578c-4f01-8a00-b6a537333089": pickle.loads(b"\x80\x04]\x94."),
        "48acdbb5-793d-4527-8b6f-61d56cd8f5c9": pickle.loads(b"\x80\x04]\x94."),
        "39ac2fc7-df87-424d-8b8a-bc1a5384ca5e": pickle.loads(
            b"\x80\x04\x95\x9e\x00\x00\x00\x00\x00\x00\x00\x8c\x9aBajo la luna, el r\xc3\xado cant\xc3\xb3,  \nUn secreto antiguo en su agua dej\xc3\xb3.  \nLa ni\xc3\xb1a lo escuch\xc3\xb3 y empez\xc3\xb3 a so\xc3\xb1ar,  \nQue el mundo era suyo, listo para amar.\n\x94."
        ),
        "6097c6f4-6688-48a8-975f-ababb4bac7a2": pickle.loads(b"\x80\x04\x89."),
        "8fcfa1bc-b83f-4621-a3a3-8997f51f8041": pickle.loads(b"\x80\x04}\x94."),
        "38c87df4-d5a7-45b0-abc1-2851a25483b2": pickle.loads(
            b"\x80\x04\x95\xb6\x00\x00\x00\x00\x00\x00\x00\x8c\xb2rewrite this in english\n\nBajo la luna, el r\xc3\xado cant\xc3\xb3,  \nUn secreto antiguo en su agua dej\xc3\xb3.  \nLa ni\xc3\xb1a lo escuch\xc3\xb3 y empez\xc3\xb3 a so\xc3\xb1ar,  \nQue el mundo era suyo, listo para amar.\x94."
        ),
        "9075cdcd-54e4-418b-ab12-4e2455e26cc2": pickle.loads(b"\x80\x04]\x94."),
        "9ea4a5c6-6077-4ecd-a04c-890416f541e9": pickle.loads(b"\x80\x04]\x94."),
        "fe2ab08b-9871-4025-8d9f-f6a382cfd69e": pickle.loads(
            b"\x80\x04\x95\xa4\x00\x00\x00\x00\x00\x00\x00\x8c\xa0Beneath the moon, the river sang,  \nAn ancient secret in its waters it rang.  \nThe girl heard it and began to dream,  \nThat the world was hers, ready to gleam.\n\x94."
        ),
        "990ec3bd-894e-48e8-b988-ba6ca8c01cb9": pickle.loads(
            b"\x80\x04\x95\x1b\x00\x00\x00\x00\x00\x00\x00\x8c\x17rewrite this in english\x94."
        ),
        "a5c4790a-b4cb-45d5-9b27-f135e3a8f8d0": pickle.loads(
            b"\x80\x04\x95\x06\x00\x00\x00\x00\x00\x00\x00\x8c\x02\n\n\x94."
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
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="ReadMe",
                    metadata={
                        "position": {"x": -550, "y": -400},
                        "size": {"width": 1000, "height": 350},
                        "library_node_metadata": {
                            "category": "misc",
                            "description": "Create a note node to provide helpful context in your workflow",
                            "display_name": "Note",
                            "tags": ["workflow", "annotation", "note"],
                            "icon": "notepad-text",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="NextStep",
                    metadata={
                        "position": {"x": 2370.745229127131, "y": 599},
                        "size": {"width": 1100, "height": 232},
                        "library_node_metadata": {
                            "category": "misc",
                            "description": "Create a note node to provide helpful context in your workflow",
                            "display_name": "Note",
                            "tags": ["workflow", "annotation", "note"],
                            "icon": "notepad-text",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "category": "misc",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Agent",
                    specific_library_name="Griptape Nodes Library",
                    node_name="spanish_story",
                    metadata={
                        "library_node_metadata": {
                            "category": "agents",
                            "description": "Creates an AI agent with conversation memory and the ability to use tools",
                            "display_name": "Agent",
                            "tags": ["agent", "ai", "llm", "conversation", "memory"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": [
                                        "gtc_claude_sonnet_5",
                                        "gtc_claude_opus_5",
                                        "gtc_claude_haiku_4_5",
                                        "gtc_gemini_3_6_flash",
                                        "gtc_gemini_3_5_flash",
                                        "gtc_gemini_3_5_flash_lite",
                                        "gtc_gemini_3_1_pro",
                                        "gtc_gemini_3_1_flash_lite",
                                        "gtc_gemini_3_flash",
                                        "gtc_gemini_2_5_pro",
                                        "gtc_gemini_2_5_flash",
                                        "gtc_gemini_2_5_flash_lite",
                                        "gtc_gpt_5_2",
                                        "gtc_gpt_5_2_chat",
                                        "gtc_gpt_5_1",
                                        "gtc_gpt_5",
                                        "gtc_gpt_5_mini",
                                        "gtc_gpt_5_nano",
                                        "gtc_gpt_4_1",
                                        "gtc_gpt_4_1_mini",
                                        "gtc_gpt_4_1_nano",
                                        "gtc_gpt_4o",
                                        "gtc_o4_mini",
                                        "gtc_o3",
                                        "gtc_o3_mini",
                                        "gtc_o1",
                                        "gtc_deepseek_v3",
                                        "gtc_deepseek_r1",
                                        "gtc_llama_3_3_70b",
                                        "gtc_llama_3_1_70b",
                                    ],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Agent",
                        "category": "agents",
                        "position": {"x": -550, "y": 0},
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 864},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Agent",
                    specific_library_name="Griptape Nodes Library",
                    node_name="to_english",
                    metadata={
                        "library_node_metadata": {
                            "category": "agents",
                            "description": "Creates an AI agent with conversation memory and the ability to use tools",
                            "display_name": "Agent",
                            "tags": ["agent", "ai", "llm", "conversation", "memory"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": [
                                        "gtc_claude_sonnet_5",
                                        "gtc_claude_opus_5",
                                        "gtc_claude_haiku_4_5",
                                        "gtc_gemini_3_6_flash",
                                        "gtc_gemini_3_5_flash",
                                        "gtc_gemini_3_5_flash_lite",
                                        "gtc_gemini_3_1_pro",
                                        "gtc_gemini_3_1_flash_lite",
                                        "gtc_gemini_3_flash",
                                        "gtc_gemini_2_5_pro",
                                        "gtc_gemini_2_5_flash",
                                        "gtc_gemini_2_5_flash_lite",
                                        "gtc_gpt_5_2",
                                        "gtc_gpt_5_2_chat",
                                        "gtc_gpt_5_1",
                                        "gtc_gpt_5",
                                        "gtc_gpt_5_mini",
                                        "gtc_gpt_5_nano",
                                        "gtc_gpt_4_1",
                                        "gtc_gpt_4_1_mini",
                                        "gtc_gpt_4_1_nano",
                                        "gtc_gpt_4o",
                                        "gtc_o4_mini",
                                        "gtc_o3",
                                        "gtc_o3_mini",
                                        "gtc_o1",
                                        "gtc_deepseek_v3",
                                        "gtc_deepseek_r1",
                                        "gtc_llama_3_3_70b",
                                        "gtc_llama_3_1_70b",
                                    ],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Agent",
                        "category": "agents",
                        "position": {"x": 887.3951454918155, "y": 0},
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 865},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="MergeTexts",
                    specific_library_name="Griptape Nodes Library",
                    node_name="prompt_header",
                    metadata={
                        "library_node_metadata": {
                            "category": "text",
                            "description": "MergeTexts node",
                            "display_name": "Merge Texts",
                            "tags": ["text", "combine"],
                            "icon": "merge",
                            "color": None,
                            "group": "merge",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "MergeTexts",
                        "category": "text",
                        "position": {"x": 171.32755537785516, "y": 296},
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 568},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="DisplayText",
                    specific_library_name="Griptape Nodes Library",
                    node_name="english_story",
                    metadata={
                        "library_node_metadata": {
                            "category": "text",
                            "description": "DisplayText node",
                            "display_name": "Display Text",
                            "tags": ["text", "display"],
                            "icon": None,
                            "color": None,
                            "group": "display",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "DisplayText",
                        "category": "text",
                        "position": {"x": 1661.6984368752724, "y": 599},
                        "size": {"width": 600, "height": 265},
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
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
                source_node_name=node2_name,
                source_parameter_name="output",
                target_node_name=node4_name,
                target_parameter_name="input_2",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="output",
                target_node_name=node5_name,
                target_parameter_name="text",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="output",
                target_node_name=node3_name,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node0_name,
                    value=top_level_unique_values_dict["15d872a2-3381-47a6-952c-602c425316fe"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node1_name,
                    value=top_level_unique_values_dict["77c89870-326b-4e8d-8d55-ef20e1ca8867"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model_provider",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["926aa940-fdc8-4fa1-b2c8-61530d6f4b0e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["87e6664d-9ec8-4192-a21b-b4db2e435860"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="agent_memory",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["a6cbe774-374f-48f8-ab3e-1dab344e6590"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["f6e85633-8f7c-4301-8e03-17260ba81330"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_context",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["577dedcb-eb49-4094-a9e4-1caca4e84063"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="tools",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["ece96dc8-578c-4f01-8a00-b6a537333089"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="rulesets",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["48acdbb5-793d-4527-8b6f-61d56cd8f5c9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["39ac2fc7-df87-424d-8b8a-bc1a5384ca5e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="include_details",
                    node_name=node2_name,
                    value=top_level_unique_values_dict["6097c6f4-6688-48a8-975f-ababb4bac7a2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model_provider",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["926aa940-fdc8-4fa1-b2c8-61530d6f4b0e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["87e6664d-9ec8-4192-a21b-b4db2e435860"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="agent_memory",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["8fcfa1bc-b83f-4621-a3a3-8997f51f8041"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["38c87df4-d5a7-45b0-abc1-2851a25483b2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_context",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["577dedcb-eb49-4094-a9e4-1caca4e84063"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="tools",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["9075cdcd-54e4-418b-ab12-4e2455e26cc2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="rulesets",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["9ea4a5c6-6077-4ecd-a04c-890416f541e9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["fe2ab08b-9871-4025-8d9f-f6a382cfd69e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="include_details",
                    node_name=node3_name,
                    value=top_level_unique_values_dict["6097c6f4-6688-48a8-975f-ababb4bac7a2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_1",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["990ec3bd-894e-48e8-b988-ba6ca8c01cb9"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_2",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["39ac2fc7-df87-424d-8b8a-bc1a5384ca5e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="merge_string",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["a5c4790a-b4cb-45d5-9b27-f135e3a8f8d0"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="whitespace",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["6097c6f4-6688-48a8-975f-ababb4bac7a2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["38c87df4-d5a7-45b0-abc1-2851a25483b2"],
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node4_name,
                    value=top_level_unique_values_dict["38c87df4-d5a7-45b0-abc1-2851a25483b2"],
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node5_name,
                    value=top_level_unique_values_dict["fe2ab08b-9871-4025-8d9f-f6a382cfd69e"],
                    initial_setup=True,
                    is_output=False,
                )
            )
