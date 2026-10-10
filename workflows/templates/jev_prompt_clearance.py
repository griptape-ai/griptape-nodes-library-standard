# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "jev_prompt_clearance"
# schema_version = "0.21.0"
# engine_version_created_with = "0.104.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "Agent"], ["Griptape Nodes Library", "CreateVariable"], ["Griptape Nodes Library", "GrokImageGeneration"], ["Griptape Nodes Library", "JevAskYesNo"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "SetVariable"], ["Griptape Nodes Library", "TextInput"]]
# description = "Checks an image prompt for real people and trademarked characters before generating. If JEV finds any, an Agent rewrites the prompt with original characters, so the image comes from a prompt that's safe to use."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_jev_prompt_clearance.webp"
# is_griptape_provided = true
# is_template = true
# is_internal = false
# creation_date = 2026-09-28T04:37:07.591842Z
# last_modified_date = 2026-10-08T01:05:53.584005Z
#
# ///

from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AlterParameterDetailsRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.events.variable_events import CreateVariableRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.serialization.values import decode_value


async def build_workflow() -> None:
    await GriptapeNodes.ahandle_request(
        RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
    )
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)
    # Every unique parameter value, stored once and keyed by a hash of its content.
    # Values that aren't plain data carry a '$type' naming their class; decode_value rebuilds them.
    top_level_unique_values_dict = {
        "7d58514cc83cc992037e56dc1f6f745d": "PROMPT",
        "fcbcf165908dd18a9e49f7ff27810176": False,
        "1e1a6bbb783263b9b6247e30b217d6ba": {
            "$type": "griptape_nodes.node_libraries.griptape_nodes_library.create_variable:CaseStyle",
            "$value": "snake_case",
        },
        "72495b0e1f3c6c961f46d3f249ce968a": "str",
        "d147124a81a1f5634d37f37626c91677": "Count Dracula standing on the stone balcony of a crumbling castle during a lightning storm, wind tearing at his cloak, rain sheeting across the frame, cold blue moonlight with warm candlelight spilling from the doorway behind him, low angle, anamorphic 2.39:1, shot on 35mm film, gothic horror concept art",
        "25a6de226672088d980bed6d8e0c2162": "hierarchical",
        "43b1ebed9a4dfca12dbf36745428bb30": "## Prompt Clearance with JEV\n\nChecks an image prompt for real people and trademarked characters before you pay to generate it.\n\n- **No:** the prompt goes straight to **Generate Image**.\n- **Yes:** an Agent rewrites the prompt with original characters first.\n\nEither way you get one image, from a prompt that's safe to use.\n\n**Try it:** run the flow as it is and Count Dracula takes the Yes branch. Then replace Dracula with an original character and run it again to see the No branch.\n\n",
        "99b125e433b7a10e130c9a5e9b33d791": "## Store the prompt\n\n**Store Prompt** saves your prompt in a variable called `PROMPT`. Other nodes read it when you type `{PROMPT}` into a text field.\n\nThe variable is what lets both branches share one image node. If the prompt gets rewritten, **Replace Prompt** updates the variable, and **Generate Image** uses whichever version is there when it runs.",
        "3fa455b032a56c96534ec11db4716100": "## Ask JEV\n\n**Ask Yes/No (JEV)** reads `{PROMPT}` and answers the question with a probability from 0 to 1.\n\n- **Say Yes at or above** is set to 0.6. Lower it to catch more borderline prompts. Raise it to rewrite less often.\n- After a run, check **Probability** to see how sure JEV was.\n- JEV flags public-domain characters like Dracula too. To let them through, open **Define Yes and No** and write *Public-domain characters like Dracula or Sherlock Holmes* in **No means**.",
        "3699446c655cfdf0e2e26dfb1090b2b2": "## Rewrite on Yes\n\n**Rewrite Without IP** runs only when JEV says Yes. It swaps each person or character for an original description and keeps the rest of the scene.\n\nAlways look over the rewrite before using the image. For an extra check, add a second **Ask Yes/No** after the Agent with the same question.",
        "d9335d9e14f4c5b9caedd0f993e59c09": "## One image node for both branches\n\nThe No branch and Replace Prompt both connect to **Generate Image**'s Flow In, so it runs once whichever way JEV answered, and reads `{PROMPT}` when it runs.\n\nKeep expensive nodes like image generation after the Yes/No branch so they only run when needed.",
        "7e54864de79e88805b058e70b2a47535": "## Replace Prompt Variable\n\n**Replace Prompt** writes the rewrite into the `PROMPT` variable.",
        "284b7e6d788f363f910f7beb1910473e": 600,
        "7117fe6b6193e23a141ffc11d62b3f9e": "grok-imagine-image",
        "ea54873a3d5ac8b24056edfd2f87a2d3": "{PROMPT}",
        "0c607131a93856ed93d33dad0c3829fb": "16:9",
        "6b86b273ff34fce19d6b804eff5a3f57": 1,
        "60d4c90eee5e731df8d3ef2891de541d": "medium",
        "16d2724fe0a9971ec931e119652d59e5": "1k",
        "ee0b3aa7846d4f2b6fcd984dd714432a": "grok_image.jpg",
        "12ae32cb1ec02d01eda3581b127c1fee": "",
        "406b626b318c054d61c9debe1e232629": "Does this image prompt ask for a real, identifiable person or a trademarked character?",
        "b4af4e0d40391d3a00179d935c63b7e1": 0.6,
        "93d926fb5a12ad7dc7179ed59e1b66ac": "jev-latest",
        "d99102b775c65ed8a31e518cbd14adc8": "griptape_cloud",
        "30c1f92362fbf9ad1f8aae0e41fc431d": "claude-sonnet-5",
        "44136fa355b3678a1146ad16f7e8649e": {},
        "cecd9ea72dc43bd93e723e24dea4d61f": "You rewrite image generation prompts so they contain no real people and no copyrighted or trademarked characters, brands, or logos.\n\nFor each person, character, or brand in the prompt, replace the name with a short visual description of a new, original version: age, build, clothing, hair, mood. The result must not be recognizable as the original, so leave out signature costumes, masks, helmets, props, sounds, and catchphrases.\n\nKeep everything else in the prompt exactly as written: setting, action, lighting, camera, and style.\n\nExample:  \nInput: Sherlock Holmes examining a clue on a foggy London street at night, cinematic concept art  \nOutput: A sharp-featured man in his 40s in a dark wool overcoat, examining a clue on a foggy London street at night, cinematic concept art\n\nReply with only the rewritten prompt, with no quotes and no explanation.\n\nPrompt to rewrite: {PROMPT}",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5": [],
        "c9cb3a7dbd3efd2a3d5a6c6bbd354e4b": "A tall, pale-skinned man in his 50s with slicked-back dark hair and a long black cloak, standing on the stone balcony of a crumbling castle during a lightning storm, wind tearing at his cloak, rain sheeting across the frame, cold blue moonlight with warm candlelight spilling from the doorway behind him, low angle, anamorphic 2.39:1, shot on 35mm film, gothic horror concept art",
    }
    # Create the Flow, then do work within it as context.
    flow0_name = (
        await GriptapeNodes.ahandle_request(
            CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
        )
    ).flow_name
    with GriptapeNodes.ContextManager().flow(flow0_name):
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="PROMPT",
                type="str",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["c9cb3a7dbd3efd2a3d5a6c6bbd354e4b"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Prompt",
                    metadata={
                        "position": {"x": -352, "y": 0},
                        "tempId": "placing-1790571513501-kr1h3",
                        "library_node_metadata": {
                            "category": "variables",
                            "description": "Create or update a variable with a specified name, type, and value. Creates new variables or updates existing ones intelligently.",
                            "display_name": "Create Variable",
                            "tags": ["data", "variable", "workflow"],
                            "icon": "PlusCircle",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "CreateVariable",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 328},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(parameter_name="value", type="str", output_type="str", initial_setup=True)
            )
        node1_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="SetVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Replace Prompt",
                    metadata={
                        "position": {"x": 2068, "y": 176},
                        "tempId": "placing-1790571586968-pxww19",
                        "library_node_metadata": {
                            "category": "variables",
                            "description": "Set the value of a variable, creating it if it does not exist",
                            "display_name": "Set Variable",
                            "tags": ["data", "variable", "workflow"],
                            "icon": "ArrowUp",
                            "color": None,
                            "group": "edit",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "SetVariable",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 344},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="str", input_types=["str"], output_type="str", initial_setup=True
                )
            )
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="About This Workflow",
                    metadata={
                        "position": {"x": -1144, "y": -620},
                        "size": {"width": 644, "height": 540},
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
                        "color": "#047857",
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Store the Prompt",
                    metadata={
                        "position": {"x": -352, "y": -620},
                        "size": {"width": 600, "height": 540},
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
                        "color": "#0e7490",
                        "showaddparameter": False,
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Ask JEV",
                    metadata={
                        "position": {"x": 528, "y": -620},
                        "size": {"width": 600, "height": 540},
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
                        "color": "#0e7490",
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Rewrite on Yes",
                    metadata={
                        "position": {"x": 1364, "y": -616},
                        "size": {"width": 600, "height": 540},
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
                        "color": "#0e7490",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: One Image Node",
                    metadata={
                        "position": {"x": 2860, "y": -616},
                        "size": {"width": 600, "height": 500},
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
                        "color": "#047857",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="IP Prompt",
                    metadata={
                        "library_node_metadata": {
                            "category": "text",
                            "description": "Create text with an input node.",
                            "display_name": "Text Input",
                            "tags": ["text", "input", "create"],
                            "icon": "text-cursor",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "TextInput",
                        "position": {"x": -1144, "y": 0},
                        "showaddparameter": False,
                        "size": {"width": 644, "height": 280},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Replace Prompt",
                    metadata={
                        "position": {"x": 2068, "y": -616},
                        "size": {"width": 600, "height": 540},
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
                        "color": "#0e7490",
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="GrokImageGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Generate Image",
                    metadata={
                        "library_node_metadata": {
                            "category": "image",
                            "description": "Generate images using Grok models via Griptape model proxy",
                            "display_name": "Grok Image Generation",
                            "tags": ["image", "generation", "ai", "api", "grok", "xai"],
                            "icon": "Sparkles",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [{"type": "model_usage", "model_ids": ["gtc_grok_imagine_image"]}],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "GrokImageGeneration",
                        "position": {"x": 2860, "y": 0},
                        "size": {"width": 600, "height": 904},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="prompt",
                    ui_options={
                        "multiline": False,
                        "placeholder_text": "Describe the image you want to generate...",
                        "hide_label": False,
                        "hide_property": False,
                        "hide": False,
                    },
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="output_file",
                    traits=[
                        {"trait_name": "Button", "trait_module": "griptape_nodes.traits.button", "trait_state": {}},
                        {
                            "trait_name": "FileSystemPicker",
                            "trait_module": "griptape_nodes.traits.file_system_picker",
                            "trait_state": {},
                        },
                    ],
                    initial_setup=True,
                )
            )
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevAskYesNo",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Ask Yes/No (JEV)",
                    metadata={
                        "position": {"x": 504, "y": 0},
                        "tempId": "placing-1791419695239-ykoxjd",
                        "library_node_metadata": {
                            "category": "classification",
                            "description": "Ask a yes/no question about text using TypeSafe JEV.",
                            "display_name": "Ask Yes/No (JEV)",
                            "tags": None,
                            "icon": "check-circle",
                            "color": None,
                            "group": "classification",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevAskYesNo",
                        "showaddparameter": False,
                        "size": {"width": 636, "height": 940},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Agent",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Rewrite Without IP",
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
                                        "gtc_claude_sonnet_5_5",
                                        "gtc_claude_opus_5_5",
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
                                        "gtc_gpt_6_sol",
                                        "gtc_gpt_6_luna",
                                        "gtc_gpt_5_6_sol",
                                        "gtc_gpt_5_6_terra",
                                        "gtc_gpt_5_6_luna",
                                        "gtc_gpt_5_5",
                                        "gtc_gpt_5_4",
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
                        "position": {"x": 1364, "y": 176},
                        "size": {"width": 600, "height": 908},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node7_name,
                source_parameter_name="exec_out",
                target_node_name=node0_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node7_name,
                source_parameter_name="text",
                target_node_name=node0_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node1_name,
                source_parameter_name="exec_out",
                target_node_name=node9_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node10_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="no",
                target_node_name=node9_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="yes",
                target_node_name=node11_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node10_name,
                source_parameter_name="failure",
                target_node_name=node11_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="output",
                target_node_name=node1_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="exec_out",
                target_node_name=node1_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["d147124a81a1f5634d37f37626c91677"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["d147124a81a1f5634d37f37626c91677"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="scope",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["25a6de226672088d980bed6d8e0c2162"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["43b1ebed9a4dfca12dbf36745428bb30"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["99b125e433b7a10e130c9a5e9b33d791"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["3fa455b032a56c96534ec11db4716100"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=decode_value(top_level_unique_values_dict["3699446c655cfdf0e2e26dfb1090b2b2"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["d9335d9e14f4c5b9caedd0f993e59c09"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["d147124a81a1f5634d37f37626c91677"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["d147124a81a1f5634d37f37626c91677"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["7e54864de79e88805b058e70b2a47535"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["7117fe6b6193e23a141ffc11d62b3f9e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["ea54873a3d5ac8b24056edfd2f87a2d3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="aspect_ratio",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["0c607131a93856ed93d33dad0c3829fb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="n",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["6b86b273ff34fce19d6b804eff5a3f57"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quality",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["60d4c90eee5e731df8d3ef2891de541d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolution",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["16d2724fe0a9971ec931e119652d59e5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["ee0b3aa7846d4f2b6fcd984dd714432a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["ea54873a3d5ac8b24056edfd2f87a2d3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["406b626b318c054d61c9debe1e232629"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="yes_means",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="no_means",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="threshold",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["b4af4e0d40391d3a00179d935c63b7e1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model_provider",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["d99102b775c65ed8a31e518cbd14adc8"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["30c1f92362fbf9ad1f8aae0e41fc431d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="agent_memory",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["44136fa355b3678a1146ad16f7e8649e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["cecd9ea72dc43bd93e723e24dea4d61f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_context",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="tools",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="rulesets",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="include_details",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
