# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "jev_shot_check"
# schema_version = "0.21.0"
# engine_version_created_with = "0.104.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "Agent"], ["Griptape Nodes Library", "CreateVariable"], ["Griptape Nodes Library", "GrokImageGeneration"], ["Griptape Nodes Library", "JevRate"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "SetVariable"], ["Griptape Nodes Library", "TextInput"]]
# description = "JEV rates how ready a shot description is for an image generator. Vague or partial shots get an Agent rewrite that fills in the subject, action, camera angle, and lighting. Ready shots go straight to the image."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_jev_shot_check.webp"
# is_griptape_provided = true
# is_template = true
# is_internal = false
# creation_date = 2026-09-28T04:37:07.591842Z
# last_modified_date = 2026-10-08T01:04:54.308908Z
#
# ///

from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import (
    AddParameterToNodeRequest,
    AlterParameterDetailsRequest,
    SetParameterValueRequest,
)
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
        "0af5395a77978c1dbafd1add9349f4f3": "Low-angle wide shot looking up the spiral staircase of a lighthouse as an old keeper climbs toward the lamp room, lantern raised, his yellow oilskin coat dripping. Rain streams down the tall windows and lightning flashes outside. His lantern gives a warm amber key light, and the cold blue storm light from the windows fills the rest. Heavy shadows, 35mm film look, tense and lonely.",
        "1813c1dc118fdfd2e27110f8fa148f62": "## Shot Check with JEV\n\nJEV rates how ready a shot description is for an image generator, then sends it down a branch based on the result.\n\n- **Vague** or **Partial:** an Agent fills in what's missing, like the camera angle, framing, and lighting. The image is made from the rewrite.\n- **Ready:** the shot goes straight to the image node, with no rewrite.\n\nGood shots skip the Agent, and weak ones get fixed before you spend an image generation.\n\n**Try it:** run the flow as it is. Then drag the **Text** output of **Vague Shot**, **Partial Shot**, or **Ready Shot** onto **Store Prompt**'s **Value** and run it again to see where each one goes. Or write your own shot in any of them.\n\n",
        "0ee5b563eaa24ecf4369b0499be7eeb9": "## Store the prompt\n\n**Store Prompt** saves the shot description in a variable called `PROMPT`. Other nodes read it when you type `{PROMPT}` into a text field.\n\n**Rate** and the **Agent** read `{PROMPT}`, and the image node's prompt is just `{PROMPT}`, so every node works from the same shot.\n\nStore Prompt runs first on every run, so each run starts from the shot you connected, not from the last run's rewrite.",
        "4cc36534b2474d787bf3a98a53bacd7a": "A lighthouse keeper during a storm.",
        "3d617e36e7a8e10ec0622af73ba94c91": "An old lighthouse keeper climbs the spiral stairs with a lantern as a storm batters the windows, rain streaming down the glass.",
        "25a6de226672088d980bed6d8e0c2162": "hierarchical",
        "284b7e6d788f363f910f7beb1910473e": 600,
        "7117fe6b6193e23a141ffc11d62b3f9e": "grok-imagine-image",
        "ea54873a3d5ac8b24056edfd2f87a2d3": "{PROMPT}",
        "0c607131a93856ed93d33dad0c3829fb": "16:9",
        "6b86b273ff34fce19d6b804eff5a3f57": 1,
        "60d4c90eee5e731df8d3ef2891de541d": "medium",
        "16d2724fe0a9971ec931e119652d59e5": "1k",
        "ee0b3aa7846d4f2b6fcd984dd714432a": "grok_image.jpg",
        "12ae32cb1ec02d01eda3581b127c1fee": "",
        "bb6cb47bd207dc9b9aaa1e847bcbdcc3": '## Rate the shot\n\n**Rate (JEV)** reads `{PROMPT}` and rates it against the rows in **Levels**, lowest first. Each row is a label, a colon, and a description of what a shot at that level contains. JEV only sees the descriptions.\n\nEach label becomes a flow output, and the flow continues from the level the shot scores. **Vague** and **Partial** both connect to the Agent, so they share one branch.\n\n- After a run, check **Score**, **Confidence**, and **Probabilities** to see how sure JEV was. A score of 2.4 means it leaned toward Partial, with some pull toward Ready.\n- Describe what a shot contains at each level, not how good it is. "Leaves out the camera angle" works better than "Okay".',
        "a89f881edc3b69b9590403dee6236b3b": "## Generate the image\n\nThe image node's prompt is just `{PROMPT}`. Its Flow In has two wires, one from **Ready** and one from **Set Variable**, and it runs when either one arrives.\n\nEither way, `PROMPT` holds the shot to draw: the original if it scored Ready, or the Agent's rewrite if it didn't.\n\n- To use a different model, swap the node, set its prompt to `{PROMPT}`, and connect both wires to its Flow In.",
        "3f457135186609800e5a046ad4b64df1": "## Fill in the shot\n\nThe **Agent** only runs when the shot scores **Vague** or **Partial**. Its prompt includes `{PROMPT}` and asks it to add the camera angle, framing, lighting, and mood, while keeping everything already in the shot.\n\nIt's told to reply with only the shot description, because its output becomes the new prompt. Any extra text would end up in the image.\n\n- To change what gets added, edit the list in the Agent's prompt.",
        "86e021496d4679f80b41df308e5ba55b": "## Update the prompt\n\n**Set Variable** writes the Agent's rewrite into `PROMPT`, replacing the original shot. Then it passes the flow on to the image node.\n\nTo see what the Agent added, compare its **Output** with the shot you started from.",
        "400bcab8a5bd3d2860d187621258c84f": "How ready is this shot description to hand to an image generation node?",
        "85cd8fa13c956b16a587184fb13600dc": [
            "Vague: names a subject or idea, but not what's happening, where, or how it's framed",
            "Partial: describes the subject, the action, and the setting, but leaves out the camera angle, framing, or lighting",
            "Ready: gives the subject, the action, the setting, the camera angle and framing, and the lighting or mood",
        ],
        "4045ffd1f333f4a1b9baefc8b2d596e6": "Vague: names a subject or idea, but not what's happening, where, or how it's framed",
        "963bc9903099118869c5465bcb80058b": "Partial: describes the subject, the action, and the setting, but leaves out the camera angle, framing, or lighting",
        "6205a921f274a35f2dcb0d6e56d4db89": "Ready: gives the subject, the action, the setting, the camera angle and framing, and the lighting or mood",
        "93d926fb5a12ad7dc7179ed59e1b66ac": "jev-latest",
        "d99102b775c65ed8a31e518cbd14adc8": "griptape_cloud",
        "30c1f92362fbf9ad1f8aae0e41fc431d": "claude-sonnet-5",
        "44136fa355b3678a1146ad16f7e8649e": {},
        "986851cbf56208077627bb6d090a1041": "Rewrite this shot description so it's ready to hand to an image generator.\n\nShot description:  \n{PROMPT}\n\nKeep every detail that's already there: the subject, the action, the setting, and any style notes. Don't change what happens or who's in it. Fill in whatever is missing:\n\n- Camera angle and framing, like low-angle wide shot or over-the-shoulder close-up\n- Lighting, including where it comes from and its color\n- Mood and look, like 35mm film, shallow depth of field, or tense and lonely\n\nChoose details that suit the scene. Write one paragraph of plain description, under 100 words. Reply with only the shot description, with no title, preamble, or notes.",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5": [],
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
                value=decode_value(top_level_unique_values_dict["0af5395a77978c1dbafd1add9349f4f3"]),
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
                        "position": {"x": -352, "y": -44},
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
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="About This Workflow",
                    metadata={
                        "position": {"x": -1144, "y": -616},
                        "size": {"width": 666, "height": 496},
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
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Store the Prompt",
                    metadata={
                        "position": {"x": -352, "y": -528},
                        "size": {"width": 600, "height": 364},
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
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Vague Shot",
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
                        "position": {"x": -1144, "y": -44},
                        "showaddparameter": False,
                        "size": {"width": 644, "height": 214},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Partial Shot",
                    metadata={
                        "position": {"x": -1144, "y": 220},
                        "size": {"width": 644, "height": 236},
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
                        "showaddparameter": False,
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Ready Shot",
                    metadata={
                        "position": {"x": -1144, "y": 506},
                        "size": {"width": 644, "height": 412},
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
                        "showaddparameter": False,
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="SetVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Set Variable",
                    metadata={
                        "position": {"x": 2200, "y": 242},
                        "tempId": "placing-1790736707610-aj2cto",
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
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="str", input_types=["str"], output_type="str", initial_setup=True
                )
            )
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="GrokImageGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Grok Image Generation",
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
                        "position": {"x": 3014, "y": -44},
                        "size": {"width": 688, "height": 970},
                        "showaddparameter": False,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node7_name):
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
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Rate the Shot",
                    metadata={
                        "position": {"x": 396, "y": -528},
                        "size": {"width": 688, "height": 408},
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
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Generate the Image",
                    metadata={
                        "position": {"x": 3014, "y": -528},
                        "size": {"width": 666, "height": 456},
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
                    initial_setup=True,
                )
            )
        ).node_name
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Fill In the Shot",
                    metadata={
                        "position": {"x": 1430, "y": -528},
                        "size": {"width": 710, "height": 408},
                        "color": "#0e7490",
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
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Update the Prompt",
                    metadata={
                        "position": {"x": 2200, "y": -528},
                        "size": {"width": 600, "height": 408},
                        "color": "#0e7490",
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
        node12_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevRate",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Rate (JEV)",
                    metadata={
                        "position": {"x": 396, "y": -36},
                        "tempId": "placing-1791421240152-scmvj",
                        "library_node_metadata": {
                            "category": "classification",
                            "description": "Rate text against levels you describe using TypeSafe JEV.",
                            "display_name": "Rate (JEV)",
                            "tags": None,
                            "icon": "bar-chart-2",
                            "color": None,
                            "group": "classification",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevRate",
                        "showaddparameter": False,
                        "size": {"width": 708, "height": 1044},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="levels_ParameterListUniqueParamID_fb32fb28efad43e08a5c0a86dd0d8865",
                    tooltip="One level per row, lowest first, up to 10. Describe the situation at each level. To name a level, put a label before a colon: 'Minor: broken but a workaround exists'. Each level gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "Describe this level", "display_name": "Levels"},
                    parent_container_name="levels",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="levels_ParameterListUniqueParamID_a084fb2a37104c47b304af808d610943",
                    tooltip="One level per row, lowest first, up to 10. Describe the situation at each level. To name a level, put a label before a colon: 'Minor: broken but a workaround exists'. Each level gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "Describe this level", "display_name": "Levels"},
                    parent_container_name="levels",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="levels_ParameterListUniqueParamID_4690489319b84b308937c8f9ae1f838b",
                    tooltip="One level per row, lowest first, up to 10. Describe the situation at each level. To name a level, put a label before a colon: 'Minor: broken but a workaround exists'. Each level gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "Describe this level", "display_name": "Levels"},
                    parent_container_name="levels",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="rate_level_fb32fb28efad43e08a5c0a86dd0d8865",
                    tooltip="Taken when the score rounds to Vague.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Vague"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="rate_level_a084fb2a37104c47b304af808d610943",
                    tooltip="Taken when the score rounds to Partial.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Partial"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="rate_level_4690489319b84b308937c8f9ae1f838b",
                    tooltip="Taken when the score rounds to Ready.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Ready"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
        node13_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Agent",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Agent",
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
                        "position": {"x": 1430, "y": 242},
                        "size": {"width": 710, "height": 1040},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="exec_out",
                target_node_name=node0_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="exec_out",
                target_node_name=node0_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="exec_out",
                target_node_name=node0_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node5_name,
                source_parameter_name="text",
                target_node_name=node0_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node6_name,
                source_parameter_name="exec_out",
                target_node_name=node7_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node12_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="rate_level_4690489319b84b308937c8f9ae1f838b",
                target_node_name=node7_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="rate_level_fb32fb28efad43e08a5c0a86dd0d8865",
                target_node_name=node13_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="rate_level_a084fb2a37104c47b304af808d610943",
                target_node_name=node13_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node12_name,
                source_parameter_name="failure",
                target_node_name=node13_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="exec_out",
                target_node_name=node6_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="output",
                target_node_name=node6_name,
                target_parameter_name="value",
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
                    value=decode_value(top_level_unique_values_dict["0af5395a77978c1dbafd1add9349f4f3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["0af5395a77978c1dbafd1add9349f4f3"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["1813c1dc118fdfd2e27110f8fa148f62"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["0ee5b563eaa24ecf4369b0499be7eeb9"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["4cc36534b2474d787bf3a98a53bacd7a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["4cc36534b2474d787bf3a98a53bacd7a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["3d617e36e7a8e10ec0622af73ba94c91"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["3d617e36e7a8e10ec0622af73ba94c91"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node5_name,
                    value=decode_value(top_level_unique_values_dict["0af5395a77978c1dbafd1add9349f4f3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node5_name,
                    value=decode_value(top_level_unique_values_dict["0af5395a77978c1dbafd1add9349f4f3"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["7d58514cc83cc992037e56dc1f6f745d"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="scope",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["25a6de226672088d980bed6d8e0c2162"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["7117fe6b6193e23a141ffc11d62b3f9e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["ea54873a3d5ac8b24056edfd2f87a2d3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="aspect_ratio",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["0c607131a93856ed93d33dad0c3829fb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="n",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["6b86b273ff34fce19d6b804eff5a3f57"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quality",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["60d4c90eee5e731df8d3ef2891de541d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolution",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["16d2724fe0a9971ec931e119652d59e5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["ee0b3aa7846d4f2b6fcd984dd714432a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["bb6cb47bd207dc9b9aaa1e847bcbdcc3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["a89f881edc3b69b9590403dee6236b3b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["3f457135186609800e5a046ad4b64df1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["86e021496d4679f80b41df308e5ba55b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["ea54873a3d5ac8b24056edfd2f87a2d3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["400bcab8a5bd3d2860d187621258c84f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["85cd8fa13c956b16a587184fb13600dc"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_fb32fb28efad43e08a5c0a86dd0d8865",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["4045ffd1f333f4a1b9baefc8b2d596e6"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_a084fb2a37104c47b304af808d610943",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["963bc9903099118869c5465bcb80058b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_4690489319b84b308937c8f9ae1f838b",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["6205a921f274a35f2dcb0d6e56d4db89"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model_provider",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["d99102b775c65ed8a31e518cbd14adc8"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["30c1f92362fbf9ad1f8aae0e41fc431d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="agent_memory",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["44136fa355b3678a1146ad16f7e8649e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["986851cbf56208077627bb6d090a1041"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="additional_context",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="tools",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="rulesets",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="include_details",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
