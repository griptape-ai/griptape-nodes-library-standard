# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "jev_review_note_workflow"
# schema_version = "0.21.0"
# engine_version_created_with = "0.104.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "CreateVariable"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "GetVariable"], ["Griptape Nodes Library", "JevAskYesNo"], ["Griptape Nodes Library", "JevPickOne"], ["Griptape Nodes Library", "JevRate"], ["Griptape Nodes Library", "JsonInput"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "StartFlow"]]
# description = "JEV triages a director's note: is it actionable, which department owns it, and how big is the fix?"
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_jev_review_note_workflow.webp"
# is_griptape_provided = true
# is_template = true
# is_internal = false
# creation_date = 2026-10-08T16:32:56.352453Z
# last_modified_date = 2026-10-10T01:22:01.061593Z
# workflow_shape = "{\"inputs\":{\"Start Flow\":{\"exec_out\":{\"name\":\"exec_out\",\"tooltip\":\"Connection to the next node in the execution chain\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":false,\"mode_allowed_output\":true,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Flow Out\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"note\":{\"name\":\"note\",\"tooltip\":\"New parameter\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":\"Bank the dragon's motion between frames 200 and 264 a bit more - it should feel heavier, like it's really tough to turn.\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true,\"hide\":false,\"display_name\":\"Note\",\"multiline\":true,\"placeholder_text\":\"Enter your note here\",\"step\":1},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}},\"outputs\":{\"End Flow\":{\"exec_in\":{\"name\":\"exec_in\",\"tooltip\":\"Control path when the flow completed successfully\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Succeeded\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"failed\":{\"name\":\"failed\",\"tooltip\":\"Control path when the flow failed\",\"type\":\"parametercontroltype\",\"input_types\":[\"parametercontroltype\"],\"output_type\":\"parametercontroltype\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"parameter_render_location\":\"top\",\"display_name\":\"Failed\"},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":null},\"was_successful\":{\"name\":\"was_successful\",\"tooltip\":\"Indicates whether it completed without errors.\",\"type\":\"bool\",\"input_types\":[\"bool\"],\"output_type\":\"bool\",\"default_value\":false,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":false,\"mode_allowed_property\":true,\"mode_allowed_output\":false,\"ui_options\":{},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"result_details\":{\"name\":\"result_details\",\"tooltip\":\"Details about the operation result\",\"type\":\"str\",\"input_types\":[\"str\"],\"output_type\":\"str\",\"default_value\":null,\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":false,\"mode_allowed_output\":false,\"ui_options\":{\"multiline\":true,\"placeholder_text\":\"Details about the completion or failure will be shown here.\"},\"settable\":false,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":null,\"parent_element_name\":\"Status\"},\"categorized_note\":{\"name\":\"categorized_note\",\"tooltip\":\"New parameter\",\"type\":\"json\",\"input_types\":[\"json\"],\"output_type\":\"json\",\"default_value\":\"\",\"tooltip_as_input\":null,\"tooltip_as_property\":null,\"tooltip_as_output\":null,\"mode_allowed_input\":true,\"mode_allowed_property\":true,\"mode_allowed_output\":true,\"ui_options\":{\"is_custom\":true,\"is_user_added\":true,\"hide\":false,\"display_name\":\"Categorized Note\",\"step\":1},\"settable\":true,\"is_user_defined\":true,\"private\":false,\"parent_container_name\":\"\",\"parent_element_name\":null}}}}"
#
# ///

import argparse
import asyncio
import json
import logging
from typing import Any

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
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
        "d0b8e382b70ec7dd721471ae541466f0": '## Get the output\n\n**Get Result** reads `RESULT` and passes it to **End Flow**, so other workflows can call this one and use the record.\n\n- To sort a whole review, call this workflow from a **For Each** loop over the notes, then send the results to **Group By Key**. You get one list per department, plus "Need More Data".',
        "c62623d9701d7e99a922a93e65c351c6": "## Director's Note Review with JEV\n\nReads one director's note and works out three things: whether an artist can act on it, which department owns it, and how big the fix is.\n\n- **Actionable:** JEV picks a department and sizes the fix, and the answers are saved as one record under that department.\n- **Not actionable:** the note is saved under \"Need More Data\" for a supervisor to follow up.\n\nEither way you get one record, ready to sort with the rest of the review.\n\n**Try it:** run the flow as it is, and the dragon note comes back as Animation, size S. Then replace **note** on **Start Flow** with *The whole thing feels a bit dark. Maybe that's fine? Let's see it brighter and decide.* and run it again to see the Need More Data branch.",
        "fcbcf165908dd18a9e49f7ff27810176": False,
        "284b7e6d788f363f910f7beb1910473e": 600,
        "af0149f4795512f48b831611ab857133": "{NOTE}",
        "6a3e4213392eefd2aa5dbd49e17c4cd9": "Does this note from a director give the artist enough information to start working?",
        "c690dbeac089eff72d607afffe82ce9a": "The note identifies what's wrong or what the result should look or feel like, even if it doesn't say how to fix it.",
        "1b3a1944333c65a44294409423bfa7c5": 'The note is a question, and undecided thought ("maybe", "let\'s see") or too vague to know what to change.',
        "d2cbad71ff333de67d07ec676e352ab7": 0.5,
        "b5bea41b6c623f7c09f1bf24dcae58eb": True,
        "5bcccbeaf5d2a2118d29892cbe54534c": 0.82,
        "b2aa969b03b3bcd1a7aa48f0b8d31149": 0.86,
        "93d926fb5a12ad7dc7179ed59e1b66ac": "jev-latest",
        "2da0845e26883f1eed7ab85e3b5ed699": "JEV answered.",
        "83d0c8985446dd5a19193e38d572452e": "ded54626-77a1-4ad9-9d79-182e311ca47b",
        "bb0c96dd509ab0ce9339f027fed1034e": "fdf28998-755a-4fc9-8aa1-c9adf156abd5",
        "ba0ae16a200116e07f05885dfbad38d0": "COMPLETED",
        "27dce8063083bd72b6eac7bbcc123cab": "Which department should handle this note?",
        "c97e4a8294a6ffcbc3d058e0af95000c": [
            "Layout:  camera position, movement, lens and framing",
            "Animation: character and creature performance, timing, weight, poses",
            "CFX: cloth, hair and fur simulation",
            "FX: smoke, fire, dust, rain, water, explosions, debris",
            "Lighting: light direction, colour, intensity, rim, shadows, reflections",
            "Comp: edges, mattes, grain, colour matching, paint-outs, integration",
            "Editorial: shot length, cut points and trims",
        ],
        "cf1643261d855210b8e4af0bcffd5381": "Layout:  camera position, movement, lens and framing",
        "7ffd1c7f425062a64210a4a75c213c6e": "Animation: character and creature performance, timing, weight, poses",
        "42196e4794d235918cc58339b56b064b": "CFX: cloth, hair and fur simulation",
        "fe7d30ef9bd00d8b5f8b5e8648829063": "FX: smoke, fire, dust, rain, water, explosions, debris",
        "e6a641d229f094fe99109c86c9c011e8": "Lighting: light direction, colour, intensity, rim, shadows, reflections",
        "7ef311d59e079339fc7ebd910629d0fc": "Comp: edges, mattes, grain, colour matching, paint-outs, integration",
        "012a9ead6e35bc5db2c887bacf6c7126": "Editorial: shot length, cut points and trims",
        "fd25f2fd68d9e875d5cc9728dfc33406": "CFX",
        "564e1768c817348118e3e19c6a5d67fa": "Animation",
        "ad8910bf9e4df9d87c93d566bf70693c": "cloth, hair and fur simulation",
        "a534673e66cde6545291da4988cecc3d": "character and creature performance, timing, weight, poses",
        "d0ff5974b6aa52cf562bea5921840c03": 1.0,
        "89b22109c2bc402b59333d48e927f657": {
            "Layout": 0.0,
            "Animation": 0.0,
            "CFX": 1.0,
            "FX": 0.0,
            "Lighting": 0.0,
            "Comp": 0.0,
            "Editorial": 0.0,
        },
        "3e882d54cc5aefef7601bf25e03a8e29": {
            "Layout": 0.0,
            "Animation": 1.0,
            "CFX": 0.0,
            "FX": 0.0,
            "Lighting": 0.0,
            "Comp": 0.0,
            "Editorial": 0.0,
        },
        "e73d6cf886e132c6abbf86704c544c63": "JEV picked 'Animation'.",
        "aec5e7d5a0f65b6fa5cd6ec0b81e130e": "4b04711b-0b4f-41c3-b329-21dd2a4dc011",
        "036f7bfa0cda08d494422bca549e2f08": "6a7f5a00-5445-4c6f-9c39-81abb11b5b7e",
        "71720ab9549cdb47077bcc907a01d274": "How big is the fix?",
        "fbada25c844da2665d5398ba11711df9": [
            "S: a small adjustment to existing work, like tweaking a value, nudging timing, or re-rendering",
            "M: rework part of the shot within a single department",
            "L: rebuild the work from scratch, or a change that affects several departments",
        ],
        "59a715e11c7fbdfd2bce0b6781722327": "S: a small adjustment to existing work, like tweaking a value, nudging timing, or re-rendering",
        "9d5f4c5a3ebffc7dec7a9537d12665dd": "M: rework part of the shot within a single department",
        "0ac7e200954f0be3649f7356f01e4de9": "L: rebuild the work from scratch, or a change that affects several departments",
        "eef6078543b58ffe3e293157b177da2f": 1.3599999999999999,
        "28b478ce49706c4e6689e2f3a9a964ee": 1.09,
        "6b86b273ff34fce19d6b804eff5a3f57": 1,
        "52bd5f3d03badf80f7ab61b0ebd226fa": "S",
        "c6984194dc7d49de5b0626a4f8aaa35d": "a small adjustment to existing work, like tweaking a value, nudging timing, or re-rendering",
        "ad9dadb817c4ea957beef59cc28b49cc": 0.46,
        "6e17042abf6be9872401b72669a5a738": {"1": 0.67, "2": 0.3, "3": 0.03},
        "c01d4001f6c75261fa4004c4f474a6b4": {"1": 0.91, "2": 0.09, "3": 0.0},
        "0f2d839a586242b3a5e115d903914c76": "JEV scored level 1.",
        "c015b3833f29e129282c9bfdc0d0da86": "7db01146-fe8c-45da-8d8d-067750cda647",
        "f9ab1c276d488b770e0d37cf0aac79d9": "aee14b1b-a574-46bd-bbd7-9bbfddd6e54e",
        "aa9ed61b8342a2d499def3b1d1f8c162": "## Store the note\n\n**Store Note** saves the note in a variable called `NOTE`. Other nodes read it when you type `{NOTE}` into a text field.\n\n**Ask Yes/No**, **Pick One** and **Rate** all read `{NOTE}` in their **Context**, so every node works from the same note.",
        "fe6cdd3d8ddb75f0a2202ddb01b501ae": "## Is it actionable?\n\n**Ask Yes/No (JEV)** reads `{NOTE}` and answers *Does this note from a director give the artist enough information to start working?* with a probability from 0 to 1.\n\n**Yes** continues to **Pick One**. **No** goes to the Need More Data branch.\n\n- Open **Define Yes and No** to see what each answer means. If notes land on the wrong side, edit these definitions before you edit the question. They make the biggest difference.\n- After a run, check **Probability** to see how close the call was.\n- **Say Yes at or above** is set to 0.5. Raise it to send more notes back to the director.",
        "9fcaef44a9fa87fefb2b9cd4c224ca3a": "## Pick a department\n\n**Pick One (JEV)** reads `{NOTE}` and picks the option in **Options** that fits it best. Each row is a label, a colon, and a description of the work that department owns.\n\nEach label becomes a flow output. Every output connects to **Store Department**, which saves the department in `DEPT`. **Store Department Probabilities** then saves the probabilities in `DEPT_PROBABILITIES`.\n\n- After a run, check **Confidence** and **Probabilities**. A close split between two departments means a supervisor should assign the note.\n- To add a department, click **Add item to Options**, then connect its new output to **Store Department** like the others.\n- When two departments overlap, make their descriptions say where the line between them is.",
        "2a7c2b414ce665f875d03de3773501f1": "## Rate the size\n\n**Rate (JEV)** reads `{NOTE}` and rates the fix against the rows in **Levels**, smallest first. Each row is a label, a colon, and a description of a fix at that size. JEV only sees the descriptions.\n\n**S**, **M** and **L** all connect to **Store Size**, which saves the size in `SIZE`. **Store Size Probabilities** then saves the probabilities in `SIZE_PROBABILITIES`.\n\n- After a run, check **Score** and **Confidence**. A score of 1.4 means mostly S, with some pull toward M. Low confidence means the size is a guess worth checking.\n- Describe what a fix at each size involves, not how important it is.",
        "18edbfddad2497f054608a4938230103": '## Need More Data\n\nWhen **Ask Yes/No** says No, the department and size steps are skipped. **Build Need More Data Record** makes a record with just the note, filed under "Need More Data", and **Store Need More Data Result** saves it as `RESULT`.\n\nThese are the notes to take back to the director before anyone starts work.',
        "e31a04d5b91a750cca2e50f17266b7d1": "## Store the result\n\n**Build Record** fills in a record from the variables, keyed by department: the note, its size, and the probabilities behind both answers. **Store Result** saves it as `RESULT`.\n\nBoth branches write `RESULT`, so the output is always in the same place.\n\n- To add a field, add a **Create Variable** after the node that produces it, then add a matching line to **Build Record**.\n- Avoid double quotes inside notes. They break the JSON.",
        "b0cd1f713601eb24e5ad75f94db067b6": "Bank the dragon's motion between frames 200 and 264 a bit more - it should feel heavier, like it's really tough to turn.",
        "13d5bb126f2602fdfac6dceeb7c847be": "Control Input Selection",
        "eec80c33d01a30822ada8a44d78f23c2": "[SUCCEEDED]\n[SUCCEEDED]\nNo details supplied by flow",
        "12ae32cb1ec02d01eda3581b127c1fee": "",
        "d6e22df44ca9e5c23dcbc22898ff9ca4": "NOTE",
        "1e1a6bbb783263b9b6247e30b217d6ba": {
            "$type": "griptape_nodes.node_libraries.griptape_nodes_library.create_variable:CaseStyle",
            "$value": "snake_case",
        },
        "72495b0e1f3c6c961f46d3f249ce968a": "str",
        "576524616398bdd29319795fcbb6849a": "DEPT",
        "0251d2361522750dfd117e8e20f21b46": "DEPT_PROBABILITIES",
        "4e23b392048439acfc9c40dd5d4e908f": "json",
        "945e24664eb96485e9dc10299080cec1": "SIZE",
        "23eb49e7e62ac7249a17d01ff2391973": "SIZE_PROBABILITIES",
        "a5b98b8f8467186ea3a6175fc61a21cb": '{\n  "{DEPT}": {\n    "Note": "{NOTE}",\n    "Size": "{SIZE}",\n    "Probabilities": {\n      "Department": {DEPT_PROBABILITIES},\n      "Size": {SIZE_PROBABILITIES}\n    }\n  }\n}',
        "5d8419df992a3db0e28003735f556b9b": '{"Animation": {"Note": "Bank the dragon\'s motion between frames 200 and 264 a bit more - it should feel heavier, like it\'s really tough to turn.", "Size": "S", "Probabilities": {"Department": {"Layout": 0.0, "Animation": 1.0, "CFX": 0.0, "FX": 0.0, "Lighting": 0.0, "Comp": 0.0, "Editorial": 0.0}, "Size": {"1": 0.91, "2": 0.09, "3": 0.0}}}}',
        "cd25336a439c3c6432d0b787f6982d8b": "RESULT",
        "633cb86d4548f92908b0076929f27025": {"Need More Data": {"Note": "{NOTE}"}},
        "2e29814d5bcf0a0dba2e14c54f15adf5": {
            "Need More Data": {"Note": "Her hair should be whipping around in the wind, it's barely moving."}
        },
        "25a6de226672088d980bed6d8e0c2162": "hierarchical",
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
                name="DEPT",
                type="str",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["564e1768c817348118e3e19c6a5d67fa"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="DEPT_PROBABILITIES",
                type="json",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["3e882d54cc5aefef7601bf25e03a8e29"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="NOTE",
                type="str",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["b0cd1f713601eb24e5ad75f94db067b6"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="RESULT",
                type="json",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="SIZE",
                type="str",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["52bd5f3d03badf80f7ab61b0ebd226fa"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        GriptapeNodes.handle_request(
            CreateVariableRequest(
                name="SIZE_PROBABILITIES",
                type="json",
                is_global=False,
                value=decode_value(top_level_unique_values_dict["c01d4001f6c75261fa4004c4f474a6b4"]),
                owning_flow="ControlFlow_1",
                initial_setup=True,
            )
        )
        node0_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_6",
                    metadata={
                        "position": {"x": 5364, "y": 756},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 1464, "height": 228},
                        "font_size": None,
                        "color": "#1d4ed8",
                        "text_align": None,
                        "text_color": None,
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
                    node_name="Note_7",
                    metadata={
                        "position": {"x": -2196, "y": 216},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 636, "height": 408},
                        "font_size": None,
                        "color": "#1d4ed8",
                        "text_align": None,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node2_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevAskYesNo",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Ask Yes/No (JEV)",
                    metadata={
                        "position": {"x": -216, "y": 648},
                        "tempId": "placing-1791477552030-p3tzwg",
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
                            "declarations": [
                                {"type": "model_usage", "model_ids": ["gtc_jev_latest", "gtc_jev_preview"]}
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevAskYesNo",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 976},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevPickOne",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Pick One (JEV)",
                    metadata={
                        "position": {"x": 684, "y": 648},
                        "tempId": "placing-1791478645450-74087k",
                        "library_node_metadata": {
                            "category": "classification",
                            "description": "Pick the option that best fits some text using TypeSafe JEV.",
                            "display_name": "Pick One (JEV)",
                            "tags": None,
                            "icon": "list-checks",
                            "color": None,
                            "group": "classification",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {"type": "model_usage", "model_ids": ["gtc_jev_latest", "gtc_jev_preview"]}
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevPickOne",
                        "showaddparameter": False,
                        "size": {"width": 636, "height": 1252},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_a72a480a025d4647bc124b5a81fffe66",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_dc4e41d392474ea4824be9cf1af064dc",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_db762755675847c5bfd0882e625a123b",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_0ad83f1692ab42fbb84c0750ee823a10",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_2f9d4c1858c34e888c1e79f70873999b",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_87d35114e02f446da004e9e5149f7130",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_0e809fab83b74e1f976d7267515a1fc0",
                    tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what the option means. Each option gets its own flow output.",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={"placeholder_text": "label: description", "display_name": "Options"},
                    parent_container_name="options",
                    traits=[],
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_a72a480a025d4647bc124b5a81fffe66",
                    tooltip="Taken when JEV picks Layout.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Layout"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_dc4e41d392474ea4824be9cf1af064dc",
                    tooltip="Taken when JEV picks Animation.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Animation"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_db762755675847c5bfd0882e625a123b",
                    tooltip="Taken when JEV picks CFX.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "CFX"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_0ad83f1692ab42fbb84c0750ee823a10",
                    tooltip="Taken when JEV picks FX.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "FX"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_2f9d4c1858c34e888c1e79f70873999b",
                    tooltip="Taken when JEV picks Lighting.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Lighting"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_87d35114e02f446da004e9e5149f7130",
                    tooltip="Taken when JEV picks Comp.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Comp"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_0e809fab83b74e1f976d7267515a1fc0",
                    tooltip="Taken when JEV picks Editorial.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Editorial"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
        node4_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevRate",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Rate (JEV)",
                    metadata={
                        "position": {"x": 2268, "y": 648},
                        "tempId": "placing-1791479540449-0q1g9d",
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
                            "declarations": [
                                {"type": "model_usage", "model_ids": ["gtc_jev_latest", "gtc_jev_preview"]}
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevRate",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 1044},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="levels_ParameterListUniqueParamID_44b8bce01033490c976da648e92059ce",
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
                    parameter_name="levels_ParameterListUniqueParamID_dc8f86e521e945d4a7ee68ca1047536c",
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
                    parameter_name="levels_ParameterListUniqueParamID_9b878c64b7f540a286f1eea6ef0281e4",
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
                    parameter_name="rate_level_44b8bce01033490c976da648e92059ce",
                    tooltip="Taken when the score rounds to S.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "S"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="rate_level_dc8f86e521e945d4a7ee68ca1047536c",
                    tooltip="Taken when the score rounds to M.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "M"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="rate_level_9b878c64b7f540a286f1eea6ef0281e4",
                    tooltip="Taken when the score rounds to L.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "L"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note",
                    metadata={
                        "position": {"x": -1296, "y": 216},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 408},
                        "font_size": None,
                        "color": "#1d4ed8",
                        "text_align": None,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_1",
                    metadata={
                        "position": {"x": -216, "y": 216},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 408},
                        "font_size": None,
                        "color": "#6d28d9",
                        "text_align": None,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_2",
                    metadata={
                        "position": {"x": 684, "y": 252},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 1284, "height": 372},
                        "font_size": None,
                        "color": "#047857",
                        "text_align": None,
                        "text_color": None,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note_3",
                    metadata={
                        "position": {"x": 2268, "y": 252},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 1284, "height": 372},
                        "font_size": None,
                        "color": "#047857",
                        "text_align": None,
                        "text_color": None,
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
                    node_name="Note_4",
                    metadata={
                        "position": {"x": 684, "y": 2340},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 1320, "height": 228},
                        "font_size": None,
                        "color": "#be185d",
                        "text_align": None,
                        "text_color": None,
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
                    node_name="Note_5",
                    metadata={
                        "position": {"x": 3852, "y": 252},
                        "tempId": "placing-1791517197727-hdcqq",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "Note",
                        "showaddparameter": False,
                        "size": {"width": 1284, "height": 372},
                        "font_size": None,
                        "color": "#1d4ed8",
                        "text_align": None,
                        "text_color": None,
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="StartFlow",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Start Flow",
                    metadata={
                        "position": {"x": -2196, "y": 648},
                        "tempId": "placing-1791593831700-6wrjbd",
                        "library_node_metadata": {
                            "category": "workflows",
                            "description": "Define the start of a workflow and pass parameters into the flow",
                            "display_name": "Start Flow",
                            "tags": ["workflow", "execution"],
                            "icon": None,
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "StartFlow",
                        "showaddparameter": True,
                        "size": {"width": 636, "height": 298},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="note",
                    default_value="Bank the dragon's motion between frames 200 and 264 a bit more - it should feel heavier, like it's really tough to turn.",
                    tooltip="New parameter",
                    type="str",
                    input_types=["str"],
                    output_type="str",
                    ui_options={
                        "is_custom": True,
                        "is_user_added": True,
                        "hide": False,
                        "display_name": "Note",
                        "multiline": True,
                        "placeholder_text": "Enter your note here",
                        "step": 1,
                    },
                    mode_allowed_input=False,
                    parent_container_name="",
                    traits=[],
                    initial_setup=True,
                )
            )
        node12_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="EndFlow",
                    specific_library_name="Griptape Nodes Library",
                    node_name="End Flow",
                    metadata={
                        "position": {"x": 6156, "y": 1116},
                        "tempId": "placing-1791593952823-y3orpb",
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
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "EndFlow",
                        "showaddparameter": True,
                        "size": {"width": 672, "height": 840},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="categorized_note",
                    default_value="",
                    tooltip="New parameter",
                    type="json",
                    input_types=["json"],
                    output_type="json",
                    ui_options={
                        "is_custom": True,
                        "is_user_added": True,
                        "hide": False,
                        "display_name": "Categorized Note",
                        "step": 1,
                    },
                    parent_container_name="",
                    traits=[],
                    initial_setup=True,
                )
            )
        node13_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Note",
                    metadata={
                        "position": {"x": -1296, "y": 648},
                        "tempId": "placing-1791513917732-u14jmq",
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
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(parameter_name="value", type="str", output_type="str", initial_setup=True)
            )
        node14_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Department",
                    metadata={
                        "position": {"x": 1404, "y": 1044},
                        "tempId": "placing-1791514020966-9sre9h",
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
        with GriptapeNodes.ContextManager().node(node14_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(parameter_name="value", type="str", output_type="str", initial_setup=True)
            )
        node15_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Department Probabilities",
                    metadata={
                        "position": {"x": 1404, "y": 1440},
                        "tempId": "placing-1791514020966-9sre9h",
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
                        "size": {"width": 600, "height": 364},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node15_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="json", output_type="json", initial_setup=True
                )
            )
        node16_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Size",
                    metadata={
                        "position": {"x": 2952, "y": 1044},
                        "tempId": "placing-1791514147703-6i98jr",
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
        with GriptapeNodes.ContextManager().node(node16_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(parameter_name="value", type="str", output_type="str", initial_setup=True)
            )
        node17_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Size Probabilities",
                    metadata={
                        "position": {"x": 2952, "y": 1404},
                        "tempId": "placing-1791514147703-6i98jr",
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
        with GriptapeNodes.ContextManager().node(node17_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="json", output_type="json", initial_setup=True
                )
            )
        node18_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JsonInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Build Record",
                    metadata={
                        "position": {"x": 3852, "y": 720},
                        "tempId": "placing-1791515894397-17gam",
                        "library_node_metadata": {
                            "category": "json",
                            "description": "Create JSON data with an input node.",
                            "display_name": "JSON Input",
                            "tags": ["json", "data", "input", "create"],
                            "icon": "file-json",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JsonInput",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 396},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node19_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Result",
                    metadata={
                        "position": {"x": 4608, "y": 720},
                        "tempId": "placing-1791516072255-ujlue",
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
        with GriptapeNodes.ContextManager().node(node19_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="json", output_type="json", initial_setup=True
                )
            )
        node20_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JsonInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Build Need More Data Record",
                    metadata={
                        "position": {"x": 684, "y": 2628},
                        "tempId": "placing-1791515894397-17gam",
                        "library_node_metadata": {
                            "category": "json",
                            "description": "Create JSON data with an input node.",
                            "display_name": "JSON Input",
                            "tags": ["json", "data", "input", "create"],
                            "icon": "file-json",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JsonInput",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 324},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node21_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="CreateVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Store Need More Data Result",
                    metadata={
                        "position": {"x": 1476, "y": 2628},
                        "tempId": "placing-1791516072255-ujlue",
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
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node21_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_type", mode_allowed_input=False, settable=False, initial_setup=True
                )
            )
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="value", type="json", output_type="json", initial_setup=True
                )
            )
        node22_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="GetVariable",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Get Result",
                    metadata={
                        "position": {"x": 5364, "y": 1116},
                        "tempId": "placing-1791500735784-m7uy1",
                        "library_node_metadata": {
                            "category": "variables",
                            "description": "Retrieve the value of an existing variable",
                            "display_name": "Get Variable",
                            "tags": ["data", "variable", "workflow"],
                            "icon": "ArrowDown",
                            "color": None,
                            "group": "describe",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "GetVariable",
                        "showaddparameter": False,
                        "size": {"width": 600, "height": 344},
                    },
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node22_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="variable_name",
                    default_value="NOTE",
                    traits=[
                        {
                            "trait_name": "Options",
                            "trait_module": "griptape_nodes.traits.options",
                            "trait_state": {
                                "choices": [
                                    "RESULT",
                                    "backups",
                                    "griptape-nodes-metadata",
                                    "griptape-nodes-previews",
                                    "griptape-nodes-thumbnails",
                                    "inputs",
                                    "outputs",
                                    "project_dir",
                                    "static_files_dir",
                                    "temp",
                                    "workflow_dir",
                                    "workflow_name",
                                    "workflow_run_failures",
                                    "workspace_dir",
                                ]
                            },
                        },
                        {"trait_name": "Button", "trait_module": "griptape_nodes.traits.button", "trait_state": {}},
                    ],
                    initial_setup=True,
                )
            )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node13_name,
                source_parameter_name="exec_out",
                target_node_name=node2_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="yes",
                target_node_name=node3_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_dc4e41d392474ea4824be9cf1af064dc",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_db762755675847c5bfd0882e625a123b",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_a72a480a025d4647bc124b5a81fffe66",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_0ad83f1692ab42fbb84c0750ee823a10",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_2f9d4c1858c34e888c1e79f70873999b",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_87d35114e02f446da004e9e5149f7130",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="option_0e809fab83b74e1f976d7267515a1fc0",
                target_node_name=node14_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node14_name,
                source_parameter_name="exec_out",
                target_node_name=node15_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node15_name,
                source_parameter_name="exec_out",
                target_node_name=node4_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="rate_level_44b8bce01033490c976da648e92059ce",
                target_node_name=node16_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="rate_level_dc8f86e521e945d4a7ee68ca1047536c",
                target_node_name=node16_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="rate_level_9b878c64b7f540a286f1eea6ef0281e4",
                target_node_name=node16_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node16_name,
                source_parameter_name="exec_out",
                target_node_name=node17_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node17_name,
                source_parameter_name="exec_out",
                target_node_name=node18_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node18_name,
                source_parameter_name="json",
                target_node_name=node19_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node18_name,
                source_parameter_name="exec_out",
                target_node_name=node19_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node19_name,
                source_parameter_name="exec_out",
                target_node_name=node22_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node21_name,
                source_parameter_name="exec_out",
                target_node_name=node22_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node20_name,
                source_parameter_name="exec_out",
                target_node_name=node21_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node20_name,
                source_parameter_name="json",
                target_node_name=node21_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node2_name,
                source_parameter_name="no",
                target_node_name=node20_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="probabilities",
                target_node_name=node17_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="probabilities",
                target_node_name=node15_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node3_name,
                source_parameter_name="label",
                target_node_name=node14_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node4_name,
                source_parameter_name="label",
                target_node_name=node16_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="note",
                target_node_name=node13_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node22_name,
                source_parameter_name="exec_out",
                target_node_name=node12_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node22_name,
                source_parameter_name="value",
                target_node_name=node12_name,
                target_parameter_name="categorized_note",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="exec_out",
                target_node_name=node13_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        with GriptapeNodes.ContextManager().node(node0_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["d0b8e382b70ec7dd721471ae541466f0"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["c62623d9701d7e99a922a93e65c351c6"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["af0149f4795512f48b831611ab857133"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["6a3e4213392eefd2aa5dbd49e17c4cd9"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="yes_means",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["c690dbeac089eff72d607afffe82ce9a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="no_means",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["1b3a1944333c65a44294409423bfa7c5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="threshold",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["d2cbad71ff333de67d07ec676e352ab7"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="answer",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="answer",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probability",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["5bcccbeaf5d2a2118d29892cbe54534c"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probability",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["b2aa969b03b3bcd1a7aa48f0b8d31149"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["2da0845e26883f1eed7ab85e3b5ed699"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["2da0845e26883f1eed7ab85e3b5ed699"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["83d0c8985446dd5a19193e38d572452e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["bb0c96dd509ab0ce9339f027fed1034e"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["af0149f4795512f48b831611ab857133"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["27dce8063083bd72b6eac7bbcc123cab"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["c97e4a8294a6ffcbc3d058e0af95000c"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_a72a480a025d4647bc124b5a81fffe66",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["cf1643261d855210b8e4af0bcffd5381"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_dc4e41d392474ea4824be9cf1af064dc",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["7ffd1c7f425062a64210a4a75c213c6e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_db762755675847c5bfd0882e625a123b",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["42196e4794d235918cc58339b56b064b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_0ad83f1692ab42fbb84c0750ee823a10",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["fe7d30ef9bd00d8b5f8b5e8648829063"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_2f9d4c1858c34e888c1e79f70873999b",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["e6a641d229f094fe99109c86c9c011e8"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_87d35114e02f446da004e9e5149f7130",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["7ef311d59e079339fc7ebd910629d0fc"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_0e809fab83b74e1f976d7267515a1fc0",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["012a9ead6e35bc5db2c887bacf6c7126"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="label",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["fd25f2fd68d9e875d5cc9728dfc33406"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="label",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["564e1768c817348118e3e19c6a5d67fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="description",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["ad8910bf9e4df9d87c93d566bf70693c"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="description",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["a534673e66cde6545291da4988cecc3d"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="confidence",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["d0ff5974b6aa52cf562bea5921840c03"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="confidence",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["d0ff5974b6aa52cf562bea5921840c03"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probabilities",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["89b22109c2bc402b59333d48e927f657"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probabilities",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["3e882d54cc5aefef7601bf25e03a8e29"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["e73d6cf886e132c6abbf86704c544c63"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["e73d6cf886e132c6abbf86704c544c63"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["aec5e7d5a0f65b6fa5cd6ec0b81e130e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["036f7bfa0cda08d494422bca549e2f08"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["af0149f4795512f48b831611ab857133"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["71720ab9549cdb47077bcc907a01d274"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["fbada25c844da2665d5398ba11711df9"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_44b8bce01033490c976da648e92059ce",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["59a715e11c7fbdfd2bce0b6781722327"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_dc8f86e521e945d4a7ee68ca1047536c",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["9d5f4c5a3ebffc7dec7a9537d12665dd"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="levels_ParameterListUniqueParamID_9b878c64b7f540a286f1eea6ef0281e4",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["0ac7e200954f0be3649f7356f01e4de9"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="score",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["eef6078543b58ffe3e293157b177da2f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="score",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["28b478ce49706c4e6689e2f3a9a964ee"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="level",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["6b86b273ff34fce19d6b804eff5a3f57"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="level",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["6b86b273ff34fce19d6b804eff5a3f57"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="label",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["52bd5f3d03badf80f7ab61b0ebd226fa"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="label",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["52bd5f3d03badf80f7ab61b0ebd226fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="description",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["c6984194dc7d49de5b0626a4f8aaa35d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="description",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["c6984194dc7d49de5b0626a4f8aaa35d"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="confidence",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["ad9dadb817c4ea957beef59cc28b49cc"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="confidence",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["b2aa969b03b3bcd1a7aa48f0b8d31149"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probabilities",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["6e17042abf6be9872401b72669a5a738"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="probabilities",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["c01d4001f6c75261fa4004c4f474a6b4"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["0f2d839a586242b3a5e115d903914c76"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["0f2d839a586242b3a5e115d903914c76"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["c015b3833f29e129282c9bfdc0d0da86"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["f9ab1c276d488b770e0d37cf0aac79d9"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["ba0ae16a200116e07f05885dfbad38d0"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=decode_value(top_level_unique_values_dict["aa9ed61b8342a2d499def3b1d1f8c162"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["fe6cdd3d8ddb75f0a2202ddb01b501ae"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["9fcaef44a9fa87fefb2b9cd4c224ca3a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["2a7c2b414ce665f875d03de3773501f1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node9_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["18edbfddad2497f054608a4938230103"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["e31a04d5b91a750cca2e50f17266b7d1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["b0cd1f713601eb24e5ad75f94db067b6"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node12_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="exec_in",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["13d5bb126f2602fdfac6dceeb7c847be"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["eec80c33d01a30822ada8a44d78f23c2"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="result_details",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["eec80c33d01a30822ada8a44d78f23c2"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="categorized_note",
                    node_name=node12_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node13_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["d6e22df44ca9e5c23dcbc22898ff9ca4"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["d6e22df44ca9e5c23dcbc22898ff9ca4"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["b0cd1f713601eb24e5ad75f94db067b6"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node13_name,
                    value=decode_value(top_level_unique_values_dict["b0cd1f713601eb24e5ad75f94db067b6"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node14_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["576524616398bdd29319795fcbb6849a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["576524616398bdd29319795fcbb6849a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["564e1768c817348118e3e19c6a5d67fa"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node14_name,
                    value=decode_value(top_level_unique_values_dict["564e1768c817348118e3e19c6a5d67fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node15_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["0251d2361522750dfd117e8e20f21b46"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["0251d2361522750dfd117e8e20f21b46"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["3e882d54cc5aefef7601bf25e03a8e29"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node15_name,
                    value=decode_value(top_level_unique_values_dict["3e882d54cc5aefef7601bf25e03a8e29"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node16_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["945e24664eb96485e9dc10299080cec1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["945e24664eb96485e9dc10299080cec1"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["72495b0e1f3c6c961f46d3f249ce968a"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["52bd5f3d03badf80f7ab61b0ebd226fa"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node16_name,
                    value=decode_value(top_level_unique_values_dict["52bd5f3d03badf80f7ab61b0ebd226fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node17_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["23eb49e7e62ac7249a17d01ff2391973"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["23eb49e7e62ac7249a17d01ff2391973"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["c01d4001f6c75261fa4004c4f474a6b4"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node17_name,
                    value=decode_value(top_level_unique_values_dict["c01d4001f6c75261fa4004c4f474a6b4"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node18_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="json",
                    node_name=node18_name,
                    value=decode_value(top_level_unique_values_dict["a5b98b8f8467186ea3a6175fc61a21cb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="json",
                    node_name=node18_name,
                    value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node19_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node19_name,
                    value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node20_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="json",
                    node_name=node20_name,
                    value=decode_value(top_level_unique_values_dict["633cb86d4548f92908b0076929f27025"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="json",
                    node_name=node20_name,
                    value=decode_value(top_level_unique_values_dict["2e29814d5bcf0a0dba2e14c54f15adf5"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node21_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_name_case",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["1e1a6bbb783263b9b6247e30b217d6ba"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_type",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["4e23b392048439acfc9c40dd5d4e908f"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["2e29814d5bcf0a0dba2e14c54f15adf5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node21_name,
                    value=decode_value(top_level_unique_values_dict["2e29814d5bcf0a0dba2e14c54f15adf5"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node22_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node22_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="variable_name",
                    node_name=node22_name,
                    value=decode_value(top_level_unique_values_dict["cd25336a439c3c6432d0b787f6982d8b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node22_name,
                    value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node22_name,
                    value=decode_value(top_level_unique_values_dict["5d8419df992a3db0e28003735f556b9b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="scope",
                    node_name=node22_name,
                    value=decode_value(top_level_unique_values_dict["25a6de226672088d980bed6d8e0c2162"]),
                    initial_setup=True,
                    is_output=False,
                )
            )


async def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_request = GetTopLevelFlowRequest()
        top_level_flow_result = await GriptapeNodes.ahandle_request(top_level_flow_request)
        if (
            isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess)
            and top_level_flow_result.flow_name is not None
        ):
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(input: dict, *, workflow_executor: WorkflowExecutor | None = None, **kwargs: Any) -> dict | None:
    return asyncio.run(aexecute_workflow(input=input, workflow_executor=workflow_executor, **kwargs))


async def aexecute_workflow(
    input: dict, *, workflow_executor: WorkflowExecutor | None = None, **kwargs: Any
) -> dict | None:
    if workflow_executor is None:
        workflow_executor = LocalWorkflowExecutor(skip_library_loading=True, workflows_to_register=[__file__], **kwargs)
    async with workflow_executor as executor:
        await build_workflow()
        await _ensure_workflow_context()
        await executor.arun(flow_input=input, **kwargs)
    return executor.output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    LocalWorkflowExecutor.add_cli_arguments(parser)
    parser.add_argument(
        "--json-input",
        default=None,
        help="JSON string containing parameter values. Takes precedence over individual parameter arguments if provided.",
    )
    parser.add_argument(
        "--exec_out", dest="exec_out", default=None, help="Connection to the next node in the execution chain"
    )
    parser.add_argument("--note", dest="note", default=None, help="New parameter")
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.exec_out is not None:
            flow_input["Start Flow"]["exec_out"] = args.exec_out
        if args.note is not None:
            flow_input["Start Flow"]["note"] = args.note
    executor = LocalWorkflowExecutor.from_cli_args(args, skip_library_loading=True, workflows_to_register=[__file__])
    workflow_output = execute_workflow(input=flow_input, workflow_executor=executor)
    print(workflow_output)
