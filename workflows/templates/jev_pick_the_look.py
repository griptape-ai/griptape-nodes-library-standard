# /// script
# dependencies = []
#
# [tool.griptape-nodes]
# name = "jev_pick_the_look"
# schema_version = "0.21.0"
# engine_version_created_with = "0.104.0"
# node_libraries_referenced = [["Griptape Nodes Library", "0.88.0"]]
# node_types_used = [["Griptape Nodes Library", "CreateVariable"], ["Griptape Nodes Library", "GoogleImageGeneration"], ["Griptape Nodes Library", "GrokImageGeneration"], ["Griptape Nodes Library", "JevPickOne"], ["Griptape Nodes Library", "Note"], ["Griptape Nodes Library", "OpenAiImageGeneration"], ["Griptape Nodes Library", "TextInput"]]
# workflows_referenced = []
# description = "Reads a scene description and has JEV pick the image style that suits it best: an influencer photo, a storybook illustration, or a cinematic film still. Each style has its own image model and look, and only the picked one runs, so you get one image in the right style."
# image = "https://raw.githubusercontent.com/griptape-ai/griptape-nodes-library-standard/main/workflows/templates/thumbnail_jev_pick_the_look.webp"
# is_griptape_provided = true
# is_template = true
# is_internal = false
# creation_date = 2026-10-08T00:44:54.009070Z
# last_modified_date = 2026-10-08T00:55:24.077440Z
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
        "e6f038a4c642653369765ae6d82fe34b": "A solitary figure turns to face the camera with genuine curiosity, rain-slicked and radiant, their expression open and inviting. Behind them, a crumbling tower rises against a sky that shifts from decay to beauty—corroded metal bleeding into painted stone, shadow pooling into luminescence. Their face is brightly lit, eyes engaging directly with the viewer, and they're posed naturally but intentionally—one hand gesturing as if mid-discovery, their clothing styled to catch light and attention. Soft watercolor washes blend with sharp details, giving it the feeling of a storybook illustration brought to life. Everything feels ancient yet accessible, broken yet hopeful—the kind of image that feels both like a snapshot of adventure and a carefully composed invitation to explore.",
        "dfeec53ff4f9dc6c2c24a5e1cac409fa": "A young woman laughs at a sunny café window in Lisbon, holding an oat milk latte with a half-eaten almond croissant on the plate in front of her. Her linen shirt is rolled at the sleeves, gold hoop earrings catch the light, and a woven tote bag hangs off the back of her chair. Outside, a yellow tram passes on a busy cobblestone street.",
        "5d78fb7025741a1e625939f57218869a": "## Pick the Look with JEV\n\nReads a scene description, picks the image style that suits it best, and sends it to an image model that's good at that style.\n\n- **Influencer Photo:** Grok Image Generation\n- **Storybook Illustration:** OpenAI Image Generation\n- **Cinematic Film Still:** Google Nano Banana\n\nOnly the picked branch runs, so you get one image and pay for one image.\n\n**Try it:** run the flow as it is. **Story Prompt** mixes a few looks, but it mentions a storybook illustration, so JEV picks Storybook. Then drag the **Text** output of **Influencer Prompt** or **Cinematic Prompt** onto **Store Prompt**'s **Value** and run it again. Or write your own scene in any of them.\n\n",
        "424fcea245c10eb04462d2b962a5e9ca": "## Store the prompt\n\n**Store Prompt** saves the scene in a variable called `PROMPT`. Other nodes read it when you type `{PROMPT}` into a text field.\n\n**Pick One** reads `{PROMPT}` to choose a style, and each image node starts its prompt with `{PROMPT}`, so every node works from the same scene.",
        "10fa7bb925ce1577a033065543662d30": "## Pick One\n\n**Pick One (Choice)** reads `{PROMPT}` and picks the option in **Options** that fits it best. Each row is a label, a colon, and a description that tells JEV which scenes fit that option.\n\nEach label becomes a flow output, and the flow continues from the one JEV picks.\n\n- After a run, check **Confidence** and **Probabilities** to see how sure JEV was. A close split means the scene could suit more than one style.\n- To add a style, click **Add item to Options**, then connect its new output to another image node's Flow In.\n- When two options overlap, make their descriptions say where the line between them is.",
        "39f90d039a73793e43c2e0db1ba7a596": "## One image node per style\n\nEach style has its own image node, with a model that suits it. Its prompt is `{PROMPT}` followed by a paragraph describing the look.\n\nOnly the node for the picked style runs, because it's the only one whose Flow In receives the flow.\n\n- To change a look, edit the paragraph in that node's prompt.\n- To use a different model, swap the node and connect the option's output to the new node's Flow In.",
        "fd5fbd295fd0f7e0ad1c2e25ded84628": "A detective stands alone in a rain-soaked alley at 3 a.m., coat collar turned up, staring at a torn photograph in his hand. Steam rises from a grate at his feet. At the far end of the alley, a car idles with its headlights off, and someone inside is watching him. A single sodium streetlight flickers overhead.",
        "284b7e6d788f363f910f7beb1910473e": 600,
        "d2d429df37c58824865faf79edf2c20a": "gemini-3-pro-image",
        "39865b5645d8728f011522a978d08eb1": "{PROMPT} \n\nA cinematic film still, shot on 35mm anamorphic lens with a 2.39:1 widescreen feel. Dramatic, motivated lighting with deep shadows and a strong key light, and atmospheric haze catching the light. A teal-and-amber color grade with rich contrast and slightly crushed blacks. Shallow depth of field with subtle anamorphic lens flares and oval bokeh. Natural film grain and a composition framed like a carefully blocked shot from a prestige drama, with a sense of tension and story just before or after a pivotal moment.",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5": [],
        "0c607131a93856ed93d33dad0c3829fb": "16:9",
        "188664f31f62aa67b5d53f8e3c3fb45d": "1K",
        "b5bea41b6c623f7c09f1bf24dcae58eb": True,
        "d0ff5974b6aa52cf562bea5921840c03": 1.0,
        "22fa3ce4995af8d96fcd771f0e1f5d74": 0.95,
        "cc839368e679afc919e0c1e2912fc997": "cinematic.jpeg",
        "12ae32cb1ec02d01eda3581b127c1fee": "",
        "02396d2880f71d65510b43bbdfcc7d08": "gpt-image-2",
        "eb9fedac049676bcec51de7bfe9a4b22": "{PROMPT}\n\nA hand-painted children's picture book illustration in soft watercolor and gouache, with delicate ink linework. Gentle washes of color bleed softly at the edges over visible textured paper grain. It uses a warm, muted palette of honey yellows, sage greens, dusty rose, and cream, with cozy golden light. Characters have simple, rounded, expressive shapes. The composition is uncluttered with plenty of breathing room, and it feels nostalgic, tender, and inviting, like a classic bedtime story.\n\n",
        "1bb96b086f4966a51e5b312279a7af1a": "1792x1024",
        "e39eef82f61b21e2e7f762fcc4307358": 1024,
        "6b86b273ff34fce19d6b804eff5a3f57": 1,
        "60d4c90eee5e731df8d3ef2891de541d": "medium",
        "634635a2166efa2ae765abd31d4ae807": "auto",
        "6e8a45b4224910a4ff6d68a52d1d3fd5": "png",
        "48449a14a4ff7d79bb7a1b6f3d488eba": 80,
        "8ce44073d896ca3895e50bb731f397b8": "storybook.png",
        "7117fe6b6193e23a141ffc11d62b3f9e": "grok-imagine-image",
        "f646b0d810825a78856e1889cc42bbad": "{PROMPT}  Shot on an iPhone in portrait mode, as a candid lifestyle photo for Instagram. Bright, soft natural daylight, warm and slightly lifted tones, creamy background blur, a clean and airy composition with the subject off-center. It feels authentic and effortless, not staged, with a subtle film-like color grade and true-to-life skin texture.",
        "932acd98ec68395c221af1504122c9ac": "3:4",
        "16d2724fe0a9971ec931e119652d59e5": "1k",
        "db7890940ab1d057e3ce2c7cfa13e2b4": "influencer.jpg",
        "6112273ba12000fe4aaee5de7c74b0e3": "{PROMPT}\n\n",
        "d3ff2dd3fb3d53697a6e8d6959968f18": [
            "Influencer Photo: everyday life, people, food, fashion, travel, or products; bright natural light, candid smartphone photo, shallow depth of field, social media aesthetic",
            "Storybook Illustration: gentle, whimsical, or childlike scenes, animals, or fairy tales; soft watercolor washes, hand-drawn ink lines, warm colors",
            "Cinematic Film Still: drama, tension, mystery, action, or epic scale; anamorphic widescreen, moody lighting, film grain, shallow focus",
        ],
        "872607824c60a2f1ed93be2b03666ea8": "Influencer Photo: everyday life, people, food, fashion, travel, or products; bright natural light, candid smartphone photo, shallow depth of field, social media aesthetic",
        "fcd9e26cbc4d5b087573476a1c172f17": "Storybook Illustration: gentle, whimsical, or childlike scenes, animals, or fairy tales; soft watercolor washes, hand-drawn ink lines, warm colors",
        "d86b9bd4197295e6d8ec3d89c2cb8422": "Cinematic Film Still: drama, tension, mystery, action, or epic scale; anamorphic widescreen, moody lighting, film grain, shallow focus",
        "93d926fb5a12ad7dc7179ed59e1b66ac": "jev-latest",
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
                value=decode_value(top_level_unique_values_dict["dfeec53ff4f9dc6c2c24a5e1cac409fa"]),
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
                        "position": {"x": -1144, "y": -704},
                        "size": {"width": 644, "height": 584},
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
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node3_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Story Prompt",
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
                        "size": {"width": 644, "height": 412},
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
                    node_name="Note: Pick One",
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
                    resolution="resolved",
                    initial_setup=True,
                )
            )
        ).node_name
        node5_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="Note",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Note: Image Nodes",
                    metadata={
                        "position": {"x": 1408, "y": -528},
                        "size": {"width": 1832, "height": 412},
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
        node6_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Influencer Prompt",
                    metadata={
                        "position": {"x": -1144, "y": 396},
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
        node7_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="TextInput",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Cinematic Prompt",
                    metadata={
                        "position": {"x": -1144, "y": 836},
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
        node8_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="GoogleImageGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Cinematic Film Still (Nano Banana)",
                    metadata={
                        "library_node_metadata": {
                            "category": "image",
                            "description": "Generate images using Google models via Griptape model proxy",
                            "display_name": "Google Nano Banana Image Generation",
                            "tags": ["image", "generation", "ai", "api", "google", "nano-banana"],
                            "icon": "sparkles",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": ["gtc_gemini_3_pro_image", "gtc_gemini_3_1_flash_image"],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "GoogleImageGeneration",
                        "position": {"x": 1408, "y": 176},
                        "size": {"width": 600, "height": 1032},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="output_file",
                    traits=[
                        {
                            "trait_name": "FileSystemPicker",
                            "trait_module": "griptape_nodes.traits.file_system_picker",
                            "trait_state": {},
                        },
                        {"trait_name": "Button", "trait_module": "griptape_nodes.traits.button", "trait_state": {}},
                    ],
                    initial_setup=True,
                )
            )
        node9_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="OpenAiImageGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Storybook Illustration (OpenAI)",
                    metadata={
                        "library_node_metadata": {
                            "category": "image",
                            "description": "Generate images using OpenAI GPT Image models via Griptape model proxy",
                            "display_name": "OpenAI Image Generation",
                            "tags": [
                                "image",
                                "generation",
                                "ai",
                                "api",
                                "openai",
                                "gpt-image",
                                "gpt-image-2",
                                "gpt-image-1",
                                "gpt-image-1.5",
                            ],
                            "icon": "sparkles",
                            "color": None,
                            "group": "create",
                            "deprecation": None,
                            "is_node_group": None,
                            "declarations": [
                                {
                                    "type": "model_usage",
                                    "model_ids": [
                                        "gtc_gpt_image_1",
                                        "gtc_gpt_image_1_5",
                                        "gtc_gpt_image_2",
                                        "gtc_gpt_image_2_5_sunburst",
                                        "gtc_gpt_image_2_5_flare",
                                    ],
                                }
                            ],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "OpenAiImageGeneration",
                        "position": {"x": 2024, "y": 44},
                        "size": {"width": 600, "height": 1032},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        node10_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="GrokImageGeneration",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Influencer Photo (Grok)",
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
                        "position": {"x": 2684, "y": -44},
                        "size": {"width": 600, "height": 1036},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node10_name):
            await GriptapeNodes.ahandle_request(
                AlterParameterDetailsRequest(
                    parameter_name="output_file",
                    traits=[
                        {
                            "trait_name": "FileSystemPicker",
                            "trait_module": "griptape_nodes.traits.file_system_picker",
                            "trait_state": {},
                        },
                        {"trait_name": "Button", "trait_module": "griptape_nodes.traits.button", "trait_state": {}},
                    ],
                    initial_setup=True,
                )
            )
        node11_name = (
            await GriptapeNodes.ahandle_request(
                CreateNodeRequest(
                    node_type="JevPickOne",
                    specific_library_name="Griptape Nodes Library",
                    node_name="Pick One (JEV)",
                    metadata={
                        "position": {"x": 396, "y": -36},
                        "tempId": "placing-1791420328998-m2jooe",
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
                            "declarations": [],
                            "resolved_model_usage": [],
                        },
                        "library": "Griptape Nodes Library",
                        "node_type": "JevPickOne",
                        "showaddparameter": False,
                        "size": {"width": 672, "height": 1036},
                    },
                    initial_setup=True,
                )
            )
        ).node_name
        with GriptapeNodes.ContextManager().node(node11_name):
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="options_ParameterListUniqueParamID_d27c784024384c4b97150fe6e4d949f7",
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
                    parameter_name="options_ParameterListUniqueParamID_56b6dac6c2e94981b0dcc7088ba06d5a",
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
                    parameter_name="options_ParameterListUniqueParamID_0233afd813d847f8a78b4aab82ea3a2d",
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
                    parameter_name="option_d27c784024384c4b97150fe6e4d949f7",
                    tooltip="Taken when JEV picks Influencer Photo.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Influencer Photo"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_56b6dac6c2e94981b0dcc7088ba06d5a",
                    tooltip="Taken when JEV picks Storybook Illustration.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Storybook Illustration"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
            await GriptapeNodes.ahandle_request(
                AddParameterToNodeRequest(
                    parameter_name="option_0233afd813d847f8a78b4aab82ea3a2d",
                    tooltip="Taken when JEV picks Cinematic Film Still.",
                    type="parametercontroltype",
                    input_types=["parametercontroltype"],
                    output_type="parametercontroltype",
                    ui_options={"parameter_render_location": "top", "display_name": "Cinematic Film Still"},
                    mode_allowed_input=False,
                    mode_allowed_property=False,
                    is_user_defined=False,
                    initial_setup=True,
                )
            )
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
                source_node_name=node6_name,
                source_parameter_name="exec_out",
                target_node_name=node0_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
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
                source_node_name=node3_name,
                source_parameter_name="text",
                target_node_name=node0_name,
                target_parameter_name="value",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node0_name,
                source_parameter_name="exec_out",
                target_node_name=node11_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="option_d27c784024384c4b97150fe6e4d949f7",
                target_node_name=node10_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="option_56b6dac6c2e94981b0dcc7088ba06d5a",
                target_node_name=node9_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="option_0233afd813d847f8a78b4aab82ea3a2d",
                target_node_name=node8_name,
                target_parameter_name="exec_in",
                initial_setup=True,
            )
        )
        await GriptapeNodes.ahandle_request(
            CreateConnectionRequest(
                source_node_name=node11_name,
                source_parameter_name="failure",
                target_node_name=node8_name,
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
                    value=decode_value(top_level_unique_values_dict["e6f038a4c642653369765ae6d82fe34b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="value",
                    node_name=node0_name,
                    value=decode_value(top_level_unique_values_dict["dfeec53ff4f9dc6c2c24a5e1cac409fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node1_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node1_name,
                    value=decode_value(top_level_unique_values_dict["5d78fb7025741a1e625939f57218869a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node2_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node2_name,
                    value=decode_value(top_level_unique_values_dict["424fcea245c10eb04462d2b962a5e9ca"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node3_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["e6f038a4c642653369765ae6d82fe34b"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node3_name,
                    value=decode_value(top_level_unique_values_dict["e6f038a4c642653369765ae6d82fe34b"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node4_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node4_name,
                    value=decode_value(top_level_unique_values_dict["10fa7bb925ce1577a033065543662d30"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node5_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="note",
                    node_name=node5_name,
                    value=decode_value(top_level_unique_values_dict["39f90d039a73793e43c2e0db1ba7a596"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
        with GriptapeNodes.ContextManager().node(node6_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["dfeec53ff4f9dc6c2c24a5e1cac409fa"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node6_name,
                    value=decode_value(top_level_unique_values_dict["dfeec53ff4f9dc6c2c24a5e1cac409fa"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node7_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["fd5fbd295fd0f7e0ad1c2e25ded84628"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="text",
                    node_name=node7_name,
                    value=decode_value(top_level_unique_values_dict["fd5fbd295fd0f7e0ad1c2e25ded84628"]),
                    initial_setup=True,
                    is_output=True,
                )
            )
        with GriptapeNodes.ContextManager().node(node8_name):
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="api_key_provider",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["d2d429df37c58824865faf79edf2c20a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["39865b5645d8728f011522a978d08eb1"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_images",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="object_images",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="human_images",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="aspect_ratio",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["0c607131a93856ed93d33dad0c3829fb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="image_size",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["188664f31f62aa67b5d53f8e3c3fb45d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="auto_image_resize",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["b5bea41b6c623f7c09f1bf24dcae58eb"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="temperature",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["d0ff5974b6aa52cf562bea5921840c03"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="top_p",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["22fa3ce4995af8d96fcd771f0e1f5d74"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="use_google_search",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="use_google_image_search",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["cc839368e679afc919e0c1e2912fc997"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node8_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
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
                    value=decode_value(top_level_unique_values_dict["02396d2880f71d65510b43bbdfcc7d08"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["eb9fedac049676bcec51de7bfe9a4b22"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="size",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["1bb96b086f4966a51e5b312279a7af1a"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="custom_width",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["e39eef82f61b21e2e7f762fcc4307358"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="custom_height",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["e39eef82f61b21e2e7f762fcc4307358"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="input_images",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["4f53cda18c2baa0c0354bb5f9a3ecbe5"]),
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
                    parameter_name="background",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["634635a2166efa2ae765abd31d4ae807"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="moderation",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["634635a2166efa2ae765abd31d4ae807"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_format",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["6e8a45b4224910a4ff6d68a52d1d3fd5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_compression",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["48449a14a4ff7d79bb7a1b6f3d488eba"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node9_name,
                    value=decode_value(top_level_unique_values_dict["8ce44073d896ca3895e50bb731f397b8"]),
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
                    parameter_name="model",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["7117fe6b6193e23a141ffc11d62b3f9e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="prompt",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["f646b0d810825a78856e1889cc42bbad"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="aspect_ratio",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["932acd98ec68395c221af1504122c9ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="n",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["6b86b273ff34fce19d6b804eff5a3f57"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="quality",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["60d4c90eee5e731df8d3ef2891de541d"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="resolution",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["16d2724fe0a9971ec931e119652d59e5"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="output_file",
                    node_name=node10_name,
                    value=decode_value(top_level_unique_values_dict["db7890940ab1d057e3ce2c7cfa13e2b4"]),
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
                    parameter_name="api_key_provider",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="timeout",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["284b7e6d788f363f910f7beb1910473e"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="context",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["6112273ba12000fe4aaee5de7c74b0e3"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="question",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["d3ff2dd3fb3d53697a6e8d6959968f18"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_d27c784024384c4b97150fe6e4d949f7",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["872607824c60a2f1ed93be2b03666ea8"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_56b6dac6c2e94981b0dcc7088ba06d5a",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["fcd9e26cbc4d5b087573476a1c172f17"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="options_ParameterListUniqueParamID_0233afd813d847f8a78b4aab82ea3a2d",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["d86b9bd4197295e6d8ec3d89c2cb8422"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="model",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["93d926fb5a12ad7dc7179ed59e1b66ac"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="was_successful",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["fcbcf165908dd18a9e49f7ff27810176"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_id",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
            await GriptapeNodes.ahandle_request(
                SetParameterValueRequest(
                    parameter_name="generation_status",
                    node_name=node11_name,
                    value=decode_value(top_level_unique_values_dict["12ae32cb1ec02d01eda3581b127c1fee"]),
                    initial_setup=True,
                    is_output=False,
                )
            )
