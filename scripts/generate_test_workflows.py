"""Generator script for integration test workflows.

Produces one workflow .py file per node in tests/workflows/integration_tests/.
Run with: python scripts/generate_test_workflows.py
"""

from pathlib import Path

# Existing text_prompt configs: (NodeType, prompt_param, output_param, prompt_value, filename_suffix)
TEXT_PROMPT_CONFIGS = [
    # Image generation
    ("FluxImageGeneration", "prompt", "image_url", "A red circle", "flux_image_generation"),
    ("Flux2ImageGeneration", "prompt", "image_url", "A red circle", "flux_2_image_generation"),
    ("GrokImageGeneration", "prompt", "image_url", "A red circle", "grok_image_generation"),
    ("GoogleImageGeneration", "prompt", "image", "A red circle", "google_image_generation"),
    ("SeedreamImageGeneration", "prompt", "image_url", "A red circle", "seedream_image_generation"),
    ("QwenImageGeneration", "prompt", "image_url", "A red circle", "qwen_image_generation"),
    ("GenerateImage", "prompt", "output", "A red circle", "generate_image"),
    # Video generation
    ("GrokVideoGeneration", "prompt", "video_url", "A ball bouncing", "grok_video_generation"),
    ("KlingTextToVideoGeneration", "prompt", "video_url", "A ball bouncing", "kling_text_to_video_generation"),
    ("Veo3VideoGeneration", "prompt", "video_url", "A ball bouncing", "veo3_video_generation"),
    ("SoraVideoGeneration", "prompt", "video_url", "A ball bouncing", "sora_video_generation"),
    ("LTXTextToVideoGeneration", "prompt", "video_url", "A ball bouncing", "ltx_text_to_video_generation"),
    ("WanTextToVideoGeneration", "prompt", "video_url", "A ball bouncing", "wan_text_to_video_generation"),
    ("MinimaxHailuoVideoGeneration", "prompt", "video_url", "A ball bouncing", "minimax_hailuo_video_generation"),
    ("SeedanceVideoGeneration", "prompt", "video_url", "A ball bouncing", "seedance_video_generation"),
    # Audio generation
    ("ElevenLabsTextToSpeechGeneration", "text", "audio_url", "Hello world", "elevenlabs_text_to_speech_generation"),
    ("ElevenLabsSoundEffectGeneration", "text", "audio_url", "Thunder clap", "elevenlabs_sound_effect_generation"),
    ("ElevenLabsMusicGeneration", "text", "audio_url", "Upbeat jazz", "elevenlabs_music_generation"),
    # 3D generation
    ("Rodin23DGeneration", "prompt", "model_url", "A simple chair", "rodin2_3d_generation"),
]

# Advanced configs — each dict must include "template" and "suffix" plus template-specific keys.
ADVANCED_CONFIGS = [
    # no_input — node generates output with no required inputs
    {
        "template": "no_input",
        "node_type": "CreateColorBars",
        "output_param": "image",
        "suffix": "create_color_bars",
    },
    # single_image — standard params: image_in=input_image, output=output
    {"template": "single_image", "node_type": "AdjustImageEQ", "image_in": "input_image", "output_param": "output", "suffix": "adjust_image_eq"},
    {"template": "single_image", "node_type": "AdjustImageLevels", "image_in": "input_image", "output_param": "output", "suffix": "adjust_image_levels"},
    {"template": "single_image", "node_type": "BloomEffect", "image_in": "input_image", "output_param": "output", "suffix": "bloom_effect"},
    {"template": "single_image", "node_type": "FlipImage", "image_in": "input_image", "output_param": "output", "suffix": "flip_image"},
    {"template": "single_image", "node_type": "GaussianBlurImage", "image_in": "input_image", "output_param": "output", "suffix": "gaussian_blur_image"},
    {"template": "single_image", "node_type": "GrayscaleImage", "image_in": "input_image", "output_param": "output", "suffix": "grayscale_image"},
    {"template": "single_image", "node_type": "CropImage", "image_in": "input_image", "output_param": "output", "suffix": "crop_image"},
    {"template": "single_image", "node_type": "DisplayChannel", "image_in": "input_image", "output_param": "output", "suffix": "display_channel"},
    {"template": "single_image", "node_type": "InvertImage", "image_in": "input_image", "output_param": "output", "suffix": "invert_image"},
    {"template": "single_image", "node_type": "SetColorToTransparent", "image_in": "input_image", "output_param": "output", "suffix": "set_color_to_transparent"},
    # single_image — non-standard params
    {"template": "single_image", "node_type": "ExtendCanvas", "image_in": "input_image", "output_param": "extended_image", "suffix": "extend_canvas"},
    {"template": "single_image", "node_type": "GaussianEdgeFade", "image_in": "input_image", "output_param": "output_image", "suffix": "gaussian_edge_fade"},
    {"template": "single_image", "node_type": "DisplayMask", "image_in": "input_image", "output_param": "output_mask", "suffix": "display_mask"},
    {"template": "single_image", "node_type": "InvertMask", "image_in": "input_mask", "output_param": "output_mask", "suffix": "invert_mask"},
    {"template": "single_image", "node_type": "ImageGridSplitter", "image_in": "input_image", "output_param": "preview", "suffix": "image_grid_splitter"},
    {"template": "single_image", "node_type": "WriteImageMetadataNode", "image_in": "input_image", "output_param": "output_image", "suffix": "write_image_metadata"},
    # single_image — API proxy nodes
    {"template": "single_image", "node_type": "SeedVRImageUpscale", "image_in": "image_url", "output_param": "image", "suffix": "seedvr_image_upscale"},
    {"template": "single_image", "node_type": "TopazImageEnhance", "image_in": "image_input", "output_param": "image_output", "suffix": "topaz_image_enhance"},
    # dual_image — two CreateColorBars instances feed two image inputs
    {"template": "dual_image", "node_type": "ApplyMask", "image_in_1": "input_image", "image_in_2": "input_mask", "output_param": "output", "suffix": "apply_mask"},
    {"template": "dual_image", "node_type": "ColorMatch", "image_in_1": "reference_image", "image_in_2": "target_image", "output_param": "output", "suffix": "color_match"},
    {"template": "dual_image", "node_type": "ImageBlendCompositor", "image_in_1": "input_image", "image_in_2": "blend_image", "output_param": "output", "suffix": "image_blend_compositor"},
    # image_and_text — CreateColorBars + a text param set directly on the node
    {"template": "image_and_text", "node_type": "AddTextToExistingImage", "image_in": "input_image", "text_param": "text", "text_value": "Hello", "output_param": "output", "suffix": "add_text_to_existing_image"},
    {"template": "image_and_text", "node_type": "GrokImageEdit", "image_in": "image", "text_param": "prompt", "text_value": "Make it blue", "output_param": "image_url", "suffix": "grok_image_edit"},
    # video_input — LTXTextToVideoGeneration generates a video, node receives it
    {"template": "video_input", "node_type": "CropVideo", "video_in": "video", "output_param": "output", "suffix": "crop_video"},
    {"template": "video_input", "node_type": "ExtractAudio", "video_in": "video", "output_param": "extracted_audio", "suffix": "extract_audio"},
    {"template": "video_input", "node_type": "ExtractLastFrame", "video_in": "video", "output_param": "last_frame_image", "suffix": "extract_last_frame"},
    {"template": "video_input", "node_type": "ResizeVideo", "video_in": "video", "output_param": "resized_video", "suffix": "resize_video"},
    {"template": "video_input", "node_type": "SeedVRVideoUpscale", "video_in": "video_url", "output_param": "video", "suffix": "seedvr_video_upscale"},
    # video_and_prompt — LTXTextToVideoGeneration video + a prompt set on the node
    {"template": "video_and_prompt", "node_type": "GrokVideoEdit", "video_in": "video", "prompt_value": "Make it slow motion", "output_param": "video_url", "suffix": "grok_video_edit"},
    {"template": "video_and_prompt", "node_type": "LTXVideoRetake", "video_in": "video", "prompt_value": "A ball bouncing", "output_param": "video_url", "suffix": "ltx_video_retake"},
    # image_to_video — CreateColorBars provides an image; prompt_param=None means no prompt needed
    {"template": "image_to_video", "node_type": "KlingImageToVideoGeneration", "image_in": "image", "prompt_param": "prompt", "prompt_value": "Animate this", "output_param": "video_url", "suffix": "kling_image_to_video_generation"},
    {"template": "image_to_video", "node_type": "LTXImageToVideoGeneration", "image_in": "image", "prompt_param": "prompt", "prompt_value": "Animate this", "output_param": "video_url", "suffix": "ltx_image_to_video_generation"},
    {"template": "image_to_video", "node_type": "WanImageToVideoGeneration", "image_in": "input_image", "prompt_param": None, "prompt_value": None, "output_param": "video", "suffix": "wan_image_to_video_generation"},
]

# ---------------------------------------------------------------------------
# Shared code blocks embedded in every generated workflow file
# ---------------------------------------------------------------------------

_IMPORTS = """\
import asyncio
import logging
from pathlib import Path

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
from griptape_nodes.drivers.storage.storage_backend import StorageBackend
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes"""

_SETUP = """\
GriptapeNodes.handle_request(RegisterLibraryFromFileRequest(
    library_name="Griptape Nodes Library", perform_discovery_if_not_found=True))

context_manager = GriptapeNodes.ContextManager()
if not context_manager.has_current_workflow():
    context_manager.push_workflow(file_path=__file__)

flow_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1",
                      set_as_new_context=False, metadata={})
).flow_name
"""

_FOOTER = """\

def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess) and top_level_flow_result.flow_name is not None:
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False):
    return asyncio.run(aexecute_workflow(input=input, storage_backend=storage_backend, project_file_path=project_file_path, workflow_executor=workflow_executor, pickle_control_flow_result=pickle_control_flow_result))


async def aexecute_workflow(input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False):
    _ensure_workflow_context()
    storage_backend_enum = StorageBackend(storage_backend)
    project_file_path_resolved = Path(project_file_path) if project_file_path is not None else None
    workflow_executor = workflow_executor or LocalWorkflowExecutor(
        storage_backend=storage_backend_enum, project_file_path=project_file_path_resolved,
        skip_library_loading=True, workflows_to_register=[__file__])
    async with workflow_executor as executor:
        await executor.arun(flow_input=input, pickle_control_flow_result=pickle_control_flow_result)
    return executor.output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    workflow_output = execute_workflow(input={})
    print(workflow_output)
"""

# ---------------------------------------------------------------------------
# Helpers that produce the indented "with flow:" block for each template type
# ---------------------------------------------------------------------------

_CREATE_NODE = """\
    {var} = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="{node_type}", specific_library_name="Griptape Nodes Library",
        node_name="{node_name}", metadata={{}}, resolution="resolved", initial_setup=True,
    )).node_name"""

_ADD_END_FLOW_PARAM = """\
    with GriptapeNodes.ContextManager().node(end_node):
        GriptapeNodes.handle_request(AddParameterToNodeRequest(
            parameter_name="str", default_value="", tooltip="Result",
            type="str", input_types=["str"], output_type="str",
            ui_options={}, parent_container_name="", initial_setup=True,
        ))"""

_CONNECT = """\
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name={src}, source_parameter_name="{src_param}",
        target_node_name={tgt}, target_parameter_name="{tgt_param}", initial_setup=True))"""

_SET_PARAM = """\
    with GriptapeNodes.ContextManager().node({node_var}):
        GriptapeNodes.handle_request(SetParameterValueRequest(
            parameter_name="{param}", node_name={node_var},
            value="{value}", initial_setup=True, is_output=False))"""

_TAIL_CONNECTIONS = """\
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=gen_node, source_parameter_name="{output_param}",
        target_node_name=to_text_node, target_parameter_name="from", initial_setup=True))
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=to_text_node, source_parameter_name="output",
        target_node_name=assert_node, target_parameter_name="file_path", initial_setup=True))
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=assert_node, source_parameter_name="result_details",
        target_node_name=end_node, target_parameter_name="str", initial_setup=True))"""


def _create_node(var: str, node_type: str, node_name: str) -> str:
    return _CREATE_NODE.format(var=var, node_type=node_type, node_name=node_name)


def _common_utility_nodes() -> str:
    """Create ToText, AssertFileExists, and EndFlow nodes with the EndFlow param."""
    lines = []
    lines.append(_create_node("to_text_node", "ToText", "To Text"))
    lines.append(_create_node("assert_node", "AssertFileExists", "Assert File Exists"))
    lines.append(_create_node("end_node", "EndFlow", "End Flow"))
    lines.append(_ADD_END_FLOW_PARAM)
    return "\n".join(lines)


def _tail_connections(output_param: str) -> str:
    return _TAIL_CONNECTIONS.format(output_param=output_param)


def _set_param(node_var: str, param: str, value: str) -> str:
    return _SET_PARAM.format(node_var=node_var, param=param, value=value)


def _connect(src: str, src_param: str, tgt: str, tgt_param: str) -> str:
    return _CONNECT.format(src=src, src_param=src_param, tgt=tgt, tgt_param=tgt_param)


# ---------------------------------------------------------------------------
# Body builders for each template type
# ---------------------------------------------------------------------------

def _body_no_input(cfg: dict) -> str:
    node_type = cfg["node_type"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_single_image(cfg: dict) -> str:
    node_type = cfg["node_type"]
    image_in = cfg["image_in"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("source_node", "CreateColorBars", "Create Color Bars"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _connect("source_node", "image", "gen_node", image_in),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_dual_image(cfg: dict) -> str:
    node_type = cfg["node_type"]
    image_in_1 = cfg["image_in_1"]
    image_in_2 = cfg["image_in_2"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("source_node_1", "CreateColorBars", "Create Color Bars 1"),
        _create_node("source_node_2", "CreateColorBars", "Create Color Bars 2"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _connect("source_node_1", "image", "gen_node", image_in_1),
        _connect("source_node_2", "image", "gen_node", image_in_2),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_image_and_text(cfg: dict) -> str:
    node_type = cfg["node_type"]
    image_in = cfg["image_in"]
    text_param = cfg["text_param"]
    text_value = cfg["text_value"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("source_node", "CreateColorBars", "Create Color Bars"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _connect("source_node", "image", "gen_node", image_in),
        _set_param("gen_node", text_param, text_value),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_video_input(cfg: dict) -> str:
    node_type = cfg["node_type"]
    video_in = cfg["video_in"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("ltx_node", "LTXTextToVideoGeneration", "LTX Text To Video Generation"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _set_param("ltx_node", "prompt", "A ball bouncing"),
        _connect("ltx_node", "video_url", "gen_node", video_in),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_video_and_prompt(cfg: dict) -> str:
    node_type = cfg["node_type"]
    video_in = cfg["video_in"]
    prompt_value = cfg["prompt_value"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("ltx_node", "LTXTextToVideoGeneration", "LTX Text To Video Generation"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _set_param("ltx_node", "prompt", "A ball bouncing"),
        _connect("ltx_node", "video_url", "gen_node", video_in),
        _set_param("gen_node", "prompt", prompt_value),
        _tail_connections(output_param),
    ]
    return "\n".join(parts)


def _body_image_to_video(cfg: dict) -> str:
    node_type = cfg["node_type"]
    image_in = cfg["image_in"]
    prompt_param = cfg["prompt_param"]
    prompt_value = cfg["prompt_value"]
    output_param = cfg["output_param"]
    parts = [
        "with GriptapeNodes.ContextManager().flow(flow_name):",
        _create_node("source_node", "CreateColorBars", "Create Color Bars"),
        _create_node("gen_node", node_type, node_type),
        _common_utility_nodes(),
        _connect("source_node", "image", "gen_node", image_in),
    ]
    if prompt_param is not None and prompt_value is not None:
        parts.append(_set_param("gen_node", prompt_param, prompt_value))
    parts.append(_tail_connections(output_param))
    return "\n".join(parts)


_BODY_BUILDERS = {
    "no_input": _body_no_input,
    "single_image": _body_single_image,
    "dual_image": _body_dual_image,
    "image_and_text": _body_image_and_text,
    "video_input": _body_video_input,
    "video_and_prompt": _body_video_and_prompt,
    "image_to_video": _body_image_to_video,
}

# ---------------------------------------------------------------------------
# node_types_used lists per template type (sorted for determinism)
# ---------------------------------------------------------------------------

_NODE_TYPES_USED = {
    "no_input": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "single_image": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "CreateColorBars"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "dual_image": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "CreateColorBars"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "image_and_text": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "CreateColorBars"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "video_input": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "LTXTextToVideoGeneration"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "video_and_prompt": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "LTXTextToVideoGeneration"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
    "image_to_video": '["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "CreateColorBars"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "ToText"]',
}

# ---------------------------------------------------------------------------
# Existing text_prompt template (unchanged from original implementation)
# ---------------------------------------------------------------------------

WORKFLOW_TEMPLATE = """\
# /// script
# dependencies = []
# [tool.griptape-nodes]
# name = "test_{node_snake}"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "AssertFileExists"], ["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "{NodeType}"], ["Griptape Nodes Library", "StartFlow"], ["Griptape Nodes Library", "ToText"]]
# is_griptape_provided = false
# is_template = false
# ///
import argparse
import asyncio
import json
import logging
from pathlib import Path

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.bootstrap.workflow_executors.workflow_executor import WorkflowExecutor
from griptape_nodes.drivers.storage.storage_backend import StorageBackend
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, GetTopLevelFlowRequest, GetTopLevelFlowResultSuccess
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

GriptapeNodes.handle_request(RegisterLibraryFromFileRequest(
    library_name="Griptape Nodes Library", perform_discovery_if_not_found=True))

context_manager = GriptapeNodes.ContextManager()
if not context_manager.has_current_workflow():
    context_manager.push_workflow(file_path=__file__)

flow_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1",
                      set_as_new_context=False, metadata={})
).flow_name

with GriptapeNodes.ContextManager().flow(flow_name):
    gen_node = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="{NodeType}", specific_library_name="Griptape Nodes Library",
        node_name="{NodeType}", metadata={}, resolution="resolved", initial_setup=True,
    )).node_name
    to_text_node = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="ToText", specific_library_name="Griptape Nodes Library",
        node_name="To Text", metadata={}, resolution="resolved", initial_setup=True,
    )).node_name
    assert_node = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="AssertFileExists", specific_library_name="Griptape Nodes Library",
        node_name="Assert File Exists", metadata={}, resolution="resolved", initial_setup=True,
    )).node_name
    start_node = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="StartFlow", specific_library_name="Griptape Nodes Library",
        node_name="Start Flow", metadata={}, resolution="resolved", initial_setup=True,
    )).node_name
    with GriptapeNodes.ContextManager().node(start_node):
        GriptapeNodes.handle_request(AddParameterToNodeRequest(
            parameter_name="prompt", default_value="", tooltip="Prompt",
            type="str", input_types=["any"], output_type="str",
            ui_options={"display_name": "Prompt", "multiline": True},
            parent_container_name="", initial_setup=True,
        ))
    end_node = GriptapeNodes.handle_request(CreateNodeRequest(
        node_type="EndFlow", specific_library_name="Griptape Nodes Library",
        node_name="End Flow", metadata={}, resolution="resolved", initial_setup=True,
    )).node_name
    with GriptapeNodes.ContextManager().node(end_node):
        GriptapeNodes.handle_request(AddParameterToNodeRequest(
            parameter_name="str", default_value="", tooltip="Result",
            type="str", input_types=["str"], output_type="str",
            ui_options={}, parent_container_name="", initial_setup=True,
        ))

    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=start_node, source_parameter_name="prompt",
        target_node_name=gen_node, target_parameter_name="{prompt_param}", initial_setup=True))
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=gen_node, source_parameter_name="{output_param}",
        target_node_name=to_text_node, target_parameter_name="from", initial_setup=True))
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=to_text_node, source_parameter_name="output",
        target_node_name=assert_node, target_parameter_name="file_path", initial_setup=True))
    GriptapeNodes.handle_request(CreateConnectionRequest(
        source_node_name=assert_node, source_parameter_name="result_details",
        target_node_name=end_node, target_parameter_name="str", initial_setup=True))

    with GriptapeNodes.ContextManager().node(start_node):
        GriptapeNodes.handle_request(SetParameterValueRequest(
            parameter_name="prompt", node_name=start_node,
            value="{prompt_value}", initial_setup=True, is_output=False))


def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess) and top_level_flow_result.flow_name is not None:
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False):
    return asyncio.run(aexecute_workflow(input=input, storage_backend=storage_backend, project_file_path=project_file_path, workflow_executor=workflow_executor, pickle_control_flow_result=pickle_control_flow_result))


async def aexecute_workflow(input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False):
    _ensure_workflow_context()
    storage_backend_enum = StorageBackend(storage_backend)
    project_file_path_resolved = Path(project_file_path) if project_file_path is not None else None
    workflow_executor = workflow_executor or LocalWorkflowExecutor(
        storage_backend=storage_backend_enum, project_file_path=project_file_path_resolved,
        skip_library_loading=True, workflows_to_register=[__file__])
    async with workflow_executor as executor:
        await executor.arun(flow_input=input, pickle_control_flow_result=pickle_control_flow_result)
    return executor.output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage-backend", choices=["local", "gtc"], default="local")
    parser.add_argument("--project-file-path", default=None)
    parser.add_argument("--json-input", default=None)
    parser.add_argument("--prompt", default=None)
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.prompt is not None:
            flow_input["Start Flow"]["prompt"] = args.prompt
    workflow_output = execute_workflow(
        input=flow_input, storage_backend=args.storage_backend,
        project_file_path=args.project_file_path)
    print(workflow_output)
"""


def generate_text_prompt_workflow(node_type: str, prompt_param: str, output_param: str, prompt_value: str, filename_suffix: str) -> str:
    return (
        WORKFLOW_TEMPLATE
        .replace("{node_snake}", filename_suffix)
        .replace("{NodeType}", node_type)
        .replace("{prompt_param}", prompt_param)
        .replace("{output_param}", output_param)
        .replace("{prompt_value}", prompt_value)
    )


def generate_advanced_workflow(cfg: dict) -> str:
    template = cfg["template"]
    node_type = cfg["node_type"]
    suffix = cfg["suffix"]

    node_types_list = (
        "[" + _NODE_TYPES_USED[template].replace("{NodeType}", node_type) + "]"
    )

    body = _BODY_BUILDERS[template](cfg)

    content = "\n".join([
        f"# /// script",
        f"# dependencies = []",
        f"# [tool.griptape-nodes]",
        f"# name = \"test_{suffix}\"",
        f"# schema_version = \"0.16.0\"",
        f"# engine_version_created_with = \"0.77.3\"",
        f"# node_libraries_referenced = [[\"Griptape Nodes Library\", \"0.67.0\"]]",
        f"# node_types_used = {node_types_list}",
        f"# is_griptape_provided = false",
        f"# is_template = false",
        f"# ///",
        _IMPORTS,
        "",
        _SETUP,
        body,
        _FOOTER,
    ])
    return content


def main() -> None:
    output_dir = Path(__file__).parents[1] / "tests" / "workflows" / "integration_tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = []

    for node_type, prompt_param, output_param, prompt_value, filename_suffix in TEXT_PROMPT_CONFIGS:
        content = generate_text_prompt_workflow(node_type, prompt_param, output_param, prompt_value, filename_suffix)
        output_path = output_dir / f"test_{filename_suffix}.py"
        output_path.write_text(content)
        generated.append(output_path.name)
        print(f"Generated {output_path}")

    for cfg in ADVANCED_CONFIGS:
        content = generate_advanced_workflow(cfg)
        output_path = output_dir / f"test_{cfg['suffix']}.py"
        output_path.write_text(content)
        generated.append(output_path.name)
        print(f"Generated {output_path}")

    print(f"\nTotal: {len(generated)} workflow files generated in {output_dir}")


if __name__ == "__main__":
    main()
