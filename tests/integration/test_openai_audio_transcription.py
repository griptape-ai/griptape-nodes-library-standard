# /// script
# dependencies = []
# [tool.griptape-nodes]
# name = "test_openai_audio_transcription"
# schema_version = "0.16.0"
# engine_version_created_with = "0.77.3"
# node_libraries_referenced = [["Griptape Nodes Library", "0.67.0"]]
# node_types_used = [["Griptape Nodes Library", "EndFlow"], ["Griptape Nodes Library", "OpenAiAudioTranscription"], ["Griptape Nodes Library", "StartFlow"]]
# is_griptape_provided = false
# is_template = false
# ///
import argparse
import asyncio
import json
import logging
import math
import struct
import tempfile
import wave
from pathlib import Path

from griptape_nodes.bootstrap.workflow_executors.local_workflow_executor import LocalWorkflowExecutor
from griptape_nodes.drivers.storage.storage_backend import StorageBackend
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    GetTopLevelFlowRequest,
    GetTopLevelFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.library_events import RegisterLibraryFromFileRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import AddParameterToNodeRequest, SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

GriptapeNodes.handle_request(
    RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
)

context_manager = GriptapeNodes.ContextManager()
if not context_manager.has_current_workflow():
    context_manager.push_workflow(file_path=__file__)

flow_name = GriptapeNodes.handle_request(
    CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=False, metadata={})
).flow_name


def _create_test_wav_file() -> str:
    """Create a minimal WAV file with a sine wave tone for testing."""
    sample_rate = 16000
    duration_seconds = 1
    frequency = 440.0
    num_samples = sample_rate * duration_seconds

    samples = []
    for i in range(num_samples):
        t = i / sample_rate
        sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(struct.pack("<h", sample))

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(samples))
    return tmp.name


test_wav_path = _create_test_wav_file()

with GriptapeNodes.ContextManager().flow(flow_name):
    transcribe_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="OpenAiAudioTranscription",
            specific_library_name="Griptape Nodes Library",
            node_name="OpenAiAudioTranscription",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    start_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="StartFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="Start Flow",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(start_node):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="language",
                default_value="",
                tooltip="Language",
                type="str",
                input_types=["any"],
                output_type="str",
                ui_options={"display_name": "Language"},
                parent_container_name="",
                initial_setup=True,
            )
        )
    end_node = GriptapeNodes.handle_request(
        CreateNodeRequest(
            node_type="EndFlow",
            specific_library_name="Griptape Nodes Library",
            node_name="End Flow",
            metadata={},
            resolution="resolved",
            initial_setup=True,
        )
    ).node_name
    with GriptapeNodes.ContextManager().node(end_node):
        GriptapeNodes.handle_request(
            AddParameterToNodeRequest(
                parameter_name="str",
                default_value="",
                tooltip="Result",
                type="str",
                input_types=["str"],
                output_type="str",
                ui_options={},
                parent_container_name="",
                initial_setup=True,
            )
        )

    # Set audio directly on the transcription node
    with GriptapeNodes.ContextManager().node(transcribe_node):
        GriptapeNodes.handle_request(
            SetParameterValueRequest(
                parameter_name="audio",
                node_name=transcribe_node,
                value={"value": test_wav_path, "name": "test.wav"},
                initial_setup=True,
                is_output=False,
            )
        )

    # StartFlow.language -> Transcribe.language creates execution dependency
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=start_node,
            source_parameter_name="language",
            target_node_name=transcribe_node,
            target_parameter_name="language",
            initial_setup=True,
        )
    )
    GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=transcribe_node,
            source_parameter_name="output",
            target_node_name=end_node,
            target_parameter_name="str",
            initial_setup=True,
        )
    )


def _ensure_workflow_context():
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_flow():
        top_level_flow_result = GriptapeNodes.handle_request(GetTopLevelFlowRequest())
        if (
            isinstance(top_level_flow_result, GetTopLevelFlowResultSuccess)
            and top_level_flow_result.flow_name is not None
        ):
            flow_manager = GriptapeNodes.FlowManager()
            flow_obj = flow_manager.get_flow_by_name(top_level_flow_result.flow_name)
            context_manager.push_flow(flow_obj)


def execute_workflow(
    input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False
):
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
    input, storage_backend="local", project_file_path=None, workflow_executor=None, pickle_control_flow_result=False
):
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
    parser.add_argument("--storage-backend", choices=["local", "gtc"], default="local")
    parser.add_argument("--project-file-path", default=None)
    parser.add_argument("--json-input", default=None)
    parser.add_argument("--language", default="en")
    args = parser.parse_args()
    flow_input = {}
    if args.json_input is not None:
        flow_input = json.loads(args.json_input)
    if args.json_input is None:
        if "Start Flow" not in flow_input:
            flow_input["Start Flow"] = {}
        if args.language is not None:
            flow_input["Start Flow"]["language"] = args.language
    workflow_output = execute_workflow(
        input=flow_input, storage_backend=args.storage_backend, project_file_path=args.project_file_path
    )
    print(workflow_output)
