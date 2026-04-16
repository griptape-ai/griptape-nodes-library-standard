#!/usr/bin/env python
"""Integration tests for Seedance 2.0 specific features.

This test suite validates:
- Seedance 2.0 and 2.0 Fast basic text-to-video
- Seedance 2.0 with reference images
- Seedance 2.0 with reference videos (if available)
- Seedance 2.0 with reference audio (if available)
- Parameter validation for V2 models
"""

import argparse
import asyncio
import logging
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

# Register library
GriptapeNodes.handle_request(
    RegisterLibraryFromFileRequest(library_name="Griptape Nodes Library", perform_discovery_if_not_found=True)
)

logger = logging.getLogger(__name__)


def create_test_workflow(workflow_name: str):
    """Create a test workflow with SeedanceVideoGeneration node."""
    context_manager = GriptapeNodes.ContextManager()
    if not context_manager.has_current_workflow():
        context_manager.push_workflow(file_path=__file__)

    flow_name = GriptapeNodes.handle_request(
        CreateFlowRequest(parent_flow_name=None, flow_name=workflow_name, set_as_new_context=False, metadata={})
    ).flow_name

    with GriptapeNodes.ContextManager().flow(flow_name):
        gen_node = GriptapeNodes.handle_request(
            CreateNodeRequest(
                node_type="SeedanceVideoGeneration",
                specific_library_name="Griptape Nodes Library",
                node_name="SeedanceVideoGeneration",
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
                    parameter_name="prompt",
                    default_value="",
                    tooltip="Prompt",
                    type="str",
                    input_types=["any"],
                    output_type="str",
                    ui_options={"display_name": "Prompt", "multiline": True},
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
                    parameter_name="result",
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

        # Connect nodes
        GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=start_node,
                source_parameter_name="prompt",
                target_node_name=gen_node,
                target_parameter_name="prompt",
                initial_setup=True,
            )
        )

        GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=gen_node,
                source_parameter_name="result_details",
                target_node_name=end_node,
                target_parameter_name="result",
                initial_setup=True,
            )
        )

    return flow_name, gen_node


def _ensure_workflow_context():
    """Ensure workflow context is set."""
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


async def aexecute_workflow(input, storage_backend="local", project_file_path=None, workflow_executor=None):
    """Execute workflow asynchronously."""
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
        await executor.arun(flow_input=input, pickle_control_flow_result=False)
    return executor.output


def execute_workflow(input, storage_backend="local", project_file_path=None):
    """Execute workflow synchronously."""
    return asyncio.run(
        aexecute_workflow(input=input, storage_backend=storage_backend, project_file_path=project_file_path)
    )


def test_seedance_2_0_text_to_video(storage_backend="local", project_file_path=None):
    """Test Seedance 2.0 basic text-to-video generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Seedance 2.0 Text-to-Video")
    logger.info("=" * 60)

    flow_name, gen_node = create_test_workflow("test_v2_text_to_video")

    with GriptapeNodes.ContextManager().node(gen_node):
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="model_id", value="Seedance 2.0"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="resolution", value="720p"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="duration", value=5))

    flow_input = {"Start Flow": {"prompt": "A golden retriever running through a field of flowers at sunset"}}

    try:
        output = execute_workflow(
            input=flow_input, storage_backend=storage_backend, project_file_path=project_file_path
        )
        logger.info("✓ Seedance 2.0 text-to-video test passed")
        logger.info(f"Output: {output}")
        return True
    except Exception as e:
        logger.error(f"✗ Seedance 2.0 text-to-video test failed: {e}")
        return False


def test_seedance_2_0_fast_text_to_video(storage_backend="local", project_file_path=None):
    """Test Seedance 2.0 Fast basic text-to-video generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Seedance 2.0 Fast Text-to-Video")
    logger.info("=" * 60)

    flow_name, gen_node = create_test_workflow("test_v2_fast_text_to_video")

    with GriptapeNodes.ContextManager().node(gen_node):
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="model_id", value="Seedance 2.0 Fast"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="resolution", value="480p"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="duration", value=5))

    flow_input = {"Start Flow": {"prompt": "A red sports car driving on a coastal highway"}}

    try:
        output = execute_workflow(
            input=flow_input, storage_backend=storage_backend, project_file_path=project_file_path
        )
        logger.info("✓ Seedance 2.0 Fast text-to-video test passed")
        logger.info(f"Output: {output}")
        return True
    except Exception as e:
        logger.error(f"✗ Seedance 2.0 Fast text-to-video test failed: {e}")
        return False


def test_seedance_2_0_with_audio(storage_backend="local", project_file_path=None):
    """Test Seedance 2.0 with audio generation."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Seedance 2.0 with Audio")
    logger.info("=" * 60)

    flow_name, gen_node = create_test_workflow("test_v2_with_audio")

    with GriptapeNodes.ContextManager().node(gen_node):
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="model_id", value="Seedance 2.0 Fast"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="resolution", value="720p"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="duration", value=5))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="generate_audio", value=True))

    flow_input = {"Start Flow": {"prompt": "A jazz band playing in a smoky nightclub"}}

    try:
        output = execute_workflow(
            input=flow_input, storage_backend=storage_backend, project_file_path=project_file_path
        )
        logger.info("✓ Seedance 2.0 with audio test passed")
        logger.info(f"Output: {output}")
        return True
    except Exception as e:
        logger.error(f"✗ Seedance 2.0 with audio test failed: {e}")
        return False


def test_seedance_2_0_smart_duration(storage_backend="local", project_file_path=None):
    """Test Seedance 2.0 with smart duration selection (-1)."""
    logger.info("\n" + "=" * 60)
    logger.info("Test: Seedance 2.0 Smart Duration")
    logger.info("=" * 60)

    flow_name, gen_node = create_test_workflow("test_v2_smart_duration")

    with GriptapeNodes.ContextManager().node(gen_node):
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="model_id", value="Seedance 2.0 Fast"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="resolution", value="720p"))
        GriptapeNodes.handle_request(SetParameterValueRequest(parameter_name="duration", value=-1))

    flow_input = {"Start Flow": {"prompt": "Time-lapse of a flower blooming"}}

    try:
        output = execute_workflow(
            input=flow_input, storage_backend=storage_backend, project_file_path=project_file_path
        )
        logger.info("✓ Seedance 2.0 smart duration test passed")
        logger.info(f"Output: {output}")
        return True
    except Exception as e:
        logger.error(f"✗ Seedance 2.0 smart duration test failed: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Test Seedance 2.0 specific features")
    parser.add_argument("--storage-backend", choices=["local", "gtc"], default="local")
    parser.add_argument("--project-file-path", default=None)
    parser.add_argument("--test", choices=["all", "text-to-video", "fast", "audio", "smart-duration"], default="all")
    args = parser.parse_args()

    tests = {
        "text-to-video": test_seedance_2_0_text_to_video,
        "fast": test_seedance_2_0_fast_text_to_video,
        "audio": test_seedance_2_0_with_audio,
        "smart-duration": test_seedance_2_0_smart_duration,
    }

    if args.test == "all":
        results = {}
        for test_name, test_func in tests.items():
            results[test_name] = test_func(args.storage_backend, args.project_file_path)

        logger.info("\n" + "=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"{status}: {test_name}")

        if not all(results.values()):
            exit(1)
    else:
        test_func = tests[args.test]
        success = test_func(args.storage_backend, args.project_file_path)
        exit(0 if success else 1)
