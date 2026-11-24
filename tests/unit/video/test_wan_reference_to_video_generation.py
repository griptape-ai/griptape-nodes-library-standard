from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest, DeleteConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.wan_reference_to_video_generation import WanReferenceToVideoGeneration


def _is_hidden(node: Any, parameter_name: str) -> bool:
    parameter = node.get_parameter_by_name(parameter_name)
    assert parameter is not None
    return parameter.hide


def _connection_request(request_type: type, target_parameter: str) -> Any:
    return request_type(
        source_node_name="source",
        source_parameter_name="video",
        target_node_name="wan",
        target_parameter_name=target_parameter,
    )


@pytest.fixture
def real_graph(open_workflow: None) -> Any:
    """A Wan node fed by another node's video output, both built by the engine in a real flow."""
    result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name="TestFlow"))
    assert result.succeeded(), result.result_details
    for name in ("source", "wan"):
        result = GriptapeNodes.handle_request(
            CreateNodeRequest(node_type="WanReferenceToVideoGeneration", node_name=name)
        )
        assert result.succeeded(), result.result_details
    node = GriptapeNodes.ObjectManager().attempt_get_object_by_name("wan")
    assert node is not None
    return node


def test_reference_video_inputs_progressively_appear_as_values_are_set() -> None:
    node = WanReferenceToVideoGeneration(name="WanR2V")

    assert _is_hidden(node, "reference_video_1") is False
    assert _is_hidden(node, "reference_video_2") is True
    assert _is_hidden(node, "reference_video_3") is True

    node.set_parameter_value("reference_video_1", "https://example.com/reference-1.mp4")
    assert _is_hidden(node, "reference_video_2") is False
    assert _is_hidden(node, "reference_video_3") is True

    node.set_parameter_value("reference_video_2", "https://example.com/reference-2.mp4")
    assert _is_hidden(node, "reference_video_3") is False


def test_reference_video_inputs_appear_and_hide_with_connections(real_graph: Any) -> None:
    def connect(target_parameter: str) -> None:
        result = GriptapeNodes.handle_request(_connection_request(CreateConnectionRequest, target_parameter))
        assert result.succeeded(), result.result_details

    def disconnect(target_parameter: str) -> None:
        result = GriptapeNodes.handle_request(_connection_request(DeleteConnectionRequest, target_parameter))
        assert result.succeeded(), result.result_details

    connect("reference_video_1")
    assert _is_hidden(real_graph, "reference_video_2") is False
    assert _is_hidden(real_graph, "reference_video_3") is True

    connect("reference_video_2")
    assert _is_hidden(real_graph, "reference_video_3") is False

    disconnect("reference_video_2")
    assert _is_hidden(real_graph, "reference_video_2") is False
    assert _is_hidden(real_graph, "reference_video_3") is True

    disconnect("reference_video_1")
    assert _is_hidden(real_graph, "reference_video_2") is True
    assert _is_hidden(real_graph, "reference_video_3") is True
