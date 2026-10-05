from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    DeleteFlowRequest,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import (
    SetParameterValueRequest,
    SetParameterValueResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from collections.abc import Generator

FLOW_NAME = "canvas"


@pytest.fixture
def flow(griptape_nodes: GriptapeNodes) -> Generator[str, None, None]:  # noqa: ARG001
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="test_numeric_option_strings_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=FLOW_NAME))
        assert isinstance(result, CreateFlowResultSuccess)
        yield FLOW_NAME
        GriptapeNodes.handle_request(DeleteFlowRequest(flow_name=FLOW_NAME))
    finally:
        context_manager.pop_workflow()


@pytest.mark.parametrize(
    ("node_type", "parameter_name", "value", "expected"),
    [
        ("KlingOmniVideoGeneration", "duration", "8", 8),
        ("KlingTextToVideoGeneration", "duration", " 12 ", 12),
        ("SoraVideoGeneration", "seconds", "8", 8),
        ("LTXTextToVideoGeneration", "fps", "48", 48),
    ],
)
def test_numeric_string_selects_matching_int_choice(
    flow: str, node_type: str, parameter_name: str, value: str, expected: int
) -> None:
    created = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow))
    assert isinstance(created, CreateNodeResultSuccess)

    result = GriptapeNodes.handle_request(
        SetParameterValueRequest(parameter_name=parameter_name, node_name=created.node_name, value=value)
    )

    assert isinstance(result, SetParameterValueResultSuccess)
    node = GriptapeNodes.NodeManager().get_node_by_name(created.node_name)
    assert node.get_parameter_value(parameter_name) == expected
