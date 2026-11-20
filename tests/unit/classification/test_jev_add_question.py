"""The add buttons on Questions (JEV) create a question node and wire it into a new row."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest
from griptape_nodes.exe_types.node_types import NODE_DEFAULT_SIZE
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    CreateNodeResultSuccess,
    SetNodeMetadataRequest,
    SetNodeMetadataResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import ButtonDetailsMessagePayload

from griptape_nodes_library.classification.jev_ask import QUESTION_NODE_GAP, JevAsk

CANVAS = "jev_add_question_canvas"


@pytest.fixture
def canvas(griptape_nodes: Any) -> Generator[str, None, None]:  # noqa: ARG001
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="jev_add_question_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=CANVAS))
        assert isinstance(result, CreateFlowResultSuccess)
        yield CANVAS
    finally:
        context_manager.pop_workflow()


def _create(node_type: str, flow: str, position: dict[str, float] | None = None) -> Any:
    result = GriptapeNodes.handle_request(
        CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow, metadata={"position": position})
    )
    assert isinstance(result, CreateNodeResultSuccess), result.result_details
    return GriptapeNodes.NodeManager().get_node_by_name(result.node_name)


def _details() -> ButtonDetailsMessagePayload:
    return ButtonDetailsMessagePayload(label="+", variant="secondary", size="default", state="normal")


def _incoming(node: Any) -> list[tuple[str, str]]:
    result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    assert isinstance(result, ListConnectionsForNodeResultSuccess)
    return [(c.source_node_name, c.target_parameter_name) for c in result.incoming_connections]


def _box(node: Any) -> tuple[float, float, float, float]:
    metadata = node.metadata
    size = metadata.get("size") or NODE_DEFAULT_SIZE
    return metadata["position"]["x"], metadata["position"]["y"], size["width"], size["height"]


def _resize(node: Any, width: int, height: int) -> None:
    result = GriptapeNodes.handle_request(
        SetNodeMetadataRequest(node_name=node.name, metadata={"size": {"width": width, "height": height}})
    )
    assert isinstance(result, SetNodeMetadataResultSuccess)


@pytest.mark.parametrize("node_type", ["JevYesNoQuestion", "JevPickOneQuestion", "JevRateQuestion"])
def test_each_button_adds_a_connected_question_node_left_of_the_questions_node(canvas: str, node_type: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas, {"x": 1000, "y": 200}))
    _resize(ask, 600, 800)

    result = ask._add_question(node_type, _details())

    assert result.success, result.details
    ((source, row),) = _incoming(ask)
    added = GriptapeNodes.NodeManager().get_node_by_name(source)
    assert type(added).__name__ == node_type
    assert row.startswith("questions_")
    x, y, width, _ = _box(added)
    assert x + width + QUESTION_NODE_GAP == 1000
    assert y == 200


def test_new_questions_stack_below_the_measured_height_of_the_ones_already_connected(canvas: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas, {"x": 1000, "y": 200}))

    assert ask._add_question("JevYesNoQuestion", _details()).success
    first = GriptapeNodes.NodeManager().get_node_by_name(_incoming(ask)[0][0])
    _resize(first, 450, 900)
    assert ask._add_question("JevYesNoQuestion", _details()).success

    incoming = _incoming(ask)
    assert len({source for source, _ in incoming}) == 2
    assert len({row for _, row in incoming}) == 2
    second = GriptapeNodes.NodeManager().get_node_by_name(incoming[1][0])
    assert _box(second)[1] == 200 + 900 + QUESTION_NODE_GAP
