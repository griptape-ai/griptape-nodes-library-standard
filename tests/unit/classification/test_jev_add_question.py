"""The add buttons on Questions (JEV) create a question node and wire it into a new row."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import ButtonDetailsMessagePayload

from griptape_nodes_library.classification.jev_ask import (
    QUESTION_NODE_OFFSET_X,
    QUESTION_NODE_OFFSET_Y,
    JevAsk,
)

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


@pytest.mark.parametrize("node_type", ["JevYesNoQuestion", "JevPickOneQuestion", "JevRateQuestion"])
def test_each_button_adds_a_connected_question_node(canvas: str, node_type: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas, {"x": 1000, "y": 200}))

    result = ask._add_question(node_type, _details())

    assert result.success, result.details
    ((source, row),) = _incoming(ask)
    added = GriptapeNodes.NodeManager().get_node_by_name(source)
    assert type(added).__name__ == node_type
    assert row.startswith("questions_")
    assert added.metadata["position"] == {"x": 1000 - QUESTION_NODE_OFFSET_X, "y": 200}


def test_each_new_question_gets_its_own_row_below_the_last(canvas: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas, {"x": 1000, "y": 200}))

    assert ask._add_question("JevYesNoQuestion", _details()).success
    assert ask._add_question("JevYesNoQuestion", _details()).success

    incoming = _incoming(ask)
    assert len({source for source, _ in incoming}) == 2
    assert len({row for _, row in incoming}) == 2
    second = GriptapeNodes.NodeManager().get_node_by_name(incoming[1][0])
    assert second.metadata["position"]["y"] == 200 + QUESTION_NODE_OFFSET_Y
