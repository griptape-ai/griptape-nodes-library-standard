"""The add buttons on Questions (JEV) create a question node and wire it into a new row."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, cast

import pytest
from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.exe_types.node_types import NODE_DEFAULT_SIZE
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultFailure,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
    ListNodesInFlowRequest,
    ListNodesInFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    CreateNodeResultSuccess,
    SetNodeMetadataRequest,
    SetNodeMetadataResultSuccess,
)
from griptape_nodes.retained_mode.events.object_events import RenameObjectRequest, RenameObjectResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import ButtonDetailsMessagePayload

from griptape_nodes_library.classification.jev_ask import QUESTION_NODE_GAP, JevAsk
from griptape_nodes_library.classification.jev_common import question_key

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


def _node_names(flow: str) -> set[str]:
    result = GriptapeNodes.handle_request(ListNodesInFlowRequest(flow_name=flow))
    assert isinstance(result, ListNodesInFlowResultSuccess)
    return set(result.node_names)


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


def test_a_button_added_node_gets_its_own_name_so_renaming_the_node_keeps_the_key(canvas: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas))

    for node_type in ("JevYesNoQuestion", "JevYesNoQuestion", "JevPickOneQuestion", "JevRateQuestion"):
        assert ask._add_question(node_type, _details()).success

    nodes = [GriptapeNodes.NodeManager().get_node_by_name(source) for source, _ in _incoming(ask)]
    assert [question_key(node) for node in nodes] == ["yes_no_1", "yes_no_2", "pick_one_1", "rate_1"]

    renamed = GriptapeNodes.handle_request(RenameObjectRequest(object_name=nodes[0].name, requested_name="tidy"))
    assert isinstance(renamed, RenameObjectResultSuccess), renamed.result_details
    assert question_key(nodes[0]) == "yes_no_1"


def test_a_name_that_is_in_use_is_skipped(canvas: str) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas))
    assert ask._add_question("JevYesNoQuestion", _details()).success
    ((source, _),) = _incoming(ask)
    first = GriptapeNodes.NodeManager().get_node_by_name(source)
    first.set_parameter_value("key", "yes_no_2")

    assert ask._add_question("JevYesNoQuestion", _details()).success

    keys = [question_key(GriptapeNodes.NodeManager().get_node_by_name(node)) for node, _ in _incoming(ask)]
    assert sorted(keys) == ["yes_no_1", "yes_no_2"]


def test_a_failed_connection_leaves_no_node_and_no_empty_row(canvas: str, monkeypatch: pytest.MonkeyPatch) -> None:
    ask = cast(JevAsk, _create("JevAsk", canvas))
    questions = ask.get_parameter_by_name("questions")
    assert isinstance(questions, ParameterList)
    nodes_before = _node_names(canvas)
    original = GriptapeNodes.handle_request

    def refuse_connections(request: Any, **kwargs: Any) -> Any:
        if isinstance(request, CreateConnectionRequest):
            return CreateConnectionResultFailure(result_details="refused")
        return original(request, **kwargs)

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(refuse_connections))

    result = ask._add_question("JevYesNoQuestion", _details())

    assert not result.success
    assert "refused" in result.details
    assert questions.get_child_parameters() == []
    assert _node_names(canvas) == nodes_before
