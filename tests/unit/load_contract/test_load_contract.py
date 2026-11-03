"""Nodes load without touching the graph, and a loaded copy runs like the node it was copied from.

Each test builds a node by normal edits, then loads a copy the way a saved workflow (or a worker)
does: into a scratch flow holding only the copy, with no connections.
"""

from collections.abc import Generator
from typing import Any, cast

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import (
    CreateFlowRequest,
    CreateFlowResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import (
    AddNodesToNodeGroupRequest,
    AddNodesToNodeGroupResultSuccess,
    CreateNodeRequest,
    CreateNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.variable_events import (
    CreateVariableRequest,
    CreateVariableResultSuccess,
    HasVariableRequest,
    HasVariableResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.variable_types import VariableScope

from griptape_nodes_library.classification.row_outputs import RowOutputsMixin
from griptape_nodes_library.image.describe_image import LEGACY_IMAGE_ROW, DescribeImage
from griptape_nodes_library.lists.display_list import DisplayList
from griptape_nodes_library.variables.create_variable import CreateVariable
from griptape_nodes_library.variables.set_variable import SetVariable

from .harness import create_scratch_flow, load_copy

CANVAS = "canvas"
SCRATCH = "scratch"
IMAGE = ImageArtifact(b"\x89PNG\r\n\x1a\nfake", format="png", width=1, height=1)

# Every variable node lists the visible variables from __init__ to fill its dropdown. __init__ cannot
# tell a load from a user creating the node, so that listing still runs while loading.
VARIABLE_DROPDOWN_LISTING = {"GetFlowForNodeRequest", "ListVariablesRequest"}


@pytest.fixture
def canvas(griptape_nodes: Any) -> Generator[str, None, None]:  # noqa: ARG001
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="load_contract_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=CANVAS))
        assert isinstance(result, CreateFlowResultSuccess)
        yield CANVAS
    finally:
        context_manager.pop_workflow()


@pytest.fixture
def scratch(canvas: str) -> str:  # noqa: ARG001
    return create_scratch_flow(SCRATCH)


def _create(node_type: str, flow: str) -> Any:
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess), result.result_details
    return GriptapeNodes.NodeManager().get_node_by_name(result.node_name)


def _connect(
    source: BaseNode, source_param: str, target: BaseNode, target_param: str, *, initial_setup: bool = False
) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source.name,
            source_parameter_name=source_param,
            target_node_name=target.name,
            target_parameter_name=target_param,
            initial_setup=initial_setup,
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess), result.result_details


def _outgoing_targets(node: BaseNode) -> list[tuple[str, str]]:
    result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    assert isinstance(result, ListConnectionsForNodeResultSuccess)
    return [(c.target_node_name, c.target_parameter_name) for c in result.outgoing_connections]


def _incoming_targets(node: BaseNode) -> list[str]:
    result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=node.name))
    assert isinstance(result, ListConnectionsForNodeResultSuccess)
    return [c.target_parameter_name for c in result.incoming_connections]


def _has_variable(name: str, flow: str) -> bool:
    result = GriptapeNodes.handle_request(
        HasVariableRequest(name=name, lookup_scope=VariableScope.CURRENT_FLOW_ONLY, starting_flow=flow)
    )
    assert isinstance(result, HasVariableResultSuccess)
    return result.exists


class TestSetVariable:
    def test_loading_does_not_look_up_or_create_the_variable(
        self, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        created = GriptapeNodes.handle_request(
            CreateVariableRequest(name="total", type="str", is_global=False, value="", owning_flow=canvas)
        )
        assert isinstance(created, CreateVariableResultSuccess)
        node = cast(SetVariable, _create("SetVariable", canvas))
        node._refresh_variable_names(button=None, button_details=None)  # type: ignore[arg-type]
        node.set_parameter_value("variable_name", "total")

        copy, recorder = load_copy(node.name, scratch, monkeypatch)

        assert set(recorder.forbidden()) <= VARIABLE_DROPDOWN_LISTING
        assert copy.get_parameter_value("variable_name") == "total"
        assert not _has_variable("total", scratch)

    def test_editing_the_name_still_creates_the_variable(self, canvas: str) -> None:
        node = cast(SetVariable, _create("SetVariable", canvas))
        # A name that is in the dropdown but not yet a variable visible from this node's flow.
        node._update_option_choices(param="variable_name", choices=["fresh"], default="fresh")
        node.set_parameter_value("variable_name", "fresh")

        assert _has_variable("fresh", canvas)


class TestCreateVariable:
    def test_loading_keeps_the_type_change_without_listing_connections(
        self, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = cast(CreateVariable, _create("CreateVariable", canvas))
        node.set_parameter_value("variable_type", "str")

        copy, recorder = load_copy(node.name, scratch, monkeypatch)

        assert recorder.forbidden() == []
        value_param = copy.get_parameter_by_name("value")
        assert value_param is not None
        assert value_param.output_type == "str"

    def test_editing_the_type_still_deletes_incompatible_connections(self, canvas: str) -> None:
        node = cast(CreateVariable, _create("CreateVariable", canvas))
        target = cast(CreateVariable, _create("CreateVariable", canvas))
        _connect(node, "value", target, "auto_name")
        assert (target.name, "auto_name") in _outgoing_targets(node)

        node.set_parameter_value("variable_type", "str")

        assert (target.name, "auto_name") not in _outgoing_targets(node)


class TestDescribeImage:
    def test_loading_a_legacy_image_value_moves_it_to_a_row_without_graph_requests(
        self, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = cast(DescribeImage, _create("DescribeImage", canvas))
        # As a workflow saved before 'images' existed holds it.
        node.parameter_values["image"] = IMAGE

        copy, recorder = load_copy(node.name, scratch, monkeypatch)

        assert recorder.forbidden() == []
        images = copy.get_parameter_by_name("images")
        assert isinstance(images, ParameterList)
        rows = images.get_child_parameters()
        assert len(rows) == 1
        assert copy.get_parameter_value(LEGACY_IMAGE_ROW) == rows[0].name
        assert "image" not in copy.parameter_values
        assert len(cast(DescribeImage, copy)._collect_image_contents()) == 1

    def test_a_value_arriving_on_image_moves_to_images(self, canvas: str) -> None:
        node = cast(DescribeImage, _create("DescribeImage", canvas))

        node.set_parameter_value("image", IMAGE)

        images = node.get_parameter_by_name("images")
        assert isinstance(images, ParameterList)
        assert len(images.get_child_parameters()) == 1
        assert "image" not in node.parameter_values
        assert len(node._collect_image_contents()) == 1

    def test_a_value_arriving_after_a_load_replaces_the_row_the_load_made(
        self, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = cast(DescribeImage, _create("DescribeImage", canvas))
        node.parameter_values["image"] = IMAGE
        copy, _ = load_copy(node.name, scratch, monkeypatch)

        copy.set_parameter_value("image", IMAGE)

        images = copy.get_parameter_by_name("images")
        assert isinstance(images, ParameterList)
        assert len(images.get_child_parameters()) == 1
        assert len(cast(DescribeImage, copy)._collect_image_contents()) == 1

    @pytest.mark.parametrize("connect_first", [True, False], ids=["connect_then_set", "set_then_connect"])
    def test_a_connection_and_value_on_image_share_one_row(self, canvas: str, *, connect_first: bool) -> None:
        """A saved workflow connects before it sets values; a paste sets values before it connects."""
        source = _create("LoadImage", canvas)
        node = cast(DescribeImage, _create("DescribeImage", canvas))

        if connect_first:
            _connect(source, "image", node, "image", initial_setup=True)
            node.set_parameter_value("image", IMAGE, initial_setup=True)
        else:
            node.set_parameter_value("image", IMAGE, initial_setup=True)
            _connect(source, "image", node, "image", initial_setup=True)

        images = node.get_parameter_by_name("images")
        assert isinstance(images, ParameterList)
        rows = images.get_child_parameters()
        assert len(rows) == 1
        assert node.get_parameter_value(LEGACY_IMAGE_ROW) == rows[0].name
        assert _incoming_targets(node) == [rows[0].name]
        assert "image" not in node.parameter_values
        assert len(node._collect_image_contents()) == 1

    def test_a_connection_from_outside_its_group_moves_to_a_row_through_the_proxy(self, canvas: str) -> None:
        source = _create("LoadImage", canvas)
        node = cast(DescribeImage, _create("DescribeImage", canvas))
        group = _create("StandardSubflowNodeGroup", canvas)
        added = GriptapeNodes.handle_request(
            AddNodesToNodeGroupRequest(node_names=[node.name], node_group_name=group.name)
        )
        assert isinstance(added, AddNodesToNodeGroupResultSuccess), added.result_details

        _connect(source, "image", node, "image", initial_setup=True)

        assert _incoming_targets(node) == [node.get_parameter_value(LEGACY_IMAGE_ROW)]
        assert [target for _, target in _outgoing_targets(source)] == _incoming_targets(group)


class TestRowOutputs:
    @pytest.mark.parametrize("node_type", ["JevPickOne", "JevRate"])
    def test_loading_keeps_row_outputs_without_graph_requests(
        self, node_type: str, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        node = cast(RowOutputsMixin, _create(node_type, canvas))
        rows = node.get_parameter_by_name(node.ROWS_PARAM)
        assert isinstance(rows, ParameterList)
        for text in ("Yes: it fits", "No"):
            node.set_parameter_value(rows.add_child_parameter().name, text)
        outputs = [p.name for p in node._row_output_params()]
        assert outputs

        copy, recorder = load_copy(node.name, scratch, monkeypatch)

        assert recorder.forbidden() == []
        assert [p.name for p in cast(RowOutputsMixin, copy)._row_output_params()] == outputs


class TestDisplayList:
    def test_loaded_copy_keeps_the_declared_element_type(
        self, canvas: str, scratch: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = cast(CreateVariable, _create("CreateVariable", canvas))
        source.set_parameter_value("variable_type", "list[Thing]")
        node = cast(DisplayList, _create("DisplayList", canvas))
        _connect(source, "value", node, "items")
        node.set_parameter_value("items", ["a", 1])
        node.process()

        copy, recorder = load_copy(node.name, scratch, monkeypatch)
        assert recorder.forbidden() == []
        copy.process()

        def children(display: BaseNode) -> list[tuple[str, Any]]:
            display_list = display.get_parameter_by_name("display_list")
            assert isinstance(display_list, ParameterList)
            return [
                (child.type, display.parameter_output_values.get(child.name, display.get_parameter_value(child.name)))
                for child in display_list.get_child_parameters()
            ]

        assert children(node) == [("Thing", "a"), ("Thing", 1)]
        assert children(copy) == children(node)
