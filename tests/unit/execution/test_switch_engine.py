"""Switch adapts its options to the type wired into 'Switch on', through the real engine."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError, ParameterList
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    DeleteConnectionRequest,
    DeleteConnectionResultSuccess,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.execution.switch import Switch

CANVAS = "switch_canvas"


@pytest.fixture
def canvas(griptape_nodes: Any) -> Generator[str, None, None]:  # noqa: ARG001
    context_manager = GriptapeNodes.ContextManager()
    context_manager.push_workflow(workflow_name="switch_workflow")
    try:
        result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name=CANVAS))
        assert isinstance(result, CreateFlowResultSuccess)
        yield CANVAS
    finally:
        context_manager.pop_workflow()


def _create(node_type: str, flow: str) -> Any:
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type=node_type, override_parent_flow_name=flow))
    assert isinstance(result, CreateNodeResultSuccess), result.result_details
    return GriptapeNodes.NodeManager().get_node_by_name(result.node_name)


def _hidden(node: Switch) -> set[str]:
    return {p.name for p in (node.path_param, node.case_sensitive_param, node.pattern_matching_param) if p.hide}


def test_a_bool_wired_in_hides_every_option_and_removing_it_brings_them_back(canvas: str) -> None:
    switch = _create("Switch", canvas)
    source = _create("ToBool", canvas)
    assert _hidden(switch) == set()

    connected = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source.name,
            source_parameter_name="output",
            target_node_name=switch.name,
            target_parameter_name="value",
        )
    )
    assert isinstance(connected, CreateConnectionResultSuccess), connected.result_details
    assert _hidden(switch) == {"path", "case_sensitive", "pattern_matching"}

    removed = GriptapeNodes.handle_request(
        DeleteConnectionRequest(
            source_node_name=source.name,
            source_parameter_name="output",
            target_node_name=switch.name,
            target_parameter_name="value",
        )
    )
    assert isinstance(removed, DeleteConnectionResultSuccess)
    assert _hidden(switch) == set()


@pytest.mark.parametrize(
    ("value_type", "hidden"),
    [
        ("str", set()),
        ("int", {"path", "case_sensitive", "pattern_matching"}),
        ("float", {"path", "case_sensitive", "pattern_matching"}),
        ("json", set()),
        ("yaml", set()),
        ("dict", set()),
        ("any", set()),
    ],
)
def test_each_type_shows_only_the_options_that_mean_something(canvas: str, value_type: str, hidden: set[str]) -> None:
    switch = _create("Switch", canvas)
    switch._show_options_for(value_type)
    assert _hidden(switch) == hidden


def _connect(source: Any, source_param: str, target: Any) -> None:
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source.name,
            source_parameter_name=source_param,
            target_node_name=target.name,
            target_parameter_name="value",
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess), result.result_details


def _exit_labels(node: Switch) -> list[str]:
    return [p.display_name or p.name for p in node.parameters if p.name.startswith("switch_case_")]


def test_wiring_in_a_bool_adds_true_and_false_cases(canvas: str) -> None:
    switch = _create("Switch", canvas)
    _connect(_create("ToBool", canvas), "output", switch)

    assert _exit_labels(switch) == ["true", "false"]
    assert switch._rows()[0][1] == "true"


def test_wiring_in_a_bool_keeps_cases_that_are_already_there(canvas: str) -> None:
    switch = _create("Switch", canvas)
    rows = switch.get_parameter_by_name("cases")
    switch.set_parameter_value(rows.add_child_parameter().name, "true")
    _connect(_create("ToBool", canvas), "output", switch)

    assert _exit_labels(switch) == ["true"]


def test_wiring_in_text_adds_no_cases(canvas: str) -> None:
    switch = _create("Switch", canvas)
    _connect(_create("ToText", canvas), "output", switch)

    assert _exit_labels(switch) == []


def test_a_bool_switch_routes_to_the_matching_exit(canvas: str) -> None:
    switch = _create("Switch", canvas)
    _connect(_create("ToBool", canvas), "output", switch)

    switch.set_parameter_value("value", False)
    switch.process()

    exit_ = switch.get_next_control_output()
    assert exit_ is not None
    assert exit_.display_name == "false"


def _case_rows(node: Switch) -> list[Any]:
    cases = node.get_parameter_by_name("cases")
    assert isinstance(cases, ParameterList)
    return cases.get_child_parameters()


def _data_rows(node: Switch) -> list[Any]:
    values = node.get_parameter_by_name("values")
    assert isinstance(values, ParameterList)
    return values.get_child_parameters()


def _add_case(node: Switch, text: str) -> Any:
    cases = node.get_parameter_by_name("cases")
    assert isinstance(cases, ParameterList)
    row = cases.add_child_parameter()
    if text:
        node.set_parameter_value(row.name, text)
    return row


def _data_labels(node: Switch) -> list[str]:
    return [row.display_name for row in _data_rows(node)]


def test_each_case_gets_a_data_row_named_after_it(canvas: str) -> None:
    switch = _create("Switch", canvas)
    lighting = _add_case(switch, "Lighting")
    _add_case(switch, "Layout")

    assert _data_labels(switch) == ["Lighting", "Layout"]
    assert _data_rows(switch)[0].name == switch._data_row_name(lighting.name)


def test_a_case_with_no_text_gets_its_data_row_when_it_gets_text(canvas: str) -> None:
    switch = _create("Switch", canvas)
    blank = _add_case(switch, "")
    assert _data_rows(switch) == []

    switch.set_parameter_value(blank.name, "Lighting")

    assert _data_labels(switch) == ["Lighting"]


def test_renaming_a_case_renames_its_data_row(canvas: str) -> None:
    switch = _create("Switch", canvas)
    row = _add_case(switch, "Lighting")
    switch.set_parameter_value(row.name, "Lights")

    assert _data_labels(switch) == ["Lights"]


def test_reordering_the_cases_reorders_the_data_rows_with_them(canvas: str) -> None:
    switch = _create("Switch", canvas)
    for text in ("Lighting", "Layout", "Animation"):
        _add_case(switch, text)
    typed = {row.display_name: row.name for row in _data_rows(switch)}
    switch.set_parameter_value(typed["Lighting"], "Sam")

    cases = switch.get_parameter_by_name("cases")
    cases.move_child(0, 2)

    assert [row.display_name for row in _data_rows(switch)] == ["Layout", "Animation", "Lighting"]
    assert switch.get_parameter_value(typed["Lighting"]) == "Sam"


def test_deleting_a_case_removes_its_data_row_and_the_wire_into_it(canvas: str) -> None:
    switch = _create("Switch", canvas)
    doomed = _add_case(switch, "Lighting")
    _add_case(switch, "Layout")
    data_name = switch._data_row_name(doomed.name)
    source = _create("ToText", canvas)
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name=source.name,
            source_parameter_name="output",
            target_node_name=switch.name,
            target_parameter_name=data_name,
        )
    )
    assert isinstance(result, CreateConnectionResultSuccess), result.result_details

    switch.get_parameter_by_name("cases").remove_child(doomed)

    assert _data_labels(switch) == ["Layout"]
    connections = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=switch.name))
    assert isinstance(connections, ListConnectionsForNodeResultSuccess)
    assert [c.target_parameter_name for c in connections.incoming_connections] == []


def test_an_extra_data_row_added_by_hand_is_removed(canvas: str) -> None:
    switch = _create("Switch", canvas)
    _add_case(switch, "Lighting")
    values = switch.get_parameter_by_name("values")
    values.add_child_parameter()
    switch.set_parameter_value("values", switch.get_parameter_value("values"))

    assert _data_labels(switch) == ["Lighting"]


def test_result_is_the_data_of_the_matching_case(canvas: str) -> None:
    switch = _create("Switch", canvas)
    for text in ("Lighting", "Layout"):
        _add_case(switch, text)
    rows = {row.display_name: row.name for row in _data_rows(switch)}
    switch.set_parameter_value(rows["Lighting"], "Sam")
    switch.set_parameter_value(rows["Layout"], "Priya")
    switch.set_parameter_value("default_result", "Triage")

    switch.set_parameter_value("value", "layout")
    switch.process()
    assert switch.parameter_output_values["result"] == "Priya"

    switch.set_parameter_value("value", "animation")
    switch.process()
    assert switch.parameter_output_values["result"] == "Triage"


def test_a_failed_run_leaves_no_stale_result(canvas: str) -> None:
    switch = _create("Switch", canvas)
    _add_case(switch, "Lighting")
    switch.set_parameter_value(_data_rows(switch)[0].name, "Sam")
    switch.set_parameter_value("value", "lighting")
    switch.process()
    assert switch.parameter_output_values["result"] == "Sam"

    switch.set_parameter_value("value", None)
    with pytest.raises(NodeError, match="'Switch on' is empty"):
        switch.process()
    assert "result" not in switch.parameter_output_values
