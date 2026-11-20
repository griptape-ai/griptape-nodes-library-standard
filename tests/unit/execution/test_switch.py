"""Switch picks the flow exit whose case matches the value, and says clearly why when it can't."""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError, ParameterList

from griptape_nodes_library.execution.switch import MatchMode, Switch, matches, value_text


@pytest.mark.parametrize(
    ("value", "text"),
    [(True, "true"), (False, "false"), (3, "3"), (3.0, "3"), (2.5, "2.5"), ("Fx", "Fx"), (None, "")],
)
def test_value_text(value: Any, text: str) -> None:
    assert value_text(value) == text


@pytest.mark.parametrize(
    ("value", "pattern", "mode", "case_sensitive", "expected"),
    [
        ("fx", "fx", MatchMode.EQUALS, True, True),
        ("Fx", "fx", MatchMode.EQUALS, True, False),
        ("Fx", "fx", MatchMode.EQUALS, False, True),
        ("cfx", "fx", MatchMode.EQUALS, False, False),
        ("fx", "*fx", MatchMode.WILDCARD, True, True),
        ("cfx", "*fx", MatchMode.WILDCARD, True, True),
        ("fxx", "*fx", MatchMode.WILDCARD, True, False),
        ("CFX", "*fx", MatchMode.WILDCARD, False, True),
        ("CFX", "*fx", MatchMode.WILDCARD, True, False),
        ("Level 2", "Level ?", MatchMode.WILDCARD, True, True),
        (True, "true", MatchMode.EQUALS, False, True),
        (False, "true", MatchMode.EQUALS, False, False),
        (3, "3.0", MatchMode.EQUALS, True, True),
        (3.0, "3", MatchMode.EQUALS, True, True),
        (3, "three", MatchMode.EQUALS, True, False),
        (3, "3*", MatchMode.WILDCARD, True, True),
        ("3", "3.0", MatchMode.EQUALS, True, False),
    ],
)
def test_matches(value: Any, pattern: str, mode: MatchMode, case_sensitive: bool, expected: bool) -> None:  # noqa: FBT001
    assert matches(value, pattern, mode, case_sensitive=case_sensitive) is expected


def test_matches_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown match mode"):
        matches("a", "a", "Regex", case_sensitive=True)  # type: ignore[arg-type]


def _switch(value: Any, cases: list[str], **settings: Any) -> Switch:
    node = Switch(name="switch")
    rows = node.get_parameter_by_name("cases")
    assert isinstance(rows, ParameterList)
    for text in cases:
        node.set_parameter_value(rows.add_child_parameter().name, text)
    node.set_parameter_value("value", value)
    for name, setting in settings.items():
        node.set_parameter_value(name, setting)
    return node


def _exit_label(node: Switch) -> str:
    node.process()
    exit_ = node.get_next_control_output()
    assert exit_ is not None
    return exit_.display_name or exit_.name


def test_each_case_row_gets_a_flow_exit_and_default_stays_last() -> None:
    node = _switch("x", ["Lighting", "", "Layout"])
    exits = [p.display_name for p in node.parameters if p.name == "default" or p.name.startswith("switch_case_")]
    assert exits == ["Lighting", "Layout", "Default"]


def test_the_first_matching_row_wins() -> None:
    assert _exit_label(_switch("cfx", ["cfx", "*fx"], match_mode="Wildcard")) == "cfx"
    assert _exit_label(_switch("cfx", ["*fx", "cfx"], match_mode="Wildcard")) == "*fx"


def test_no_match_takes_default() -> None:
    assert _exit_label(_switch("animation", ["Lighting", "Layout"])) == "Default"


def test_case_is_ignored_unless_asked() -> None:
    assert _exit_label(_switch("LIGHTING", ["Lighting"])) == "Lighting"
    assert _exit_label(_switch("LIGHTING", ["Lighting"], case_sensitive=True)) == "Default"


def test_a_bool_value_matches_true_and_false_rows() -> None:
    assert _exit_label(_switch(True, ["false", "true"])) == "true"


def test_a_path_picks_the_field_out_of_a_dictionary() -> None:
    data = {"is_animal": True, "breed": "dog", "size": "Level 2"}
    assert _exit_label(_switch(data, ["false", "true"], path="is_animal")) == "true"
    assert _exit_label(_switch(data, ["cat", "dog"], path="breed")) == "dog"
    nested = {"details": {"size": {"label": "Level 2"}}}
    assert _exit_label(_switch(nested, ["Level 1", "Level 2"], path="details.size.label")) == "Level 2"


@pytest.mark.parametrize("text", ['{"breed": "dog"}', "breed: dog"])
def test_a_path_reads_json_and_yaml_text(text: str) -> None:
    assert _exit_label(_switch(text, ["cat", "dog"], path="breed")) == "dog"


def test_a_missing_path_fails_and_names_the_fields_there() -> None:
    node = _switch({"breed": "dog", "size": "large"}, ["dog"], path="species")
    with pytest.raises(NodeError, match=r"nothing at 'species'.*top-level fields are: breed, size"):
        node.process()


def test_an_invalid_path_fails() -> None:
    node = _switch({"breed": "dog"}, ["dog"], path="[")
    with pytest.raises(NodeError, match="isn't a valid JMESPath expression"):
        node.process()


def test_text_that_isnt_json_or_yaml_fails_when_a_path_is_set() -> None:
    node = _switch("a: [", ["dog"], path="breed")
    with pytest.raises(NodeError, match="isn't valid JSON or YAML"):
        node.process()


def test_a_dictionary_without_a_path_fails() -> None:
    node = _switch({"breed": "dog"}, ["dog"])
    with pytest.raises(NodeError, match="Set 'Path' to one field"):
        node.process()


def test_an_empty_value_fails() -> None:
    node = _switch(None, ["dog"])
    with pytest.raises(NodeError, match="'Value' is empty"):
        node.process()
