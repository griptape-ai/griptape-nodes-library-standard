"""Switch picks the flow exit whose case matches the value, and says clearly why when it can't."""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError, ParameterList

from griptape_nodes_library.execution.switch import Switch, matches, value_text


@pytest.mark.parametrize(
    ("value", "text"),
    [(True, "true"), (False, "false"), (3, "3"), (3.0, "3"), (2.5, "2.5"), ("Fx", "Fx"), (None, "")],
)
def test_value_text(value: Any, text: str) -> None:
    assert value_text(value) == text


@pytest.mark.parametrize(
    ("value", "pattern", "case_sensitive", "expected"),
    [
        ("fx", "fx", True, True),
        ("Fx", "fx", True, False),
        ("Fx", "fx", False, True),
        ("cfx", "fx", False, False),
        ("fx", "*fx", True, True),
        ("cfx", "*fx", True, True),
        ("fxx", "*fx", True, False),
        ("CFX", "*fx", False, True),
        ("CFX", "*fx", True, False),
        ("fx_hero", "fx*", True, True),
        ("a_fx_b", "*fx*", True, True),
        ("Level 2", "Level ?", True, True),
        ("Level 10", "Level ?", True, False),
        ("[fx]", "[fx]", True, True),
        ("f", "[fx]", True, False),
        ("[fx]", "*[fx]", True, True),
        (True, "true", False, True),
        (False, "true", False, False),
        (3, "3.0", True, True),
        (3.0, "3", True, True),
        (3, "three", True, False),
        (3, "3*", True, True),
        ("3", "3.0", True, False),
    ],
)
def test_matches(value: Any, pattern: str, case_sensitive: bool, expected: bool) -> None:  # noqa: FBT001
    assert matches(value, pattern, case_sensitive=case_sensitive) is expected


@pytest.mark.parametrize(
    ("value", "pattern", "expected"),
    [("Why?", "Why?", True), ("Whys", "Why?", False), ("cfx", "*fx", False), ("*fx", "*fx", True), (3, "3*", False)],
)
def test_matches_with_patterns_off_reads_wildcard_characters_literally(
    value: Any, pattern: str, expected: bool
) -> None:  # noqa: FBT001
    assert matches(value, pattern, case_sensitive=True, patterns=False) is expected


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
    assert _exit_label(_switch("cfx", ["cfx", "*fx"])) == "cfx"
    assert _exit_label(_switch("cfx", ["*fx", "cfx"])) == "*fx"


def test_exact_and_pattern_rows_mix_in_one_switch() -> None:
    cases = ["layout", "*fx", "light*"]
    assert _exit_label(_switch("Layout", cases)) == "layout"
    assert _exit_label(_switch("cfx", cases)) == "*fx"
    assert _exit_label(_switch("lighting", cases)) == "light*"
    assert _exit_label(_switch("animation", cases)) == "Default"


def test_the_pattern_matching_option_turns_wildcards_off() -> None:
    assert _exit_label(_switch("cfx", ["*fx"])) == "*fx"
    assert _exit_label(_switch("cfx", ["*fx"], pattern_matching=False)) == "Default"
    assert _exit_label(_switch("*fx", ["*fx"], pattern_matching=False)) == "*fx"


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
    with pytest.raises(NodeError, match="'Switch on' is empty"):
        node.process()


def test_a_hidden_path_has_no_effect_and_comes_back_with_its_value() -> None:
    node = _switch("cartoon", ["cartoon"], path="department")
    node._show_options_for("str")
    assert _exit_label(node) == "cartoon"

    node._show_options_for("json")
    with pytest.raises(NodeError, match="nothing at 'department'"):
        node.process()


def test_hidden_case_sensitivity_has_no_effect_on_a_bool() -> None:
    node = _switch(True, ["True"], case_sensitive=True)
    node._show_options_for("bool")
    assert _exit_label(node) == "True"

    node._show_options_for("str")
    assert _exit_label(node) == "Default"


def test_hidden_pattern_matching_is_back_to_its_default() -> None:
    node = _switch("cfx", ["*fx"], pattern_matching=False)
    assert _exit_label(node) == "Default"

    node._show_options_for("int")
    assert _exit_label(node) == "*fx"
