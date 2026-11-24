"""Tests for the YamlReplace node."""

from __future__ import annotations

import pytest
from ruamel.yaml import YAML

from griptape_nodes_library.yaml.yaml_replace import YamlReplace


def _run(yaml_input: str, path: str, replacement: str) -> str:
    node = YamlReplace("yaml_replace")
    node.set_parameter_value("yaml", yaml_input)
    node.set_parameter_value("path", path)
    node.set_parameter_value("replacement_value", replacement)
    node.process()
    return node.get_parameter_value("output")


@pytest.mark.parametrize(
    ("yaml_input", "path", "expected"),
    [
        ("items: [a, b]\n", "items[0]", "items: [X, b]\n"),
        ("items:\n- title: Book\n", "items[1].title", "items:\n- title: Book\n- title: X\n"),
        ("user:\n  name: A\n", "user.name", "user:\n  name: X\n"),
        ("items:\n", "items[0]", "items:\n- X\n"),
    ],
)
def test_sets_value(yaml_input: str, path: str, expected: str) -> None:
    assert _run(yaml_input, path, "X") == expected


def test_list_index_past_end_pads_and_sets() -> None:
    assert YAML(typ="safe").load(_run("items: [a]\n", "items[3]", "X")) == {"items": ["a", None, None, "X"]}


@pytest.mark.parametrize(("yaml_input", "path"), [("a: 1\n", "a.b"), ("a: 1\n", "a[0]")])
def test_path_into_scalar_raises_on_process(yaml_input: str, path: str) -> None:
    with pytest.raises(ValueError, match="Cannot follow path"):
        _run(yaml_input, path, "X")


def test_path_into_scalar_clears_output_while_editing() -> None:
    node = YamlReplace("yaml_replace")
    node.set_parameter_value("yaml", "a: 1\n")
    node.set_parameter_value("replacement_value", "2")
    node.set_parameter_value("path", "a")
    assert node.get_parameter_value("output") == "a: 2\n"

    node.set_parameter_value("path", "a.b")

    assert not node.get_parameter_value("output")
