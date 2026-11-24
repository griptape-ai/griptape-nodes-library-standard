"""Compare Numbers routes on the chosen comparison, starting from a default that is one of its choices."""

from __future__ import annotations

import pytest
from griptape_nodes.traits.options import Options

from griptape_nodes_library.execution.compare_numbers import CompareNumbers


def test_default_evaluate_is_a_choice() -> None:
    node = CompareNumbers(name="Compare")
    evaluate = node.get_parameter_by_name("evaluate")
    assert evaluate is not None
    assert evaluate.default_value in evaluate.find_elements_by_type(Options)[0].choices


@pytest.mark.parametrize(("a", "b", "expected"), [(3, 3, True), (3, 4, False)])
def test_default_compares_for_equality(a: int, b: int, expected: bool) -> None:  # noqa: FBT001
    node = CompareNumbers(name="Compare")
    node.set_parameter_value("A", a)
    node.set_parameter_value("B", b)
    node.process()
    assert node.parameter_output_values["result"] is expected
    assert node.get_next_control_output() is node.get_parameter_by_name("True" if expected else "False")
