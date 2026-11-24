"""Tests for the RandomNumber node."""

import asyncio

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.number.random_number import NumberMode, RandomNumber

RUNS = 50


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> RandomNumber:  # noqa: ARG001
    return RandomNumber(name="test_random_number")


def _results(node: RandomNumber, mode: NumberMode, minimum: float, maximum: float) -> list[float]:
    node.set_parameter_value("mode", mode)
    node.set_parameter_value("minimum", minimum)
    node.set_parameter_value("maximum", maximum)
    results = []
    for seed in range(RUNS):
        node.set_parameter_value("seed", seed)
        asyncio.run(node.aprocess())
        results.append(node.parameter_output_values["result"])
    return results


class TestRandomNumber:
    @pytest.mark.parametrize("mode", list(NumberMode))
    @pytest.mark.parametrize(("minimum", "maximum"), [(-10, 0), (0, 0), (-5, -1)])
    def test_result_stays_within_range(
        self, node: RandomNumber, mode: NumberMode, minimum: float, maximum: float
    ) -> None:
        results = _results(node, mode, minimum, maximum)

        assert all(minimum <= result <= maximum for result in results)

    @pytest.mark.parametrize("mode", list(NumberMode))
    def test_missing_bounds_use_defaults(self, node: RandomNumber, mode: NumberMode) -> None:
        node.set_parameter_value("mode", mode)
        node.parameter_values["minimum"] = None
        node.parameter_values["maximum"] = None
        asyncio.run(node.aprocess())

        assert 0 <= node.parameter_output_values["result"] <= 100  # noqa: PLR2004
