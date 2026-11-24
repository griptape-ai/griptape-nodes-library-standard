"""Tests for the RandomText node's seeded selection."""

import random

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.text.random_text import RandomText


def _select(input_text: str, selection_type: str, seed: int) -> str:
    node = RandomText(name="test_random_text")
    node.set_parameter_value("input_text", input_text)
    node.set_parameter_value("selection_type", selection_type)
    node.set_parameter_value("randomize_seed", False)  # noqa: FBT003
    node.set_parameter_value("seed", seed)
    return node._get_random_selection()


class TestRandomTextSeed:
    @pytest.fixture(autouse=True)
    def _engine(self, griptape_nodes: GriptapeNodes) -> None:  # noqa: ARG002
        pass

    @pytest.mark.parametrize("selection_type", ["character", "word"])
    def test_empty_input_repeats_for_same_seed(self, selection_type: str) -> None:
        results = {_select("", selection_type, seed=42) for _ in range(5)}

        assert len(results) == 1

    @pytest.mark.parametrize("selection_type", ["character", "word"])
    def test_empty_input_varies_with_seed(self, selection_type: str) -> None:
        results = {_select("", selection_type, seed=seed) for seed in range(20)}

        assert len(results) > 1

    def test_input_text_repeats_for_same_seed(self) -> None:
        results = {_select("a b c d e f g h", "word", seed=42) for _ in range(5)}

        assert len(results) == 1

    def test_selection_leaves_global_random_state_alone(self) -> None:
        random.seed(0)
        expected = random.random()  # noqa: S311
        random.seed(0)

        _select("a b c d e f g h", "word", seed=42)

        assert random.random() == expected  # noqa: S311
