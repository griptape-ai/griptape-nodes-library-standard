"""Tests for the remove_duplicates and remove_blank options of the create list nodes."""

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.lists.create_image_list import CreateImageList
from griptape_nodes_library.lists.create_int_list import CreateIntList
from griptape_nodes_library.lists.create_list import CreateList
from griptape_nodes_library.lists.create_text_list import CreateTextList


def _run(node: CreateList | CreateImageList | CreateIntList | CreateTextList, items: list, **options: bool) -> list:
    node.set_parameter_value("items", items)
    for name, value in options.items():
        node.set_parameter_value(name, value)
    node.process()
    return node.parameter_output_values["output"]


class TestCreateListOptions:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> CreateList:  # noqa: ARG002
        return CreateList(name="test_create_list")

    def test_remove_duplicates_keeps_first_occurrence_order(self, node: CreateList) -> None:
        assert _run(node, ["z", "a", "z", "m", "b"], remove_duplicates=True) == ["z", "a", "m", "b"]

    def test_remove_duplicates_handles_unhashable_items(self, node: CreateList) -> None:
        items = [{"k": 1}, [1, 2], {"k": 1}, [1, 2], {"k": 2}]

        assert _run(node, items, remove_duplicates=True) == [{"k": 1}, [1, 2], {"k": 2}]

    def test_remove_blank_skips_non_text_items(self, node: CreateList) -> None:
        assert _run(node, [1, None, " ", "a", 0, {"k": 1}], remove_blank=True) == [1, "a", 0, {"k": 1}]

    def test_remove_blank_on_text(self, node: CreateList) -> None:
        assert _run(node, ["a", " ", None, "b"], remove_blank=True) == ["a", "b"]


class TestCreateImageListOptions:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> CreateImageList:  # noqa: ARG002
        return CreateImageList(name="test_create_image_list")

    def test_remove_duplicates_matches_images_by_url(self, node: CreateImageList) -> None:
        first = ImageUrlArtifact("https://example.com/a.png")
        repeat = ImageUrlArtifact("https://example.com/a.png")
        other = ImageUrlArtifact("https://example.com/b.png")

        assert _run(node, [first, other, repeat], remove_duplicates=True) == [first, other]

    def test_remove_blank_keeps_images(self, node: CreateImageList) -> None:
        images = [ImageUrlArtifact("https://example.com/a.png"), ImageUrlArtifact("https://example.com/b.png")]

        assert _run(node, [images[0], None, images[1]], remove_blank=True) == images


class TestCreateIntListOptions:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> CreateIntList:  # noqa: ARG002
        return CreateIntList(name="test_create_int_list")

    def test_both_options_on_ints(self, node: CreateIntList) -> None:
        assert _run(node, [3, 1, 3, 2], remove_duplicates=True, remove_blank=True) == [3, 1, 2]
