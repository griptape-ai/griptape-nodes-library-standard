"""Tests for how GoogleImageGeneration's dependent options follow the selected model."""

from __future__ import annotations

from typing import cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.image.google_image_generation import GoogleImageGeneration

LIBRARY_NAME = "Griptape Nodes Library"
NANO_BANANA_2_1 = "gemini-nano-banana-2.1"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> GoogleImageGeneration:  # noqa: ARG001
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast(
        "GoogleImageGeneration",
        library.create_node(node_type="GoogleImageGeneration", name="Google Image Generation"),
    )


def _choices(node: GoogleImageGeneration, name: str) -> list[str]:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    return parameter.find_elements_by_type(Options)[0].choices


def _is_hidden(node: GoogleImageGeneration, name: str) -> bool:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    return bool(parameter.ui_options.get("hide", False))


def test_nano_banana_2_1_is_offered(node: GoogleImageGeneration) -> None:
    assert NANO_BANANA_2_1 in _choices(node, "model")


def test_nano_banana_2_1_offers_no_512_size(node: GoogleImageGeneration) -> None:
    node.set_parameter_value("model", NANO_BANANA_2_1)

    assert _choices(node, "image_size") == ["1K", "2K", "4K"]


def test_switching_to_nano_banana_2_1_moves_a_512_size_to_1k(node: GoogleImageGeneration) -> None:
    node.set_parameter_value("model", "gemini-3.1-flash-image")
    node.set_parameter_value("image_size", "512")

    node.set_parameter_value("model", NANO_BANANA_2_1)

    assert node.get_parameter_value("image_size") == "1K"


def test_nano_banana_2_1_keeps_the_extreme_aspect_ratios(node: GoogleImageGeneration) -> None:
    node.set_parameter_value("model", NANO_BANANA_2_1)
    node.set_parameter_value("aspect_ratio", "8:1")

    assert {"1:4", "4:1", "1:8", "8:1"} <= set(_choices(node, "aspect_ratio"))
    assert node.get_parameter_value("aspect_ratio") == "8:1"


def test_nano_banana_2_1_shows_image_search(node: GoogleImageGeneration) -> None:
    assert _is_hidden(node, "use_google_image_search")

    node.set_parameter_value("model", NANO_BANANA_2_1)
    assert not _is_hidden(node, "use_google_image_search")

    node.set_parameter_value("model", "gemini-3-pro-image")
    assert _is_hidden(node, "use_google_image_search")
