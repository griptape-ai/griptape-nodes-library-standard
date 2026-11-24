"""Tests for Rodin23DGeneration's Bounding Box parsing and Input Images limit."""

from __future__ import annotations

from typing import cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.three_d.rodin_2_3d_generation import (
    MAX_INPUT_IMAGES,
    Rodin23DGeneration,
    parse_bbox_condition,
)

LIBRARY_NAME = "Griptape Nodes Library"

# 1x1 transparent PNG.
PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> Rodin23DGeneration:  # noqa: ARG001
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("Rodin23DGeneration", library.create_node(node_type="Rodin23DGeneration", name="Rodin"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", None),
        ("   ", None),
        ("100,200,300", [100, 200, 300]),
        (" 1 , 2 , 2048 ", [1, 2, 2048]),
        ("1.0,2.0,3.0", [1, 2, 3]),
    ],
)
def test_parse_bbox_condition_accepts_whole_numbers(text: str, expected: list[int] | None) -> None:
    assert parse_bbox_condition(text) == expected


@pytest.mark.parametrize(
    "text",
    ["1.5,2.5,3.9", "0.5,0.5,0.5", "1,2", "1,2,3,4", "a,b,c", "1;2;3", "0,1,1", "1,1,2049", "-1,1,1", "nan,1,1"],
)
def test_parse_bbox_condition_rejects_malformed_input(text: str) -> None:
    with pytest.raises(ValueError, match="Bounding Box"):
        parse_bbox_condition(text)


def test_validate_rejects_malformed_bbox_condition(node: Rodin23DGeneration) -> None:
    node.set_parameter_value("prompt", "A chair")
    node.set_parameter_value("bbox_condition", "1.5,2.5,3.9")

    exceptions = node.validate_before_node_run() or []

    assert any("Bounding Box" in str(e) for e in exceptions)


def test_validate_rejects_more_than_five_input_images(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: [PNG_DATA_URI] * (MAX_INPUT_IMAGES + 1))

    exceptions = node.validate_before_node_run() or []

    assert any("Input Images" in str(e) for e in exceptions)


def test_validate_accepts_five_input_images_and_a_whole_number_bbox(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: [PNG_DATA_URI] * MAX_INPUT_IMAGES)
    node.set_parameter_value("bbox_condition", "100,200,300")

    exceptions = node.validate_before_node_run() or []

    assert not any("Input Images" in str(e) or "Bounding Box" in str(e) for e in exceptions)


@pytest.mark.asyncio
async def test_build_payload_sends_bbox_condition_as_integers(node: Rodin23DGeneration) -> None:
    node.set_parameter_value("prompt", "A chair")
    node.set_parameter_value("bbox_condition", "100.0,200,300")

    payload = await node._build_payload()

    assert payload["bbox_condition"] == [100, 200, 300]


@pytest.mark.asyncio
async def test_build_payload_raises_on_malformed_bbox_condition(node: Rodin23DGeneration) -> None:
    node.set_parameter_value("prompt", "A chair")
    node.set_parameter_value("bbox_condition", "1,2")

    with pytest.raises(ValueError, match="Bounding Box"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_raises_instead_of_dropping_a_sixth_image(
    node: Rodin23DGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: [PNG_DATA_URI] * (MAX_INPUT_IMAGES + 1))

    with pytest.raises(ValueError, match="Input Images"):
        await node._build_payload()
