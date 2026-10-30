from __future__ import annotations

from typing import Any, cast

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import ParameterList
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.image.grok_image_edit import GrokImageEdit

LIBRARY_NAME = "Griptape Nodes Library"

# 1x1 transparent PNG.
PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def _set_additional_images(node: GrokImageEdit, images: list[Any]) -> None:
    parameter_list = next(
        parameter
        for parameter in node.parameters
        if isinstance(parameter, ParameterList) and parameter.name == "additional_images"
    )
    parameter_list.clear_list()
    for image in images:
        child = parameter_list.add_child_parameter()
        node.set_parameter_value(child.name, image)


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> GrokImageEdit:  # noqa: ARG002
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("GrokImageEdit", library.create_node(node_type="GrokImageEdit", name="Grok Image Edit"))


@pytest.mark.asyncio
async def test_build_payload_with_single_image_sends_image_field(node: GrokImageEdit) -> None:
    node.set_parameter_value("prompt", "Make it blue")
    node.set_parameter_value("image", PNG_DATA_URI)

    payload = await node._build_payload()

    assert payload["image"] == {"url": PNG_DATA_URI}
    assert "images" not in payload


@pytest.mark.asyncio
async def test_build_payload_with_additional_images_sends_images_list(node: GrokImageEdit) -> None:
    node.set_parameter_value("prompt", "Combine these")
    node.set_parameter_value("image", PNG_DATA_URI)
    _set_additional_images(node, [PNG_DATA_URI, PNG_DATA_URI])

    payload = await node._build_payload()

    assert payload["images"] == [{"url": PNG_DATA_URI}] * 3
    assert "image" not in payload


@pytest.mark.asyncio
async def test_build_payload_encodes_image_artifact_as_data_uri(node: GrokImageEdit) -> None:
    node.set_parameter_value("prompt", "Combine these")
    node.set_parameter_value("image", PNG_DATA_URI)
    _set_additional_images(node, [ImageArtifact(value=b"png-bytes", format="png", width=1, height=1)])

    payload = await node._build_payload()

    assert payload["images"][1] == {"url": "data:image/png;base64,cG5nLWJ5dGVz"}


@pytest.mark.asyncio
async def test_build_payload_raises_when_a_source_image_cannot_be_read(node: GrokImageEdit) -> None:
    node.set_parameter_value("prompt", "Combine these")
    node.set_parameter_value("image", PNG_DATA_URI)
    _set_additional_images(node, ["/no/such/file.png"])

    with pytest.raises(ValueError, match="Source image"):
        await node._build_payload()


def test_validate_rejects_more_than_five_source_images(node: GrokImageEdit, monkeypatch: pytest.MonkeyPatch) -> None:
    # Rows are capped at 4, so only a whole list arriving over a connection can exceed it.
    monkeypatch.setattr(node, "get_parameter_list_value", lambda _name: [PNG_DATA_URI] * 5)
    node.set_parameter_value("prompt", "Combine these")
    node.set_parameter_value("image", PNG_DATA_URI)

    exceptions = node.validate_before_node_run() or []

    assert any("Additional Images" in str(e) for e in exceptions)
