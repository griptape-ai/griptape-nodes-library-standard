from __future__ import annotations

from typing import cast

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.image.qwen_image_edit import QwenImageEdit

LIBRARY_NAME = "Griptape Nodes Library"

# 1x1 transparent PNG.
PNG_DATA_URI = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> QwenImageEdit:  # noqa: ARG002
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("QwenImageEdit", library.create_node(node_type="QwenImageEdit", name="Qwen Image Edit"))


@pytest.mark.asyncio
@pytest.mark.parametrize("image_count", [1, 3])
async def test_build_payload_requests_one_output_image_for_any_input_count(
    node: QwenImageEdit, image_count: int
) -> None:
    node.set_parameter_value("editing_instruction", "Combine these")
    node.set_parameter_value("images", [ImageUrlArtifact(PNG_DATA_URI)] * image_count)

    payload = await node._build_payload()

    image_items = [item for item in payload["messages"][0]["content"] if "image" in item]
    assert len(image_items) == image_count
    assert payload["n"] == 1


def test_validation_accepts_three_images(node: QwenImageEdit) -> None:
    node.set_parameter_value("editing_instruction", "Combine these")
    node.set_parameter_value("images", [ImageUrlArtifact(PNG_DATA_URI)] * 3)

    assert node.validate_before_node_run() is None


def test_validation_refuses_four_images(node: QwenImageEdit) -> None:
    node.set_parameter_value("editing_instruction", "Combine these")
    node.set_parameter_value("images", [ImageUrlArtifact(PNG_DATA_URI)] * 4)

    errors = node.validate_before_node_run()

    assert errors is not None
    assert [str(e) for e in errors] == [
        "'Images to Edit' accepts at most 3 images. 4 are connected. Disconnect 1 of them."
    ]
