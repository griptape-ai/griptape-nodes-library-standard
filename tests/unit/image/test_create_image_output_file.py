"""Tests that ``GenerateImage._create_image`` only saves output for a successful run.

A failed generation (e.g. a 403 from Griptape Cloud) must not write a corrupt image
or use up a version number for the output file.
"""

from __future__ import annotations

import sys
from typing import cast
from unittest.mock import MagicMock

import httpx
import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.node_library.library_registry import LibraryRegistry

from griptape_nodes_library.image.create_image import GenerateImage
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider

LIBRARY_NAME = "Griptape Nodes Library"
CONFIG = ImageGenerationConfig(provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1-mini")
IMAGE_BYTES = b"\x89PNG fake image bytes"


@pytest.fixture
def node_with_fake_output_file(monkeypatch: pytest.MonkeyPatch) -> tuple[GenerateImage, MagicMock]:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node = cast(GenerateImage, library.create_node(node_type="GenerateImage", name="GenerateImage"))
    output_file = MagicMock()
    output_file.build_file.return_value.write_bytes.return_value.location = "saved/image_v001.png"
    monkeypatch.setattr(node, "_output_file", output_file)
    monkeypatch.setattr(node, "publish_update_to_parameter", MagicMock())
    return node, output_file


def _generate_with(node: GenerateImage, monkeypatch: pytest.MonkeyPatch, generate: object) -> None:
    # The engine imports node files under its own module names; patch the one this node came from.
    monkeypatch.setattr(sys.modules[type(node).__module__], "generate_image", generate)


def test_error_output_raises_without_saving_file(
    node_with_fake_output_file: tuple[GenerateImage, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    node, output_file = node_with_fake_output_file
    request = httpx.Request("POST", "https://cloud.griptape.ai/api/images/generations")
    error = httpx.HTTPStatusError("403 Forbidden", request=request, response=httpx.Response(403, request=request))

    def refuse(_config: ImageGenerationConfig, _prompt: str) -> bytes:
        raise error

    _generate_with(node, monkeypatch, refuse)

    with pytest.raises(httpx.HTTPStatusError, match="Forbidden"):
        node._create_image(CONFIG, "a cat")

    output_file.build_file.assert_not_called()
    cast(MagicMock, node.publish_update_to_parameter).assert_not_called()


def test_successful_output_is_saved_and_published(
    node_with_fake_output_file: tuple[GenerateImage, MagicMock], monkeypatch: pytest.MonkeyPatch
) -> None:
    node, output_file = node_with_fake_output_file
    _generate_with(node, monkeypatch, lambda _config, _prompt: IMAGE_BYTES)

    node._create_image(CONFIG, "a cat")

    output_file.build_file.assert_called_once()
    output_file.build_file.return_value.write_bytes.assert_called_once_with(IMAGE_BYTES)
    publish = cast(MagicMock, node.publish_update_to_parameter)
    publish.assert_called_once()
    name, artifact = publish.call_args.args
    assert name == "output"
    assert isinstance(artifact, ImageUrlArtifact)
    assert artifact.value == "saved/image_v001.png"
