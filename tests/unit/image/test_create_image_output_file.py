"""Tests that ``GenerateImage._create_image`` only saves output for a successful run.

A failed generation (e.g. a 403 from Griptape Cloud) returns an ``ErrorArtifact``
whose bytes are the error text. Saving it would write a corrupt image and use up
a version number for the output file.
"""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
import requests
from griptape.artifacts import BlobArtifact, ErrorArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.node_library.library_registry import LibraryRegistry

from griptape_nodes_library.image.create_image import GenerateImage

LIBRARY_NAME = "Griptape Nodes Library"


class _FakeAgent:
    def __init__(self, output: Any) -> None:
        self._output = output
        self.output: Any = None

    def run(self, _prompt: Any) -> None:
        self.output = self._output


@pytest.fixture
def node_with_fake_output_file(monkeypatch: pytest.MonkeyPatch) -> tuple[GenerateImage, MagicMock]:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node = cast(GenerateImage, library.create_node(node_type="GenerateImage", name="GenerateImage"))
    output_file = MagicMock()
    output_file.build_file.return_value.write_bytes.return_value.location = "saved/image_v001.png"
    monkeypatch.setattr(node, "_output_file", output_file)
    monkeypatch.setattr(node, "publish_update_to_parameter", MagicMock())
    return node, output_file


def _http_error(status: int, text: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = text.encode()
    return requests.HTTPError(f"{status} Client Error", response=response)


def test_error_output_raises_without_saving_file(
    node_with_fake_output_file: tuple[GenerateImage, MagicMock],
) -> None:
    node, output_file = node_with_fake_output_file
    error = ErrorArtifact("403 Client Error: Forbidden", exception=_http_error(403, "Forbidden"))

    with pytest.raises(NodeError, match="Forbidden") as raised:
        node._create_image(cast(Any, _FakeAgent(error)), "a cat")

    assert raised.value.fields == {"status_code": 403}

    output_file.build_file.assert_not_called()
    cast(MagicMock, node.publish_update_to_parameter).assert_not_called()


def test_successful_output_is_saved_and_published(
    node_with_fake_output_file: tuple[GenerateImage, MagicMock],
) -> None:
    node, output_file = node_with_fake_output_file
    image = BlobArtifact(b"\x89PNG fake image bytes")

    node._create_image(cast(Any, _FakeAgent(image)), "a cat")

    output_file.build_file.assert_called_once()
    output_file.build_file.return_value.write_bytes.assert_called_once_with(image.to_bytes())
    publish = cast(MagicMock, node.publish_update_to_parameter)
    publish.assert_called_once()
    name, artifact = publish.call_args.args
    assert name == "output"
    assert isinstance(artifact, ImageUrlArtifact)
    assert artifact.value == "saved/image_v001.png"
