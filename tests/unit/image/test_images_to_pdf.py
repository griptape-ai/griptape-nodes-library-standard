"""Tests for ImagesToPdf converting list items to ImageUrlArtifact."""

from pathlib import Path
from typing import cast

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image

from griptape_nodes_library.image.images_to_pdf import ImagesToPdf

LIBRARY_NAME = "Griptape Nodes Library"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> ImagesToPdf:  # noqa: ARG001
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("ImagesToPdf", library.create_node(node_type="ImagesToPdf", name="test_images_to_pdf"))


@pytest.fixture
def image_path(tmp_path: Path) -> Path:
    path = tmp_path / "image.png"
    Image.new("RGB", (2, 2)).save(path)
    return path


def test_row_set_to_path_is_stored_as_image_url_artifact(node: ImagesToPdf, image_path: Path) -> None:
    row_name = node.images.add_child_parameter().name

    node.set_parameter_value(row_name, str(image_path))

    stored = node.get_parameter_value(row_name)
    assert isinstance(stored, ImageUrlArtifact)


def test_whole_list_of_paths_is_stored_as_image_url_artifacts(node: ImagesToPdf, image_path: Path) -> None:
    node.set_parameter_value("images", [str(image_path), str(image_path)])

    stored = node.get_parameter_list_value("images")

    assert len(stored) == 2
    assert all(isinstance(item, ImageUrlArtifact) for item in stored)
