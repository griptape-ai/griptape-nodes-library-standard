"""Tests for LoadGLTF's handling of the values its gltf parameter can be set to."""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact

from griptape_nodes_library.three_d.gltf_artifact import GLTFUrlArtifact
from griptape_nodes_library.three_d.load_gltf import LoadGLTF

IMAGE_URL = "https://example.com/snapshot.png"


@pytest.fixture
def node() -> LoadGLTF:
    return LoadGLTF(name="test_load_gltf")


@pytest.mark.parametrize(
    "value",
    [
        None,
        "model.glb",
        [1],
        {},
        {"metadata": None},
        {"metadata": "not a dict"},
        GLTFUrlArtifact("https://example.com/model.glb"),
    ],
)
def test_value_without_snapshot_leaves_image_unset(node: LoadGLTF, value: Any) -> None:
    node.set_parameter_value("gltf", value)

    assert node.get_parameter_value("image") is None


def test_dict_with_snapshot_sets_image(node: LoadGLTF) -> None:
    node.set_parameter_value("gltf", {"type": "GLTFUrlArtifact", "value": "x", "metadata": {"imageUrl": IMAGE_URL}})

    image = node.get_parameter_value("image")
    assert isinstance(image, ImageUrlArtifact)
    assert image.value == IMAGE_URL


def test_artifact_with_snapshot_sets_image(node: LoadGLTF) -> None:
    artifact = GLTFUrlArtifact("https://example.com/model.glb", meta={"imageUrl": IMAGE_URL})
    node.set_parameter_value("gltf", artifact)

    image = node.get_parameter_value("image")
    assert isinstance(image, ImageUrlArtifact)
    assert image.value == IMAGE_URL


def test_clearing_gltf_clears_snapshot(node: LoadGLTF) -> None:
    node.set_parameter_value("gltf", {"type": "GLTFUrlArtifact", "value": "x", "metadata": {"imageUrl": IMAGE_URL}})
    node.set_parameter_value("gltf", None)

    assert node.get_parameter_value("image") is None
    assert node.parameter_output_values["image"] is None
