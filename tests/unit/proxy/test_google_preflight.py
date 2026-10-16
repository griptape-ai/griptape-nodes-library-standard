"""Tests for the pre-flight input checks on the Google proxy nodes.

Each node refuses, before submitting, an input Google documents it will reject, and lets through
anything it can't read locally.
"""

from __future__ import annotations

import base64
import sys
from io import BytesIO
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image

LIBRARY_NAME = "Griptape Nodes Library"


def _create(node_type: str, monkeypatch: pytest.MonkeyPatch) -> Any:
    node = LibraryRegistry.get_library(name=LIBRARY_NAME).create_node(node_type=node_type, name=node_type)
    # A ParameterList reads its items from child parameters; with no connections in a unit test,
    # read the value set on the list itself.
    monkeypatch.setattr(node, "get_parameter_list_value", lambda name: node.parameter_values.get(name) or [])
    return node


def _image_uri(image_format: str = "PNG", size: tuple[int, int] = (16, 16)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size).save(buffer, format=image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _node_module(node: Any) -> Any:
    """The module a node's class was loaded from. The library registry loads node modules under its
    own names, so patching an imported module would miss the one the node reads."""
    return sys.modules[type(node).__module__]


def _node_errors(node: Any) -> list[NodeError]:
    return [e for e in node.validate_before_node_run() or [] if isinstance(e, NodeError)]


class TestGoogleImageGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ARG002
        node = _create("GoogleImageGeneration", monkeypatch)
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_images(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri("PNG"), _image_uri("JPEG"), _image_uri("WEBP")])
        assert node.validate_before_node_run() is None

    def test_rejects_too_many_images(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri()] * 10)
        node.set_parameter_value("object_images", [_image_uri()] * 5)
        errors = _node_errors(node)
        assert [str(e) for e in errors] == ["15 images are connected to 'Input Images' and the limit is 14. Remove 1."]

    def test_rejects_an_unsupported_format(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri("PNG"), _image_uri("GIF")])
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "Image 2 in 'Input Images' is a GIF file. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]
        assert errors[0].links

    def test_rejects_an_oversized_image_only_with_auto_resize_off(
        self, node: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(_node_module(node), "MAX_IMAGE_SIZE_BYTES", 10)
        node.set_parameter_value("input_images", [_image_uri()])

        assert node.validate_before_node_run() is None

        node.set_parameter_value("auto_image_resize", False)
        errors = _node_errors(node)
        assert len(errors) == 1
        assert str(errors[0]).startswith("Image 1 in 'Input Images' is ")
        assert str(errors[0]).endswith("Turn on 'Auto Image Resize' or use a smaller image.")

    def test_skips_a_remote_image(self, node: Any) -> None:
        node.set_parameter_value("input_images", ["https://example.com/image.gif"])
        assert node.validate_before_node_run() is None


class TestVeo3VideoGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ARG002
        node = _create("Veo3VideoGeneration", monkeypatch)
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_frames(self, node: Any) -> None:
        node.set_parameter_value("start_frame", _image_uri("PNG"))
        node.set_parameter_value("last_frame", _image_uri("JPEG"))
        assert node.validate_before_node_run() is None

    def test_rejects_an_unsupported_frame_format(self, node: Any) -> None:
        node.set_parameter_value("start_frame", _image_uri("WEBP"))
        assert [str(e) for e in _node_errors(node)] == [
            "The image in 'start frame' is a WEBP file. Convert it to JPEG or PNG."
        ]

    def test_rejects_an_oversized_frame(self, node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_node_module(node), "MAX_FRAME_IMAGE_BYTES", 10)
        node.set_parameter_value("last_frame", _image_uri())
        errors = _node_errors(node)
        assert len(errors) == 1
        assert str(errors[0]).startswith("The image in 'last frame' is ")
        assert "and the limit is" in str(errors[0])

    def test_rejects_more_than_three_asset_references(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri()] * 4)
        assert [str(e) for e in _node_errors(node)] == [
            "4 images are connected to 'reference images' and the limit is 3. Remove 1."
        ]

    def test_rejects_an_unsupported_reference_format(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri(), _image_uri("GIF")])
        assert [str(e) for e in _node_errors(node)] == [
            "Image 2 in 'reference images' is a GIF file. Convert it to JPEG or PNG."
        ]


class TestGeminiOmniFlashGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ARG002
        node = _create("GeminiOmniFlashGeneration", monkeypatch)
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_references(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri("PNG"), _image_uri("WEBP")])
        assert node.validate_before_node_run() is None

    def test_rejects_an_unsupported_reference_format(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri("GIF")])
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "Image 1 in 'reference images' is a GIF file. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]

    def test_rejects_an_unsupported_deprecated_image(self, node: Any) -> None:
        node.set_parameter_value("image", _image_uri("GIF"))
        assert [str(e) for e in _node_errors(node)] == [
            "The image in 'image (deprecated)' is a GIF file. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]
