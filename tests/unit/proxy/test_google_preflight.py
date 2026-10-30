"""Tests for the pre-flight input checks on the Google proxy nodes.

Each node refuses, before submitting, an input Google documents it will reject, and lets through
anything it can't read locally.
"""

from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.exe_types.validators._runner import validate_node
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils import media_probe
from PIL import Image

from griptape_nodes_library.video import veo3_video_generation

LIBRARY_NAME = "Griptape Nodes Library"
# A real 64x48 HEIC photo, made with macOS's `sips -s format heic`. Pillow can't open it without a
# plugin the engine doesn't install; the engine reads its format from its file signature.
HEIC_PHOTO = Path(__file__).parent.parent / "fixtures" / "photo_64x48.heic"


def _create(node_type: str) -> Any:
    return LibraryRegistry.get_library(name=LIBRARY_NAME).create_node(node_type=node_type, name=node_type)


def _image_uri(image_format: str = "PNG", size: tuple[int, int] = (16, 16)) -> str:
    buffer = BytesIO()
    Image.new("RGB", size).save(buffer, format=image_format)
    return f"data:image/{image_format.lower()};base64,{base64.b64encode(buffer.getvalue()).decode()}"


def _report_size(monkeypatch: pytest.MonkeyPatch, size_bytes: int) -> None:
    """Have the engine's size reader report every input as `size_bytes`, so a limit can be broken without a huge file."""
    monkeypatch.setattr(media_probe, "_read_size_bytes", lambda _location: size_bytes)


def _all_errors(node: Any) -> list[Exception]:
    """What the engine reports at the node's turn: its pre-run validators', then its own checks'."""
    return validate_node(node)


def _node_errors(node: Any) -> list[NodeError]:
    return [e for e in _all_errors(node) if isinstance(e, NodeError)]


class TestGoogleImageGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> Any:  # noqa: ARG002
        node = _create("GoogleImageGeneration")
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_images(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri("PNG"), _image_uri("JPEG"), _image_uri("WEBP")])
        assert _all_errors(node) == []

    def test_accepts_a_heic_photo(self, node: Any) -> None:
        node.set_parameter_value("input_images", [str(HEIC_PHOTO)])
        assert _all_errors(node) == []

    def test_rejects_too_many_images_across_the_image_lists(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri()] * 10)
        node.set_parameter_value("object_images", [_image_uri()] * 5)
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "15 images are connected across 'Input Images', 'Object Images' and 'Human Images' and the limit is 14. "
            "Remove 1."
        ]
        assert errors[0].links

    @pytest.mark.parametrize(
        ("model", "url"),
        [
            (
                "gemini-3-pro-image",
                "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-pro-image",
            ),
            (
                "gemini-3.1-flash-image",
                "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image",
            ),
            # A legacy display name resolves to its model id before the page is chosen.
            (
                "Nano Banana 2",
                "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-image",
            ),
        ],
    )
    def test_errors_link_the_selected_models_page(self, node: Any, model: str, url: str) -> None:
        node.set_parameter_value("model", model)
        node.set_parameter_value("input_images", [_image_uri("GIF")])
        assert [link.url for error in _node_errors(node) for link in error.links] == [url]

    def test_rejects_an_unsupported_format(self, node: Any) -> None:
        node.set_parameter_value("input_images", [_image_uri("PNG"), _image_uri("GIF")])
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "Image 2 in 'Input Images' is in GIF format. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]
        assert errors[0].links

    def test_rejects_an_oversized_image_only_with_auto_resize_off(
        self, node: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _report_size(monkeypatch, 8 * 1024 * 1024)
        node.set_parameter_value("input_images", [_image_uri()])

        assert _all_errors(node) == []

        node.set_parameter_value("auto_image_resize", False)
        errors = _node_errors(node)
        assert len(errors) == 1
        assert str(errors[0]).startswith("Image 1 in 'Input Images' is ")
        assert str(errors[0]).endswith("Turn on 'Auto Image Resize' or use a smaller image.")

    def test_lets_auto_resize_convert_an_oversized_unsupported_image(
        self, node: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _report_size(monkeypatch, 8 * 1024 * 1024)
        node.set_parameter_value("input_images", [_image_uri("GIF")])
        assert _all_errors(node) == []

    @pytest.mark.parametrize("auto_image_resize", [True, False])
    def test_rejects_an_oversized_image_auto_resize_cannot_open(
        self, node: Any, monkeypatch: pytest.MonkeyPatch, *, auto_image_resize: bool
    ) -> None:
        _report_size(monkeypatch, 8 * 1024 * 1024)
        node.set_parameter_value("auto_image_resize", auto_image_resize)
        node.set_parameter_value("input_images", [str(HEIC_PHOTO)])
        errors = _node_errors(node)
        assert len(errors) == 1
        assert str(errors[0]).startswith("Image 1 in 'Input Images' is ")
        assert str(errors[0]).endswith("Use a smaller image.")

    def test_skips_a_remote_image(self, node: Any) -> None:
        node.set_parameter_value("input_images", ["https://example.com/image.gif"])
        assert _all_errors(node) == []


class TestVeo3VideoGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> Any:  # noqa: ARG002
        node = _create("Veo3VideoGeneration")
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_frames(self, node: Any) -> None:
        node.set_parameter_value("start_frame", _image_uri("PNG"))
        node.set_parameter_value("last_frame", _image_uri("JPEG"))
        assert _all_errors(node) == []

    def test_rejects_an_unsupported_frame_format(self, node: Any) -> None:
        node.set_parameter_value("start_frame", _image_uri("WEBP"))
        errors = _node_errors(node)
        assert [str(e) for e in errors] == ["The image in 'start frame' is in WEBP format. Convert it to JPEG or PNG."]
        assert errors[0].links
        assert errors[0].links[0].url == veo3_video_generation.FRAME_FORMATS_URL

    def test_rejects_an_oversized_frame(self, node: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        _report_size(monkeypatch, 21 * 1024 * 1024)
        node.set_parameter_value("last_frame", _image_uri())
        errors = _node_errors(node)
        assert len(errors) == 1
        assert str(errors[0]).startswith("The image in 'last frame' is ")
        assert "and the limit is" in str(errors[0])

    def test_checks_only_the_asset_references_the_request_sends(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri()] * 3 + [_image_uri("GIF")])
        assert _all_errors(node) == []

    def test_rejects_an_unsupported_reference_format(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri(), _image_uri("GIF")])
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "Image 2 in 'reference images' is in GIF format. Convert it to JPEG or PNG."
        ]
        assert errors[0].links
        assert errors[0].links[0].url == veo3_video_generation.REFERENCE_IMAGES_URL

    def test_skips_reference_formats_on_a_model_that_takes_no_references(self, node: Any) -> None:
        node.set_parameter_value("model_id", veo3_video_generation.ModelId.VEO_3_1_FAST_GENERATE_001.value)
        node.set_parameter_value("reference_images", [_image_uri("GIF")])
        assert not any("GIF format" in str(e) for e in _all_errors(node))

    @pytest.mark.parametrize("parameter", ["start_frame", "reference_images"])
    def test_rejects_a_heic_photo_by_its_real_format(self, node: Any, parameter: str) -> None:
        if parameter == "reference_images":
            node.set_parameter_value(parameter, [str(HEIC_PHOTO)])
            expected = "Image 1 in 'reference images' is in HEIC format. Convert it to JPEG or PNG."
        else:
            node.set_parameter_value(parameter, str(HEIC_PHOTO))
            expected = "The image in 'start frame' is in HEIC format. Convert it to JPEG or PNG."
        assert [str(e) for e in _node_errors(node)] == [expected]

    def test_checks_only_the_style_reference_the_request_sends(self, node: Any) -> None:
        node.set_parameter_value("reference_type", "style")
        node.set_parameter_value("reference_images", [_image_uri("PNG"), _image_uri("GIF")])
        assert _all_errors(node) == []


class TestGeminiOmniFlashGeneration:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> Any:  # noqa: ARG002
        node = _create("GeminiOmniFlashGeneration")
        node.set_parameter_value("prompt", "a cat")
        return node

    def test_accepts_supported_references(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri("PNG"), _image_uri("WEBP")])
        assert _all_errors(node) == []

    def test_rejects_an_unsupported_reference_format(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri("GIF")])
        errors = _node_errors(node)
        assert [str(e) for e in errors] == [
            "Image 1 in 'reference images' is in GIF format. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]

    def test_accepts_a_heic_reference_photo(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [str(HEIC_PHOTO)])
        assert _all_errors(node) == []

    def test_rejects_an_unsupported_deprecated_image(self, node: Any) -> None:
        node.set_parameter_value("image", _image_uri("GIF"))
        assert [str(e) for e in _node_errors(node)] == [
            "The image in 'image (deprecated)' is in GIF format. Convert it to PNG, JPEG, WEBP, HEIC or HEIF."
        ]

    def test_ignores_the_deprecated_image_when_references_are_connected(self, node: Any) -> None:
        node.set_parameter_value("reference_images", [_image_uri("PNG")])
        node.set_parameter_value("image", _image_uri("GIF"))
        assert _all_errors(node) == []
