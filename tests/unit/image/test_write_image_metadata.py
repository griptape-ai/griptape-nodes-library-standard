"""Tests for ``WriteImageMetadataNode``: repeated writes merge, and output lands at the image's own extension."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import cast

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.drivers.image_metadata.exif_metadata_driver import EXIF_USERCOMMENT_TAG
from griptape_nodes.files.file import File
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.events.artifact_events import RegisterArtifactProviderRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.artifact_providers.image.image_artifact_provider import (
    ImageArtifactProvider,
)
from PIL import Image

from griptape_nodes_library.image.write_image_metadata import WriteImageMetadataNode

LIBRARY_NAME = "Griptape Nodes Library"


@pytest.fixture(autouse=True)
def write_in_tmp_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_workflow: None) -> None:  # noqa: ARG001
    """Write outputs under ``tmp_path`` and register the image provider the engine registers at startup."""
    monkeypatch.chdir(tmp_path)
    GriptapeNodes.handle_request(RegisterArtifactProviderRequest(provider_class=ImageArtifactProvider))


def _image_bytes(image_format: str) -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, image_format)
    return buf.getvalue()


def _write(image_bytes: bytes, metadata: dict, name: str) -> tuple[WriteImageMetadataNode, bytes]:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node = cast(WriteImageMetadataNode, library.create_node(node_type="WriteImageMetadataNode", name=name))
    node.set_parameter_value("input_image", ImageArtifact(value=image_bytes, format="png", width=4, height=4))
    node.set_parameter_value("metadata", metadata)
    node.process()
    output = node.parameter_output_values["output_image"]
    return node, File(output.value).read_bytes()


@pytest.mark.parametrize("image_format", ["JPEG", "TIFF", "WEBP"])
def test_second_exif_write_merges_into_existing_user_comment(image_format: str) -> None:
    _, first = _write(_image_bytes(image_format), {"old": "keep", "k1": "v1"}, "first")
    _, second = _write(first, {"k1": "replaced", "k2": "v2"}, "second")

    user_comment = Image.open(BytesIO(second)).getexif()[EXIF_USERCOMMENT_TAG]

    assert json.loads(user_comment) == {"old": "keep", "k1": "replaced", "k2": "v2"}


def test_second_png_write_merges_into_existing_text_chunks() -> None:
    _, first = _write(_image_bytes("PNG"), {"old": "keep", "k1": "v1"}, "first")
    _, second = _write(first, {"k1": "replaced", "k2": "v2"}, "second")

    info = Image.open(BytesIO(second)).info

    assert {key: info[key] for key in ("old", "k1", "k2")} == {"old": "keep", "k1": "replaced", "k2": "v2"}


def test_non_json_user_comment_is_replaced() -> None:
    buf = BytesIO()
    exif = Image.Exif()
    exif[EXIF_USERCOMMENT_TAG] = "shot on a phone"
    Image.new("RGB", (4, 4)).save(buf, "JPEG", exif=exif)

    _, written = _write(buf.getvalue(), {"k1": "v1"}, "non_json")

    assert json.loads(Image.open(BytesIO(written)).getexif()[EXIF_USERCOMMENT_TAG]) == {"k1": "v1"}


def test_jpeg_output_is_saved_with_jpg_extension() -> None:
    node, written = _write(_image_bytes("JPEG"), {"k1": "v1"}, "jpeg_ext")

    location = node.parameter_output_values["output_image"].value

    assert Path(str(location)).suffix == ".jpg"
    assert Image.open(BytesIO(written)).format == "JPEG"


def test_unsupported_format_error_lists_registered_formats() -> None:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node = cast(WriteImageMetadataNode, library.create_node(node_type="WriteImageMetadataNode", name="gif"))
    node.set_parameter_value("input_image", ImageArtifact(value=_image_bytes("GIF"), format="gif", width=4, height=4))
    node.set_parameter_value("metadata", {"k1": "v1"})

    with pytest.raises(Exception, match="Unsupported format 'GIF'"):
        node.process()

    assert "WEBP" in node.get_parameter_value("result_details")
