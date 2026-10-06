"""Convert node image values into pydantic-ai multimodal prompt content."""

from __future__ import annotations

from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from pydantic_ai.messages import BinaryContent

from griptape_nodes_library.utils.image_utils import load_image_from_url_artifact


def image_content(value: Any) -> BinaryContent:
    """Turn an `ImageArtifact`, `ImageUrlArtifact`, serialized artifact dict, or path/URL string into image bytes.

    Raises:
        TypeError: `value` is not a recognized image value.
        ValueError: A URL or path image could not be loaded.
    """
    if isinstance(value, dict) and "value" in value:
        value = value["value"]
    if isinstance(value, str) and value.strip():
        value = ImageUrlArtifact(value)
    if isinstance(value, ImageUrlArtifact):
        value = load_image_from_url_artifact(value)
    if isinstance(value, ImageArtifact):
        return BinaryContent(data=value.value, media_type=value.mime_type)
    msg = f"Cannot use a value of type {type(value).__name__} as an image."
    raise TypeError(msg)
