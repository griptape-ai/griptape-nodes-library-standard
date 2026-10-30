"""Provider nodes turn every supported image input shape into the request shape their API expects."""

from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.image.flux_image_generation import FluxImageGeneration
from griptape_nodes_library.image.google_image_generation import GoogleImageGeneration
from griptape_nodes_library.image.grok_image_edit import GrokImageEdit
from griptape_nodes_library.image.qwen_image_edit import QwenImageEdit
from griptape_nodes_library.image.seedream_image_generation import SeedreamImageGeneration
from griptape_nodes_library.image.topaz_image_enhance import TopazImageEnhance
from griptape_nodes_library.three_d.rodin_2_3d_generation import Rodin23DGeneration
from griptape_nodes_library.video.grok_video_generation import GrokVideoGeneration
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.wan_image_to_video_generation import WanImageToVideoGeneration

# A 1x1 PNG.
PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
PNG_DATA_URI = f"data:image/png;base64,{PNG_B64}"

DATA_URI_LOADERS = [
    (Flux2ImageGeneration, "_process_input_image"),
    (FluxImageGeneration, "_process_input_image"),
    (GrokImageEdit, "_prepare_image_data_uri"),
    (QwenImageEdit, "_process_input_image"),
    (SeedreamImageGeneration, "_process_input_image"),
    (TopazImageEnhance, "_process_input_image"),
    (Rodin23DGeneration, "_prepare_image_data_uri"),
    (GrokVideoGeneration, "_prepare_image_data_uri"),
    (WanImageToVideoGeneration, "_process_input_image"),
]

RAW_BASE64_LOADERS = [
    (KlingImageToVideoGeneration, "_prepare_image_data_url_async"),
    (KlingOmniVideoGeneration, "_prepare_image_data_url_async"),
]

INPUT_SHAPES = [
    pytest.param(PNG_DATA_URI, id="data-uri-string"),
    pytest.param(SimpleNamespace(base64=PNG_B64), id="artifact-raw-base64"),
    pytest.param({"type": "ImageArtifact", "value": PNG_B64, "format": "png"}, id="serialized-image-artifact"),
]


async def _load(node_cls: type, method: str, image_input: Any) -> Any:
    node = node_cls(name=node_cls.__name__)
    return await getattr(node, method)(image_input)


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_cls", "method"), DATA_URI_LOADERS)
@pytest.mark.parametrize("image_input", INPUT_SHAPES)
async def test_loader_returns_data_uri(node_cls: type, method: str, image_input: Any) -> None:
    assert await _load(node_cls, method, image_input) == PNG_DATA_URI


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_cls", "method"), RAW_BASE64_LOADERS)
@pytest.mark.parametrize("image_input", INPUT_SHAPES)
async def test_kling_loader_returns_raw_base64(node_cls: type, method: str, image_input: Any) -> None:
    assert await _load(node_cls, method, image_input) == PNG_B64


@pytest.mark.asyncio
@pytest.mark.parametrize("image_input", INPUT_SHAPES)
async def test_google_loader_returns_mime_and_base64(image_input: Any) -> None:
    node = GoogleImageGeneration(name="GoogleImageGeneration")
    assert await node._process_input_image(image_input) == ("image/png", PNG_B64)


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_cls", "method"), DATA_URI_LOADERS + RAW_BASE64_LOADERS)
async def test_loader_returns_none_for_empty_input(node_cls: type, method: str) -> None:
    assert await _load(node_cls, method, None) is None


@pytest.fixture
def png_path(tmp_path: Path) -> str:
    path = tmp_path / "input.png"
    path.write_bytes(base64.b64decode(PNG_B64))
    return str(path)


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_cls", "method"), DATA_URI_LOADERS)
async def test_loader_reads_file_path_as_data_uri(node_cls: type, method: str, png_path: str) -> None:
    assert await _load(node_cls, method, png_path) == PNG_DATA_URI


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_cls", "method"), RAW_BASE64_LOADERS)
async def test_kling_loader_reads_file_path_as_raw_base64(node_cls: type, method: str, png_path: str) -> None:
    assert await _load(node_cls, method, png_path) == PNG_B64
