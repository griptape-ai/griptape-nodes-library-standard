from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.image.openai_image_generation import OpenAiImageGeneration


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> OpenAiImageGeneration:  # noqa: ARG002
    return OpenAiImageGeneration(name="OpenAI Image Generation")


@pytest.mark.asyncio
async def test_build_payload_for_jpeg_includes_output_compression(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 1.5")
    node.set_parameter_value("prompt", "A red circle")
    node.set_parameter_value("size", "1536x1024")
    node.set_parameter_value("n", 2)
    node.set_parameter_value("quality", "high")
    node.set_parameter_value("background", "opaque")
    node.set_parameter_value("moderation", "low")
    node.set_parameter_value("output_format", "jpeg")
    node.set_parameter_value("output_compression", 60)

    payload = await node._build_payload()

    assert payload == {
        "model": "gpt-image-1.5",
        "prompt": "A red circle",
        "size": "1536x1024",
        "n": 2,
        "quality": "high",
        "background": "opaque",
        "moderation": "low",
        "output_format": "jpeg",
        "output_compression": 60,
    }


def test_validate_rejects_invalid_gpt_image_1_size(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 1")
    node.set_parameter_value("prompt", "A red circle")
    node.parameter_values["size"] = "2048x2048"

    exceptions = node.validate_before_node_run()

    assert exceptions is not None
    assert any("GPT Image 1 size must be one of" in str(exception) for exception in exceptions)


def test_validate_rejects_transparent_jpeg(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("prompt", "A red circle")
    node.set_parameter_value("background", "transparent")
    node.set_parameter_value("output_format", "jpeg")

    exceptions = node.validate_before_node_run()

    assert exceptions is not None
    assert any("Transparent backgrounds require output_format" in str(exception) for exception in exceptions)


@pytest.mark.asyncio
async def test_parse_result_saves_base64_images(
    node: OpenAiImageGeneration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeDestination:
        def __init__(self, path: Path) -> None:
            self.path = path

        async def awrite_bytes(self, data: bytes) -> SimpleNamespace:
            self.path.write_bytes(data)
            return SimpleNamespace(location=str(self.path), name=self.path.name)

    def fake_build_file(_index: int = 1, **_kwargs: object) -> FakeDestination:
        return FakeDestination(tmp_path / f"openai_image_{_index}.png")

    monkeypatch.setattr(node._output_file, "build_file", fake_build_file)

    image_1_bytes = b"image-one"
    image_2_bytes = b"image-two"
    result_json = {
        "data": [
            {"b64_json": base64.b64encode(image_1_bytes).decode("utf-8")},
            {"b64_json": base64.b64encode(image_2_bytes).decode("utf-8")},
        ]
    }

    await node._parse_result(result_json, "gen_123")

    first_artifact = node.parameter_output_values["image_url"]
    second_artifact = node.parameter_output_values["image_url_2"]

    assert first_artifact is not None
    assert second_artifact is not None
    assert Path(first_artifact.value).read_bytes() == image_1_bytes
    assert Path(second_artifact.value).read_bytes() == image_2_bytes
    assert node.parameter_output_values["was_successful"] is True
