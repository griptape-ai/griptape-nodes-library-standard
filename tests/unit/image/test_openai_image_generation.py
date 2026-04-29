from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from griptape.artifacts import ImageArtifact
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


@pytest.mark.asyncio
async def test_build_payload_includes_input_images_as_data_urls(
    node: OpenAiImageGeneration, tmp_path: Path
) -> None:
    input_image_path = tmp_path / "source.png"
    input_image_bytes = b"image-input"
    input_image_path.write_bytes(input_image_bytes)

    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("prompt", "Use the reference image")
    node.set_parameter_value("size", "1024x1024")
    node.set_parameter_value("input_images", [str(input_image_path)])

    payload = await node._build_payload()

    assert payload["model"] == "gpt-image-2"
    assert payload["images"] == [
        {"image_url": f"data:image/png;base64,{base64.b64encode(input_image_bytes).decode('utf-8')}"}
    ]


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


@pytest.mark.parametrize("model_name", ["GPT Image 1", "GPT Image 1.5", "GPT Image 2"])
def test_validate_rejects_too_many_reference_images(node: OpenAiImageGeneration, model_name: str) -> None:
    node.set_parameter_value("model", model_name)
    node.set_parameter_value("prompt", "Use the reference images")
    node.set_parameter_value("size", "1024x1024")
    node.set_parameter_value("input_images", [f"image_{index}.png" for index in range(17)])

    exceptions = node.validate_before_node_run()

    assert exceptions is not None
    assert any("supports up to 16 reference images" in str(exception) for exception in exceptions)


@pytest.mark.parametrize(
    ("size", "message_fragment"),
    [
        ("3856x1024", "edge lengths must be 3840px or less"),
        ("1025x1024", "must both be multiples of 16px"),
        ("3840x1024", "aspect ratio cannot exceed 3:1"),
        ("512x1024", "total pixels must be between 655,360 and 8,294,400"),
    ],
)
def test_validate_rejects_invalid_gpt_image_2_custom_sizes(
    node: OpenAiImageGeneration, size: str, message_fragment: str
) -> None:
    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("prompt", "A red circle")
    node.parameter_values["size"] = size

    exceptions = node.validate_before_node_run()

    assert exceptions is not None
    assert any(message_fragment in str(exception) for exception in exceptions)


def test_validate_accepts_valid_gpt_image_2_custom_size(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("prompt", "A red circle")
    node.parameter_values["size"] = "2048x1152"

    exceptions = node.validate_before_node_run()

    assert exceptions is None


@pytest.mark.asyncio
async def test_build_payload_uses_base64_from_image_artifact(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 1")
    node.set_parameter_value("prompt", "Use the artifact image")
    node.set_parameter_value("size", "1024x1024")
    node.set_parameter_value(
        "input_images",
        [ImageArtifact(value=b"artifact-image-bytes", format="png", width=1, height=1)],
    )

    payload = await node._build_payload()

    assert payload["images"] == [
        {"image_url": f"data:image/png;base64,{base64.b64encode(b'artifact-image-bytes').decode('utf-8')}"}
    ]


@pytest.mark.asyncio
async def test_build_payload_raises_for_invalid_input_image(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 1.5")
    node.set_parameter_value("prompt", "Use the reference image")
    node.set_parameter_value("size", "1024x1024")
    node.set_parameter_value("input_images", ["/definitely/missing/image.png"])

    with pytest.raises(ValueError, match="Failed to read input image"):
        await node._build_payload()


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
