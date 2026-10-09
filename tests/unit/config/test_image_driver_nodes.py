from __future__ import annotations

from typing import cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry

from griptape_nodes_library.config.image.base_image_driver import BaseImageDriver
from griptape_nodes_library.llm.image_generation import ImageGenerationConfig, ImageProvider

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node(node_type: str) -> BaseImageDriver:
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast("BaseImageDriver", library.create_node(node_type=node_type, name=node_type))


def _output(node: BaseImageDriver) -> ImageGenerationConfig:
    node.process()
    output = node.parameter_output_values["image_model_config"]
    assert isinstance(output, ImageGenerationConfig)
    return output


def test_output_parameter_keeps_image_generation_driver_type() -> None:
    node = _create_node("GriptapeCloudImage")

    output_param = node.get_parameter_by_name("image_model_config")

    assert output_param is not None
    assert output_param.output_type == "Image Generation Driver"


def test_griptape_cloud_image_outputs_selected_model_size_and_quality() -> None:
    node = _create_node("GriptapeCloudImage")
    node.set_parameter_value("model", "gpt-image-1.5")
    node.set_parameter_value("image_size", "1536x1024")
    node.set_parameter_value("quality", "high")

    assert _output(node) == ImageGenerationConfig(
        provider=ImageProvider.GRIPTAPE_CLOUD, model="gpt-image-1.5", image_size="1536x1024", quality="high"
    )


def test_grok_image_outputs_xai_endpoint_and_secret_name() -> None:
    config = _output(_create_node("GrokImage"))

    assert config == ImageGenerationConfig(
        provider=ImageProvider.GROK,
        model="grok-2-image-1212",
        base_url="https://api.x.ai/v1",
        api_key_secret="GROK_API_KEY",  # noqa: S106
    )


def test_openai_image_outputs_gpt_image_options() -> None:
    node = _create_node("OpenAiImage")
    node.set_parameter_value("model", "gpt-image-1")
    node.set_parameter_value("output_format", "jpeg")
    node.set_parameter_value("output_compression", 60)

    config = _output(node)

    assert config.provider == ImageProvider.OPENAI
    assert config.model == "gpt-image-1"
    assert config.api_key_secret == "OPENAI_API_KEY"  # noqa: S105
    assert config.output_format == "jpeg"
    assert config.output_compression == 60
    assert config.background == "opaque"
    assert config.moderation == "low"


def test_openai_image_dall_e_3_omits_gpt_image_options() -> None:
    node = _create_node("OpenAiImage")
    node.set_parameter_value("model", "dall-e-3")

    config = _output(node)

    assert config.model == "dall-e-3"
    assert config.style == "vivid"
    assert config.quality == "hd"
    assert config.background is None
    assert config.output_format is None


def test_config_wire_dict_does_not_contain_secret_values() -> None:
    wire = _output(_create_node("OpenAiImage")).to_wire()

    assert wire["api_key_secret"] == "OPENAI_API_KEY"  # noqa: S105
    assert "api_key" not in wire


@pytest.mark.parametrize("node_type", ["GriptapeCloudImage", "GrokImage", "OpenAiImage"])
def test_output_round_trips_through_wire(node_type: str) -> None:
    config = _output(_create_node(node_type))

    assert ImageGenerationConfig.from_wire(config.to_wire()) == config
