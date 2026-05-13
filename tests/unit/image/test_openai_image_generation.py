from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.image.openai_image_generation import (
    GPT_IMAGE_1_MODEL_ID,
    GPT_IMAGE_1_MODEL_NAME,
    GPT_IMAGE_2_MODEL_ID,
    GPT_IMAGE_2_MODEL_NAME,
    OpenAiImageGeneration,
)


def _is_param_hidden(node: OpenAiImageGeneration, name: str) -> bool:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    return bool(parameter.ui_options.get("hide", False))


def _is_message_hidden(node: OpenAiImageGeneration, name: str) -> bool:
    message = node.get_message_by_name_or_element_id(name)
    assert message is not None
    return bool(message.ui_options.get("hide", False))


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


def _size_choices(node: OpenAiImageGeneration) -> list[str]:
    size_param = node.get_parameter_by_name("size")
    assert size_param is not None
    options_traits = size_param.find_elements_by_type(Options)
    assert options_traits, "size parameter is missing an Options trait"
    return list(options_traits[0].choices)


def test_default_model_exposes_gpt_image_2_aspect_ratios(node: OpenAiImageGeneration) -> None:
    assert _size_choices(node) == OpenAiImageGeneration.GPT_IMAGE_2_SIZE_OPTIONS


@pytest.mark.parametrize(
    ("model_name", "expected_choices"),
    [
        ("GPT Image 1", OpenAiImageGeneration.GPT_IMAGE_SIZE_OPTIONS),
        ("GPT Image 1.5", OpenAiImageGeneration.GPT_IMAGE_SIZE_OPTIONS),
        ("GPT Image 2", OpenAiImageGeneration.GPT_IMAGE_2_SIZE_OPTIONS),
    ],
)
def test_size_options_update_when_model_changes(
    node: OpenAiImageGeneration, model_name: str, expected_choices: list[str]
) -> None:
    node.set_parameter_value("model", model_name)

    assert _size_choices(node) == expected_choices


def test_size_resets_to_first_choice_when_switching_to_legacy_model(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("size", "1792x1024")

    node.set_parameter_value("model", "GPT Image 1")

    assert node.get_parameter_value("size") == OpenAiImageGeneration.GPT_IMAGE_SIZE_OPTIONS[0]


def test_switching_to_gpt_image_2_preserves_valid_custom_size(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 1")
    node.parameter_values["size"] = "2048x1152"

    node.set_parameter_value("model", "GPT Image 2")

    assert node.get_parameter_value("size") == "2048x1152"


@pytest.mark.parametrize("size", OpenAiImageGeneration.GPT_IMAGE_2_SIZE_OPTIONS)
def test_validate_accepts_all_listed_gpt_image_2_aspect_ratios(node: OpenAiImageGeneration, size: str) -> None:
    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("prompt", "A red circle")
    node.parameter_values["size"] = size

    assert node.validate_before_node_run() is None


@pytest.mark.asyncio
async def test_build_payload_sends_new_aspect_ratio_to_api(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", "GPT Image 2")
    node.set_parameter_value("prompt", "A wide landscape")
    node.set_parameter_value("size", "1792x1024")

    payload = await node._build_payload()

    assert payload["model"] == "gpt-image-2"
    assert payload["size"] == "1792x1024"


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


def test_default_model_uses_display_name(node: OpenAiImageGeneration) -> None:
    assert node.get_parameter_value("model") == GPT_IMAGE_2_MODEL_NAME


def test_payload_model_id_translates_display_name(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME)

    assert node._get_payload_model_id() == GPT_IMAGE_2_MODEL_ID


def test_size_choices_include_custom_for_gpt_image_2(node: OpenAiImageGeneration) -> None:
    assert "custom" in _size_choices(node)


def test_size_choices_omit_custom_for_legacy_models(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", GPT_IMAGE_1_MODEL_NAME)
    assert "custom" not in _size_choices(node)


def test_custom_size_params_hidden_by_default(node: OpenAiImageGeneration) -> None:
    assert _is_param_hidden(node, "custom_width")
    assert _is_param_hidden(node, "custom_height")
    assert _is_message_hidden(node, "custom_size_help")


def test_selecting_custom_size_reveals_dimension_inputs(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("size", "custom")

    assert not _is_param_hidden(node, "custom_width")
    assert not _is_param_hidden(node, "custom_height")
    assert not _is_message_hidden(node, "custom_size_help")


def test_switching_off_custom_size_hides_dimension_inputs(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("size", "1024x1024")

    assert _is_param_hidden(node, "custom_width")
    assert _is_param_hidden(node, "custom_height")
    assert _is_message_hidden(node, "custom_size_help")


def test_switching_to_legacy_model_hides_custom_size_inputs(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("model", GPT_IMAGE_1_MODEL_NAME)

    assert _is_param_hidden(node, "custom_width")
    assert _is_param_hidden(node, "custom_height")
    assert _is_message_hidden(node, "custom_size_help")


@pytest.mark.parametrize(
    ("typed_value", "snapped_value"),
    [
        (1000, 1008),
        (1008, 1008),
        (1, 16),
        (5000, 3840),
        (1009, 1008),
        (1015, 1008),
        (1016, 1024),
    ],
)
def test_custom_dimension_snaps_to_multiple_of_16(
    node: OpenAiImageGeneration, typed_value: int, snapped_value: int
) -> None:
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("custom_width", typed_value)
    node.set_parameter_value("custom_height", typed_value)

    assert node.get_parameter_value("custom_width") == snapped_value
    assert node.get_parameter_value("custom_height") == snapped_value


@pytest.mark.asyncio
async def test_build_payload_combines_custom_dimensions(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME)
    node.set_parameter_value("prompt", "A panoramic image")
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("custom_width", 2048)
    node.set_parameter_value("custom_height", 1152)

    payload = await node._build_payload()

    assert payload["size"] == "2048x1152"


def test_validate_runs_against_custom_dimensions(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME)
    node.set_parameter_value("prompt", "A red circle")
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("custom_width", 3840)
    node.set_parameter_value("custom_height", 1024)  # 3840:1024 is > 3:1

    exceptions = node.validate_before_node_run()

    assert exceptions is not None
    assert any("aspect ratio cannot exceed 3:1" in str(exception) for exception in exceptions)


def test_validate_accepts_valid_custom_dimensions(node: OpenAiImageGeneration) -> None:
    node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME)
    node.set_parameter_value("prompt", "A red circle")
    node.set_parameter_value("size", "custom")
    node.set_parameter_value("custom_width", 2048)
    node.set_parameter_value("custom_height", 1152)

    assert node.validate_before_node_run() is None


def test_initial_setup_sync_restores_custom_size_visibility(
    griptape_nodes: GriptapeNodes,  # noqa: ARG001
) -> None:
    """Workflow load uses initial_setup=True, which skips after_value_set. Visibility must still sync."""
    fresh_node = OpenAiImageGeneration(name="OpenAI Image Generation Loaded")

    fresh_node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME, initial_setup=True)
    fresh_node.set_parameter_value("size", "custom", initial_setup=True)

    assert not _is_param_hidden(fresh_node, "custom_width")
    assert not _is_param_hidden(fresh_node, "custom_height")
    assert not _is_message_hidden(fresh_node, "custom_size_help")


def test_initial_setup_does_not_snap_custom_dimensions(
    griptape_nodes: GriptapeNodes,  # noqa: ARG001
) -> None:
    """Workflow load must not mutate persisted custom_width/height values."""
    fresh_node = OpenAiImageGeneration(name="OpenAI Image Generation Loaded")

    fresh_node.set_parameter_value("model", GPT_IMAGE_2_MODEL_NAME, initial_setup=True)
    fresh_node.set_parameter_value("size", "custom", initial_setup=True)
    fresh_node.set_parameter_value("custom_width", 1000, initial_setup=True)

    assert fresh_node.get_parameter_value("custom_width") == 1000


def test_custom_size_help_message_uses_markdown(node: OpenAiImageGeneration) -> None:
    message = node.get_message_by_name_or_element_id("custom_size_help")

    assert message is not None
    assert message.ui_options.get("markdown") is True
