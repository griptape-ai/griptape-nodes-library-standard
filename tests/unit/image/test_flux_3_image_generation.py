from __future__ import annotations

import pytest
from griptape.artifacts import ImageUrlArtifact

from griptape_nodes_library.image.flux_3_image_generation import (
    Flux3ImageGeneration,
    Flux3ImageModel,
    Flux3SizeMode,
)


def _make_node() -> Flux3ImageGeneration:
    return Flux3ImageGeneration(name="Flux3Image")


def test_default_model_and_visibility() -> None:
    node = _make_node()
    assert node.get_parameter_value("model") == Flux3ImageModel.FLUX_3_IMAGE
    grounding = node.get_parameter_by_name("grounding")
    output_format = node.get_parameter_by_name("output_format")
    steps = node.get_parameter_by_name("steps")
    assert grounding is not None and not grounding.ui_options.get("hide", False)
    assert output_format is not None and output_format.ui_options.get("hide") is True
    assert steps is not None and steps.ui_options.get("hide") is True


def test_switching_to_klein_toggles_model_specific_parameters() -> None:
    node = _make_node()
    node.set_parameter_value("model", Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value)
    model_param = node.get_parameter_by_name("model")
    assert model_param is not None
    node.after_value_set(model_param, Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value)

    grounding = node.get_parameter_by_name("grounding")
    output_format = node.get_parameter_by_name("output_format")
    steps = node.get_parameter_by_name("steps")
    assert grounding is not None and grounding.ui_options.get("hide") is True
    assert output_format is not None and not output_format.ui_options.get("hide", False)
    assert steps is not None and not steps.ui_options.get("hide", False)


def test_validation_checks_prompt_and_match_input() -> None:
    node = _make_node()
    node.set_parameter_value("prompt", "")
    node.set_parameter_value("resolution", "match_input")
    errors = node.validate_before_node_run()
    assert errors is not None
    messages = [str(e) for e in errors]
    assert any("Enter a prompt" in m for m in messages)
    assert any("match_input" in m for m in messages)


def test_validation_checks_custom_dimension_bounds() -> None:
    node = _make_node()
    node.set_parameter_value("prompt", "A mountain lake")
    node.set_parameter_value("size_mode", Flux3SizeMode.CUSTOM_DIMENSIONS.value)
    node.set_parameter_value("width", 128)
    node.set_parameter_value("height", 1024)
    errors = node.validate_before_node_run()
    assert errors is not None
    assert any("'width' must be between 256 and 6256" in str(e) for e in errors)


@pytest.mark.asyncio
async def test_build_payload_for_flux_3_image_and_klein(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_prepare(image_input: object, **_kwargs: object) -> str:  # noqa: ARG001
        return "data:image/png;base64,AAA"

    monkeypatch.setattr(
        "griptape_nodes_library.image.flux_3_image_generation.prepare_media_data_uri",
        fake_prepare,
    )

    node = _make_node()
    node.set_parameter_value("prompt", "A red origami crane")
    node.set_parameter_value("resolution", "768sq")
    node.set_parameter_value("aspect_ratio", "1:1")
    node.set_parameter_value("grounding", False)
    node.set_parameter_value("input_images", [ImageUrlArtifact("https://example.com/ref.png")])

    payload = await node._build_payload()
    assert payload == {
        "prompt": "A red origami crane",
        "safety_tolerance": 2,
        "aspect_ratio": "1:1",
        "resolution": "768sq",
        "grounding": False,
        "images": ["data:image/png;base64,AAA"],
    }

    node.set_parameter_value("model", Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value)
    node.set_parameter_value("size_mode", Flux3SizeMode.CUSTOM_DIMENSIONS.value)
    node.set_parameter_value("width", 512)
    node.set_parameter_value("height", 512)
    node.set_parameter_value("steps", 12)
    node.set_parameter_value("output_format", "png")

    klein_payload = await node._build_payload()
    assert klein_payload["prompt"] == "A red origami crane"
    assert klein_payload["width"] == 512
    assert klein_payload["height"] == 512
    assert klein_payload["steps"] == 12
    assert klein_payload["output_format"] == "png"
    assert "seed" in klein_payload
    assert "grounding" not in klein_payload
