from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest
from griptape.artifacts import ImageArtifact

from griptape_nodes_library.video import kling_4
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.kling_text_to_video_generation import KlingTextToVideoGeneration

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def _image() -> ImageArtifact:
    return ImageArtifact(value=PNG, format="png", width=1, height=1)


def _hidden(node: Any) -> set[str]:
    return {parameter.name for parameter in node.parameters if parameter.ui_options.get("hide")}


def _choices(node: Any, name: str) -> list[Any]:
    return list(node.get_parameter_by_name(name).ui_options["simple_dropdown"])


def _validation_messages(node: Any) -> list[str]:
    # Outside the engine the base class reports that no model policy is loaded; that is not ours.
    return [str(error) for error in node.validate_before_node_run() or [] if "allowed to use" not in str(error)]


@pytest.mark.parametrize(
    ("node_class", "suffix"),
    [
        (KlingTextToVideoGeneration, "text2video"),
        (KlingImageToVideoGeneration, "image2video"),
        (KlingOmniVideoGeneration, "omnivideo"),
    ],
)
@pytest.mark.parametrize("model_id", kling_4.KLING_4_MODELS)
def test_kling_4_models_are_offered_and_routed(node_class: type, suffix: str, model_id: str) -> None:
    node = node_class(name="Kling")
    node.set_parameter_value("model_name", model_id)

    assert node._get_api_model_id() == f"{model_id}:{suffix}"
    assert {"mode"} <= _hidden(node)
    assert "resolution" not in _hidden(node)
    assert _choices(node, "duration")[-1] == kling_4.KLING_4_CAPABILITIES[model_id].max_duration
    assert _choices(node, "resolution") == kling_4.KLING_4_CAPABILITIES[model_id].resolutions


def test_text_payload_uses_settings_shape() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", " a cat on a train ")
    node.set_parameter_value("aspect_ratio", "21:9")
    node.set_parameter_value("duration", 12)
    node.set_parameter_value("resolution", "720p")

    assert asyncio.run(node._build_payload()) == {
        "prompt": "a cat on a train",
        "settings": {"resolution": "720p", "duration": 12, "aspect_ratio": "21:9"},
    }


def test_switching_back_to_v3_restores_legacy_controls() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("model_name", "kling-v3")

    hidden = _hidden(node)
    assert "resolution" in hidden
    assert not ({"mode", "cfg_scale", "negative_prompt"} & hidden)
    payload = asyncio.run(node._build_payload())
    assert payload["model_name"] == "kling-v3"
    assert "settings" not in payload


def test_flash_offers_720p_only() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("resolution", "1080p")
    node.set_parameter_value("model_name", "kling-4.0-flash")

    assert node.get_parameter_value("resolution") == "720p"
    assert _choices(node, "duration") == list(range(3, 21))


def test_image_payload_orders_keyframes_with_ids() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "go from @frame_1 to @frame_3")
    node.set_parameter_value("image", _image())
    node.set_parameter_value("middle_frames", [_image()])
    node.set_parameter_value("image_tail", _image())

    payload = asyncio.run(node._build_payload())

    assert [(item["type"], item.get("id")) for item in payload["contents"]] == [
        ("prompt", None),
        ("first_frame", "frame_1"),
        ("middle_frame", "frame_2"),
        ("last_frame", "frame_3"),
    ]
    assert payload["contents"][1]["url"] == base64.b64encode(PNG).decode()
    assert payload["settings"] == {"resolution": "1080p", "duration": 5}


def test_image_flash_rejects_extra_keyframes() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0-flash")
    node.set_parameter_value("prompt", "go")
    node.set_parameter_value("image", _image())
    node.set_parameter_value("middle_frames", [_image()])

    assert {"image_tail", "middle_frames"} <= _hidden(node)
    assert _validation_messages(node) == [
        "kling-4.0-flash supports a start frame only; use Kling 4.0 for more keyframes."
    ]


def test_image_payload_redacts_base64_in_logs() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    url = "A" * 5000
    assert url not in node._elide_base64_in_payload({"contents": [{"type": "first_frame", "url": url}]})


def _omni_with_references(model_id: str) -> KlingOmniVideoGeneration:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", model_id)
    node.set_parameter_value("prompt", "use @image_1 with @voice_1")
    node.set_parameter_value("first_frame_image", _image())
    node.set_parameter_value("reference_images", [_image()])
    node.set_parameter_value("end_frame_image", _image())
    node.set_parameter_value("element_ids", "11, 22")
    node.set_parameter_value("voice_ids", "829837252279803904")
    node.set_parameter_value("reference_video", "https://example.com/v.mp4")
    node._public_video_url_parameter.aget_public_url_for_parameter = AsyncMock(  # type: ignore[method-assign]
        return_value="https://public/v.mp4"
    )
    return node


def test_omni_payload_includes_every_reference_kind() -> None:
    node = _omni_with_references("kling-4.0")
    node.set_parameter_value("video_refer_type", "feature")
    node.set_parameter_value("aspect_ratio", "adaptive")

    payload = asyncio.run(node._build_payload())

    assert [(item["type"], item.get("id")) for item in payload["contents"]] == [
        ("prompt", None),
        ("first_frame", "first_frame"),
        ("refer_image", "image_1"),
        ("last_frame", "last_frame"),
        ("feature_video", "video_1"),
        ("element", "element_1"),
        ("element", "element_2"),
        ("voice", "voice_1"),
    ]
    assert payload["contents"][4]["url"] == "https://public/v.mp4"
    assert payload["settings"] == {"resolution": "1080p", "duration": 5, "aspect_ratio": "adaptive", "audio": "native"}


def test_omni_base_video_is_sent_as_base_video() -> None:
    node = _omni_with_references("kling-4.0")
    node.set_parameter_value("video_refer_type", "base")

    payload = asyncio.run(node._build_payload())

    assert "base_video" in [item["type"] for item in payload["contents"]]


def test_omni_flash_drops_voices_and_last_frame() -> None:
    node = _omni_with_references("kling-4.0-flash")

    payload = asyncio.run(node._build_payload())

    types = [item["type"] for item in payload["contents"]]
    assert "voice" not in types
    assert "last_frame" not in types
    assert payload["settings"]["resolution"] == "720p"


def test_omni_original_audio_needs_video_and_no_voices() -> None:
    node = _omni_with_references("kling-4.0")
    node.set_parameter_value("audio", "original")
    assert _validation_messages(node) == ["Audio 'original' cannot be combined with voice IDs."]

    node.set_parameter_value("voice_ids", "")
    node.set_parameter_value("reference_video", "")
    assert _validation_messages(node) == ["Audio 'original' keeps the reference video's audio; connect a video."]


def test_omni_counts_reference_limits() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("voice_ids", "1,2,3,4")
    node.set_parameter_value("element_ids", ",".join(str(i) for i in range(8)))

    messages = _validation_messages(node)
    assert "Kling 4.0 accepts up to 3 voices (got 4)." in messages
    assert "Kling 4.0 accepts up to 7 elements (got 8)." in messages


def test_omni_switching_to_v3_restores_legacy_controls() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("aspect_ratio", "adaptive")
    node.set_parameter_value("model_name", "kling-v3-omni")

    assert {"resolution", "audio", "voice_ids"} <= _hidden(node)
    assert "mode" not in _hidden(node)
    assert node.get_parameter_value("aspect_ratio") == "16:9"


def test_prompt_length_follows_mixed_language_rule() -> None:
    assert kling_4.prompt_length_error("a" * 26_000) is None
    assert kling_4.prompt_length_error("a" * 26_001) is not None
    assert kling_4.prompt_length_error("字" * 11_000) is None
    assert kling_4.prompt_length_error("a" * 13_000 + "字" * 5_501) is not None


def test_video_id_is_read_from_both_response_shapes() -> None:
    result = {"data": [{"id": "t", "outputs": [{"type": "audio"}, {"type": "video", "id": "vid-1", "url": "u"}]}]}
    assert kling_4.kling_video_id(result) == "vid-1"
    assert kling_4.kling_video_id({"data": {"task_result": {"videos": [{"id": "legacy-1"}]}}}) == "legacy-1"
    assert kling_4.kling_video_id({"data": {"task_result": {}}}) is None


def test_image_keyframe_that_fails_to_load_raises_instead_of_shifting_ids() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "go")
    node.set_parameter_value("image", _image())
    node.set_parameter_value("middle_frames", ["/no/such/frame.png", _image()])
    node.set_parameter_value("image_tail", _image())

    with pytest.raises(ValueError, match="Middle Frames"):
        asyncio.run(node._build_payload())


def test_image_keyframe_ids_follow_input_position_without_an_end_frame() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "go")
    node.set_parameter_value("image", _image())
    node.set_parameter_value("middle_frames", [_image(), _image()])

    payload = asyncio.run(node._build_payload())

    assert [(item["type"], item.get("id")) for item in payload["contents"][1:]] == [
        ("first_frame", "frame_1"),
        ("middle_frame", "frame_2"),
        ("middle_frame", "frame_3"),
    ]


def test_omni_reference_image_that_fails_to_load_raises() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("reference_images", ["/no/such/ref.png"])

    with pytest.raises(ValueError, match="reference images"):
        asyncio.run(node._build_payload())


def test_resolution_helpers() -> None:
    assert kling_4.resolution_choice("kling-4.0-flash", "1080p") == "720p"
    assert kling_4.resolution_choice("kling-4.0", "720p") == "720p"
    assert kling_4.resolution_error("kling-4.0", "1080p") is None
    assert kling_4.resolution_error("kling-4.0-flash", "1080p") == (
        "kling-4.0-flash does not support resolution '1080p'. Use 720p."
    )


def test_text_rejects_a_duration_the_model_does_not_offer() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")
    node.set_parameter_value("model_name", "kling-4.0-flash")
    node.set_parameter_value("prompt", "a cat")
    node.parameter_values["duration"] = 25  # e.g. a saved workflow or an upstream connection

    assert node.get_parameter_value("duration") == 25
    assert _validation_messages(node) == ["kling-4.0-flash supports durations of 3-20s (got 25s)."]


def test_omni_flash_rejects_voice_ids_instead_of_dropping_them() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0-flash")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("voice_ids", "829837252279803904")

    assert _validation_messages(node) == ["kling-4.0-flash does not take voice IDs; use Kling 4.0 for voices."]


def test_omni_defaults_to_adaptive_on_kling_4_and_keeps_a_chosen_ratio() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    assert node.get_parameter_value("aspect_ratio") == "16:9"

    node.set_parameter_value("model_name", "kling-4.0")
    assert node.get_parameter_value("aspect_ratio") == "adaptive"

    node.set_parameter_value("aspect_ratio", "9:16")
    node.set_parameter_value("model_name", "kling-4.0-flash")
    assert node.get_parameter_value("aspect_ratio") == "9:16"


def test_omni_saved_16_9_survives_reload_on_kling_4() -> None:
    # Saved workflows restore parameters in declaration order: model_name, then aspect_ratio.
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("aspect_ratio", "16:9")

    assert node.get_parameter_value("aspect_ratio") == "16:9"


@pytest.mark.parametrize(
    "node_class", [KlingTextToVideoGeneration, KlingImageToVideoGeneration, KlingOmniVideoGeneration]
)
def test_common_validation_is_shared_and_strips_the_prompt(node_class: type) -> None:
    node = node_class(name="Kling")
    node.set_parameter_value("model_name", "kling-4.0-flash")
    node.set_parameter_value("prompt", "   ")
    node.parameter_values["duration"] = 25

    messages = _validation_messages(node)

    assert "A prompt is required to generate video. Enter one in 'prompt'." in messages
    assert "kling-4.0-flash supports durations of 3-20s (got 25s)." in messages
    assert messages.count("A prompt is required to generate video. Enter one in 'prompt'.") == 1


def test_validate_common_reports_each_shared_check() -> None:
    errors = [
        str(error) for error in kling_4.validate_common("kling-4.0-flash", prompt="", resolution="1080p", duration=2)
    ]
    assert errors == [
        "A prompt is required to generate video. Enter one in 'prompt'.",
        "kling-4.0-flash does not support resolution '1080p'. Use 720p.",
        "kling-4.0-flash supports durations of 3-20s (got 2s).",
    ]
