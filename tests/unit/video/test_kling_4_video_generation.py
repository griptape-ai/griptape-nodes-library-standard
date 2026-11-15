from __future__ import annotations

import asyncio
import base64
from typing import Any
from unittest.mock import AsyncMock

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import ParameterList

from griptape_nodes_library.media.public_urls import PublicListItemUrl
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

    # Populated middle frames stay visible so the user can clear them; the empty end frame hides.
    assert "middle_frames" not in _hidden(node)
    assert "image_tail" in _hidden(node)
    assert _validation_messages(node) == [
        "kling-4.0-flash supports a start frame only. Clear 'End Frame' and 'Middle Frames', "
        "or select Kling 4.0 to use them."
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

    assert _validation_messages(node) == [
        "kling-4.0-flash does not take voice IDs. Clear 'voice_ids', or select Kling 4.0 to use them."
    ]


def test_omni_switching_to_flash_keeps_populated_unsupported_inputs_visible_until_cleared() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("voice_ids", "829837252279803904")
    node.set_parameter_value("end_frame_image", _image())

    node.set_parameter_value("model_name", "kling-4.0-flash")

    # Still visible and still rejected, so the user can see what blocks the run.
    assert not ({"voice_ids", "end_frame_image"} & _hidden(node))
    assert _validation_messages(node) == [
        "kling-4.0-flash does not take an end frame. Clear 'end frame', or select Kling 4.0 to use it.",
        "kling-4.0-flash does not take voice IDs. Clear 'voice_ids', or select Kling 4.0 to use them.",
    ]

    node.set_parameter_value("voice_ids", "")
    node.set_parameter_value("end_frame_image", None)

    assert {"voice_ids", "end_frame_image"} <= _hidden(node)


def test_image_switching_to_flash_keeps_populated_keyframes_visible_until_cleared() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("image_tail", _image())

    node.set_parameter_value("model_name", "kling-4.0-flash")
    assert "image_tail" not in _hidden(node)

    node.set_parameter_value("image_tail", None)
    assert "image_tail" in _hidden(node)


def test_omni_keeps_an_explicit_16_9_when_switching_between_kling_4_models() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("aspect_ratio", "16:9")

    node.set_parameter_value("model_name", "kling-4.0-flash")

    assert node.get_parameter_value("aspect_ratio") == "16:9"


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


def test_image_switching_to_v3_keeps_populated_middle_frames_visible_and_rejected() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "go")
    node.set_parameter_value("image", _image())
    node.set_parameter_value("middle_frames", [_image()])

    node.set_parameter_value("model_name", "kling-v3")

    # The v1-v3 payload has no middle frames, so the run is blocked rather than dropping them.
    assert "middle_frames" not in _hidden(node)
    assert _validation_messages(node) == [
        "kling-v3 does not take middle frames. Clear 'Middle Frames', or select Kling 4.0 to use them."
    ]

    node.set_parameter_value("middle_frames", [])

    assert "middle_frames" in _hidden(node)
    assert _validation_messages(node) == []


def test_image_v3_hides_empty_middle_frames() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("model_name", "kling-v3")

    assert "middle_frames" in _hidden(node)


def test_omni_switching_to_v3_keeps_populated_kling_4_inputs_visible_and_rejected() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "@voice_1 says hello")
    node.set_parameter_value("voice_ids", "829837252279803904")
    node.set_parameter_value("audio", "original")

    node.set_parameter_value("model_name", "kling-v3-omni")

    assert not ({"voice_ids", "audio"} & _hidden(node))
    assert "resolution" in _hidden(node)
    assert _validation_messages(node) == [
        "kling-v3-omni does not take voice IDs. Clear 'voice_ids', or select Kling 4.0 to use them.",
        "kling-v3-omni does not take 'audio'. Set it back to 'native' and use 'video_keep_sound' "
        "to keep the reference video's audio, or select Kling 4.0.",
    ]

    node.set_parameter_value("voice_ids", "")
    node.set_parameter_value("audio", "native")

    assert {"voice_ids", "audio"} <= _hidden(node)
    assert _validation_messages(node) == []


def _omni_with_videos(model_id: str, count: int) -> KlingOmniVideoGeneration:
    """`reference_video` plus `count - 1` additional videos, with uploads stubbed to https://public/v<n>.mp4."""
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", model_id)
    node.set_parameter_value("prompt", "move like @video_1, framed like @video_2")
    node.set_parameter_value("reference_video", "https://example.com/v1.mp4")
    node._public_video_url_parameter.aget_public_url_for_parameter = AsyncMock(  # type: ignore[method-assign]
        return_value="https://public/v1.mp4"
    )
    node.set_parameter_value(
        "additional_reference_videos", [f"https://example.com/v{number}.mp4" for number in range(2, count + 1)]
    )
    return node


@pytest.fixture(autouse=True)
def _stub_list_item_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The additional videos are already public URLs, so the real helper returns them untouched.

    Swap the public URL for the one the tests expect, without any network.
    """

    async def fake_public_url(self: PublicListItemUrl) -> str:
        return str(self._value).replace("https://example.com", "https://public")

    monkeypatch.setattr(PublicListItemUrl, "aget_public_url_for_parameter", fake_public_url)
    monkeypatch.setattr(PublicListItemUrl, "adelete_uploaded_artifact", AsyncMock())


@pytest.mark.parametrize("model_id", ["kling-4.0", "kling-4.0-flash"])
def test_omni_sends_up_to_five_videos_with_one_base(model_id: str) -> None:
    node = _omni_with_videos(model_id, 5)
    node.set_parameter_value("video_refer_type", "base")

    payload = asyncio.run(node._build_payload())

    videos = [item for item in payload["contents"] if item["type"].endswith("_video")]
    assert [(item["type"], item["id"], item["url"]) for item in videos] == [
        ("base_video", "video_1", "https://public/v1.mp4"),
        *[("feature_video", f"video_{n}", f"https://public/v{n}.mp4") for n in range(2, 6)],
    ]
    assert _validation_messages(node) == []


def test_omni_additional_videos_are_a_capped_list_beside_reference_video() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    names = [parameter.name for parameter in node.parameters]

    assert names.index("reference_video") < names.index("additional_reference_videos")
    assert names.index("additional_reference_videos") < names.index("element_ids")
    assert names.index("reference_images") < names.index("reference_video")
    additional_videos = node.get_parameter_by_name("additional_reference_videos")
    assert isinstance(additional_videos, ParameterList)
    assert additional_videos.max_items == kling_4.MAX_VIDEOS - 1


def test_omni_shows_the_additional_videos_on_kling_4_without_waiting_for_a_connection() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")

    assert "additional_reference_videos" not in _hidden(node)


def test_omni_counts_every_video_toward_the_reference_total() -> None:
    node = _omni_with_videos("kling-4.0", 5)
    node.set_parameter_value("element_ids", "1,2,3,4,5,6,7")
    node.set_parameter_value("voice_ids", "1,2,3")
    assert _validation_messages(node) == []

    node.set_parameter_value("reference_images", [_image()])
    assert _validation_messages(node) == [
        "Kling 4.0 accepts up to 15 images, videos, elements and voices combined (got 16)."
    ]


def test_omni_additional_videos_list_refuses_a_sixth_video() -> None:
    with pytest.raises(ValueError, match="holds at most 4 items"):
        _omni_with_videos("kling-4.0", 6)


def test_omni_rejects_additional_videos_without_a_reference_video() -> None:
    node = _omni_with_videos("kling-4.0", 3)
    node.set_parameter_value("reference_video", None)

    assert _validation_messages(node) == ["'additional_reference_videos' needs 'reference_video' to be set."]


def test_omni_v3_rejects_extra_videos_and_keeps_them_visible() -> None:
    node = _omni_with_videos("kling-4.0", 2)
    node.set_parameter_value("video_refer_type", "feature")
    node.set_parameter_value("model_name", "kling-v3-omni")

    assert "additional_reference_videos" not in _hidden(node)
    assert (
        "kling-v3-omni takes one reference video. Clear 'additional_reference_videos', or select Kling 4.0 to use more."
        in _validation_messages(node)
    )

    node.set_parameter_value("additional_reference_videos", [])
    assert "additional_reference_videos" in _hidden(node)


@pytest.mark.parametrize("run_fails", [False, True])
def test_omni_deletes_every_uploaded_video_after_the_run(*, run_fails: bool) -> None:
    node = _omni_with_videos("kling-4.0", 3)
    first_delete = AsyncMock()
    node._public_video_url_parameter.adelete_uploaded_artifact = first_delete  # type: ignore[method-assign]

    async def build_then_run() -> None:
        await node._build_payload()
        if run_fails:
            msg = "upload of video 2 failed"
            raise RuntimeError(msg)

    node._process_generation = build_then_run  # type: ignore[method-assign]

    if run_fails:
        with pytest.raises(RuntimeError, match="upload of video 2 failed"):
            asyncio.run(node.aprocess())
    else:
        asyncio.run(node.aprocess())

    first_delete.assert_awaited_once()
    assert PublicListItemUrl.adelete_uploaded_artifact.await_count == 2  # type: ignore[attr-defined]


def test_omni_sends_middle_frames_between_the_reference_images_and_the_last_frame() -> None:
    node = _omni_with_references("kling-4.0")
    node.set_parameter_value("video_refer_type", "feature")
    node.set_parameter_value("middle_frames", [_image(), _image()])

    payload = asyncio.run(node._build_payload())

    assert [(item["type"], item.get("id")) for item in payload["contents"][:6]] == [
        ("prompt", None),
        ("first_frame", "first_frame"),
        ("refer_image", "image_1"),
        ("middle_frame", "middle_frame_1"),
        ("middle_frame", "middle_frame_2"),
        ("last_frame", "last_frame"),
    ]
    assert "middle_frames" not in _hidden(node)


def test_omni_counts_middle_frames_toward_the_image_limit() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("first_frame_image", _image())
    node.set_parameter_value("reference_images", [_image()] * 4)
    node.set_parameter_value("middle_frames", [_image()] * 5)

    assert not [message for message in _validation_messages(node) if "accepts up to 10 images" in message]

    node.set_parameter_value("end_frame_image", _image())

    assert "Kling 4.0 accepts up to 10 images (got 11)." in _validation_messages(node)


def test_omni_flash_keeps_populated_middle_frames_visible_and_rejected() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("middle_frames", [_image()])

    node.set_parameter_value("model_name", "kling-4.0-flash")

    assert "middle_frames" not in _hidden(node)
    assert _validation_messages(node) == [
        "kling-4.0-flash does not take middle frames. Clear 'middle frames', or select Kling 4.0 to use them."
    ]
    assert "middle_frame" not in [item["type"] for item in asyncio.run(node._build_payload())["contents"]]

    node.set_parameter_value("middle_frames", [])

    assert "middle_frames" in _hidden(node)


def test_omni_v3_keeps_populated_middle_frames_visible_and_rejected() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    assert "middle_frames" in _hidden(node)

    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("prompt", "x")
    node.set_parameter_value("middle_frames", [_image()])
    node.set_parameter_value("model_name", "kling-v3-omni")

    assert "middle_frames" not in _hidden(node)
    assert (
        "kling-v3-omni does not take middle frames. Clear 'middle frames', or select Kling 4.0 to use them."
        in _validation_messages(node)
    )

    node.set_parameter_value("middle_frames", [])

    assert "middle_frames" in _hidden(node)


def _advanced_group_hidden(node: Any) -> bool:
    group = next(element for element in node.root_ui_element.children if element.name == "Advanced References")
    return bool(group.ui_options.get("hide"))


@pytest.mark.parametrize("model_id", ["kling-v3-omni", "kling-4.0", "kling-4.0-flash"])
def test_omni_hides_element_and_voice_ids_on_the_griptape_key(model_id: str) -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", model_id)

    assert {"element_ids", "voice_ids"} <= _hidden(node)
    assert _advanced_group_hidden(node)


def test_omni_shows_element_and_voice_ids_on_a_customer_key_and_hides_them_again() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")

    node.set_parameter_value("api_key_provider", True)
    assert not ({"element_ids", "voice_ids"} & _hidden(node))
    assert not _advanced_group_hidden(node)

    node.set_parameter_value("api_key_provider", False)
    assert {"element_ids", "voice_ids"} <= _hidden(node)
    assert _advanced_group_hidden(node)


def test_omni_customer_key_shows_voice_ids_only_where_the_model_takes_them() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("api_key_provider", True)

    for model_id, voices_hidden in (("kling-v3-omni", True), ("kling-4.0-flash", True), ("kling-4.0", False)):
        node.set_parameter_value("model_name", model_id)
        assert "element_ids" not in _hidden(node)
        assert ("voice_ids" in _hidden(node)) is voices_hidden


def test_omni_keeps_populated_element_ids_visible_on_the_griptape_key() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "kling-4.0")
    node.set_parameter_value("element_ids", "11, 22")

    assert "element_ids" not in _hidden(node)
    assert "voice_ids" in _hidden(node)
    assert not _advanced_group_hidden(node)

    node.set_parameter_value("element_ids", "")
    assert "element_ids" in _hidden(node)
    assert _advanced_group_hidden(node)


def test_omni_prompt_placeholder_follows_the_reference_syntax_of_the_model() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    prompt = node.get_parameter_by_name("prompt")
    assert prompt is not None
    assert "<<<image_1>>>" in prompt.ui_options["placeholder_text"]

    node.set_parameter_value("model_name", "kling-4.0")
    assert "@image_1" in prompt.ui_options["placeholder_text"]
    assert "<<<" not in prompt.ui_options["placeholder_text"]

    node.set_parameter_value("model_name", "kling-v3-omni")
    assert "<<<image_1>>>" in prompt.ui_options["placeholder_text"]


def test_omni_tooltips_name_the_ids_the_payload_sends() -> None:
    node = _omni_with_references("kling-4.0")
    node.set_parameter_value("middle_frames", [_image()])
    payload = asyncio.run(node._build_payload())
    sent_ids = {item["id"] for item in payload["contents"] if item.get("id")}

    tooltips = {
        name: node.get_parameter_by_name(name).tooltip  # type: ignore[union-attr]
        for name in ("prompt", "first_frame_image", "end_frame_image", "middle_frames")
    }
    assert "@first_frame" in tooltips["first_frame_image"]
    assert "@last_frame" in tooltips["end_frame_image"]
    assert "@middle_frame_1" in tooltips["middle_frames"]
    for token in ("first_frame", "last_frame", "middle_frame_1", "image_1", "video_1", "element_1", "voice_1"):
        assert token in sent_ids
        assert f"@{token}" in tooltips["prompt"]
