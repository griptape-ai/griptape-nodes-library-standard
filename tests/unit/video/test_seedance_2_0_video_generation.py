from __future__ import annotations

from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode, Trait
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_types import parameter_image
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    create_provider_asset_reference,
)
from griptape_nodes_library.video.seedance_2_0_video_generation import (
    LEGACY_REFERENCE_VIDEO_PARAMETERS,
    MAX_REFERENCE_VIDEOS,
    REFERENCE_VIDEOS_PARAMETER,
    SEEDANCE_2_0_FAST_MODEL_ID,
    SEEDANCE_2_0_MINI_MODEL_ID,
    SEEDANCE_2_0_MODEL_ID,
    SEEDANCE_2_5_MODEL_ID,
    SEEDANCE_MODEL_CAPABILITIES,
    Seedance20VideoGeneration,
    _normalize_audio_data_uri_subtype,
)


def _set_parameter_list_values(node: Seedance20VideoGeneration, parameter_name: str, values: list[object]) -> None:
    parameter_list = next(
        parameter
        for parameter in node.parameters
        if isinstance(parameter, ParameterList) and parameter.name == parameter_name
    )
    parameter_list.clear_list()
    for value in values:
        child = parameter_list.add_child_parameter()
        node.set_parameter_value(child.name, value)


def _parameter_by_name(node: Seedance20VideoGeneration, parameter_name: str):
    return next(parameter for parameter in node.parameters if parameter.name == parameter_name)


@pytest.mark.asyncio
async def test_build_payload_normalizes_local_frame_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    normalization_calls: list[str] = []
    first_frame = tmp_path / "first.png"
    last_frame = tmp_path / "last.png"
    first_frame.write_bytes(b"first-frame")
    last_frame.write_bytes(b"last-frame")

    def fake_normalize_artifact_input(value, artifact_type, *, accepted_types=None):
        if isinstance(value, str) and value.endswith(".png"):
            normalization_calls.append(value)
            return ImageUrlArtifact(f"https://example.com/{Path(value).name}")
        return value

    monkeypatch.setattr(parameter_image, "normalize_artifact_input", fake_normalize_artifact_input)

    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "First/Last Frame")
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value("first_frame", str(first_frame))
    node.set_parameter_value("last_frame", str(last_frame))

    async def fake_aread_data_uri(self: File, fallback_mime: str = "application/octet-stream") -> str:
        return "data:image/png;base64,VALID_IMAGE"

    monkeypatch.setattr(File, "aread_data_uri", fake_aread_data_uri)

    payload = await node._build_payload()
    frame_entries = [item for item in payload["content"] if item["type"] == "image_url"]

    assert normalization_calls == [str(first_frame), str(last_frame)]
    assert frame_entries == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,VALID_IMAGE"}, "role": "first_frame"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,VALID_IMAGE"}, "role": "last_frame"},
    ]
    assert all(str(first_frame) not in item["image_url"]["url"] for item in frame_entries)
    assert all(str(last_frame) not in item["image_url"]["url"] for item in frame_entries)


@pytest.mark.parametrize(
    "model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_0_FAST_MODEL_ID, SEEDANCE_2_0_MINI_MODEL_ID, SEEDANCE_2_5_MODEL_ID]
)
def test_all_models_support_last_frame(model_id: str) -> None:
    # Per the BytePlus capability matrix, first+last frame i2v is supported by all four variants.
    assert Seedance20VideoGeneration._supports_last_frame(model_id) is True


@pytest.mark.parametrize("model_name", ["Seedance 2.0 Fast", "Seedance 2.0 Mini"])
def test_fast_and_mini_accept_last_frame(model_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", model_name)
    node.set_parameter_value("input_mode", "First/Last Frame")
    node.set_parameter_value("last_frame", "data:image/png;base64,AAA")

    # Should validate without raising now that Fast/Mini support last_frame.
    node._validate_parameters(node._get_parameters())


def test_multimodal_mode_rejects_first_last_frame_inputs() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("first_frame", "data:image/png;base64,AAA")

    with pytest.raises(ValueError, match="only used in First/Last Frame mode"):
        node._validate_parameters(node._get_parameters())


def test_frame_inputs_remain_input_only() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    first_frame_parameter = next(parameter for parameter in node.parameters if parameter.name == "first_frame")
    last_frame_parameter = next(parameter for parameter in node.parameters if parameter.name == "last_frame")

    assert first_frame_parameter.allowed_modes == {ParameterMode.INPUT}
    assert last_frame_parameter.allowed_modes == {ParameterMode.INPUT}


def test_first_last_frame_mode_rejects_multimodal_reference_inputs() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "First/Last Frame")
    _set_parameter_list_values(node, "reference_videos", ["https://example.com/reference.mp4"])

    with pytest.raises(ValueError, match="only used in Multimodal References mode"):
        node._validate_parameters(node._get_parameters())


def test_reference_videos_is_an_input_only_list() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    reference_videos = _parameter_by_name(node, "reference_videos")

    assert isinstance(reference_videos, ParameterList)
    assert reference_videos._max_items == MAX_REFERENCE_VIDEOS
    assert reference_videos.allowed_modes == {ParameterMode.INPUT}


@pytest.mark.asyncio
async def test_build_payload_accepts_serialized_image_artifact_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "First/Last Frame")
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value(
        "first_frame",
        {"type": "ImageArtifact", "value": "RAW_IMAGE_BASE64", "format": "png", "width": 1, "height": 1},
    )

    async def fail_if_called(self: File, fallback_mime: str = "application/octet-stream") -> str:
        raise AssertionError("File.aread_data_uri should not be used for inline image artifact dicts")

    monkeypatch.setattr(File, "aread_data_uri", fail_if_called)

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "A fox runs through a forest"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,RAW_IMAGE_BASE64"},
            "role": "first_frame",
        },
    ]


@pytest.mark.asyncio
async def test_build_payload_accepts_image_url_artifact_with_file_path_value(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    frame_path = tmp_path / "first.png"
    frame_path.write_bytes(b"frame")

    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "First/Last Frame")
    node.set_parameter_value("prompt", "A fox runs through a forest")
    node.set_parameter_value("first_frame", ImageUrlArtifact(str(frame_path)))

    async def fake_aread_data_uri(self: File, fallback_mime: str = "application/octet-stream") -> str:
        return "data:image/png;base64,VALID_IMAGE"

    monkeypatch.setattr(File, "aread_data_uri", fake_aread_data_uri)

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "A fox runs through a forest"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,VALID_IMAGE"},
            "role": "first_frame",
        },
    ]


@pytest.mark.asyncio
async def test_build_payload_includes_multimodal_video_url_and_audio_base64() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])
    _set_parameter_list_values(
        node,
        "reference_audio",
        [{"type": "AudioArtifact", "value": "RAW_AUDIO_BASE64", "format": "wav"}],
    )

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "Use the reference video motion"},
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/reference.mp4"},
            "role": "reference_video",
        },
        {
            "type": "audio_url",
            "audio_url": {"url": "data:audio/wav;base64,RAW_AUDIO_BASE64"},
            "role": "reference_audio",
        },
    ]


@pytest.mark.parametrize(
    ("data_uri", "expected"),
    [
        # mimetypes resolves .mp3 -> audio/mpeg and .wav -> audio/x-wav; Seedance only accepts
        # audio/mp3 and audio/wav, so these aliases must be rewritten.
        ("data:audio/mpeg;base64,AAA", "data:audio/mp3;base64,AAA"),
        ("data:audio/x-wav;base64,BBB", "data:audio/wav;base64,BBB"),
        # Already-accepted subtypes pass through unchanged.
        ("data:audio/mp3;base64,CCC", "data:audio/mp3;base64,CCC"),
        ("data:audio/wav;base64,DDD", "data:audio/wav;base64,DDD"),
        # Non-audio data URIs and plain URLs are left alone.
        ("data:image/png;base64,EEE", "data:image/png;base64,EEE"),
        ("https://public.example/clip.mp3", "https://public.example/clip.mp3"),
    ],
)
def test_normalize_audio_data_uri_subtype(data_uri: str, expected: str) -> None:
    assert _normalize_audio_data_uri_subtype(data_uri) == expected


@pytest.mark.asyncio
async def test_build_payload_rewrites_mp3_audio_subtype_to_seedance_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    # A connected mp3 file (e.g. ElevenLabs Music output) loads to a data:audio/mpeg URI; the node
    # must rewrite it to data:audio/mp3 so Seedance does not reject it as "Invalid base64 audio_url".
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Mini")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the backing track")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    music = tmp_path / "music.mp3"
    music.write_bytes(b"ID3fakeaudio")
    _set_parameter_list_values(node, "reference_audio", [AudioUrlArtifact(str(music))])

    payload = await node._build_payload()
    audio_entries = [item for item in payload["content"] if item["type"] == "audio_url"]

    assert len(audio_entries) == 1
    assert audio_entries[0]["audio_url"]["url"].startswith("data:audio/mp3;base64,")


@pytest.mark.asyncio
async def test_build_payload_rejects_local_reference_video_path(tmp_path) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    reference_video = tmp_path / "reference.mp4"
    reference_video.write_bytes(b"video")

    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", [{"type": "VideoUrlArtifact", "value": str(reference_video)}])

    with pytest.raises(
        ValueError, match="reference video 1 only supports public URLs, uploaded asset URLs, or asset:// IDs"
    ):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_uploads_reference_video_without_a_public_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", ["workspace/reference.mp4"])

    monkeypatch.setattr(
        PublicArtifactUrlParameter,
        "get_public_url_for_parameter",
        lambda self: "https://public.example/reference.mp4",
    )

    payload = await node._build_payload()

    assert payload["content"] == [
        {"type": "text", "text": "Use the reference video motion"},
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/reference.mp4"},
            "role": "reference_video",
        },
    ]
    # The upload runs through a transient scratch parameter that the run's cleanup tears down.
    assert len(node._pending_asset_uploads) == 1
    scratch_name = node._pending_asset_uploads[0][1]
    assert node.get_parameter_by_name(scratch_name) is not None


# --- Private-asset reference gating (Seedance 2.0 only) -------------------------------------


@pytest.mark.parametrize(
    "model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_0_FAST_MODEL_ID, SEEDANCE_2_0_MINI_MODEL_ID, SEEDANCE_2_5_MODEL_ID]
)
def test_all_models_support_private_assets(model_id: str) -> None:
    # The GTC backend links provider assets for all Seedance variant ids, so the node allows
    # private-asset references on all four.
    assert Seedance20VideoGeneration._supports_private_assets(model_id) is True


# --- 4k resolution gating (Seedance 2.0 only) ----------------------------------------------


def test_supports_4k_only_for_seedance_2_0() -> None:
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_MODEL_ID) is True
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_FAST_MODEL_ID) is False
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_MINI_MODEL_ID) is False
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_5_MODEL_ID) is False


def test_capability_table_matches_documented_matrix() -> None:
    # Regression guard on the single source of truth: values mirror the BytePlus capability matrix.
    # All four variants support last_frame and private assets; resolution ceiling, max_duration,
    # and reference budgets differ (the three 2.0 variants cap duration at 15s and allow up to 9
    # reference images / 3 reference videos / 3 reference audio files; 2.5 extends duration to 30s
    # and raises those budgets to 30 images / 10 videos / 10 audio files).
    standard = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_MODEL_ID]
    fast = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_FAST_MODEL_ID]
    mini = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_MINI_MODEL_ID]
    seedance_2_5 = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_5_MODEL_ID]

    assert standard.resolutions == ("480p", "720p", "1080p", "4k")
    assert fast.resolutions == ("480p", "720p")
    assert mini.resolutions == ("480p", "720p")
    assert seedance_2_5.resolutions == ("480p", "720p")

    for caps in (standard, fast, mini, seedance_2_5):
        assert caps.supports_last_frame is True
        assert caps.supports_private_assets is True

    for caps in (standard, fast, mini):
        assert caps.max_duration == 15
        assert caps.max_reference_images == 9
        assert caps.max_reference_videos == 3
        assert caps.max_reference_audio == 3
    assert seedance_2_5.max_duration == 30
    assert seedance_2_5.max_reference_images == 30
    assert seedance_2_5.max_reference_videos == 10
    assert seedance_2_5.max_reference_audio == 10


# --- Reference image/audio budgets (model-dependent) ----------------------------------------


def test_seedance_2_5_accepts_30_reference_images() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.5")
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(node, "reference_images", [f"https://public.example/img{i}.png" for i in range(30)])

    # 30 reference images validates on Seedance 2.5 (its documented ceiling).
    node._validate_parameters(node._get_parameters())


@pytest.mark.parametrize("model_name", ["Seedance 2.0", "Seedance 2.0 Fast", "Seedance 2.0 Mini"])
def test_seedance_2_0_variants_reject_10_reference_images(model_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", model_name)
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(node, "reference_images", [f"https://public.example/img{i}.png" for i in range(10)])

    with pytest.raises(ValueError, match="supports up to 9 reference images"):
        node._validate_parameters(node._get_parameters())


def test_seedance_2_5_rejects_31_reference_images() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = SEEDANCE_2_5_MODEL_ID
    params["input_mode"] = "Multimodal References"
    params["reference_images"] = [f"https://public.example/img{i}.png" for i in range(31)]

    with pytest.raises(ValueError, match="supports up to 30 reference images"):
        node._validate_parameters(params)


def test_seedance_2_5_accepts_10_reference_audio() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.5")
    node.set_parameter_value("input_mode", "Multimodal References")
    # Audio cannot be used alone, so a reference image is included alongside the audio files.
    _set_parameter_list_values(node, "reference_images", ["https://public.example/portrait.png"])
    _set_parameter_list_values(node, "reference_audio", [f"https://public.example/clip{i}.wav" for i in range(10)])

    node._validate_parameters(node._get_parameters())


@pytest.mark.parametrize("model_name", ["Seedance 2.0", "Seedance 2.0 Fast", "Seedance 2.0 Mini"])
def test_seedance_2_0_variants_reject_4_reference_audio(model_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", model_name)
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(node, "reference_images", ["https://public.example/portrait.png"])
    _set_parameter_list_values(node, "reference_audio", [f"https://public.example/clip{i}.wav" for i in range(4)])

    with pytest.raises(ValueError, match="supports up to 3 reference audio files"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_max_items_equal_the_maximum_across_models() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    reference_images_param = _parameter_by_name(node, "reference_images")
    reference_videos_param = _parameter_by_name(node, "reference_videos")
    reference_audio_param = _parameter_by_name(node, "reference_audio")

    assert reference_images_param._max_items == 30
    assert reference_videos_param._max_items == 10
    assert reference_audio_param._max_items == 10


def test_seedance_2_5_accepts_10_reference_videos() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.5")
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(node, "reference_videos", [f"https://public.example/clip{i}.mp4" for i in range(10)])

    # 10 reference videos validates on Seedance 2.5 (its documented ceiling).
    node._validate_parameters(node._get_parameters())


@pytest.mark.parametrize("model_name", ["Seedance 2.0", "Seedance 2.0 Fast", "Seedance 2.0 Mini"])
def test_seedance_2_0_variants_reject_4_reference_videos(model_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", model_name)
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(node, "reference_videos", [f"https://public.example/clip{i}.mp4" for i in range(4)])

    with pytest.raises(ValueError, match="supports up to 3 reference videos"):
        node._validate_parameters(node._get_parameters())


@pytest.mark.asyncio
async def test_build_payload_truncates_reference_videos_to_the_model_cap() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Fast")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Reference many videos")
    _set_parameter_list_values(node, "reference_videos", [f"https://public.example/clip{i}.mp4" for i in range(5)])

    payload = await node._build_payload()
    reference_video_entries = [item for item in payload["content"] if item.get("role") == "reference_video"]

    # Seedance 2.0 Fast caps reference videos at 3, even though 5 were provided.
    assert len(reference_video_entries) == 3


# --- Legacy reference video slot adoption ---------------------------------------------------


@pytest.mark.parametrize("legacy_parameter_name", LEGACY_REFERENCE_VIDEO_PARAMETERS)
def test_incoming_connection_adopts_a_legacy_slot_as_a_list_child(legacy_parameter_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    node.before_incoming_connection(node, "video", legacy_parameter_name)

    child = node.get_parameter_by_name(legacy_parameter_name)
    assert child is not None
    assert child.parent_container_name == REFERENCE_VIDEOS_PARAMETER
    assert child in _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER).get_child_parameters()
    # The child carries the element type, not the list's list[...] form. A list-typed child
    # rejects every incoming single-artifact connection on a type mismatch.
    assert child.type == "VideoUrlArtifact"
    assert child.input_types == ["VideoUrlArtifact", "BytePlusVideoAssetReference"]


def test_adopting_a_legacy_slot_twice_adds_one_child() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    node.before_incoming_connection(node, "video", "reference_video_1")
    node.before_incoming_connection(node, "video", "reference_video_1")

    assert len(_parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER).get_child_parameters()) == 1


def test_incoming_connection_to_an_unrelated_parameter_adopts_nothing() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    node.before_incoming_connection(node, "video", "first_frame")

    assert _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER).get_child_parameters() == []


def test_value_set_on_an_adopted_legacy_slot_lands_on_the_list() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")

    node.before_incoming_connection(node, "video", "reference_video_1")
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/a.mp4"))

    reference_videos = node._get_parameters()[REFERENCE_VIDEOS_PARAMETER]
    assert [getattr(value, "value", value) for value in reference_videos] == ["https://public.example/a.mp4"]


def test_adoption_is_skipped_once_the_list_is_full() -> None:
    # Adoption runs while a workflow is replaying, so a full list must not raise out of
    # before_incoming_connection and abort the load.
    node = Seedance20VideoGeneration(name="Seedance20")
    reference_videos = _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER)
    for _ in range(MAX_REFERENCE_VIDEOS):
        reference_videos.add_child_parameter()

    node.before_incoming_connection(node, "video", "reference_video_1")

    assert len(reference_videos.get_child_parameters()) == MAX_REFERENCE_VIDEOS
    assert node.get_parameter_by_name("reference_video_1") is None


def test_adopted_legacy_slots_sort_by_slot_number() -> None:
    # A saved workflow replays connections in creation order, so adoption can see the slots out of
    # order. List order decides which video the prompt's `@Video N` references resolve against.
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")

    for legacy_parameter_name in ("reference_video_3", "reference_video_1", "reference_video_2"):
        node.before_incoming_connection(node, "video", legacy_parameter_name)
        node.set_parameter_value(
            legacy_parameter_name, VideoUrlArtifact(f"https://public.example/{legacy_parameter_name}.mp4")
        )

    reference_videos = _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER)
    assert [child.name for child in reference_videos.get_child_parameters()] == list(LEGACY_REFERENCE_VIDEO_PARAMETERS)
    reference_video_values = node._get_parameters()[REFERENCE_VIDEOS_PARAMETER]
    assert [getattr(value, "value", value) for value in reference_video_values] == [
        "https://public.example/reference_video_1.mp4",
        "https://public.example/reference_video_2.mp4",
        "https://public.example/reference_video_3.mp4",
    ]


def test_reorder_announces_the_list_and_only_when_it_moves_children(monkeypatch: pytest.MonkeyPatch) -> None:
    # A per-child event carries no position, so the container's event is what tells a listener the
    # order. It should fire for a reorder that moves children and stay quiet otherwise.
    node = Seedance20VideoGeneration(name="Seedance20")
    announced: list[list[str]] = []
    original_emit = Seedance20VideoGeneration._emit_parameter_lifecycle_event

    def spy(self: Seedance20VideoGeneration, parameter: object, *, remove: bool = False) -> None:
        if getattr(parameter, "name", None) == REFERENCE_VIDEOS_PARAMETER:
            announced.append([child.name for child in parameter.children])
        return original_emit(self, parameter, remove=remove)

    monkeypatch.setattr(Seedance20VideoGeneration, "_emit_parameter_lifecycle_event", spy)

    node.before_incoming_connection(node, "video", "reference_video_3")
    assert announced == []

    node.before_incoming_connection(node, "video", "reference_video_1")
    assert announced == [["reference_video_1", "reference_video_3"]]


def test_adopted_child_matches_an_editor_added_child() -> None:
    # The adopted child is built field by field instead of through add_child_parameter, which takes
    # no name. Everything except the name and the two dropped ui_options must still match, so a
    # field the framework starts copying cannot silently go missing on a migrated slot.
    node = Seedance20VideoGeneration(name="Seedance20")
    reference_videos = _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER)

    editor_child = reference_videos.add_child_parameter()
    node.before_incoming_connection(node, "video", "reference_video_1")
    adopted_child = node.get_parameter_by_name("reference_video_1")

    for attribute in (
        "type",
        "input_types",
        "output_type",
        "default_value",
        "tooltip",
        "tooltip_as_input",
        "tooltip_as_output",
        "tooltip_as_property",
        "allowed_modes",
        "settable",
        "user_defined",
        "parent_container_name",
        "converters",
        "validators",
    ):
        assert getattr(adopted_child, attribute) == getattr(editor_child, attribute), attribute

    assert {type(trait) for trait in adopted_child.find_elements_by_type(Trait)} == {
        type(trait) for trait in editor_child.find_elements_by_type(Trait)
    }
    assert adopted_child.ui_options == {
        key: value for key, value in editor_child.ui_options.items() if key not in {"hide", "display_name"}
    }


def test_adopted_and_editor_added_children_share_the_list() -> None:
    # The editor adds children with generated names; adoption adds one with a legacy slot name.
    # Both land in the same list, in insertion order, and count against the same ceiling.
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")
    reference_videos = _parameter_by_name(node, REFERENCE_VIDEOS_PARAMETER)

    editor_child = reference_videos.add_child_parameter()
    node.before_incoming_connection(node, "video", "reference_video_1")
    node.set_parameter_value(editor_child.name, VideoUrlArtifact("https://public.example/editor.mp4"))
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/legacy.mp4"))

    assert [child.name for child in reference_videos.get_child_parameters()] == [
        editor_child.name,
        "reference_video_1",
    ]
    reference_video_values = node._get_parameters()[REFERENCE_VIDEOS_PARAMETER]
    assert [getattr(value, "value", value) for value in reference_video_values] == [
        "https://public.example/editor.mp4",
        "https://public.example/legacy.mp4",
    ]


@pytest.mark.asyncio
async def test_build_payload_truncates_reference_images_to_the_model_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Fast")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Reference many images")
    _set_parameter_list_values(node, "reference_images", [f"https://public.example/img{i}.png" for i in range(12)])

    async def fake_aread_data_uri(self: File, fallback_mime: str = "application/octet-stream") -> str:
        return "data:image/png;base64,VALID_IMAGE"

    monkeypatch.setattr(File, "aread_data_uri", fake_aread_data_uri)

    payload = await node._build_payload()
    reference_image_entries = [item for item in payload["content"] if item.get("role") == "reference_image"]

    # Seedance 2.0 Fast caps reference images at 9, even though 12 were provided.
    assert len(reference_image_entries) == 9


def test_seedance_2_0_offers_4k_resolution_choice() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node._update_resolution_options(SEEDANCE_2_0_MODEL_ID)

    resolution_param = _parameter_by_name(node, "resolution")
    choices = resolution_param.find_elements_by_type(Options)[0].choices
    assert "4k" in choices
    assert "1080p" in choices


def test_seedance_2_fast_omits_4k_resolution_choice() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Fast")
    node._update_resolution_options(SEEDANCE_2_0_FAST_MODEL_ID)

    resolution_param = _parameter_by_name(node, "resolution")
    choices = resolution_param.find_elements_by_type(Options)[0].choices
    assert "4k" not in choices
    assert "1080p" not in choices
    assert choices == ["480p", "720p"]


def test_seedance_2_mini_omits_4k_resolution_choice() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Mini")
    node._update_resolution_options(SEEDANCE_2_0_MINI_MODEL_ID)

    resolution_param = _parameter_by_name(node, "resolution")
    choices = resolution_param.find_elements_by_type(Options)[0].choices
    assert "4k" not in choices
    assert "1080p" not in choices
    assert choices == ["480p", "720p"]


def test_seedance_2_5_omits_4k_resolution_choice() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.5")
    node._update_resolution_options(SEEDANCE_2_5_MODEL_ID)

    resolution_param = _parameter_by_name(node, "resolution")
    choices = resolution_param.find_elements_by_type(Options)[0].choices
    assert "4k" not in choices
    assert "1080p" not in choices
    assert choices == ["480p", "720p"]


@pytest.mark.parametrize("model_id", [SEEDANCE_2_0_FAST_MODEL_ID, SEEDANCE_2_0_MINI_MODEL_ID])
def test_seedance_2_fast_and_mini_reject_4k_resolution(model_id: str) -> None:
    # The Options trait normally prevents selecting 4k on Fast/Mini via the UI, but resolution can
    # also arrive over an INPUT connection that bypasses the trait — _validate_parameters is the
    # backstop, mirroring the existing 1080p check.
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = model_id
    params["resolution"] = "4k"

    with pytest.raises(ValueError, match="does not support 4k resolution"):
        node._validate_parameters(params)


@pytest.mark.parametrize("resolution", ["1080p", "4k"])
def test_seedance_2_5_rejects_high_resolutions(resolution: str) -> None:
    # Seedance 2.5 tops out at 720p (no 1080p, no 4k), same backstop as Fast/Mini.
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = SEEDANCE_2_5_MODEL_ID
    params["resolution"] = resolution

    with pytest.raises(ValueError, match=f"does not support {resolution} resolution"):
        node._validate_parameters(params)


# --- Duration gating (model-dependent max) --------------------------------------------------


def test_update_duration_options_reflects_selected_model_max() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    node._update_duration_options(SEEDANCE_2_5_MODEL_ID)
    seedance_2_5_choices = _parameter_by_name(node, "duration").find_elements_by_type(Options)[0].choices
    assert 30 in seedance_2_5_choices
    assert 31 not in seedance_2_5_choices

    node._update_duration_options(SEEDANCE_2_0_MODEL_ID)
    seedance_2_0_choices = _parameter_by_name(node, "duration").find_elements_by_type(Options)[0].choices
    assert 15 in seedance_2_0_choices
    assert 16 not in seedance_2_0_choices


def test_seedance_2_5_accepts_duration_up_to_30_but_2_0_does_not() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = SEEDANCE_2_5_MODEL_ID
    params["duration"] = 30

    # 30s validates on Seedance 2.5 (its documented ceiling)...
    node._validate_parameters(params)

    # ...but exceeds the 15s ceiling on Seedance 2.0.
    params["model_id"] = SEEDANCE_2_0_MODEL_ID
    with pytest.raises(ValueError, match="supports duration between"):
        node._validate_parameters(params)


@pytest.mark.parametrize("model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_5_MODEL_ID])
def test_duration_smart_selection_validates_on_all_models(model_id: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = model_id
    params["duration"] = -1

    node._validate_parameters(params)


@pytest.mark.parametrize("model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_5_MODEL_ID])
def test_duration_below_minimum_rejected_on_all_models(model_id: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    params = node._get_parameters()
    params["model_id"] = model_id
    params["duration"] = 3

    with pytest.raises(ValueError, match="supports duration between"):
        node._validate_parameters(params)


def test_switching_from_seedance_2_5_clamps_out_of_range_duration() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.5")
    node.set_parameter_value("duration", 30)
    assert node.get_parameter_value("duration") == 30

    # Switching to Seedance 2.0 (15s max) must clamp the now out-of-range duration rather than
    # leaving an invalid value selected.
    node.set_parameter_value("model_id", "Seedance 2.0")

    assert node.get_parameter_value("duration") != 30
    assert node.get_parameter_value("duration") == 5


@pytest.mark.parametrize("model_name", ["Seedance 2.0", "Seedance 2.0 Fast", "Seedance 2.0 Mini", "Seedance 2.5"])
def test_all_models_accept_private_asset_reference(model_name: str) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", model_name)
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    # Matching kind on any supported model validates without raising.
    node._validate_parameters(node._get_parameters())


def test_seedance_2_0_rejects_private_asset_reference_kind_mismatch() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    # An Audio-kind reference wired into a video input should be rejected.
    _set_parameter_list_values(
        node,
        "reference_videos",
        [create_provider_asset_reference(value="https://public.example/clip.wav", asset_kind=ASSET_KIND_AUDIO)],
    )

    with pytest.raises(ValueError, match="private-asset reference is connected to a Video reference input"):
        node._validate_parameters(node._get_parameters())


def test_seedance_2_0_accepts_matching_private_asset_reference() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    # Matching kind on a supported model validates without raising.
    node._validate_parameters(node._get_parameters())


@pytest.mark.asyncio
async def test_build_payload_registers_private_asset_reference_for_seedance_2_0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Animate the reference portrait")
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    registered: list[tuple[str, str]] = []

    async def fake_create_provider_asset(self, public_url: str, asset_kind: str, headers: dict[str, str]) -> str:
        registered.append((public_url, asset_kind))
        return "generated-asset-id"

    monkeypatch.setattr(Seedance20VideoGeneration, "_create_provider_asset", fake_create_provider_asset)
    monkeypatch.setattr(Seedance20VideoGeneration, "_validate_api_key", lambda self: "test-key")

    payload = await node._build_payload()

    assert registered == [("https://public.example/portrait.png", ASSET_KIND_IMAGE)]
    assert payload["content"] == [
        {"type": "text", "text": "Animate the reference portrait"},
        {
            "type": "image_url",
            "image_url": {"url": "asset://generated-asset-id"},
            "role": "reference_image",
        },
    ]


@pytest.mark.asyncio
async def test_build_payload_does_not_register_assets_for_plain_media(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0 Fast")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("plain (non-private-asset) media must not register a provider asset")

    monkeypatch.setattr(Seedance20VideoGeneration, "_create_provider_asset", fail_if_called)

    payload = await node._build_payload()

    # Plain media (not a private-asset reference) flows through the standard (non-asset) path.
    assert payload["content"] == [
        {"type": "text", "text": "Use the reference video motion"},
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/reference.mp4"},
            "role": "reference_video",
        },
    ]


# --- Private-asset reference gating (Griptape auth only, not BYOK) --------------------------


def test_private_assets_inactive_when_byok_enabled() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")

    # Griptape auth (default): active on Seedance 2.0.
    assert node._private_assets_active(SEEDANCE_2_0_MODEL_ID) is True

    # BYOK (customer key): inactive even on Seedance 2.0.
    node.set_parameter_value("api_key_provider", True)
    assert node._is_byok_enabled() is True
    assert node._private_assets_active(SEEDANCE_2_0_MODEL_ID) is False


def test_byok_rejects_private_asset_reference() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("api_key_provider", True)
    _set_parameter_list_values(
        node,
        "reference_images",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    with pytest.raises(ValueError, match="require Griptape authentication"):
        node._validate_parameters(node._get_parameters())


@pytest.mark.asyncio
async def test_build_payload_does_not_register_assets_when_byok_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("api_key_provider", True)
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", [VideoUrlArtifact("https://public.example/reference.mp4")])

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("BYOK must not register private assets")

    monkeypatch.setattr(Seedance20VideoGeneration, "_create_provider_asset", fail_if_called)

    payload = await node._build_payload()

    # Normal media under BYOK still flows through the standard (non-asset) path unchanged.
    assert payload["content"] == [
        {"type": "text", "text": "Use the reference video motion"},
        {
            "type": "video_url",
            "video_url": {"url": "https://public.example/reference.mp4"},
            "role": "reference_video",
        },
    ]


def test_scratch_upload_parameters_are_removed_after_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    # Registering an asset whose media needs uploading creates a uniquely-named scratch
    # parameter. The cleanup must remove it so parameters don't accumulate across runs.
    node = Seedance20VideoGeneration(name="Seedance20")

    monkeypatch.setattr(
        PublicArtifactUrlParameter,
        "get_public_url_for_parameter",
        lambda self: "https://public.example/uploaded.png",
    )
    monkeypatch.setattr(PublicArtifactUrlParameter, "delete_uploaded_artifact", lambda self: None)

    # A non-public (data URI) value forces the upload path that mints a scratch parameter.
    public_url = node._resolve_public_url_for_asset(
        create_provider_asset_reference(value="data:image/png;base64,AAAA", asset_kind=ASSET_KIND_IMAGE),
        asset_kind=ASSET_KIND_IMAGE,
    )
    assert public_url == "https://public.example/uploaded.png"

    scratch_names = [name for _, name in node._pending_asset_uploads]
    assert scratch_names, "expected a scratch upload parameter to be created"
    assert all(node.get_parameter_by_name(name) is not None for name in scratch_names)

    # Run the cleanup the way _process_generation's finally block does.
    for helper, scratch_name in node._pending_asset_uploads:
        helper.delete_uploaded_artifact()
        node.remove_parameter_element_by_name(scratch_name)

    assert all(node.get_parameter_by_name(name) is None for name in scratch_names)
