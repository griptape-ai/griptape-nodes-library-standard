from __future__ import annotations

from pathlib import Path

import pytest
from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_types import parameter_image
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    create_provider_asset_reference,
)
from griptape_nodes_library.video.seedance_2_0_video_generation import (
    INPUT_MODE_MULTIMODAL_REFERENCE_LIST,
    INPUT_MODE_MULTIMODAL_REFERENCES,
    SEEDANCE_2_0_FAST_MODEL_ID,
    SEEDANCE_2_0_MINI_MODEL_ID,
    SEEDANCE_2_0_MODEL_ID,
    SEEDANCE_MODEL_CAPABILITIES,
    Seedance20VideoGeneration,
    _normalize_audio_data_uri_subtype,
)


def _set_parameter_list_values(node: Seedance20VideoGeneration, parameter_name: str, values: list[object]) -> None:
    parameter = _parameter_by_name(node, parameter_name)
    # reference_media is a plain list-typed Parameter (whole-list connection input), so set the
    # value directly the way a connection delivers it. Real ParameterLists (reference_images /
    # reference_audio) populate individual child slots instead.
    if not isinstance(parameter, ParameterList):
        node.set_parameter_value(parameter_name, list(values))
        return
    parameter.clear_list()
    for value in values:
        child = parameter.add_child_parameter()
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


@pytest.mark.parametrize("model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_0_FAST_MODEL_ID, SEEDANCE_2_0_MINI_MODEL_ID])
def test_all_models_support_last_frame(model_id: str) -> None:
    # Per the BytePlus capability matrix, first+last frame i2v is supported by all three variants.
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
    node.set_parameter_value("reference_video_1", "https://example.com/reference.mp4")

    with pytest.raises(ValueError, match="only used in Multimodal References mode"):
        node._validate_parameters(node._get_parameters())


def test_multimodal_reference_video_inputs_progressively_appear() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")

    assert _parameter_by_name(node, "reference_video_1").hide is False
    assert _parameter_by_name(node, "reference_video_2").hide is True
    assert _parameter_by_name(node, "reference_video_3").hide is True

    node.set_parameter_value("reference_video_1", "https://example.com/reference-1.mp4")
    assert _parameter_by_name(node, "reference_video_2").hide is False
    assert _parameter_by_name(node, "reference_video_3").hide is True

    node.set_parameter_value("reference_video_2", "https://example.com/reference-2.mp4")
    assert _parameter_by_name(node, "reference_video_3").hide is False


def test_multimodal_reference_video_inputs_require_contiguous_order() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("reference_video_2", "https://example.com/reference-2.mp4")

    with pytest.raises(ValueError, match="reference_video_2 requires reference_video_1"):
        node._validate_parameters(node._get_parameters())


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
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/reference.mp4"))
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
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/reference.mp4"))

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
    node.set_parameter_value("reference_video_1", {"type": "VideoUrlArtifact", "value": str(reference_video)})

    with pytest.raises(
        ValueError, match="reference_video_1 only supports public URLs, uploaded asset URLs, or asset:// IDs"
    ):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_uses_public_artifact_url_parameter_for_reference_videos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    node.set_parameter_value("reference_video_1", "workspace/reference.mp4")

    monkeypatch.setattr(
        node._public_reference_video_parameter_1,
        "get_public_url_for_parameter",
        lambda: "https://public.example/reference.mp4",
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


# --- Private-asset reference gating (Seedance 2.0 only) -------------------------------------


@pytest.mark.parametrize("model_id", [SEEDANCE_2_0_MODEL_ID, SEEDANCE_2_0_FAST_MODEL_ID, SEEDANCE_2_0_MINI_MODEL_ID])
def test_all_models_support_private_assets(model_id: str) -> None:
    # The GTC backend links provider assets for all Seedance 2.0 variant ids, so the node allows
    # private-asset references on all three.
    assert Seedance20VideoGeneration._supports_private_assets(model_id) is True


# --- 4k resolution gating (Seedance 2.0 only) ----------------------------------------------


def test_supports_4k_only_for_seedance_2_0() -> None:
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_MODEL_ID) is True
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_FAST_MODEL_ID) is False
    assert Seedance20VideoGeneration._supports_4k(SEEDANCE_2_0_MINI_MODEL_ID) is False


def test_capability_table_matches_documented_matrix() -> None:
    # Regression guard on the single source of truth: values mirror the BytePlus capability matrix.
    # All three variants support last_frame and private assets; only resolution ceiling differs.
    standard = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_MODEL_ID]
    fast = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_FAST_MODEL_ID]
    mini = SEEDANCE_MODEL_CAPABILITIES[SEEDANCE_2_0_MINI_MODEL_ID]

    assert standard.resolutions == ("480p", "720p", "1080p", "4k")
    assert fast.resolutions == ("480p", "720p")
    assert mini.resolutions == ("480p", "720p")

    for caps in (standard, fast, mini):
        assert caps.supports_last_frame is True
        assert caps.supports_private_assets is True


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


@pytest.mark.parametrize("model_name", ["Seedance 2.0", "Seedance 2.0 Fast", "Seedance 2.0 Mini"])
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
    node.set_parameter_value(
        "reference_video_1",
        create_provider_asset_reference(value="https://public.example/clip.wav", asset_kind=ASSET_KIND_AUDIO),
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
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/reference.mp4"))

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
    node.set_parameter_value("reference_video_1", VideoUrlArtifact("https://public.example/reference.mp4"))

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


# --- _classify_media_item tests ---------------------------------------------------------------


def test_classify_image_url_artifact() -> None:
    assert Seedance20VideoGeneration._classify_media_item(ImageUrlArtifact("https://ex.com/a.png")) == ASSET_KIND_IMAGE


def test_classify_image_artifact() -> None:
    assert (
        Seedance20VideoGeneration._classify_media_item(ImageArtifact(b"\x89PNG", format="png", width=1, height=1))
        == ASSET_KIND_IMAGE
    )


def test_classify_video_url_artifact() -> None:
    assert Seedance20VideoGeneration._classify_media_item(VideoUrlArtifact("https://ex.com/v.mp4")) == ASSET_KIND_VIDEO


def test_classify_audio_url_artifact() -> None:
    assert Seedance20VideoGeneration._classify_media_item(AudioUrlArtifact("https://ex.com/a.wav")) == ASSET_KIND_AUDIO


def test_classify_audio_artifact() -> None:
    assert Seedance20VideoGeneration._classify_media_item(AudioArtifact(b"\x00", format="wav")) == ASSET_KIND_AUDIO


def test_classify_provider_asset_image() -> None:
    ref = create_provider_asset_reference(value="https://ex.com/portrait.png", asset_kind=ASSET_KIND_IMAGE)
    assert Seedance20VideoGeneration._classify_media_item(ref) == ASSET_KIND_IMAGE


def test_classify_provider_asset_video() -> None:
    ref = create_provider_asset_reference(value="https://ex.com/clip.mp4", asset_kind=ASSET_KIND_VIDEO)
    assert Seedance20VideoGeneration._classify_media_item(ref) == ASSET_KIND_VIDEO


def test_classify_provider_asset_audio() -> None:
    ref = create_provider_asset_reference(value="https://ex.com/clip.mp3", asset_kind=ASSET_KIND_AUDIO)
    assert Seedance20VideoGeneration._classify_media_item(ref) == ASSET_KIND_AUDIO


@pytest.mark.parametrize(
    ("url", "expected_kind"),
    [
        ("https://ex.com/clip.mp3", ASSET_KIND_AUDIO),
        ("https://ex.com/clip.wav", ASSET_KIND_AUDIO),
        ("https://ex.com/vid.mp4", ASSET_KIND_VIDEO),
        ("https://ex.com/vid.mov", ASSET_KIND_VIDEO),
        ("https://ex.com/photo.jpg", ASSET_KIND_IMAGE),
        ("https://ex.com/photo.png", ASSET_KIND_IMAGE),
        ("https://ex.com/photo.webp", ASSET_KIND_IMAGE),
    ],
)
def test_classify_string_by_extension(url: str, expected_kind: str) -> None:
    assert Seedance20VideoGeneration._classify_media_item(url) == expected_kind


def test_classify_string_with_query_params() -> None:
    assert Seedance20VideoGeneration._classify_media_item("https://ex.com/vid.mp4?token=abc") == ASSET_KIND_VIDEO


def test_classify_unknown_string_falls_back_to_image() -> None:
    assert Seedance20VideoGeneration._classify_media_item("https://ex.com/unknown") == ASSET_KIND_IMAGE


def test_classify_ogg_not_treated_as_seedance_audio() -> None:
    # Seedance only accepts mp3/wav; .ogg should NOT classify as audio
    assert Seedance20VideoGeneration._classify_media_item("https://ex.com/clip.ogg") != ASSET_KIND_AUDIO


def test_classify_none_falls_back_to_image() -> None:
    assert Seedance20VideoGeneration._classify_media_item(None) == ASSET_KIND_IMAGE


# --- _split_media_list tests ------------------------------------------------------------------


def test_split_media_list_empty() -> None:
    images, videos, audio = Seedance20VideoGeneration._split_media_list([])
    assert images == []
    assert videos == []
    assert audio == []


def test_split_media_list_mixed() -> None:
    items = [
        ImageUrlArtifact("https://ex.com/a.png"),
        VideoUrlArtifact("https://ex.com/v.mp4"),
        AudioUrlArtifact("https://ex.com/a.wav"),
        ImageUrlArtifact("https://ex.com/b.png"),
    ]
    images, videos, audio = Seedance20VideoGeneration._split_media_list(items)
    assert len(images) == 2
    assert len(videos) == 1
    assert len(audio) == 1


def test_split_media_list_strings() -> None:
    items = [
        "https://ex.com/photo.jpg",
        "https://ex.com/clip.mp4",
        "https://ex.com/sound.mp3",
    ]
    images, videos, audio = Seedance20VideoGeneration._split_media_list(items)
    assert len(images) == 1
    assert len(videos) == 1
    assert len(audio) == 1


# --- Multimodal Reference List mode: validation tests ----------------------------------------


def test_reference_list_mode_rejects_first_frame() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("first_frame", "data:image/png;base64,AAA")

    with pytest.raises(ValueError, match="only used in First/Last Frame mode"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_individual_reference_images() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(node, "reference_images", [ImageUrlArtifact("https://ex.com/a.png")])

    with pytest.raises(ValueError, match="not used in.*Multimodal Reference List.*mode"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_individual_reference_audio() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(node, "reference_audio", [AudioUrlArtifact("https://ex.com/a.wav")])

    with pytest.raises(ValueError, match="not used in.*Multimodal Reference List.*mode"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_individual_reference_video() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("reference_video_1", "https://ex.com/v.mp4")

    with pytest.raises(ValueError, match="not used in.*Multimodal Reference List.*mode"):
        node._validate_parameters(node._get_parameters())


def test_multimodal_references_mode_rejects_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCES)
    _set_parameter_list_values(node, "reference_media", [ImageUrlArtifact("https://ex.com/a.png")])

    with pytest.raises(ValueError, match="reference_media is not used in.*Multimodal References.*mode"):
        node._validate_parameters(node._get_parameters())


def test_text_only_mode_rejects_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", "Text Only")
    _set_parameter_list_values(node, "reference_media", [ImageUrlArtifact("https://ex.com/a.png")])

    with pytest.raises(ValueError, match="does not accept any media inputs"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_audio_alone() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(node, "reference_media", [AudioUrlArtifact("https://ex.com/a.wav")])

    with pytest.raises(ValueError, match="requires at least one reference image or video"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_more_than_9_images() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(
        node, "reference_media", [ImageUrlArtifact(f"https://ex.com/{i}.png") for i in range(10)]
    )

    with pytest.raises(ValueError, match="up to 9 reference images"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_more_than_3_videos() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(node, "reference_media", [VideoUrlArtifact(f"https://ex.com/{i}.mp4") for i in range(4)])

    with pytest.raises(ValueError, match="up to 3 reference videos"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_rejects_more_than_3_audio() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(
        node,
        "reference_media",
        [ImageUrlArtifact("https://ex.com/img.png")] + [AudioUrlArtifact(f"https://ex.com/{i}.wav") for i in range(4)],
    )

    with pytest.raises(ValueError, match="up to 3 reference audio"):
        node._validate_parameters(node._get_parameters())


def test_reference_list_mode_accepts_valid_mixed_list() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(
        node,
        "reference_media",
        [
            ImageUrlArtifact("https://ex.com/a.png"),
            VideoUrlArtifact("https://ex.com/v.mp4"),
            AudioUrlArtifact("https://ex.com/a.wav"),
        ],
    )

    # Should not raise
    node._validate_parameters(node._get_parameters())


# --- Multimodal Reference List mode: visibility tests ----------------------------------------


def test_reference_list_mode_shows_only_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)

    assert _parameter_by_name(node, "reference_media").hide is False
    assert _parameter_by_name(node, "first_frame").hide is True
    assert _parameter_by_name(node, "last_frame").hide is True
    assert _parameter_by_name(node, "reference_images").hide is True
    assert _parameter_by_name(node, "reference_audio").hide is True
    assert _parameter_by_name(node, "reference_video_1").hide is True


def test_other_modes_hide_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")

    for mode in ["Text Only", "First/Last Frame", "Multimodal References"]:
        node.set_parameter_value("input_mode", mode)
        assert _parameter_by_name(node, "reference_media").hide is True, f"reference_media should be hidden in {mode}"


def test_input_mode_choices_include_reference_list() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    input_mode_param = _parameter_by_name(node, "input_mode")
    choices = input_mode_param.find_elements_by_type(Options)[0].choices
    assert INPUT_MODE_MULTIMODAL_REFERENCE_LIST in choices


# --- Multimodal Reference List mode: whole-list connection acceptance (#488) -----------------


def test_reference_media_accepts_whole_list_connection() -> None:
    """The core #488 requirement: a list-producing node (output type bare ``list``) must be
    wireable into reference_media. It is a plain list-typed Parameter, so the connection
    type-check (Parameter.is_incoming_type_allowed, the same gate the engine's FlowManager uses)
    accepts any list-shaped source — bare ``list`` and typed ``list[X]`` alike.
    """
    node = Seedance20VideoGeneration(name="Seedance20")
    reference_media = _parameter_by_name(node, "reference_media")

    # CreateList / CreateImageList expose an ``output`` of type ``list``.
    assert reference_media.is_incoming_type_allowed("list")
    # Typed list outputs still connect.
    assert reference_media.is_incoming_type_allowed("list[ImageUrlArtifact]")
    assert reference_media.is_incoming_type_allowed("list[any]")
    # A single (non-list) artifact is not accepted — this mode takes a whole list.
    assert not reference_media.is_incoming_type_allowed("ImageUrlArtifact")


def test_reference_media_whole_list_value_round_trips() -> None:
    """A connection delivers the whole list via set_parameter_value; the node must read it back
    intact (a ParameterList would collapse it to an empty list — the bug behind #488)."""
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    incoming = [ImageUrlArtifact("https://ex.com/a.png"), AudioUrlArtifact("https://ex.com/b.wav")]
    node.set_parameter_value("reference_media", incoming)

    assert node.get_parameter_value("reference_media") == incoming
    assert node._get_parameters()["reference_media"] == incoming


def test_reference_media_has_media_upload_badge() -> None:
    """Parity with the per-slot reference-video inputs: reference_media carries a Media Upload badge
    warning that local videos are uploaded to a short-lived public URL."""
    node = Seedance20VideoGeneration(name="Seedance20")
    badge = _parameter_by_name(node, "reference_media").get_badge()
    assert badge is not None
    assert badge.title == "Media Upload"
    assert badge.variant == "cloud-upload"


# --- Multimodal Reference List mode: _iter_reference_asset_checks ----------------------------


def test_iter_reference_asset_checks_includes_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    img_ref = create_provider_asset_reference(value="https://ex.com/img.png", asset_kind=ASSET_KIND_IMAGE)
    vid_ref = create_provider_asset_reference(value="https://ex.com/vid.mp4", asset_kind=ASSET_KIND_VIDEO)
    params = node._get_parameters()
    params["reference_media"] = [img_ref, vid_ref]

    checks = node._iter_reference_asset_checks(params)
    assert (img_ref, ASSET_KIND_IMAGE) in checks
    assert (vid_ref, ASSET_KIND_VIDEO) in checks


# --- Multimodal Reference List mode: _get_parameters includes reference_media ----------------


def test_get_parameters_includes_reference_media() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    _set_parameter_list_values(
        node,
        "reference_media",
        [ImageUrlArtifact("https://ex.com/a.png")],
    )

    params = node._get_parameters()
    assert "reference_media" in params
    assert len(params["reference_media"]) == 1


# --- Multimodal Reference List mode: payload building ----------------------------------------


@pytest.mark.asyncio
async def test_build_payload_reference_list_images_and_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("prompt", "Animate the references")

    async def fake_aread_data_uri(self: File, fallback_mime: str = "application/octet-stream") -> str:
        return "data:image/png;base64,VALID_IMAGE"

    monkeypatch.setattr(File, "aread_data_uri", fake_aread_data_uri)

    _set_parameter_list_values(
        node,
        "reference_media",
        [
            ImageUrlArtifact("https://ex.com/photo.png"),
            AudioUrlArtifact("data:audio/wav;base64,RAW_AUDIO_BASE64"),
        ],
    )

    payload = await node._build_payload()

    image_entries = [item for item in payload["content"] if item["type"] == "image_url"]
    audio_entries = [item for item in payload["content"] if item["type"] == "audio_url"]

    assert len(image_entries) == 1
    assert image_entries[0]["role"] == "reference_image"
    assert len(audio_entries) == 1
    assert audio_entries[0]["role"] == "reference_audio"


@pytest.mark.asyncio
async def test_build_payload_reference_list_video_url(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("prompt", "Use the reference video")
    _set_parameter_list_values(
        node,
        "reference_media",
        [VideoUrlArtifact("https://public.example/reference.mp4")],
    )

    payload = await node._build_payload()

    video_entries = [item for item in payload["content"] if item["type"] == "video_url"]
    assert len(video_entries) == 1
    assert video_entries[0] == {
        "type": "video_url",
        "video_url": {"url": "https://public.example/reference.mp4"},
        "role": "reference_video",
    }


@pytest.mark.asyncio
async def test_build_payload_reference_list_uploads_local_video(monkeypatch: pytest.MonkeyPatch) -> None:
    # Parity with Multimodal References: a local (non-public) video in the list is uploaded to a
    # short-lived public URL rather than rejected. The upload goes through a transient
    # PublicArtifactUrlParameter tracked for cleanup, mirroring the private-asset path.
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("prompt", "Animate")

    monkeypatch.setattr(
        PublicArtifactUrlParameter,
        "get_public_url_for_parameter",
        lambda self: "https://public.example/uploaded.mp4",
    )
    monkeypatch.setattr(PublicArtifactUrlParameter, "delete_uploaded_artifact", lambda self: None)

    _set_parameter_list_values(
        node,
        "reference_media",
        [VideoUrlArtifact("/local/path/video.mp4")],
    )

    payload = await node._build_payload()

    video_entries = [item for item in payload["content"] if item["type"] == "video_url"]
    assert video_entries == [
        {"type": "video_url", "video_url": {"url": "https://public.example/uploaded.mp4"}, "role": "reference_video"}
    ]
    # A transient scratch upload parameter was minted (and will be cleaned up by _process_generation).
    assert [name for _, name in node._pending_asset_uploads], "expected a scratch upload parameter"


@pytest.mark.asyncio
async def test_build_payload_reference_list_local_video_upload_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("prompt", "Animate")

    def _boom(self: PublicArtifactUrlParameter) -> str:
        raise RuntimeError("upload failed")

    monkeypatch.setattr(PublicArtifactUrlParameter, "get_public_url_for_parameter", _boom)
    monkeypatch.setattr(PublicArtifactUrlParameter, "delete_uploaded_artifact", lambda self: None)

    _set_parameter_list_values(
        node,
        "reference_media",
        [VideoUrlArtifact("/local/path/video.mp4")],
    )

    with pytest.raises(ValueError, match="could not be resolved to a public URL"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_build_payload_reference_list_private_asset(monkeypatch: pytest.MonkeyPatch) -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    node.set_parameter_value("prompt", "Animate the portrait")

    registered: list[tuple[str, str]] = []

    async def fake_create_provider_asset(self, public_url: str, asset_kind: str, headers: dict[str, str]) -> str:
        registered.append((public_url, asset_kind))
        return "generated-asset-id"

    monkeypatch.setattr(Seedance20VideoGeneration, "_create_provider_asset", fake_create_provider_asset)
    monkeypatch.setattr(Seedance20VideoGeneration, "_validate_api_key", lambda self: "test-key")

    _set_parameter_list_values(
        node,
        "reference_media",
        [create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)],
    )

    payload = await node._build_payload()

    assert registered == [("https://public.example/portrait.png", ASSET_KIND_IMAGE)]
    assert payload["content"] == [
        {"type": "text", "text": "Animate the portrait"},
        {
            "type": "image_url",
            "image_url": {"url": "asset://generated-asset-id"},
            "role": "reference_image",
        },
    ]


def test_reference_list_mode_validates_private_asset_kind_mismatch() -> None:
    node = Seedance20VideoGeneration(name="Seedance20")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", INPUT_MODE_MULTIMODAL_REFERENCE_LIST)
    # Audio asset reference classified as audio, but then tested against its own kind —
    # should validate fine since the classifier matches
    audio_ref = create_provider_asset_reference(value="https://ex.com/clip.wav", asset_kind=ASSET_KIND_AUDIO)
    img_ref = create_provider_asset_reference(value="https://ex.com/portrait.png", asset_kind=ASSET_KIND_IMAGE)
    _set_parameter_list_values(node, "reference_media", [img_ref, audio_ref])

    # Should not raise — kinds match their classified types
    node._validate_parameters(node._get_parameters())
