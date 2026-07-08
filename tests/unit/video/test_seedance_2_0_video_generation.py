from __future__ import annotations

from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact
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
    create_provider_asset_reference,
)
from griptape_nodes_library.video.seedance_2_0_video_generation import (
    SEEDANCE_2_0_FAST_MODEL_ID,
    SEEDANCE_2_0_MINI_MODEL_ID,
    SEEDANCE_2_0_MODEL_ID,
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


@pytest.fixture(autouse=True)
def stub_public_artifact_bucket_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        PublicArtifactUrlParameter, "_get_bucket_id", staticmethod(lambda *_args, **_kwargs: "test-bucket")
    )


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
