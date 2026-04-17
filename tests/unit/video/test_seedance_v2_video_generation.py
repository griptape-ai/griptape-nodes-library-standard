from __future__ import annotations

from pathlib import Path

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.files.file import File

import griptape_nodes_library.video.seedance_v2_video_generation as seedance_v2_module
from griptape_nodes_library.video.seedance_v2_video_generation import SeedanceV2VideoGeneration


def _set_parameter_list_values(node: SeedanceV2VideoGeneration, parameter_name: str, values: list[object]) -> None:
    parameter_list = next(
        parameter for parameter in node.parameters if isinstance(parameter, ParameterList) and parameter.name == parameter_name
    )
    parameter_list.clear_list()
    for value in values:
        child = parameter_list.add_child_parameter()
        node.set_parameter_value(child.name, value)


@pytest.mark.asyncio
async def test_build_payload_normalizes_local_frame_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
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

    monkeypatch.setattr(seedance_v2_module, "normalize_artifact_input", fake_normalize_artifact_input)

    node.set_parameter_value("model_id", "Seedance 2.0")
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


def test_seedance_2_fast_rejects_last_frame() -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    node.set_parameter_value("model_id", "Seedance 2.0 Fast")
    node.set_parameter_value("last_frame", "data:image/png;base64,AAA")

    with pytest.raises(ValueError, match="does not support last_frame"):
        node._validate_parameters(node._get_parameters())


def test_multimodal_mode_rejects_first_last_frame_inputs() -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("first_frame", "data:image/png;base64,AAA")

    with pytest.raises(ValueError, match="only used in First/Last Frame mode"):
        node._validate_parameters(node._get_parameters())


def test_frame_inputs_remain_input_only() -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")

    first_frame_parameter = next(parameter for parameter in node.parameters if parameter.name == "first_frame")
    last_frame_parameter = next(parameter for parameter in node.parameters if parameter.name == "last_frame")

    assert first_frame_parameter.allowed_modes == {ParameterMode.INPUT}
    assert last_frame_parameter.allowed_modes == {ParameterMode.INPUT}


def test_first_last_frame_mode_rejects_multimodal_reference_inputs() -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    node.set_parameter_value("input_mode", "First/Last Frame")
    _set_parameter_list_values(node, "reference_videos", ["https://example.com/reference.mp4"])

    with pytest.raises(ValueError, match="only used in Multimodal References mode"):
        node._validate_parameters(node._get_parameters())


@pytest.mark.asyncio
async def test_build_payload_accepts_serialized_image_artifact_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    node.set_parameter_value("model_id", "Seedance 2.0")
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
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    frame_path = tmp_path / "first.png"
    frame_path.write_bytes(b"frame")

    node.set_parameter_value("model_id", "Seedance 2.0")
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
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(
        node,
        "reference_videos",
        [{"type": "VideoUrlArtifact", "value": "https://public.example/reference.mp4"}],
    )
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


@pytest.mark.asyncio
async def test_build_payload_rejects_local_reference_video_path(tmp_path) -> None:
    node = SeedanceV2VideoGeneration(name="SeedanceV2")
    reference_video = tmp_path / "reference.mp4"
    reference_video.write_bytes(b"video")

    node.set_parameter_value("model_id", "Seedance 2.0")
    node.set_parameter_value("input_mode", "Multimodal References")
    node.set_parameter_value("prompt", "Use the reference video motion")
    _set_parameter_list_values(node, "reference_videos", [str(reference_video)])

    with pytest.raises(ValueError, match="reference_videos only support public URLs or asset:// IDs"):
        await node._build_payload()
