"""Tests for ``TranscribeAudio`` targeting the proxy-based architecture."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

import griptape_nodes_library.audio.transcribe_audio as transcribe_audio_module
from griptape_nodes_library.audio.transcribe_audio import TranscribeAudio

DATA_URI = "data:audio/mpeg;base64,QUFB"


def _make_node() -> TranscribeAudio:
    node = TranscribeAudio(name="transcribe")
    node.set_parameter_value("model", "whisper-1")
    return node


@pytest.fixture
def stub_file(monkeypatch: pytest.MonkeyPatch) -> Mock:
    mock_cls = Mock(return_value=Mock(spec=File, aread_data_uri=AsyncMock(return_value=DATA_URI)))
    monkeypatch.setattr(transcribe_audio_module, "File", mock_cls)
    return mock_cls


class TestResolveAudioDataUri:
    """Exercises ``_resolve_audio_data_uri``: the method that turns audio param values
    into base64 data URIs via ``File.aread_data_uri``.
    """

    @pytest.mark.asyncio
    async def test_audio_url_artifact_with_macro_path(self, stub_file: Mock) -> None:
        node = _make_node()
        artifact = AudioUrlArtifact("{outputs}/Extract Audio_output.mp4")

        result = await node._resolve_audio_data_uri(artifact)

        stub_file.assert_called_once_with("{outputs}/Extract Audio_output.mp4")
        stub_file.return_value.aread_data_uri.assert_awaited_once_with(fallback_mime="audio/mpeg")
        assert result == DATA_URI

    @pytest.mark.asyncio
    async def test_audio_url_artifact_with_filesystem_path(self, stub_file: Mock) -> None:
        node = _make_node()
        artifact = AudioUrlArtifact("/abs/path/clip.mp3")

        result = await node._resolve_audio_data_uri(artifact)

        stub_file.assert_called_once_with("/abs/path/clip.mp3")
        stub_file.return_value.aread_data_uri.assert_awaited_once_with(fallback_mime="audio/mpeg")
        assert result == DATA_URI

    @pytest.mark.asyncio
    async def test_audio_url_artifact_with_http_url(self, stub_file: Mock) -> None:
        node = _make_node()
        artifact = AudioUrlArtifact("https://example.com/clip.mp3")

        result = await node._resolve_audio_data_uri(artifact)

        stub_file.assert_called_once_with("https://example.com/clip.mp3")
        stub_file.return_value.aread_data_uri.assert_awaited_once_with(fallback_mime="audio/mpeg")
        assert result == DATA_URI

    @pytest.mark.asyncio
    async def test_dict_serialized_audio_url_artifact(self, stub_file: Mock) -> None:
        node = _make_node()
        audio_dict: dict[str, Any] = {"type": "AudioUrlArtifact", "value": "{outputs}/clip.mp3"}

        result = await node._resolve_audio_data_uri(audio_dict)

        stub_file.assert_called_once_with("{outputs}/clip.mp3")
        stub_file.return_value.aread_data_uri.assert_awaited_once_with(fallback_mime="audio/mpeg")
        assert result == DATA_URI

    @pytest.mark.asyncio
    async def test_file_load_error_returns_none(self, stub_file: Mock) -> None:
        node = _make_node()
        artifact = AudioUrlArtifact("{outputs}/missing.mp3")
        stub_file.return_value.aread_data_uri.side_effect = FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "missing")

        result = await node._resolve_audio_data_uri(artifact)

        assert result is None

    @pytest.mark.asyncio
    async def test_none_audio_returns_none(self) -> None:
        node = _make_node()
        result = await node._resolve_audio_data_uri(None)
        assert result is None

    @pytest.mark.asyncio
    async def test_string_audio_value(self, stub_file: Mock) -> None:
        node = _make_node()

        result = await node._resolve_audio_data_uri("/some/path.wav")

        stub_file.assert_called_once_with("/some/path.wav")
        stub_file.return_value.aread_data_uri.assert_awaited_once_with(fallback_mime="audio/mpeg")
        assert result == DATA_URI

    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self) -> None:
        node = _make_node()
        result = await node._resolve_audio_data_uri("")
        assert result is None


class TestBuildPayload:
    """Exercises ``_build_payload``: constructs the request dict for the proxy."""

    @pytest.mark.asyncio
    async def test_none_audio_raises_value_error(self) -> None:
        node = _make_node()
        node.set_parameter_value("audio", None)

        with pytest.raises(ValueError, match="No audio provided"):
            await node._build_payload()

    @pytest.mark.asyncio
    async def test_failed_audio_load_raises_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_node()
        node.set_parameter_value("audio", AudioUrlArtifact("{outputs}/missing.mp3"))

        monkeypatch.setattr(node, "_resolve_audio_data_uri", AsyncMock(return_value=None))
        with pytest.raises(ValueError, match="Failed to load audio"):
            await node._build_payload()

    @pytest.mark.asyncio
    async def test_minimal_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_node()
        node.set_parameter_value("audio", AudioUrlArtifact("test.mp3"))

        monkeypatch.setattr(node, "_resolve_audio_data_uri", AsyncMock(return_value=DATA_URI))
        payload = await node._build_payload()

        assert payload["file"] == DATA_URI
        assert payload["response_format"] == "json"
        assert "language" not in payload
        assert "prompt" not in payload
        assert "temperature" not in payload

    @pytest.mark.asyncio
    async def test_payload_with_optional_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        node = _make_node()
        node.set_parameter_value("audio", AudioUrlArtifact("test.mp3"))
        node.set_parameter_value("language", "en")
        node.set_parameter_value("prompt", "Meeting notes")
        node.set_parameter_value("response_format", "verbose_json")
        node.set_parameter_value("temperature", 0.5)

        monkeypatch.setattr(node, "_resolve_audio_data_uri", AsyncMock(return_value=DATA_URI))
        payload = await node._build_payload()

        assert payload["language"] == "en"
        assert payload["prompt"] == "Meeting notes"
        assert payload["response_format"] == "verbose_json"
        assert payload["temperature"] == 0.5


class TestParseResult:
    """Exercises ``_parse_result``: extracts transcription text and verbose_json fields."""

    @pytest.mark.asyncio
    async def test_basic_text_output(self) -> None:
        node = _make_node()
        await node._parse_result({"text": "Hello world"}, generation_id="gen-1")

        assert node.parameter_output_values["output"] == "Hello world"

    @pytest.mark.asyncio
    async def test_verbose_json_fields(self) -> None:
        node = _make_node()
        result_json = {
            "text": "Hello",
            "words": [{"word": "Hello", "start": 0.0, "end": 0.5}],
            "segments": [{"id": 0, "text": "Hello"}],
            "language": "en",
            "duration": 1.5,
        }

        await node._parse_result(result_json, generation_id="gen-2")

        assert node.parameter_output_values["output"] == "Hello"
        assert node.parameter_output_values["words"] == [{"word": "Hello", "start": 0.0, "end": 0.5}]
        assert node.parameter_output_values["segments"] == [{"id": 0, "text": "Hello"}]
        assert node.parameter_output_values["detected_language"] == "en"
        assert node.parameter_output_values["duration"] == 1.5

    @pytest.mark.asyncio
    async def test_missing_text_sets_safe_defaults(self) -> None:
        node = _make_node()
        node.parameter_output_values["output"] = "stale"
        node.parameter_output_values["agent"] = "stale"
        node.parameter_output_values["words"] = [{"word": "stale"}]
        node.parameter_output_values["segments"] = [{"id": 0}]
        node.parameter_output_values["detected_language"] = "en"
        node.parameter_output_values["duration"] = 99.0

        await node._parse_result({}, generation_id="gen-3")

        assert node.parameter_output_values["output"] is None
        assert node.parameter_output_values["agent"] is None
        assert node.parameter_output_values["words"] is None
        assert node.parameter_output_values["segments"] is None
        assert node.parameter_output_values["detected_language"] is None
        assert node.parameter_output_values["duration"] is None


class TestSetSafeDefaults:
    """Exercises ``_set_safe_defaults``: clears all output parameters."""

    def test_clears_all_outputs(self) -> None:
        node = _make_node()
        node.parameter_output_values["output"] = "stale"
        node.parameter_output_values["words"] = [{"word": "stale"}]
        node.parameter_output_values["segments"] = [{"id": 0}]
        node.parameter_output_values["detected_language"] = "en"
        node.parameter_output_values["duration"] = 99.0

        node._set_safe_defaults()

        assert node.parameter_output_values["output"] is None
        assert node.parameter_output_values["words"] is None
        assert node.parameter_output_values["segments"] is None
        assert node.parameter_output_values["detected_language"] is None
        assert node.parameter_output_values["duration"] is None
        assert node.parameter_output_values["agent"] is None
