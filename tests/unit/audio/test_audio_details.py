"""Tests that ``AudioDetails._get_audio_url`` resolves macro paths via ``File``.

The ffprobe subprocess used by ``AudioDetails`` cannot understand project macro
paths like ``{outputs}/clip.mp3`` that upstream nodes emit when they write
through ``ProjectFileParameter``. The fix routes ``AudioUrlArtifact`` values
through ``File(value).resolve()`` so macro paths become absolute paths and
HTTP URLs pass through unchanged.
"""

from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files import file as file_module

from griptape_nodes_library.audio import audio_details as audio_details_module
from griptape_nodes_library.audio.audio_details import AudioDetails


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    def fake_resolve(self: file_module.File) -> str:
        return f"/resolved{paths[-1]}"

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "resolve", fake_resolve)
    return paths


def test_audio_url_artifact_macro_path_is_resolved_via_file(captured_paths: list[str]) -> None:
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    artifact = AudioUrlArtifact("{outputs}/clip.mp3")

    result = node._get_audio_url(artifact)

    assert result == "/resolved{outputs}/clip.mp3"
    assert "{outputs}/clip.mp3" in captured_paths


def test_audio_url_artifact_filesystem_path_is_resolved_via_file(captured_paths: list[str]) -> None:
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    artifact = AudioUrlArtifact("/abs/path/clip.mp3")

    result = node._get_audio_url(artifact)

    assert result == "/resolved/abs/path/clip.mp3"
    assert "/abs/path/clip.mp3" in captured_paths


def test_audio_url_artifact_http_url_is_resolved_via_file(captured_paths: list[str]) -> None:
    """ffprobe handles HTTP URLs natively; ``File.resolve`` is a pass-through."""
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    artifact = AudioUrlArtifact("https://example.com/clip.mp3")

    result = node._get_audio_url(artifact)

    assert result == "/resolvedhttps://example.com/clip.mp3"
    assert "https://example.com/clip.mp3" in captured_paths


@pytest.mark.parametrize(
    "audio",
    [{"type": "AudioUrlArtifact", "value": "{outputs}/clip.mp3"}, "{outputs}/clip.mp3"],
    ids=["serialized_dict", "bare_string"],
)
def test_unconverted_macro_path_is_resolved_via_file(audio: Any, captured_paths: list[str]) -> None:
    """``set_parameter_value`` runs before the ParameterAudio converter, so these shapes reach the node."""
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"

    result = node._get_audio_url(audio)

    assert result == "/resolved{outputs}/clip.mp3"
    assert captured_paths == ["{outputs}/clip.mp3"]


@pytest.mark.parametrize(
    "audio",
    [{"type": "AudioUrlArtifact", "value": "data:audio/mpeg;base64,SUQz"}, "data:audio/mpeg;base64,SUQz"],
    ids=["serialized_dict", "bare_string"],
)
def test_data_uri_reaches_ffprobe_unchanged(audio: Any) -> None:
    """Uses the real ``File``, which would anchor a data URI under the workspace."""
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"

    assert node._get_audio_url(audio) == "data:audio/mpeg;base64,SUQz"


def test_falsy_audio_returns_none() -> None:
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    assert node._get_audio_url(None) is None


def test_file_load_error_raises_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

    def fail_resolve(self: file_module.File) -> str:
        raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "boom")

    monkeypatch.setattr(file_module.File, "resolve", fail_resolve)

    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    artifact = AudioUrlArtifact("{outputs}/missing.mp3")

    with pytest.raises(ValueError, match="Failed to resolve audio path"):
        node._get_audio_url(artifact)


PROBE_OUTPUT = {
    "format": {"duration": "2.5", "size": "40000", "bit_rate": "128000", "format_name": "mp3"},
    "streams": [{"codec_type": "audio", "sample_rate": "44100", "channels": 2, "codec_name": "mp3"}],
}


def _node() -> AudioDetails:
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    node.parameter_output_values = {}  # pyright: ignore[reportAttributeAccessIssue]
    return node


@pytest.fixture
def bundled_ffprobe(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Point the node at a fake bundled ffprobe and record the commands it runs."""
    commands: list[list[str]] = []
    monkeypatch.setattr(audio_details_module, "get_ffprobe_path", lambda: "/bundled/ffprobe")

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(PROBE_OUTPUT), stderr="")

    monkeypatch.setattr(audio_details_module.subprocess, "run", fake_run)
    return commands


def test_uses_bundled_ffprobe(bundled_ffprobe: list[list[str]]) -> None:
    info = _node()._analyze_audio_with_ffprobe("/abs/clip.mp3")

    assert [cmd[0] for cmd in bundled_ffprobe] == ["/bundled/ffprobe"]
    assert info == {
        "duration": 2.5,
        "format": "mp3",
        "file_size": 40000,
        "sample_rate": 44100,
        "channels": 2,
        "bitrate": 128000,
        "codec": "mp3",
    }


def test_ffprobe_failure_raises_with_stderr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_details_module, "get_ffprobe_path", lambda: "/bundled/ffprobe")

    def fail_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd, stderr="/abs/missing.mp3: No such file or directory")

    monkeypatch.setattr(audio_details_module.subprocess, "run", fail_run)

    with pytest.raises(ValueError, match="No such file or directory"):
        _node()._analyze_audio_with_ffprobe("/abs/missing.mp3")


def test_no_audio_stream_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio_details_module, "get_ffprobe_path", lambda: "/bundled/ffprobe")
    output = {"format": {}, "streams": [{"codec_type": "video"}]}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(output), stderr="")

    monkeypatch.setattr(audio_details_module.subprocess, "run", fake_run)

    with pytest.raises(ValueError, match="No audio stream"):
        _node()._analyze_audio_with_ffprobe("/abs/clip.mp4")


def test_process_fails_when_audio_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    monkeypatch.setattr(node, "get_parameter_value", lambda _name: "/abs/clip.mp3")
    monkeypatch.setattr(audio_details_module, "resolve_media_location", lambda location: location)

    def fail(_url: str) -> dict[str, Any]:
        msg = "ffprobe failed to read the audio: invalid data"
        raise ValueError(msg)

    monkeypatch.setattr(node, "_analyze_audio_with_ffprobe", fail)

    with pytest.raises(ValueError, match="invalid data"):
        node.process()


def test_set_parameter_value_logs_and_resets_when_audio_unreadable(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    monkeypatch.setattr(audio_details_module.DataNode, "set_parameter_value", lambda *_args, **_kwargs: None)
    node.parameter_output_values["duration"] = 9.0

    def fail(_audio: Any) -> None:
        msg = "ffprobe failed to read the audio: invalid data"
        raise ValueError(msg)

    monkeypatch.setattr(node, "_update_audio_details", fail)

    node.set_parameter_value("audio", "/abs/clip.mp3")

    assert node.parameter_output_values["duration"] == 0.0
    assert node.parameter_output_values["codec"] == "UNKNOWN"
