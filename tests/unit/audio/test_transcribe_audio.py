"""Tests that ``TranscribeAudio.process`` reads bytes via ``File`` for
``AudioUrlArtifact`` inputs (issue #215).

``AudioUrlArtifact.to_bytes()`` issues an HTTP GET against ``self.value`` and
fails when the value is a project macro path (``{outputs}/Extract Audio.mp4``)
emitted by upstream nodes that write through ``ProjectFileParameter`` (e.g.
``ExtractAudio``). The fix routes ``AudioUrlArtifact`` inputs through
``File(value).read_bytes()`` so macro paths and plain filesystem paths
resolve correctly.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files import file as file_module

import griptape_nodes_library.audio.transcribe_audio as transcribe_audio_module
from griptape_nodes_library.audio.transcribe_audio import TranscribeAudio


@pytest.fixture(autouse=True)
def _stub_external_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace external dependencies (driver, agent, loader, task) with no-ops
    so the test can drive ``process()`` up to the ``File`` call without
    requiring API keys, real audio bytes, or actual transcription.
    """

    class FakeDriver:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    monkeypatch.setattr(transcribe_audio_module, "OpenAiAudioTranscriptionDriver", FakeDriver)

    class FakeAgent:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.output = type("Out", (), {"value": "transcribed-text"})()

        def from_dict(self, _data: dict) -> FakeAgent:
            return self

        def swap_task(self, _task: Any) -> None:
            pass

        def run(self, _tasks: list) -> None:
            pass

        def insert_false_memory(self, **_kwargs: Any) -> None:
            pass

        def restore_task(self) -> None:
            pass

        def to_dict(self) -> dict:
            return {}

    monkeypatch.setattr(transcribe_audio_module, "GtAgent", FakeAgent)

    class FakeLoader:
        def parse(self, _bytes: bytes) -> Any:
            return type("AudioArtifact", (), {})()

    monkeypatch.setattr(transcribe_audio_module, "AudioLoader", FakeLoader)

    monkeypatch.setattr(
        transcribe_audio_module,
        "AudioTranscriptionTask",
        lambda *_args, **_kwargs: object(),
    )


def _drive_process(node: TranscribeAudio) -> None:
    """Drive ``process()`` past its single yield (the agent runner)."""
    result = node.process()
    if result is None:
        return
    try:
        runner = next(result)
    except StopIteration:
        # Early return path (e.g. ``audio is None``) yields nothing.
        return
    runner()
    try:
        next(result)
    except StopIteration:
        pass


def _make_node() -> TranscribeAudio:
    node = TranscribeAudio(name="transcribe")
    node.set_parameter_value("model", "whisper-1")
    return node


def test_audio_url_artifact_with_macro_path_reads_via_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"paths": []}
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        captured["paths"].append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", lambda self: b"AUDIO-BYTES")

    node = _make_node()
    node.set_parameter_value("audio", AudioUrlArtifact("{outputs}/Extract Audio_output.mp4"))

    _drive_process(node)

    assert "{outputs}/Extract Audio_output.mp4" in captured["paths"]


def test_audio_url_artifact_with_filesystem_path_reads_via_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", lambda self: b"AUDIO-BYTES")

    node = _make_node()
    node.set_parameter_value("audio", AudioUrlArtifact("/abs/path/clip.mp3"))

    _drive_process(node)

    assert "/abs/path/clip.mp3" in paths


def test_audio_url_artifact_with_http_url_reads_via_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", lambda self: b"AUDIO-BYTES")

    node = _make_node()
    node.set_parameter_value("audio", AudioUrlArtifact("https://example.com/clip.mp3"))

    _drive_process(node)

    assert "https://example.com/clip.mp3" in paths


def test_dict_serialized_audio_url_artifact_with_macro_path_reads_via_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", lambda self: b"AUDIO-BYTES")

    node = _make_node()
    node.set_parameter_value("audio", {"type": "AudioUrlArtifact", "value": "{outputs}/clip.mp3"})

    _drive_process(node)

    assert "{outputs}/clip.mp3" in paths


def test_file_load_error_is_wrapped_as_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

    def fail_read(self: file_module.File) -> bytes:
        raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "missing")

    monkeypatch.setattr(file_module.File, "read_bytes", fail_read)

    node = _make_node()
    node.set_parameter_value("audio", AudioUrlArtifact("{outputs}/missing.mp3"))

    with pytest.raises(ValueError, match="Failed to read audio from"):
        _drive_process(node)


def test_none_audio_returns_early_without_calling_file(monkeypatch: pytest.MonkeyPatch) -> None:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)

    node = _make_node()
    node.set_parameter_value("audio", None)

    _drive_process(node)

    assert paths == []
    assert node.parameter_output_values.get("output") == "No audio provided"
