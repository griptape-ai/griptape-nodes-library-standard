"""Tests that ``AudioDetails._get_audio_url`` resolves macro paths via ``File``.

The ffprobe subprocess used by ``AudioDetails`` cannot understand project macro
paths like ``{outputs}/clip.mp3`` that upstream nodes emit when they write
through ``ProjectFileParameter``. The fix routes ``AudioUrlArtifact`` values
through ``File(value).resolve()`` so macro paths become absolute paths and
HTTP URLs pass through unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files import file as file_module

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


def test_falsy_audio_returns_none() -> None:
    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    assert node._get_audio_url(None) is None  # pyright: ignore[reportArgumentType]


def test_file_load_error_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

    def fail_resolve(self: file_module.File) -> str:
        raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "boom")

    monkeypatch.setattr(file_module.File, "resolve", fail_resolve)

    node = AudioDetails.__new__(AudioDetails)
    node.name = "audio_details"
    artifact = AudioUrlArtifact("{outputs}/missing.mp3")

    assert node._get_audio_url(artifact) is None
