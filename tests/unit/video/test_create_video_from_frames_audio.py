"""Create Video from Frames resolves the audio input through ``File`` and reports resolution failures.

The node used to fall back to a helper that doesn't understand macro paths, handing FFmpeg a broken
``<workspace>/{outputs}/...`` path and failing with an unrelated error.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.files import file as file_module
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

from griptape_nodes_library.video.create_video_from_frames import CreateVideoFromFrames


def _node() -> CreateVideoFromFrames:
    node = CreateVideoFromFrames.__new__(CreateVideoFromFrames)
    node.name = "create_video_from_frames"
    return node


@pytest.mark.parametrize(
    "audio_input",
    [
        AudioUrlArtifact("{outputs}/track.mp3"),
        {"type": "AudioUrlArtifact", "value": "{outputs}/track.mp3"},
        "{outputs}/track.mp3",
    ],
    ids=["audio_url_artifact", "serialized_dict", "bare_string"],
)
def test_audio_macro_path_is_resolved_via_file(monkeypatch: pytest.MonkeyPatch, audio_input: Any) -> None:
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "resolve", lambda _self: "/project/outputs/track.mp3")

    assert _node()._extract_audio_url(audio_input) == "/project/outputs/track.mp3"
    assert paths == ["{outputs}/track.mp3"]


def test_audio_resolution_failure_is_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_resolve(self: file_module.File) -> str:  # noqa: ARG001
        raise file_module.FileLoadError(FileIOFailureReason.MISSING_MACRO_VARIABLES, "no project loaded")

    monkeypatch.setattr(file_module.File, "resolve", fail_resolve)

    with pytest.raises(file_module.FileLoadError, match="no project loaded"):
        _node()._extract_audio_url(AudioUrlArtifact("{outputs}/track.mp3"))
