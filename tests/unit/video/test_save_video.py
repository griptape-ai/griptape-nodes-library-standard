from __future__ import annotations

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.retained_mode.events.artifact_events import (
    CheckArtifactReadPermissionRequest,
    CheckArtifactReadPermissionResultFailure,
    CheckArtifactReadPermissionResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial, CheckpointFailure

import griptape_nodes_library.video.save_video as save_video_module
from griptape_nodes_library.video.save_video import SaveVideo


def _stub_handle_request(monkeypatch: pytest.MonkeyPatch, result: object) -> list[CheckArtifactReadPermissionRequest]:
    """Replace ``GriptapeNodes.handle_request`` and capture the requests it was sent."""
    sent: list[CheckArtifactReadPermissionRequest] = []

    def fake_handle_request(request: CheckArtifactReadPermissionRequest) -> object:
        sent.append(request)
        return result

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(fake_handle_request))
    return sent


class _FileCalls:
    """Records how the stubbed ``File`` was exercised."""

    def __init__(self) -> None:
        self.constructed: list[str] = []
        self.resolved: list[str] = []
        self.read: list[str] = []


def _stub_file(
    monkeypatch: pytest.MonkeyPatch,
    *,
    data: bytes = b"fake video bytes",
    resolved_path: str | None = None,
) -> _FileCalls:
    """Replace ``File`` with a fake that records construction, resolve, and read.

    ``resolve()`` returns ``resolved_path`` when provided -- simulating a macro
    path (e.g. ``{inputs}/videos/clip.mp4``) resolving to an absolute path --
    otherwise it echoes the constructor argument, matching how the real
    ``File.resolve()`` passes plain paths and URLs through unchanged.
    """
    calls = _FileCalls()

    class _FakeFile:
        def __init__(self, url: str) -> None:
            self._url = url
            calls.constructed.append(url)

        def resolve(self) -> str:
            calls.resolved.append(self._url)
            return resolved_path if resolved_path is not None else self._url

        def read_bytes(self) -> bytes:
            calls.read.append(self._url)
            return data

    monkeypatch.setattr(save_video_module, "File", _FakeFile)
    return calls


def test_normalize_input_loads_bytes_when_read_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SaveVideo(name="Save")
    sent = _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=None, result_details="Read allowed."),
    )
    calls = _stub_file(monkeypatch, data=b"video-bytes")

    video_input = node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert video_input.data == b"video-bytes"
    assert video_input.source_url == "https://example.com/input.mp4"
    assert sent == [CheckArtifactReadPermissionRequest(source_path="https://example.com/input.mp4")]
    assert calls.read == ["https://example.com/input.mp4"]


def test_normalize_input_checks_resolved_macro_path_not_the_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    """The read check must receive the resolved absolute path, not the raw macro.

    A workspace artifact carries a macro value like ``{inputs}/videos/clip.mp4``.
    The engine ffprobes ``source_path`` directly, so the macro has to be resolved
    first or the probe fails and the check denies even permitted codecs.
    """
    node = SaveVideo(name="Save")
    sent = _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=None, result_details="Read allowed."),
    )
    calls = _stub_file(
        monkeypatch,
        data=b"video-bytes",
        resolved_path="/abs/workspace/inputs/videos/clip.mp4",
    )

    video_input = node._normalize_input(VideoUrlArtifact(value="{inputs}/videos/clip.mp4"))

    assert sent == [CheckArtifactReadPermissionRequest(source_path="/abs/workspace/inputs/videos/clip.mp4")]
    assert calls.read == ["{inputs}/videos/clip.mp4"]
    # The human-facing source_url keeps the portable macro form.
    assert video_input.source_url == "{inputs}/videos/clip.mp4"


def test_normalize_input_raises_on_denial_without_reading_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SaveVideo(name="Save")
    denial = CheckpointDenial(failures=(CheckpointFailure(detail="Codec not permitted by license policy."),))
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=denial, result_details="Read denied."),
    )
    calls = _stub_file(monkeypatch)

    with pytest.raises(ValueError, match="Codec not permitted by license policy"):
        node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert calls.read == []


def test_normalize_input_raises_when_permission_check_fails_without_reading_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SaveVideo(name="Save")
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultFailure(result_details="Attempted to check read permission. Failed."),
    )
    calls = _stub_file(monkeypatch, data=b"video-bytes")

    with pytest.raises(ValueError, match="video codec permission check failed"):
        node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert calls.read == []
