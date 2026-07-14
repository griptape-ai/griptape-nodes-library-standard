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


def _stub_file_read_bytes(monkeypatch: pytest.MonkeyPatch, data: bytes = b"fake video bytes") -> list[str]:
    """Replace ``File`` with a fake that records the URL it was asked to read."""
    read_urls: list[str] = []

    class _FakeFile:
        def __init__(self, url: str) -> None:
            read_urls.append(url)

        def read_bytes(self) -> bytes:
            return data

    monkeypatch.setattr(save_video_module, "File", _FakeFile)
    return read_urls


def test_normalize_input_loads_bytes_when_read_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SaveVideo(name="Save")
    sent = _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=None, result_details="Read allowed."),
    )
    read_urls = _stub_file_read_bytes(monkeypatch, data=b"video-bytes")

    video_input = node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert video_input.data == b"video-bytes"
    assert video_input.source_url == "https://example.com/input.mp4"
    assert sent == [CheckArtifactReadPermissionRequest(source_path="https://example.com/input.mp4")]
    assert read_urls == ["https://example.com/input.mp4"]


def test_normalize_input_raises_on_denial_without_reading_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    node = SaveVideo(name="Save")
    denial = CheckpointDenial(failures=(CheckpointFailure(detail="Codec not permitted by license policy."),))
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=denial, result_details="Read denied."),
    )
    read_urls = _stub_file_read_bytes(monkeypatch)

    with pytest.raises(ValueError, match="Codec not permitted by license policy"):
        node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert read_urls == []


def test_normalize_input_raises_when_permission_check_fails_without_reading_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = SaveVideo(name="Save")
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultFailure(result_details="Attempted to check read permission. Failed."),
    )
    read_urls = _stub_file_read_bytes(monkeypatch, data=b"video-bytes")

    with pytest.raises(ValueError, match="video codec permission check failed"):
        node._normalize_input(VideoUrlArtifact(value="https://example.com/input.mp4"))

    assert read_urls == []
