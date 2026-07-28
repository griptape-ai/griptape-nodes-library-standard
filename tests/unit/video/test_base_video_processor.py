from __future__ import annotations

import pytest
from griptape_nodes.retained_mode.events.artifact_events import (
    CheckArtifactReadPermissionRequest,
    CheckArtifactReadPermissionResultFailure,
    CheckArtifactReadPermissionResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial, CheckpointFailure

from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor


class _StubVideoProcessor(BaseVideoProcessor):
    """Minimal concrete subclass so the base class's read-permission gate can be tested without a real ffmpeg pipeline."""

    def _setup_custom_parameters(self) -> None:
        pass

    def _get_processing_description(self) -> str:
        return "stub processing"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        return []


def _stub_handle_request(monkeypatch: pytest.MonkeyPatch, result: object) -> list[CheckArtifactReadPermissionRequest]:
    """Replace ``GriptapeNodes.handle_request`` and capture the requests it was sent."""
    sent: list[CheckArtifactReadPermissionRequest] = []

    def fake_handle_request(request: CheckArtifactReadPermissionRequest) -> object:
        sent.append(request)
        return result

    monkeypatch.setattr(GriptapeNodes, "handle_request", staticmethod(fake_handle_request))
    return sent


def test_check_read_codec_permission_allows_when_no_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=None, result_details="Read allowed for 'input.mp4'."),
    )
    node = _StubVideoProcessor(name="Stub")

    node._check_read_codec_permission("/tmp/input.mp4")

    assert sent == [CheckArtifactReadPermissionRequest(source_path="/tmp/input.mp4")]


def test_check_read_codec_permission_raises_on_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    denial = CheckpointDenial(failures=(CheckpointFailure(detail="Codec not permitted by license policy."),))
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultSuccess(denial=denial, result_details="Read denied."),
    )
    node = _StubVideoProcessor(name="Stub")

    with pytest.raises(ValueError, match="Codec not permitted by license policy"):
        node._check_read_codec_permission("/tmp/input.mp4")


def test_check_read_codec_permission_raises_when_permission_check_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_handle_request(
        monkeypatch,
        CheckArtifactReadPermissionResultFailure(result_details="Attempted to check read permission. Failed."),
    )
    node = _StubVideoProcessor(name="Stub")

    with pytest.raises(ValueError, match="video codec permission check failed"):
        node._check_read_codec_permission("/tmp/input.mp4")
