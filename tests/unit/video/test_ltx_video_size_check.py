"""Unit tests for the two-tier video-URI resolution in LTX retake and extend nodes.

Tier 1: upload the video to Griptape Cloud via ``PublicArtifactUrlParameter`` and hand LTX a
        guaranteed-public HTTPS URL it fetches server-side (no base64 inflation → no 413).
Tier 2: fall back to a base64 ``data:`` URI with a pre-flight size check (local dev / no cloud).
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.video import public_video_url_mixin
from griptape_nodes_library.video.ltx_video_extend import LTXVideoExtend
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.public_video_url_mixin import (
    MAX_VIDEO_DATA_URI_SIZE_BYTES,
    decoded_data_uri_size,
)

# Minimal valid base64 payload whose decoded size is well under the limit (~15 bytes).
_SMALL_DATA_URI = "data:video/mp4;base64," + base64.b64encode(b"tiny video data").decode()

# Large fake data URI whose decoded size exceeds the limit.
# Formula: (b64_len // 4) * 3 - padding. 'A' chars carry no '=' padding, and b64_len is a
# multiple of 4, so decoded size == (b64_len // 4) * 3 > MAX_VIDEO_DATA_URI_SIZE_BYTES.
_LARGE_B64_LEN = ((MAX_VIDEO_DATA_URI_SIZE_BYTES // 3) + 1) * 4
_LARGE_DATA_URI = "data:video/mp4;base64," + "A" * _LARGE_B64_LEN

_PUBLIC_URL = "https://storage.griptapecloud.com/bucket/video.mp4?sig=abc123"


# ---------------------------------------------------------------------------
# Shared helpers: decoded_data_uri_size
# ---------------------------------------------------------------------------


def test_decoded_data_uri_size_returns_decoded_byte_count() -> None:
    payload = b"hello world, this is some video bytes"
    data_uri = "data:video/mp4;base64," + base64.b64encode(payload).decode()
    assert decoded_data_uri_size(data_uri) == len(payload)


def test_decoded_data_uri_size_raises_on_malformed_uri() -> None:
    """A data URI with no ',' separator is unmeasurable — must raise, not silently pass."""
    with pytest.raises(ValueError, match="Malformed data URI"):
        decoded_data_uri_size("data:video/mp4;base64-no-comma-here")


# ---------------------------------------------------------------------------
# LTXVideoRetake — base64 fallback path (tier 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retake_rejects_video_exceeding_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    node.set_parameter_value("resolution", "1920x1080")

    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_DATA_URI))

    with pytest.raises(ValueError, match="too large"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_retake_error_message_includes_size_and_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    node.set_parameter_value("resolution", "1920x1080")

    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_DATA_URI))

    with pytest.raises(ValueError) as exc_info:
        await node._build_payload()

    message = str(exc_info.value)
    assert "MB" in message
    assert str(MAX_VIDEO_DATA_URI_SIZE_BYTES // (1024 * 1024)) in message


@pytest.mark.asyncio
async def test_retake_accepts_video_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/small.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    node.set_parameter_value("resolution", "1920x1080")

    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_SMALL_DATA_URI))

    payload = await node._build_payload()
    assert payload["video_uri"] == _SMALL_DATA_URI


# ---------------------------------------------------------------------------
# LTXVideoRetake — public-URL path (tier 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retake_uses_public_url_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    node.set_parameter_value("resolution", "1920x1080")

    prepare_mock = AsyncMock()
    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: _PUBLIC_URL)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", prepare_mock)

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL
    prepare_mock.assert_not_called()  # base64 path skipped entirely


# ---------------------------------------------------------------------------
# LTXVideoExtend — base64 fallback path (tier 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_rejects_video_exceeding_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_DATA_URI))

    with pytest.raises(ValueError, match="too large"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_extend_accepts_video_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/small.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_SMALL_DATA_URI))

    payload = await node._build_payload()
    assert payload["video_uri"] == _SMALL_DATA_URI


# ---------------------------------------------------------------------------
# LTXVideoExtend — public-URL path (tier 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_uses_public_url_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    prepare_mock = AsyncMock()
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: _PUBLIC_URL)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", prepare_mock)

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL
    prepare_mock.assert_not_called()


# ---------------------------------------------------------------------------
# _upload_video_to_public_url — routing + fallback behaviour
# ---------------------------------------------------------------------------


def test_upload_passes_through_remote_https_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-public remote https URL is handed to LTX as-is — no upload attempted."""
    node = LTXVideoRetake(name="Retake")
    node._reset_video_uploads()

    ctor = MagicMock()
    monkeypatch.setattr(public_video_url_mixin, "PublicArtifactUrlParameter", ctor)

    result = node._upload_video_to_public_url(VideoUrlArtifact(_PUBLIC_URL))

    assert result == _PUBLIC_URL
    ctor.assert_not_called()


def test_upload_uploads_localhost_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """A localhost URL is not publicly reachable, so it is uploaded to Griptape Cloud."""
    node = LTXVideoRetake(name="Retake")
    node._reset_video_uploads()

    helper = MagicMock()
    helper.get_public_url_for_parameter.return_value = _PUBLIC_URL
    monkeypatch.setattr(public_video_url_mixin, "PublicArtifactUrlParameter", MagicMock(return_value=helper))
    monkeypatch.setattr(node, "set_parameter_value", lambda *_args, **_kwargs: None)

    result = node._upload_video_to_public_url(VideoUrlArtifact("http://localhost:9999/static/video.mp4?token=abc"))

    assert result == _PUBLIC_URL
    assert node._pending_video_uploads  # tracked for cleanup


def test_upload_falls_back_to_none_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the cloud upload raises, resolution degrades to base64 (returns None), not an error."""
    node = LTXVideoRetake(name="Retake")
    node._reset_video_uploads()

    monkeypatch.setattr(
        public_video_url_mixin,
        "PublicArtifactUrlParameter",
        MagicMock(side_effect=RuntimeError("no cloud creds")),
    )

    result = node._upload_video_to_public_url(VideoUrlArtifact("http://localhost:9999/static/video.mp4"))

    assert result is None
    assert not node._pending_video_uploads


# ---------------------------------------------------------------------------
# _cleanup_video_uploads — deletes artifacts and removes scratch parameters
# ---------------------------------------------------------------------------


def test_cleanup_deletes_artifacts_and_removes_scratch_params() -> None:
    node = LTXVideoRetake(name="Retake")
    node._reset_video_uploads()

    helper = MagicMock()
    node._pending_video_uploads.append((helper, "_video_upload_deadbeef"))

    with patch.object(node, "remove_parameter_element_by_name") as remove_mock:
        node._cleanup_video_uploads()

    helper.delete_uploaded_artifact.assert_called_once()
    remove_mock.assert_called_once_with("_video_upload_deadbeef")
    assert node._pending_video_uploads == []


# ---------------------------------------------------------------------------
# API key validation error — failure edge must propagate
# ---------------------------------------------------------------------------


def test_retake_api_key_error_triggers_failure_exception() -> None:
    """_handle_api_key_validation_error must call _handle_failure_exception so the
    SuccessFailureNode failure edge fires and the user sees the error in the UI."""
    node = LTXVideoRetake(name="Retake")
    err = ValueError("LTX Video Retake is missing GT_CLOUD_API_KEY.")
    with patch.object(node, "_handle_failure_exception") as mock_failure:
        node._handle_api_key_validation_error(err)
    mock_failure.assert_called_once_with(err)


def test_extend_api_key_error_triggers_failure_exception() -> None:
    """Same check for LTXVideoExtend."""
    node = LTXVideoExtend(name="Extend")
    err = ValueError("LTX Video Extend is missing GT_CLOUD_API_KEY.")
    with patch.object(node, "_handle_failure_exception") as mock_failure:
        node._handle_api_key_validation_error(err)
    mock_failure.assert_called_once_with(err)
