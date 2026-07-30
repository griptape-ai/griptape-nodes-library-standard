"""Unit tests for the two-tier video-URI resolution in LTX retake and extend nodes.

Tier 1: resolve to a public presigned HTTPS URL (cloud path — no 413 risk).
Tier 2: fall back to base64 encoding with a pre-flight size check (local dev).
"""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, patch

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.video.ltx_video_extend import LTXVideoExtend
from griptape_nodes_library.video.ltx_video_retake import LTXVideoRetake
from griptape_nodes_library.video.public_video_url_mixin import MAX_VIDEO_FILE_SIZE_BYTES

# Minimal valid base64 payload whose decoded size is well under the limit (~15 bytes).
_SMALL_DATA_URI = "data:video/mp4;base64," + base64.b64encode(b"tiny video data").decode()

# Large fake data URI whose decoded size exceeds the limit.
# Formula used in the nodes: (b64_len // 4) * 3 - padding.
# 'A' chars have no '=' padding. We need (b64_len // 4) * 3 > MAX_VIDEO_FILE_SIZE_BYTES.
# b64_len must be a multiple of 4 for the formula to be consistent.
_LARGE_B64_LEN = ((MAX_VIDEO_FILE_SIZE_BYTES // 3) + 1) * 4  # first multiple-of-4 that decodes to > MAX
_LARGE_DATA_URI = "data:video/mp4;base64," + "A" * _LARGE_B64_LEN

_PUBLIC_URL = "https://storage.griptapecloud.com/bucket/video.mp4?sig=abc123"


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
    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: None)
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
    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_DATA_URI))

    with pytest.raises(ValueError) as exc_info:
        await node._build_payload()

    message = str(exc_info.value)
    assert "MB" in message
    assert str(MAX_VIDEO_FILE_SIZE_BYTES // (1024 * 1024)) in message


@pytest.mark.asyncio
async def test_retake_accepts_video_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/small.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    node.set_parameter_value("resolution", "1920x1080")

    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_SMALL_DATA_URI))

    payload = await node._build_payload()
    assert payload["video_uri"] == _SMALL_DATA_URI


# ---------------------------------------------------------------------------
# LTXVideoRetake — presigned URL path (tier 1)
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
    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: _PUBLIC_URL)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", prepare_mock)

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL
    prepare_mock.assert_not_called()


@pytest.mark.asyncio
async def test_retake_resolve_to_public_url_returns_https_url_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolve_to_public_url returns any https URL as-is."""
    node = LTXVideoRetake(name="Retake")
    result = node._resolve_to_public_url(VideoUrlArtifact(_PUBLIC_URL))
    assert result == _PUBLIC_URL


@pytest.mark.asyncio
async def test_retake_resolve_to_public_url_returns_none_for_localhost(monkeypatch: pytest.MonkeyPatch) -> None:
    """_resolve_to_public_url returns None for localhost URLs so they fall through to base64."""
    node = LTXVideoRetake(name="Retake")
    result = node._resolve_to_public_url(VideoUrlArtifact("http://localhost:9999/static/video.mp4?token=abc"))
    assert result is None


@pytest.mark.asyncio
async def test_retake_resolve_to_public_url_returns_none_for_data_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_resolve_to_public_url returns None for data URIs (skip to base64 path)."""
    node = LTXVideoRetake(name="Retake")
    result = node._resolve_to_public_url(_SMALL_DATA_URI)
    assert result is None


# ---------------------------------------------------------------------------
# LTXVideoExtend — base64 fallback path (tier 2)
# ---------------------------------------------------------------------------

_LARGE_EXTEND_B64_LEN = ((MAX_VIDEO_FILE_SIZE_BYTES // 3) + 1) * 4
_LARGE_EXTEND_DATA_URI = "data:video/mp4;base64," + "A" * _LARGE_EXTEND_B64_LEN


@pytest.mark.asyncio
async def test_extend_rejects_video_exceeding_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_EXTEND_DATA_URI))

    with pytest.raises(ValueError, match="too large"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_extend_accepts_video_within_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/small.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: None)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_SMALL_DATA_URI))

    payload = await node._build_payload()
    assert payload["video_uri"] == _SMALL_DATA_URI


# ---------------------------------------------------------------------------
# LTXVideoExtend — presigned URL path (tier 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extend_uses_public_url_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoExtend(name="Extend")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/big.mp4"))
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("duration", 2)
    node.set_parameter_value("context", 1)

    prepare_mock = AsyncMock()
    monkeypatch.setattr(node, "_resolve_to_public_url", lambda _video: _PUBLIC_URL)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", prepare_mock)

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL
    prepare_mock.assert_not_called()


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
