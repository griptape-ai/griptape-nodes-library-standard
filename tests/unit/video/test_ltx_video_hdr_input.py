"""LTX Video to Video HDR input handling: URL tier, data-URI fallback and frame limits."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from griptape_nodes_library.video import ltx_video_to_video_hdr as hdr_module
from griptape_nodes_library.video.ltx_video_to_video_hdr import LTXVideoToVideoHDR
from griptape_nodes_library.video.public_video_url_mixin import MAX_VIDEO_DATA_URI_SIZE_BYTES

_SMALL_DATA_URI = "data:video/mp4;base64," + base64.b64encode(b"tiny video data").decode()
_LARGE_B64_LEN = ((MAX_VIDEO_DATA_URI_SIZE_BYTES // 3) + 1) * 4
_LARGE_DATA_URI = "data:video/mp4;base64," + "A" * _LARGE_B64_LEN
_PUBLIC_URL = "https://storage.griptapecloud.com/bucket/video.mp4?sig=abc123"


def _node(monkeypatch: pytest.MonkeyPatch) -> LTXVideoToVideoHDR:
    node = LTXVideoToVideoHDR(name="HDR")
    node.set_parameter_value("video", VideoUrlArtifact("https://example.com/in.mp4"))
    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    return node


@pytest.mark.asyncio
async def test_hdr_uses_public_url_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch)
    monkeypatch.setattr(node, "_upload_video_to_public_url", AsyncMock(return_value=_PUBLIC_URL))
    prepare = AsyncMock(return_value=_SMALL_DATA_URI)
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", prepare)

    payload = await node._build_payload()

    assert payload == {"video_uri": _PUBLIC_URL, "output_colorspace": "acescg"}
    prepare.assert_not_called()


@pytest.mark.asyncio
async def test_hdr_falls_back_to_data_uri_when_upload_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch)
    monkeypatch.setattr(node, "_upload_video_to_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_SMALL_DATA_URI))

    assert await node._build_payload() == {"video_uri": _SMALL_DATA_URI, "output_colorspace": "acescg"}


@pytest.mark.asyncio
async def test_hdr_rejects_data_uri_exceeding_size_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch)
    monkeypatch.setattr(node, "_upload_video_to_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=_LARGE_DATA_URI))

    with pytest.raises(ValueError, match="too large"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_hdr_reports_unreadable_video(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(monkeypatch)
    monkeypatch.setattr(node, "_upload_video_to_public_url", AsyncMock(return_value=None))
    monkeypatch.setattr(node, "_prepare_video_data_uri_async", AsyncMock(return_value=None))

    with pytest.raises(ValueError, match="^Could not read the input video\\."):
        await node._build_payload()


@pytest.mark.asyncio
async def test_hdr_process_generation_resets_and_cleans_up_uploads(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoToVideoHDR(name="HDR")
    calls: list[str] = []
    monkeypatch.setattr(node, "_reset_video_uploads", lambda: calls.append("reset"))
    monkeypatch.setattr(node, "_cleanup_video_uploads", AsyncMock(side_effect=lambda: calls.append("cleanup")))
    parent = hdr_module.GriptapeProxyNode
    monkeypatch.setattr(parent, "_process_generation", AsyncMock(side_effect=RuntimeError("boom")))

    with pytest.raises(RuntimeError, match="boom"):
        await node._process_generation()

    assert calls == ["reset", "cleanup"]


@pytest.mark.parametrize(
    ("width", "height", "bucket", "max_frames"),
    [
        (1920, 1080, "1080p", 273),
        (1080, 1920, "1080p", 273),
        (2560, 1440, "1440p", 193),
        (3840, 2160, "2160p", 65),
    ],
)
def test_hdr_frame_limit_per_tier(width: int, height: int, bucket: str, max_frames: int) -> None:
    assert LTXVideoToVideoHDR._pick_input_tier(width, height) == (bucket, max_frames)


@pytest.mark.parametrize(
    ("width", "height", "max_frames"),
    [(1920, 1080, 273), (2560, 1440, 193), (3840, 2160, 65)],
)
def test_hdr_frame_limit_boundaries(monkeypatch: pytest.MonkeyPatch, width: int, height: int, max_frames: int) -> None:
    node = LTXVideoToVideoHDR(name="HDR")
    video = VideoUrlArtifact("https://example.com/in.mp4")

    monkeypatch.setattr(node, "_probe_video", lambda _url: (width, height, max_frames))
    assert node._validate_video_input(video) is None

    monkeypatch.setattr(node, "_probe_video", lambda _url: (width, height, max_frames + 1))
    error = node._validate_video_input(video)
    assert error is not None
    assert f"limit of {max_frames} frames" in error


def test_hdr_rejects_input_above_4k(monkeypatch: pytest.MonkeyPatch) -> None:
    node = LTXVideoToVideoHDR(name="HDR")
    monkeypatch.setattr(node, "_probe_video", lambda _url: (4096, 2304, 10))

    error = node._validate_video_input(VideoUrlArtifact("https://example.com/in.mp4"))

    assert error is not None
    assert "larger than LTX HDR upscale accepts" in error
