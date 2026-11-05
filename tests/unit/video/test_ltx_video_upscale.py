"""Unit tests for the LTX Video Upscale node: payload shape and local input validation.

The ffprobe-backed probes are stubbed, so these tests run without ffmpeg and check the node's
own rules. Probe-level behaviour (macro paths, data URIs, the duration x fps fallback, fps parsing) is covered by
test_video_probe_macro_paths.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.video.ltx_video_upscale import (
    LTXVideoUpscale,
    VideoProbe,
    _frame_count,
    _parse_frame_rate,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_VIDEO = VideoUrlArtifact("https://example.com/clip.mp4")
_PUBLIC_URI = "https://cdn.example.com/clip.mp4"
_H264_1080P_25FPS = {"codec": "h264", "fps": 25.0, "dims": (1920, 1080), "frames": 100}


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Stub the single ffprobe-backed probe. Call the returned setter to change the input.

    Pass ``unreadable=True`` to model a failed probe, which the node must treat as "skip checks".
    """
    state: dict[str, Any] = {"unreadable": False, **_H264_1080P_25FPS}

    def set_probe(
        *,
        codec: str | None = "h264",
        fps: float | None = 25.0,
        dims: tuple[int, int] = (1920, 1080),
        frames: int | None = 100,
        unreadable: bool = False,
    ) -> None:
        state.update(codec=codec, fps=fps, dims=dims, frames=frames, unreadable=unreadable)

    def fake_probe(self: LTXVideoUpscale, _url: str) -> VideoProbe | None:  # noqa: ARG001
        if state["unreadable"]:
            return None
        width, height = state["dims"]
        return VideoProbe(
            width=width,
            height=height,
            frames=state["frames"],
            codec=state["codec"],
            fps=state["fps"],
        )

    monkeypatch.setattr(LTXVideoUpscale, "_probe_video", fake_probe)
    set_probe()
    return set_probe


def _node(resolution: str = "4k") -> LTXVideoUpscale:
    node = LTXVideoUpscale(name="Upscale")
    node.set_parameter_value("video", _VIDEO)
    node.set_parameter_value("resolution", resolution)
    return node


def _validate(node: LTXVideoUpscale, resolution: str) -> str | None:
    return node._validate_video_input(_VIDEO, resolution)


# ---------------------------------------------------------------------------
# Model and offered choices
# ---------------------------------------------------------------------------


def test_api_model_id_is_the_upscale_endpoint() -> None:
    assert _node()._get_api_model_id() == "ltx:video-upscale"


def test_resolution_choices_are_1080p_4k_8k_with_4k_default() -> None:
    node = LTXVideoUpscale(name="Upscale")
    parameter = node.get_parameter_by_name("resolution")
    assert parameter is not None
    assert list(parameter.ui_options["simple_dropdown"]) == ["1080p", "4k", "8k"]
    assert node.get_parameter_value("resolution") == "4k"


# ---------------------------------------------------------------------------
# Payload
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("resolution", ["1080p", "4k", "8k"])
async def test_payload_carries_video_uri_and_resolution(
    monkeypatch: pytest.MonkeyPatch, probe: Callable[..., None], resolution: str
) -> None:
    probe()
    node = _node(resolution)
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value=_PUBLIC_URI))

    payload = await node._build_payload()

    assert payload == {"video_uri": _PUBLIC_URI, "resolution": resolution}


@pytest.mark.asyncio
async def test_payload_sends_no_model_field(monkeypatch: pytest.MonkeyPatch, probe: Callable[..., None]) -> None:
    probe()
    node = _node("4k")
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value=_PUBLIC_URI))

    payload = await node._build_payload()

    assert "model" not in payload


@pytest.mark.asyncio
async def test_payload_build_raises_node_error_naming_the_limit(
    monkeypatch: pytest.MonkeyPatch, probe: Callable[..., None]
) -> None:
    probe(frames=602)
    node = _node("4k")
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value=_PUBLIC_URI))

    with pytest.raises(NodeError, match="601 frame limit"):
        await node._build_payload()


# ---------------------------------------------------------------------------
# Accepted input passes validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["h264", "hevc"])
def test_accepts_h264_and_hevc(probe: Callable[..., None], codec: str) -> None:
    probe(codec=codec)
    assert _validate(_node(), "4k") is None


def test_skips_codec_check_when_codec_unknown(probe: Callable[..., None]) -> None:
    probe(codec=None)
    assert _validate(_node(), "4k") is None


def test_skips_all_checks_when_probe_fails(probe: Callable[..., None]) -> None:
    probe(unreadable=True, codec="prores", fps=0.1, dims=(100, 100), frames=9999)
    assert _validate(_node(), "8k") is None


def test_skips_frame_check_when_frame_count_unknown(probe: Callable[..., None]) -> None:
    probe(frames=None)
    assert _validate(_node(), "8k") is None


def test_error_is_a_node_error_with_the_user_facing_message(probe: Callable[..., None]) -> None:
    probe(codec="prores")
    error = _validate(_node(), "4k")
    assert error is not None
    assert "'input video'" in error
    assert "H.264 or HEVC" in error


@pytest.mark.asyncio
async def test_payload_build_node_error_is_not_prefixed(
    monkeypatch: pytest.MonkeyPatch, probe: Callable[..., None]
) -> None:
    """A NodeError raised during payload build reaches the failure path with its message unchanged."""
    probe(codec="prores")
    node = _node("4k")
    monkeypatch.setattr(node, "_resolve_video_uri", AsyncMock(return_value=_PUBLIC_URI))
    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_handle_failure_exception", lambda _e: None)
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs))

    with pytest.raises(NodeError) as excinfo:
        await node._build_payload()

    node._handle_payload_build_error(excinfo.value)

    assert len(status_calls) == 1
    details = status_calls[0]["result_details"]
    assert "'input video' uses prores" in details
    assert not details.startswith("Failed to build the request")


# ---------------------------------------------------------------------------
# Single-probe frame count: nb_frames first, then duration x fps
# ---------------------------------------------------------------------------


def test_frame_count_prefers_nb_frames() -> None:
    assert _frame_count({"nb_frames": "120", "duration": "9.0"}, 25.0) == 120


def test_frame_count_falls_back_to_duration_times_fps() -> None:
    assert _frame_count({"duration": "4.0"}, 25.0) == 100


def test_frame_count_is_none_without_nb_frames_or_duration() -> None:
    assert _frame_count({}, 25.0) is None


def test_frame_count_is_none_when_fps_unknown_and_no_nb_frames() -> None:
    assert _frame_count({"duration": "4.0"}, None) is None


def test_skips_fps_check_when_fps_unknown(probe: Callable[..., None]) -> None:
    probe(fps=None)
    assert _validate(_node(), "4k") is None


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codec", ["prores", "ffv1", "rawvideo", "vp9"])
def test_rejects_codecs_other_than_h264_or_hevc(probe: Callable[..., None], codec: str) -> None:
    probe(codec=codec)
    error = _validate(_node(), "4k")
    assert error is not None
    assert codec in error
    assert "H.264 or HEVC" in error


# ---------------------------------------------------------------------------
# Short and long side
# ---------------------------------------------------------------------------


def test_short_side_of_480_passes(probe: Callable[..., None]) -> None:
    probe(dims=(854, 480))
    assert _validate(_node("1080p"), "1080p") is None


def test_short_side_below_480_fails(probe: Callable[..., None]) -> None:
    probe(dims=(854, 479))
    error = _validate(_node("1080p"), "1080p")
    assert error is not None
    assert "480" in error


def test_portrait_short_side_is_checked_the_same_way(probe: Callable[..., None]) -> None:
    probe(dims=(479, 854))
    error = _validate(_node("1080p"), "1080p")
    assert error is not None
    assert "480" in error


def test_long_side_of_7680_passes_when_short_side_within_4320(probe: Callable[..., None]) -> None:
    probe(dims=(7680, 4320))
    assert _validate(_node("8k"), "8k") is None


def test_long_side_above_7680_fails(probe: Callable[..., None]) -> None:
    probe(dims=(7681, 4320))
    error = _validate(_node("8k"), "8k")
    assert error is not None
    assert "7680x4320" in error


def test_short_side_above_4320_fails(probe: Callable[..., None]) -> None:
    probe(dims=(7680, 4321))
    error = _validate(_node("8k"), "8k")
    assert error is not None
    assert "7680x4320" in error


def test_portrait_long_side_above_7680_fails(probe: Callable[..., None]) -> None:
    probe(dims=(4320, 7681))
    assert _validate(_node("8k"), "8k") is not None


# ---------------------------------------------------------------------------
# Frame count: the cap depends on the target resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("resolution", "at_cap", "over_cap"),
    [("1080p", 2401, 2402), ("4k", 601, 602), ("8k", 241, 242)],
)
def test_frame_cap_boundary_per_resolution(
    probe: Callable[..., None], resolution: str, at_cap: int, over_cap: int
) -> None:
    probe(frames=at_cap, dims=(1920, 1080))
    assert _validate(_node(resolution), resolution) is None

    probe(frames=over_cap, dims=(1920, 1080))
    error = _validate(_node(resolution), resolution)
    assert error is not None
    assert str(at_cap) in error
    assert resolution in error


def test_frame_count_of_2_passes(probe: Callable[..., None]) -> None:
    probe(frames=2)
    assert _validate(_node("4k"), "4k") is None


def test_frame_count_of_1_fails(probe: Callable[..., None]) -> None:
    probe(frames=1)
    error = _validate(_node("4k"), "4k")
    assert error is not None
    assert "at least 2" in error


# ---------------------------------------------------------------------------
# Frame rate
# ---------------------------------------------------------------------------


def test_fps_of_1_passes(probe: Callable[..., None]) -> None:
    probe(fps=1.0)
    assert _validate(_node("4k"), "4k") is None


def test_fps_below_1_fails(probe: Callable[..., None]) -> None:
    probe(fps=0.5)
    error = _validate(_node("4k"), "4k")
    assert error is not None
    assert "1 fps" in error


# ---------------------------------------------------------------------------
# Frame-rate parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("25/1", 25.0), ("30000/1001", 30000 / 1001), ("24", 24.0), ("0/0", None), ("", None), (None, None)],
)
def test_parse_frame_rate(raw: str | None, expected: float | None) -> None:
    assert _parse_frame_rate(raw) == expected


# --- Submission wait for large URL inputs --------------------------------------------------------


def test_waits_longer_than_the_proxys_url_download_for_the_create_call() -> None:
    # Griptape Cloud allows up to 150 s to download a URL input while handling the create call.
    assert LTXVideoUpscale.SUBMIT_TIMEOUT_SECONDS > 150


@pytest.mark.parametrize(
    ("connected", "sent", "warns"),
    [
        # A public https link the node passes through: the proxy fetches it from its own host.
        ("https://videos.example.com/clip.mp4", "https://videos.example.com/clip.mp4", True),
        # A local file the node uploaded first: the proxy fetches Griptape storage, which is fast.
        ("/workspace/clip.mp4", "https://storage.example.net/uploads/abc.mp4", False),
        # A base64 fallback: nothing to download.
        ("/workspace/clip.mp4", "data:video/mp4;base64,AAAA", False),
    ],
)
def test_warns_only_when_the_proxy_downloads_the_users_own_link(connected: str, sent: str, *, warns: bool) -> None:
    node = LTXVideoUpscale(name="Upscale")
    node.status_component.clear_execution_status(initial_message="")

    node._warn_if_proxy_downloads_external_url(VideoUrlArtifact(value=connected), sent)

    details = node.get_parameter_value("result_details") or ""
    assert ("link to another website" in details) is warns
