"""Unit tests for the LTX Video Reframe node: payload building, validation and placement visibility."""

from __future__ import annotations

import json
import subprocess
import threading
from typing import Any
from unittest.mock import AsyncMock

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.files import file as file_module
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, SerializedParameterValueTracker
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason
from griptape_nodes.retained_mode.managers.node_manager import NodeManager

from griptape_nodes_library.video import ltx_video_reframe
from griptape_nodes_library.video.ltx_video_reframe import (
    SUPPORTED_RESOLUTIONS,
    LTXVideoReframe,
    PlacementMode,
    build_placement,
)

_PUBLIC_URL = "https://storage.griptapecloud.com/bucket/video.mp4?sig=abc123"
_RECTANGLE_PARAMETERS = ["x", "y", "width", "height"]


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> LTXVideoReframe:
    reframe = LTXVideoReframe(name="Reframe")
    reframe.set_parameter_value("video", VideoUrlArtifact("https://example.com/in.mp4"))
    monkeypatch.setattr(reframe, "_probe_video", lambda _url: (1920, 1080, 300, 10.0))
    monkeypatch.setattr(reframe, "_upload_video_to_public_url", AsyncMock(return_value=_PUBLIC_URL))
    return reframe


def _is_hidden(node: LTXVideoReframe, name: str) -> bool:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    return bool(parameter.ui_options.get("hide"))


# ---------------------------------------------------------------------------
# Model id
# ---------------------------------------------------------------------------


def test_api_model_id_is_ltx_reframe(node: LTXVideoReframe) -> None:
    assert node._get_api_model_id() == "ltx:video-to-video-reframe"


# ---------------------------------------------------------------------------
# Payload building
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_center_mode_omits_placement(node: LTXVideoReframe) -> None:
    node.set_parameter_value("resolution", "720x720")

    payload = await node._build_payload()

    assert payload == {"video_uri": _PUBLIC_URL, "resolution": "720x720"}


@pytest.mark.asyncio
async def test_center_mode_ignores_stale_anchor_and_rectangle_values(node: LTXVideoReframe) -> None:
    node.set_parameter_value("anchor", "top")
    node.set_parameter_value("x", 0.5)

    payload = await node._build_payload()

    assert "placement" not in payload


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "anchor",
    ["center", "top", "bottom", "left", "right", "top_left", "top_right", "bottom_left", "bottom_right"],
)
async def test_anchor_mode_sends_anchor(node: LTXVideoReframe, anchor: str) -> None:
    node.set_parameter_value("placement_mode", "anchor")
    node.set_parameter_value("anchor", anchor)

    payload = await node._build_payload()

    assert payload["placement"] == {"anchor": anchor}


@pytest.mark.asyncio
async def test_rectangle_mode_with_width_only(node: LTXVideoReframe) -> None:
    node.set_parameter_value("resolution", "1080x1920")
    node.set_parameter_value("placement_mode", "rectangle")
    node.set_parameter_value("x", 0.0)
    node.set_parameter_value("y", 0.35)
    node.set_parameter_value("width", 1.0)
    node.set_parameter_value("height", None)

    payload = await node._build_payload()

    assert payload == {
        "video_uri": _PUBLIC_URL,
        "resolution": "1080x1920",
        "placement": {"x": 0.0, "y": 0.35, "width": 1.0},
    }


@pytest.mark.asyncio
async def test_rectangle_mode_with_height_only(node: LTXVideoReframe) -> None:
    node.set_parameter_value("placement_mode", "rectangle")
    node.set_parameter_value("x", 0.1)
    node.set_parameter_value("y", 0.1)
    node.set_parameter_value("width", None)
    node.set_parameter_value("height", 0.25)

    payload = await node._build_payload()

    assert payload["placement"] == {"x": 0.1, "y": 0.1, "height": 0.25}


@pytest.mark.asyncio
async def test_rectangle_mode_with_width_and_height(node: LTXVideoReframe) -> None:
    node.set_parameter_value("placement_mode", "rectangle")
    node.set_parameter_value("width", 0.5)
    node.set_parameter_value("height", 0.5)

    payload = await node._build_payload()

    assert payload["placement"] == {"x": 0.0, "y": 0.0, "width": 0.5, "height": 0.5}


@pytest.mark.asyncio
async def test_rejects_unsupported_resolution(node: LTXVideoReframe) -> None:
    # The Options trait snaps unsupported values on assignment, so write the stored value directly
    # to reach the payload builder's own guard (e.g. a stale value from an older saved workflow).
    node.parameter_values["resolution"] = "640x480"

    with pytest.raises(NodeError, match="'Resolution' must be one of"):
        await node._build_payload()


def test_every_documented_resolution_is_offered() -> None:
    assert SUPPORTED_RESOLUTIONS == [
        "720x720",
        "1080x1080",
        "720x900",
        "1080x1350",
        "900x720",
        "1350x1080",
        "720x1280",
        "1080x1920",
        "1280x720",
        "1920x1080",
    ]


# ---------------------------------------------------------------------------
# Placement validation
# ---------------------------------------------------------------------------


def test_placement_center_returns_none() -> None:
    assert build_placement(PlacementMode.CENTER) is None


def test_placement_rejects_unknown_mode() -> None:
    with pytest.raises(NodeError, match="'Placement Mode' must be one of"):
        build_placement("diagonal")


def test_placement_rejects_unknown_anchor() -> None:
    with pytest.raises(NodeError, match="'Anchor' must be one of"):
        build_placement(PlacementMode.ANCHOR, anchor="middle")


@pytest.mark.parametrize("x", [-0.1, 1.0, 1.5])
def test_rectangle_rejects_x_out_of_range(x: float) -> None:
    with pytest.raises(NodeError, match="'X' must be at least 0 and less than 1"):
        build_placement(PlacementMode.RECTANGLE, x=x, y=0, width=1)


@pytest.mark.parametrize("y", [-0.1, 1.0, 2.0])
def test_rectangle_rejects_y_out_of_range(y: float) -> None:
    with pytest.raises(NodeError, match="'Y' must be at least 0 and less than 1"):
        build_placement(PlacementMode.RECTANGLE, x=0, y=y, width=1)


@pytest.mark.parametrize("value", [None, ""])
def test_rectangle_requires_width_or_height(value: Any) -> None:
    with pytest.raises(NodeError, match="Set 'Width', 'Height' or both"):
        build_placement(PlacementMode.RECTANGLE, x=0, y=0, width=value, height=value)


@pytest.mark.parametrize("width", [0, -0.5, 1.01])
def test_rectangle_rejects_width_out_of_range(width: float) -> None:
    with pytest.raises(NodeError, match="'Width' must be greater than 0 and at most 1"):
        build_placement(PlacementMode.RECTANGLE, x=0, y=0, width=width)


@pytest.mark.parametrize("height", [0, -0.5, 1.01])
def test_rectangle_rejects_height_out_of_range(height: float) -> None:
    with pytest.raises(NodeError, match="'Height' must be greater than 0 and at most 1"):
        build_placement(PlacementMode.RECTANGLE, x=0, y=0, height=height)


def test_rectangle_rejects_non_numeric_value() -> None:
    with pytest.raises(NodeError, match="'Width' must be a number"):
        build_placement(PlacementMode.RECTANGLE, x=0, y=0, width="wide")


def test_rectangle_accepts_boundary_values() -> None:
    placement = build_placement(PlacementMode.RECTANGLE, x=0, y=0.999, width=1, height=1)

    assert placement == {"x": 0.0, "y": 0.999, "width": 1.0, "height": 1.0}


def test_rectangle_treats_blank_width_as_omitted() -> None:
    placement = build_placement(PlacementMode.RECTANGLE, x=0, y=0, width="  ", height=0.5)

    assert placement == {"x": 0.0, "y": 0.0, "height": 0.5}


# ---------------------------------------------------------------------------
# Input video validation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_requires_a_video(node: LTXVideoReframe) -> None:
    node.set_parameter_value("video", None)

    with pytest.raises(NodeError, match="Connect a video to 'input video'"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_rejects_video_over_duration_limit(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_probe_video", lambda _url: (1920, 1080, 1000, 60.5))

    with pytest.raises(NodeError, match="over the 60s limit"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_rejects_video_over_frame_limit(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_probe_video", lambda _url: (1920, 1080, 1801, 30.0))

    with pytest.raises(NodeError, match="over the 1800 frame limit"):
        await node._build_payload()


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(63, 1080), (1920, 63), (4097, 1080), (1920, 4097)])
async def test_rejects_video_with_side_out_of_range(
    node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch, size: tuple[int, int]
) -> None:
    width, height = size
    monkeypatch.setattr(node, "_probe_video", lambda _url: (width, height, 100, 5.0))

    with pytest.raises(NodeError, match="Each side must be between 64 and 4096 pixels"):
        await node._build_payload()


@pytest.mark.asyncio
async def test_accepts_duration_within_tolerance(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_probe_video", lambda _url: (1920, 1080, 1800, 60.02))

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL


@pytest.mark.asyncio
async def test_probe_runs_off_the_event_loop(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    threads: list[int] = []

    def probe(_url: str) -> tuple[int, int, int, float]:
        threads.append(threading.get_ident())
        return 1920, 1080, 300, 10.0

    monkeypatch.setattr(node, "_probe_video", probe)

    await node._build_payload()

    assert threads
    assert threads[0] != threading.get_ident()


@pytest.mark.asyncio
async def test_accepts_video_at_the_limits(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_probe_video", lambda _url: (64, 4096, 1800, 60.0))

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL


@pytest.mark.asyncio
async def test_skips_validation_when_probe_fails(node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_probe_video", lambda _url: None)

    payload = await node._build_payload()

    assert payload["video_uri"] == _PUBLIC_URL


def test_probe_returns_none_when_ffprobe_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise() -> None:
        msg = "ffprobe missing"
        raise ValueError(msg)

    monkeypatch.setattr(ltx_video_reframe.run, "get_or_fetch_platform_executables_else_raise", _raise)

    assert LTXVideoReframe(name="Reframe")._probe_video("https://example.com/in.mp4") is None


# ---------------------------------------------------------------------------
# Visibility toggling
# ---------------------------------------------------------------------------


def test_placement_controls_are_hidden_by_default() -> None:
    node = LTXVideoReframe(name="Reframe")

    assert _is_hidden(node, "anchor")
    for name in _RECTANGLE_PARAMETERS:
        assert _is_hidden(node, name)


def test_anchor_mode_shows_only_anchor() -> None:
    node = LTXVideoReframe(name="Reframe")

    node.set_parameter_value("placement_mode", "anchor")

    assert not _is_hidden(node, "anchor")
    for name in _RECTANGLE_PARAMETERS:
        assert _is_hidden(node, name)


def test_rectangle_mode_shows_only_rectangle_controls() -> None:
    node = LTXVideoReframe(name="Reframe")

    node.set_parameter_value("placement_mode", "rectangle")

    assert _is_hidden(node, "anchor")
    for name in _RECTANGLE_PARAMETERS:
        assert not _is_hidden(node, name)


def test_returning_to_center_hides_everything_again() -> None:
    node = LTXVideoReframe(name="Reframe")
    node.set_parameter_value("placement_mode", "rectangle")

    node.set_parameter_value("placement_mode", "center")

    assert _is_hidden(node, "anchor")
    for name in _RECTANGLE_PARAMETERS:
        assert _is_hidden(node, name)


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_generation_cleans_up_uploads_even_on_failure(
    node: LTXVideoReframe, monkeypatch: pytest.MonkeyPatch
) -> None:
    cleanup = AsyncMock()
    monkeypatch.setattr(node, "_cleanup_video_uploads", cleanup)
    monkeypatch.setattr(
        ltx_video_reframe.GriptapeProxyNode, "_process_generation", AsyncMock(side_effect=RuntimeError("boom"))
    )

    with pytest.raises(RuntimeError, match="boom"):
        await node._process_generation()

    cleanup.assert_awaited_once()


# ---------------------------------------------------------------------------
# Probe parsing
# ---------------------------------------------------------------------------


def test_parse_uses_nb_frames_when_present() -> None:
    data = {"streams": [{"width": 1920, "height": 1080, "duration": "4.0", "nb_frames": "96"}]}

    assert LTXVideoReframe._parse_probe_output(data) == (1920, 1080, 96, 4.0)


def test_parse_falls_back_to_duration_times_frame_rate() -> None:
    data = {"streams": [{"width": 1280, "height": 720, "duration": "10.0", "avg_frame_rate": "30000/1001"}]}

    assert LTXVideoReframe._parse_probe_output(data) == (1280, 720, 300, 10.0)


def test_parse_falls_back_to_format_duration() -> None:
    data = {
        "streams": [{"width": 640, "height": 480, "avg_frame_rate": "25/1"}],
        "format": {"duration": "2.0"},
    }

    assert LTXVideoReframe._parse_probe_output(data) == (640, 480, 50, 2.0)


@pytest.mark.parametrize(
    ("rate", "expected"),
    [("30000/1001", 29.97002997), ("25/1", 25.0), ("24", 24.0), ("0/0", 0.0), ("N/A", 0.0), (None, 0.0)],
)
def test_frame_rate_parsing(rate: Any, expected: float) -> None:
    assert ltx_video_reframe._frame_rate(rate) == pytest.approx(expected)


@pytest.mark.parametrize("data", [{}, {"streams": []}, {"streams": [{"width": 0, "height": 100}]}])
def test_parse_returns_none_for_unusable_output(data: dict[str, Any]) -> None:
    assert LTXVideoReframe._parse_probe_output(data) is None


@pytest.mark.parametrize("rotation", [90, -90, 270])
def test_parse_swaps_sides_for_quarter_turned_video(rotation: int) -> None:
    stream = {
        "width": 1920,
        "height": 1080,
        "nb_frames": "10",
        "duration": "1.0",
        "side_data_list": [{"rotation": rotation}],
    }

    assert LTXVideoReframe._parse_probe_output({"streams": [stream]}) == (1080, 1920, 10, 1.0)


def test_parse_keeps_sides_for_half_turn() -> None:
    stream = {"width": 1920, "height": 1080, "nb_frames": "10", "side_data_list": [{"rotation": 180}]}

    parsed = LTXVideoReframe._parse_probe_output({"streams": [stream]})

    assert parsed is not None
    assert parsed[:2] == (1920, 1080)


def _stub_ffprobe(monkeypatch: pytest.MonkeyPatch, stdout: str) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append(cmd)
        assert kwargs["timeout"] == ltx_video_reframe.PROBE_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(
        ltx_video_reframe.run, "get_or_fetch_platform_executables_else_raise", lambda: ("ffmpeg", "ffprobe")
    )
    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def test_probe_resolves_macro_path_and_parses_output(monkeypatch: pytest.MonkeyPatch) -> None:
    stream = {"width": 1920, "height": 1080, "duration": "4.0", "nb_frames": "96"}
    calls = _stub_ffprobe(monkeypatch, json.dumps({"streams": [stream]}))
    monkeypatch.setattr(file_module.File, "resolve", lambda _self: "/resolved/outputs/clip.mp4")

    probed = LTXVideoReframe(name="Reframe")._probe_video("{outputs}/clip.mp4")

    assert calls[0][-1] == "/resolved/outputs/clip.mp4"
    assert probed == (1920, 1080, 96, 4.0)


def test_probe_passes_data_uri_through(monkeypatch: pytest.MonkeyPatch) -> None:
    data_uri = "data:video/mp4;base64,AAAA"
    calls = _stub_ffprobe(monkeypatch, json.dumps({"streams": []}))

    assert LTXVideoReframe(name="Reframe")._probe_video(data_uri) is None
    assert calls[0][-1] == data_uri


def test_probe_is_skipped_when_macro_resolution_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _stub_ffprobe(monkeypatch, "{}")

    def fail(_self: file_module.File) -> str:
        raise file_module.FileLoadError(FileIOFailureReason.MISSING_MACRO_VARIABLES, "no project loaded")

    monkeypatch.setattr(file_module.File, "resolve", fail)

    assert LTXVideoReframe(name="Reframe")._probe_video("{outputs}/clip.mp4") is None
    assert calls == []


# ---------------------------------------------------------------------------
# Save / reload
# ---------------------------------------------------------------------------


def test_cleared_width_survives_the_engines_save_rule() -> None:
    """A height-only rectangle must not come back with a width after save and reload.

    The engine saves a parameter value only when it was explicitly set or equals a non-None
    default (`NodeManager.handle_parameter_value_saving`), and never saves None. A cleared
    width therefore reloads as the parameter's default, which must be None.
    """
    node = LTXVideoReframe(name="Reframe")
    node.set_parameter_value("placement_mode", "rectangle")
    node.set_parameter_value("height", 0.25)
    node.set_parameter_value("width", None)

    width_param = node.get_parameter_by_name("width")
    assert width_param is not None
    assert width_param.default_value is None

    reloaded = LTXVideoReframe(name="Reframe")
    assert reloaded.get_parameter_value("width") is None


def test_unset_width_and_height_default_to_blank() -> None:
    node = LTXVideoReframe(name="Reframe")

    assert node.get_parameter_value("width") is None
    assert node.get_parameter_value("height") is None


def test_engine_save_omits_cleared_width_and_reload_restores_none() -> None:
    """Run the engine's real per-parameter save rule on a height-only rectangle."""
    node = LTXVideoReframe(name="Reframe")
    node.set_parameter_value("placement_mode", "rectangle")
    node.set_parameter_value("height", 0.25)
    node.set_parameter_value("width", None)
    width_param = node.get_parameter_by_name("width")
    assert width_param is not None

    values: dict[Any, Any] = {}
    commands = NodeManager.handle_parameter_value_saving(
        parameter=width_param,
        node=node,
        unique_parameter_uuid_to_values=values,
        serialized_parameter_value_tracker=SerializedParameterValueTracker(),
        create_node_request=CreateNodeRequest(node_type="LTXVideoReframe"),
    )

    # Nothing is saved for width, so the reloaded node uses the default, which is also None.
    assert not commands
    assert LTXVideoReframe(name="Reframe").get_parameter_value("width") is None
