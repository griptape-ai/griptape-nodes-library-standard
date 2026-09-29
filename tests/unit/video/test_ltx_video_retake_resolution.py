"""The resolutions LTX Video Retake offers and auto-selects are ones its endpoint accepts.

The endpoint takes 1080p in either orientation only, so orientation is the whole choice the
node makes — and the probed orientation has to account for the display matrix that stores
phone footage landscape.
"""

from __future__ import annotations

import json
import subprocess
from typing import TYPE_CHECKING, Any

import pytest

from griptape_nodes_library.video.ltx_video_retake import (
    DEFAULT_RESOLUTION,
    PORTRAIT_RESOLUTION,
    SUPPORTED_RESOLUTIONS,
    LTXVideoRetake,
)

if TYPE_CHECKING:
    from pathlib import Path

# ffmpeg encodes to even dimensions, and the 4:2:0 chroma subsampling below needs them.
_LANDSCAPE = (1920, 1080)
_PORTRAIT = (1080, 1920)
_OVER_1440P = (2056, 1440)  # the reported source: closer to 1440p than 1080p by pixel area


def _resolution_choices(node: LTXVideoRetake) -> list[str]:
    parameter = node.get_parameter_by_name("resolution")
    assert parameter is not None
    return list(parameter.ui_options["simple_dropdown"])


def _encode_video(path: Path, size: tuple[int, int], *, rotation: int | None = None) -> Path:
    """Write a one-second test video, optionally carrying a display-matrix rotation."""
    from static_ffmpeg import run  # type: ignore[import-untyped]  # noqa: PLC0415

    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    width, height = size
    source = [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:duration=1:rate=25",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(source, capture_output=True, check=True, timeout=120)  # noqa: S603

    if rotation is None:
        return path

    # `-display_rotation` is an input option, so the rotation is applied by remuxing the
    # encoded file rather than by encoding it in one pass.
    rotated = path.with_name(f"rotated_{path.name}")
    subprocess.run(  # noqa: S603
        [ffmpeg_path, "-y", "-display_rotation", str(rotation), "-i", str(path), "-c", "copy", str(rotated)],
        capture_output=True,
        check=True,
        timeout=120,
    )
    return rotated


# ---------------------------------------------------------------------------
# The offered choices
# ---------------------------------------------------------------------------


def test_only_provider_accepted_resolutions_are_offered() -> None:
    """1440p and 2160p are rejected by the endpoint, so the node must not offer them."""
    assert SUPPORTED_RESOLUTIONS == (DEFAULT_RESOLUTION, PORTRAIT_RESOLUTION)
    assert _resolution_choices(LTXVideoRetake(name="Retake")) == ["1920x1080", "1080x1920"]


def test_portrait_is_offered() -> None:
    node = LTXVideoRetake(name="Retake")

    node.set_parameter_value("resolution", PORTRAIT_RESOLUTION)

    assert node.get_parameter_value("resolution") == PORTRAIT_RESOLUTION


@pytest.mark.parametrize("stale_resolution", ["2560x1440", "3840x2160", "1440x2560", "2160x3840"])
def test_resolution_saved_from_a_wider_list_snaps_to_an_accepted_value(stale_resolution: str) -> None:
    """A workflow saved while the dropdown offered more must not load a value that fails."""
    node = LTXVideoRetake(name="Retake")

    node.set_parameter_value("resolution", stale_resolution)

    assert node.get_parameter_value("resolution") in SUPPORTED_RESOLUTIONS


@pytest.mark.asyncio
async def test_build_payload_rejects_a_resolution_the_endpoint_would_reject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-flight validation must fail the request the endpoint fails, before it is sent."""
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("prompt", "test prompt")
    node.set_parameter_value("retake_segment", [0.0, 2.0])
    # Past the Options converter, standing in for a value reaching the payload by another path.
    node.parameter_values["resolution"] = "2560x1440"

    monkeypatch.setattr(node, "_validate_video_input", lambda _video: None)
    monkeypatch.setattr(node, "_upload_video_to_public_url", lambda _video: "https://example.com/v.mp4")

    with pytest.raises(ValueError, match="Unsupported resolution"):
        await node._build_payload()


# ---------------------------------------------------------------------------
# Auto-detection from the connected video
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (*_LANDSCAPE, DEFAULT_RESOLUTION),
        (*_PORTRAIT, PORTRAIT_RESOLUTION),
        (*_OVER_1440P, DEFAULT_RESOLUTION),
        (3840, 2160, DEFAULT_RESOLUTION),
        (2160, 3840, PORTRAIT_RESOLUTION),
        (1000, 1000, DEFAULT_RESOLUTION),  # square takes the landscape default
    ],
)
def test_snap_picks_the_orientation_matching_accepted_resolution(width: int, height: int, expected: str) -> None:
    assert LTXVideoRetake._snap_to_supported_resolution(width, height) == expected


@pytest.mark.parametrize(
    ("size", "rotation", "expected"),
    [
        (_LANDSCAPE, None, DEFAULT_RESOLUTION),
        (_PORTRAIT, None, PORTRAIT_RESOLUTION),
        (_OVER_1440P, None, DEFAULT_RESOLUTION),
        # Stored landscape, displayed portrait — the shape phone footage arrives in.
        (_LANDSCAPE, 90, PORTRAIT_RESOLUTION),
        (_LANDSCAPE, 270, PORTRAIT_RESOLUTION),
        # A half turn leaves the orientation alone.
        (_LANDSCAPE, 180, DEFAULT_RESOLUTION),
    ],
)
def test_connected_video_auto_selects_an_accepted_resolution(
    tmp_path: Path, size: tuple[int, int], rotation: int | None, expected: str
) -> None:
    """The end-to-end path a user hits: connect a video, read the resolution it picked."""
    video_path = _encode_video(tmp_path / "input.mp4", size, rotation=rotation)
    node = LTXVideoRetake(name="Retake")

    node._update_segment_range_from_video(str(video_path))

    assert node.get_parameter_value("resolution") == expected


def test_probe_reports_displayed_orientation_for_rotated_video(tmp_path: Path) -> None:
    """A quarter turn swaps the stored dimensions; ffprobe reports them unswapped."""
    video_path = _encode_video(tmp_path / "input.mp4", _LANDSCAPE, rotation=90)
    node = LTXVideoRetake(name="Retake")

    info = node._get_video_stream_info(str(video_path))

    assert info is not None
    assert (info["width"], info["height"]) == _PORTRAIT


def test_probe_without_rotation_metadata_reports_stored_dimensions(tmp_path: Path) -> None:
    video_path = _encode_video(tmp_path / "input.mp4", _PORTRAIT)
    node = LTXVideoRetake(name="Retake")

    info = node._get_video_stream_info(str(video_path))

    assert info is not None
    assert (info["width"], info["height"]) == _PORTRAIT


def test_unreadable_video_leaves_resolution_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed probe must not move the resolution off an accepted value."""
    node = LTXVideoRetake(name="Retake")
    node.set_parameter_value("resolution", PORTRAIT_RESOLUTION)

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(subprocess, "run", fail)

    node._update_segment_range_from_video("https://example.com/unreadable.mp4")

    assert node.get_parameter_value("resolution") == PORTRAIT_RESOLUTION


def test_probe_tolerates_stream_without_side_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """ffprobe omits `side_data_list` entirely for most video; that is not an error."""
    node = LTXVideoRetake(name="Retake")
    probe_output = json.dumps({"streams": [{"duration": "3.0", "width": 1080, "height": 1920}]})

    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(args=[], returncode=0, stdout=probe_output, stderr=""),
    )

    info = node._get_video_stream_info("https://example.com/v.mp4")

    assert info is not None
    assert (info["width"], info["height"]) == _PORTRAIT
