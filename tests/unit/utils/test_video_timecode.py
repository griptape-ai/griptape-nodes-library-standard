import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.utils.ffmpeg_utils import (
    build_video_segment_cmd,
    detect_video_properties,
    seconds_to_ts,
)
from griptape_nodes_library.utils.video_utils import smpte_to_seconds


def _probe(monkeypatch: pytest.MonkeyPatch, r_frame_rate: str) -> bool:
    payload = {"streams": [{"r_frame_rate": r_frame_rate}], "format": {"duration": "600.0"}}

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    return detect_video_properties("video.mp4", "ffprobe").drop_frame


@pytest.mark.parametrize(
    ("r_frame_rate", "expected"),
    [
        ("30/1", False),
        ("60/1", False),
        ("24/1", False),
        ("30000/1001", True),
        ("60000/1001", True),
        ("2997/100", True),
    ],
)
def test_drop_frame_detected_only_for_ntsc_rates(
    monkeypatch: pytest.MonkeyPatch, r_frame_rate: str, *, expected: bool
) -> None:
    assert _probe(monkeypatch, r_frame_rate) is expected


@pytest.mark.parametrize(
    ("tc", "expected"),
    [
        ("00:00:10:00", 10.0),
        ("00:00:59:00", 59.0),
        ("00:01:00:00", 60.0),
        ("00:01:59:00", 119.0),
        ("00:05:00:00", 300.0),
    ],
)
@pytest.mark.parametrize("rate", [30, 60])
def test_timecode_at_true_30_and_60_fps(monkeypatch: pytest.MonkeyPatch, tc: str, expected: float, rate: int) -> None:
    drop_frame = _probe(monkeypatch, f"{rate}/1")
    assert smpte_to_seconds(tc, rate, drop_frame=drop_frame) == pytest.approx(expected)


def test_drop_frame_timecode_at_ntsc_rate() -> None:
    # 00:01:00;02 is the first frame after the two numbers dropped at minute one: frame 1800.
    assert smpte_to_seconds("00:01:00;02", 30000 / 1001) == pytest.approx(1800 * 1001 / 30000)


def test_seconds_to_ts_carries_rounded_microseconds_into_seconds() -> None:
    assert seconds_to_ts(59.9999999) == "00:01:00.000000"


# Each source frame's luma is LUMA_STEP * frame index, so a decoded frame identifies itself.
LUMA_STEP = 4
SOURCE_FRAMES = 64
SIZE = 16


def _make_numbered_clip(ffmpeg_path: str, rate: str, path: Path) -> None:
    cmd = [
        ffmpeg_path,
        "-hide_banner",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"nullsrc=s={SIZE}x{SIZE}:r={rate},format=gray,geq=lum='N*{LUMA_STEP}'",
        "-frames:v",
        str(SOURCE_FRAMES),
        "-c:v",
        "libx264",
        "-qp",
        "0",
        str(path),
    ]
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)  # noqa: S603


def _decoded_frame_indices(ffmpeg_path: str, path: Path) -> list[int]:
    cmd = [ffmpeg_path, "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True, timeout=60).stdout  # noqa: S603
    frame_size = SIZE * SIZE
    centre = frame_size // 2 + SIZE // 2
    return [round(raw[i + centre] / LUMA_STEP) for i in range(0, len(raw), frame_size)]


@pytest.mark.parametrize(("rate", "fps"), [("24", 24.0), ("30", 30.0), ("30000/1001", 30000 / 1001), ("60", 60.0)])
@pytest.mark.parametrize(("start_frame", "end_frame"), [(0, 10), (1, 2), (2, 3), (2, 5), (4, 6), (7, 41)])
def test_frame_range_cut_keeps_exactly_the_requested_frames(
    tmp_path: Path, rate: str, fps: float, start_frame: int, end_frame: int
) -> None:
    ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
    source = tmp_path / "source.mp4"
    output = tmp_path / "trimmed.mp4"
    _make_numbered_clip(ffmpeg_path, rate, source)

    cmd = build_video_segment_cmd(
        ffmpeg_path, str(source), start_frame / fps, end_frame / fps, str(output), frame_accurate=True
    )
    subprocess.run(cmd, capture_output=True, check=True, timeout=60)  # noqa: S603

    assert _decoded_frame_indices(ffmpeg_path, output) == list(range(start_frame, end_frame))
