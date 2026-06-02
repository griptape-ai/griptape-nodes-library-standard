"""FFmpeg utility functions for cross-platform executable path resolution."""

import json
import subprocess
from collections.abc import Callable
from typing import NamedTuple

import static_ffmpeg.run  # type: ignore[import-untyped]

RATE_TOLERANCE = 0.1
MIN_SEGMENT_DURATION_FOR_STREAM_COPY = 2.0  # seconds — below this stream copy is unreliable


class FfmpegPaths(NamedTuple):
    ffmpeg: str
    ffprobe: str


class VideoProperties(NamedTuple):
    frame_rate: float
    drop_frame: bool
    duration: float


def get_ffmpeg_path() -> str:
    """Get the path to ffmpeg executable using static_ffmpeg for cross-platform compatibility.

    This function handles finding the ffmpeg executable across different platforms:
    - Windows: Uses static-ffmpeg's platform-specific resolution
    - Linux: Uses static-ffmpeg's platform-specific resolution
    - macOS: Uses static-ffmpeg's platform-specific resolution

    Returns:
        Path to the ffmpeg executable

    Raises:
        RuntimeError: If ffmpeg is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return ffmpeg_path


def get_ffprobe_path() -> str:
    """Get the path to ffprobe executable using static_ffmpeg for cross-platform compatibility.

    This function handles finding the ffprobe executable across different platforms:
    - Windows: Uses static-ffmpeg's platform-specific resolution
    - Linux: Uses static-ffmpeg's platform-specific resolution
    - macOS: Uses static-ffmpeg's platform-specific resolution

    Returns:
        Path to the ffprobe executable

    Raises:
        RuntimeError: If ffprobe is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        _, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFprobe not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return ffprobe_path


def get_ffmpeg_paths() -> FfmpegPaths:
    """Get both ffmpeg and ffprobe executable paths using static_ffmpeg.

    This is a convenience function that returns both paths in a single call,
    which is useful when you need both executables.

    Returns:
        FfmpegPaths(ffmpeg, ffprobe)

    Raises:
        RuntimeError: If either executable is not found or static-ffmpeg is not properly installed
    """
    # FAILURE CASES FIRST
    try:
        ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        error_msg = f"FFmpeg/FFprobe not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(error_msg) from e

    # SUCCESS PATH AT END
    return FfmpegPaths(ffmpeg_path, ffprobe_path)


def run_ffmpeg_cmd(
    cmd: list[str],
    *,
    log: Callable[[str], None] | None = None,
    timeout: int = 300,
) -> None:
    """Run an ffmpeg command with standard error handling and optional logging."""
    if log:
        log(f"Running ffmpeg command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)  # noqa: S603
        if result.stderr and log:
            log(f"FFmpeg stderr: {result.stderr}\n")
    except subprocess.TimeoutExpired as e:
        error_msg = f"FFmpeg timed out after {timeout}s"
        if log:
            log(f"ERROR: {error_msg}\n")
        raise ValueError(error_msg) from e
    except subprocess.CalledProcessError as e:
        error_msg = f"FFmpeg error: {e.stderr}"
        if log:
            log(f"ERROR: {error_msg}\n")
        raise ValueError(error_msg) from e


def detect_video_properties(
    input_url: str,
    ffprobe_path: str,
    *,
    log: Callable[[str], None] | None = None,
) -> VideoProperties:
    """Return VideoProperties(frame_rate, drop_frame, duration) for a video via ffprobe.

    Falls back to VideoProperties(24.0, False, 0.0) if ffprobe fails to run, logging a
    warning via `log`. Raises ValueError if ffprobe succeeds but finds no video streams.
    """
    _DEFAULTS = VideoProperties(24.0, False, 0.0)
    _DEFAULT_MSG = "24 fps, non-drop-frame, duration 0.0s"

    cmd = [
        ffprobe_path,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        "-select_streams",
        "v:0",
        input_url,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        if log:
            log(
                f"WARNING: Could not detect video properties ({e}) — using defaults: {_DEFAULT_MSG}. Timecode/frame-range calculations may be inaccurate.\n"
            )
        return _DEFAULTS

    if not data.get("streams"):
        msg = f"Attempted to detect video properties for {input_url!r}. Failed because ffprobe found no video streams — the file may not be a valid video or may be corrupt."
        raise ValueError(msg)

    stream = data["streams"][0]
    r_frame_rate = stream.get("r_frame_rate", "24/1")
    if "/" in r_frame_rate:
        num, den = map(int, r_frame_rate.split("/"))
        frame_rate = num / den
    else:
        frame_rate = float(r_frame_rate)

    drop_frame = abs(frame_rate - 29.97) < RATE_TOLERANCE or abs(frame_rate - 59.94) < RATE_TOLERANCE

    duration = 0.0
    if "format" in data and "duration" in data["format"]:
        duration = float(data["format"]["duration"])
    elif "duration" in stream:
        duration = float(stream["duration"])

    return VideoProperties(frame_rate, drop_frame, duration)


def seconds_to_ts(sec: float) -> str:
    """Return HH:MM:SS.mmm for ffmpeg."""
    sec = max(sec, 0)
    whole = int(sec)
    ms = round((sec - whole) * 1000)
    h = whole // 3600
    m = (whole % 3600) // 60
    s = whole % 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_video_segment_cmd(
    ffmpeg_path: str,
    input_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str,
) -> list[str]:
    """Build an ffmpeg command to extract a video segment.

    Uses stream copy with fast seek (pre-input -ss/-to) for segments starting at 0
    with sufficient duration. Uses re-encoding with accurate seek (post-input) otherwise.

    Stream copy requires fast seek — post-input accurate seek with -c copy silently
    drops video frames for many codecs/containers (e.g. H.264 in MOV/MP4).
    Data streams like timecode tracks (tmcd) are excluded since MP4 doesn't support them.
    """
    ss = seconds_to_ts(start_sec)
    to = seconds_to_ts(end_sec)
    duration = end_sec - start_sec

    base = [ffmpeg_path, "-hide_banner", "-y"]
    maps = ["-map", "0:v", "-map", "0:a?"]
    tail = ["-movflags", "+faststart", output_path]

    if duration >= MIN_SEGMENT_DURATION_FOR_STREAM_COPY and start_sec < 1e-5:
        return base + ["-ss", ss, "-to", to, "-i", input_path] + maps + ["-c", "copy"] + tail

    return base + ["-i", input_path, "-ss", ss, "-to", to] + maps + ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k"] + tail
