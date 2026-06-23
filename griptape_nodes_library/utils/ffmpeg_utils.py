"""FFmpeg utility functions for cross-platform executable path resolution."""

import json
import math
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

import static_ffmpeg.run  # type: ignore[import-untyped]

DEFAULT_FRAME_RATE = 24.0
RATE_TOLERANCE = 0.1
MIN_SEGMENT_DURATION_FOR_STREAM_COPY = 2.0  # seconds — below this stream copy is unreliable


class FfmpegPaths(NamedTuple):
    ffmpeg: str
    ffprobe: str


class VideoProperties(NamedTuple):
    frame_rate: float
    drop_frame: bool
    duration: float


@dataclass
class FileDetails:
    """File-level metadata with guaranteed and optional fields."""

    codec_name: str
    codec_type: str  # Always "video" for video streams

    optional_duration: float | None = None
    optional_bit_rate: int | None = None
    optional_codec_long_name: str | None = None
    optional_profile: str | None = None
    optional_level: int | None = None
    optional_pixel_format: str | None = None
    optional_file_size: int | None = None
    optional_format_name: str | None = None


@dataclass
class Dimensions:
    """Dimension-related metadata with guaranteed and optional fields."""

    width: int
    height: int
    aspect_ratio_decimal: float
    aspect_ratio_string: str

    optional_coded_width: int | None = None
    optional_coded_height: int | None = None
    optional_sample_aspect_ratio: str | None = None
    optional_display_aspect_ratio: str | None = None


@dataclass
class ColorDetails:
    """Color and visual quality metadata."""

    optional_color_space: str | None = None
    optional_color_transfer: str | None = None
    optional_color_primaries: str | None = None
    optional_chroma_location: str | None = None
    optional_field_order: str | None = None


@dataclass
class FrameDetails:
    """Frame rate and timing metadata with guaranteed and optional fields."""

    r_frame_rate: str  # Raw fraction string like "30/1"
    avg_frame_rate: str
    time_base: str
    frame_rate: float  # Selected playback rate (avg_frame_rate preferred, r_frame_rate or nb_frames/duration fallback)

    optional_nb_frames: int | None = None
    optional_start_time: float | None = None


@dataclass
class VideoMetadata:
    """Container for all video metadata categories."""

    file_details: FileDetails
    dimensions: Dimensions
    color_details: ColorDetails
    frame_details: FrameDetails


def _resolve_executables() -> tuple[str, str]:
    try:
        return static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
    except (FileNotFoundError, OSError, ImportError) as e:
        msg = f"FFmpeg/FFprobe not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
        raise RuntimeError(msg) from e


def get_ffmpeg_path() -> str:
    """Return the path to the ffmpeg executable."""
    return _resolve_executables()[0]


def get_ffprobe_path() -> str:
    """Return the path to the ffprobe executable."""
    return _resolve_executables()[1]


def get_ffmpeg_paths() -> FfmpegPaths:
    """Return both ffmpeg and ffprobe executable paths in a single call."""
    return FfmpegPaths(*_resolve_executables())


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
    _DEFAULTS = VideoProperties(DEFAULT_FRAME_RATE, False, 0.0)
    _DEFAULT_MSG = f"{DEFAULT_FRAME_RATE} fps, non-drop-frame, duration 0.0s"

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
    frame_rate = _parse_fraction_string(stream.get("r_frame_rate", "24/1")) or DEFAULT_FRAME_RATE

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


def _parse_fraction_string(fraction_str: str) -> float:
    """Parse ffprobe fraction string like '30/1' or '30000/1001' to float."""
    if not fraction_str or fraction_str == "0/0":
        return 0.0
    try:
        parts = fraction_str.split("/")
        if len(parts) == 2:  # noqa: PLR2004
            numerator = float(parts[0])
            denominator = float(parts[1])
            if denominator != 0:
                return numerator / denominator
    except (ValueError, ZeroDivisionError):
        pass
    return 0.0


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _calculate_aspect_ratio_string(width: int, height: int) -> str:
    divisor = math.gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _parse_file_details(video_stream: dict, fmt: dict) -> FileDetails:
    stream_duration = _safe_float(video_stream.get("duration"))
    format_duration = _safe_float(fmt.get("duration"))
    return FileDetails(
        codec_name=video_stream["codec_name"],
        codec_type=video_stream["codec_type"],
        optional_duration=stream_duration or format_duration,
        optional_bit_rate=_safe_int(video_stream.get("bit_rate")),
        optional_codec_long_name=video_stream.get("codec_long_name"),
        optional_profile=video_stream.get("profile"),
        optional_level=_safe_int(video_stream.get("level")),
        optional_pixel_format=video_stream.get("pix_fmt"),
        optional_file_size=_safe_int(fmt.get("size")),
        optional_format_name=fmt.get("format_name"),
    )


def _parse_dimensions(
    video_stream: dict, width: int, height: int, aspect_ratio_string: str, aspect_ratio_decimal: float
) -> Dimensions:
    return Dimensions(
        width=width,
        height=height,
        aspect_ratio_decimal=aspect_ratio_decimal,
        aspect_ratio_string=aspect_ratio_string,
        optional_coded_width=_safe_int(video_stream.get("coded_width")),
        optional_coded_height=_safe_int(video_stream.get("coded_height")),
        optional_sample_aspect_ratio=video_stream.get("sample_aspect_ratio"),
        optional_display_aspect_ratio=video_stream.get("display_aspect_ratio"),
    )


def _parse_color_details(video_stream: dict) -> ColorDetails:
    return ColorDetails(
        optional_color_space=video_stream.get("color_space"),
        optional_color_transfer=video_stream.get("color_transfer"),
        optional_color_primaries=video_stream.get("color_primaries"),
        optional_chroma_location=video_stream.get("chroma_location"),
        optional_field_order=video_stream.get("field_order"),
    )


def _select_frame_rate(video_stream: dict) -> float:
    """Determine playback frame rate from ffprobe stream data.

    FFmpeg's avformat.h documents r_frame_rate as the lowest rate at which all
    timestamps can be represented exactly and explicitly notes it is "just a guess".
    avg_frame_rate is the actual average playback rate and is preferred. Falls back to
    r_frame_rate when avg is unset ("0/0"), then to nb_frames/duration as a last resort.
    """
    avg_parsed = _parse_fraction_string(video_stream.get("avg_frame_rate", ""))
    if avg_parsed > 0.0:
        return avg_parsed

    r_parsed = _parse_fraction_string(video_stream.get("r_frame_rate", ""))
    if r_parsed > 0.0:
        return r_parsed

    nb_frames = _safe_int(video_stream.get("nb_frames"))
    duration = _safe_float(video_stream.get("duration"))
    if nb_frames is not None and nb_frames > 0 and duration is not None and duration > 0:
        return nb_frames / duration

    return 0.0


def _parse_frame_details(video_stream: dict) -> FrameDetails:
    return FrameDetails(
        r_frame_rate=video_stream["r_frame_rate"],
        avg_frame_rate=video_stream["avg_frame_rate"],
        time_base=video_stream["time_base"],
        frame_rate=_select_frame_rate(video_stream),
        optional_nb_frames=_safe_int(video_stream.get("nb_frames")),
        optional_start_time=_safe_float(video_stream.get("start_time")),
    )


def extract_video_metadata_structured(video_path: str) -> VideoMetadata:
    """Extract structured video metadata from a local file via ffprobe.

    Raises ValueError on any ffprobe failure so callers can decide how to surface errors.
    """
    ffprobe_path = get_ffprobe_path()

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
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
    except subprocess.TimeoutExpired as e:
        msg = f"When attempting to extract video metadata, ffprobe operation timed out: {e}"
        raise ValueError(msg) from e
    except subprocess.CalledProcessError as e:
        msg = f"When attempting to extract video metadata, ffprobe failed: {e.stderr}"
        raise ValueError(msg) from e

    try:
        probe_data = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        msg = f"When attempting to extract video metadata, ffprobe returned invalid JSON: {e}"
        raise ValueError(msg) from e

    streams = probe_data.get("streams", [])
    if not streams:
        msg = "When attempting to extract video metadata, no video streams found in file"
        raise ValueError(msg)

    video_stream = streams[0]
    fmt = probe_data.get("format", {})

    try:
        width = video_stream["width"]
        height = video_stream["height"]
    except KeyError as e:
        msg = f"When attempting to extract video metadata, required field missing from ffprobe output: {e}"
        raise ValueError(msg) from e

    aspect_ratio_decimal = width / height
    aspect_ratio_string = video_stream.get("display_aspect_ratio") or _calculate_aspect_ratio_string(width, height)

    return VideoMetadata(
        file_details=_parse_file_details(video_stream, fmt),
        dimensions=_parse_dimensions(video_stream, width, height, aspect_ratio_string, aspect_ratio_decimal),
        color_details=_parse_color_details(video_stream),
        frame_details=_parse_frame_details(video_stream),
    )


def video_metadata_to_player_dict(metadata: VideoMetadata, video_path: str) -> dict[str, Any]:
    """Map a VideoMetadata struct to the flat dict the video player details view expects."""
    meta: dict[str, Any] = {
        "width": metadata.dimensions.width,
        "height": metadata.dimensions.height,
        "codec": metadata.file_details.codec_name,
        "frame_rate": metadata.frame_details.frame_rate,
    }

    if metadata.file_details.optional_file_size is not None:
        meta["file_size"] = metadata.file_details.optional_file_size
    else:
        try:
            meta["file_size"] = os.path.getsize(video_path)
        except OSError:
            pass

    # "mp4,mov,m4a,3gp,…" — take the first token as the canonical format name.
    if metadata.file_details.optional_format_name:
        meta["format"] = metadata.file_details.optional_format_name.split(",")[0]

    if metadata.color_details.optional_color_space:
        meta["color_space"] = metadata.color_details.optional_color_space

    if metadata.file_details.optional_duration is not None:
        meta["duration_seconds"] = metadata.file_details.optional_duration

    return meta


def extract_video_player_metadata(video_path: str) -> dict[str, Any]:
    """Extract video metadata in the shape expected by the video player details view.

    Never raises — returns {} on any failure so callers can always attach the result
    to an artifact unconditionally.
    """
    try:
        metadata = extract_video_metadata_structured(video_path)
        return video_metadata_to_player_dict(metadata, video_path)
    except Exception:
        return {}


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

    return (
        base
        + ["-i", input_path, "-ss", ss, "-to", to]
        + maps
        + ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-c:a", "aac", "-b:a", "192k"]
        + tail
    )
