"""Pre-flight checks for proxy node inputs.

A proxy request a provider is certain to reject still costs the artist an upload and a wait, and
sometimes credits. These helpers let a node refuse it first, from what it can read locally.

There are two halves:

- **Readers** (`local_size_bytes`, `local_image_info`, `local_video_info`, `local_video_duration`) read one property of
  a media input from data already on this machine. None of them makes a network call or downloads a
  tool: the video reader uses ffprobe only if it is already installed. They return None when the
  value is a remote URL, can't be found, or can't be parsed, or when the tool to read it is missing
  or too slow, and the matching check then skips. A provider limit is only enforced
  where it can be read cheaply; anything else is left for the provider to enforce.
- **Checks** (`check_count`, `check_size`, `check_dimensions`, `check_aspect_ratio`,
  `check_aspect_ratio_range`, `check_mime_type`, `check_duration`) compare a value with a limit and return a `NodeError` ready to
  append to `validate_before_node_run`'s list, or None when the value is within the limit or
  unknown.

```python
def validate_before_node_run(self) -> list[Exception] | None:
    exceptions = super().validate_before_node_run() or []
    for position, image in enumerate(self.get_parameter_list_value("reference_images"), start=1):
        subject = input_subject("reference images", noun="image", position=position)
        error = check_size(local_size_bytes(image, kind="image"), maximum_bytes=MAX_BYTES, subject=subject)
        if error:
            exceptions.append(error)
    return exceptions or None
```
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
import shutil
import subprocess
import sys
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.files.path_utils import is_url
from PIL import Image, UnidentifiedImageError
from static_ffmpeg import run as static_ffmpeg_run

from griptape_nodes_library.media import MediaKind, coerce_media_url_or_data_uri

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "ImageInfo",
    "check_aspect_ratio",
    "check_aspect_ratio_range",
    "check_count",
    "check_dimensions",
    "check_duration",
    "check_mime_type",
    "check_size",
    "input_subject",
    "local_image_info",
    "local_size_bytes",
    "local_video_duration",
    "local_video_info",
]

BYTES_PER_MB = 1024 * 1024

# How far a measured width/height ratio may be from a named one, such as 16:9, and still match it.
# Covers the rounding in sizes like 1366x768.
ASPECT_RATIO_TOLERANCE = 0.01

LIMITS_LINK_LABEL = "Provider limits"

# Validation runs on the engine's event loop, so a probe that takes longer than this is abandoned
# and the duration check skipped.
FFPROBE_TIMEOUT_SECONDS = 5


class ImageInfo(NamedTuple):
    """An image's or video frame's pixel size, and an image's MIME type read from its header."""

    width: int
    height: int
    mime_type: str | None


def input_subject(parameter: str, *, noun: str, position: int | None = None) -> str:
    """The start of a message naming one input, such as "Image 2 in 'reference images'".

    Args:
        parameter: The parameter's name as the UI shows it.
        noun: What the input is, such as "image" or "video".
        position: The input's 1-based position in a list parameter, or None for a single input.
    """
    if position is None:
        return f"The {noun} in '{parameter}'"
    return f"{noun.capitalize()} {position} in '{parameter}'"


def local_size_bytes(value: Any, *, kind: MediaKind) -> int | None:
    """The size in bytes of a media input, or None if it can't be read locally.

    A data URI is measured from its base64 length. A path, a macro path, or a localhost static file
    URL is measured on disk. A remote URL returns None rather than being fetched.
    """
    location = coerce_media_url_or_data_uri(value, kind=kind)
    if not location:
        return None
    if location.startswith("data:"):
        payload = "".join(location.split(",", 1)[-1].split())
        return len(payload) * 3 // 4 - payload[-2:].count("=")
    path = _local_path(location)
    if path is None:
        return None
    try:
        return path.stat().st_size
    except OSError:
        return None


def local_image_info(value: Any) -> ImageInfo | None:
    """The pixel size and MIME type of an image input, or None if it can't be read locally.

    Reads only the image header. Returns None for a remote URL, and for a format Pillow can't open
    (HEIC, say), so the checks that use it skip rather than reject a valid image.
    """
    location = coerce_media_url_or_data_uri(value, kind="image")
    if not location:
        return None
    try:
        if location.startswith("data:"):
            payload = location.split(",", 1)[-1]
            source: Any = BytesIO(base64.b64decode(payload, validate=False))
        else:
            source = _local_path(location)
            if source is None:
                return None
        with Image.open(source) as image:
            # Phone and camera JPEGs with Multi-Picture extensions open as MPO; they are JPEG streams.
            mime_type = "image/jpeg" if image.format == "MPO" else Image.MIME.get(image.format or "")
            return ImageInfo(width=image.width, height=image.height, mime_type=mime_type)
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError, binascii.Error, ValueError) as e:
        logger.debug("Could not read image header for a pre-flight check, skipping it: %s", e)
        return None


def local_video_info(value: Any) -> ImageInfo | None:
    """The frame size of a video input, as an `ImageInfo` with no MIME type, or None if it can't be
    read locally.

    Probes the same way as `local_video_duration`. Reports the size as displayed, so a phone clip
    stored landscape with a 90-degree rotation reads as portrait.
    """
    data = _probe_local_video(value, ["-select_streams", "v:0", "-show_streams"])
    streams = data.get("streams") if data else None
    if not streams:
        return None
    stream = streams[0]
    try:
        width, height = int(stream["width"]), int(stream["height"])
    except (KeyError, TypeError, ValueError):
        return None
    rotation = stream.get("tags", {}).get("rotate")
    for side_data in stream.get("side_data_list", []):
        rotation = side_data.get("rotation", rotation)
    try:
        if rotation is not None and abs(int(float(rotation))) % 180 == 90:  # noqa: PLR2004
            width, height = height, width
    except (TypeError, ValueError):
        pass
    return ImageInfo(width=width, height=height, mime_type=None)


def local_video_duration(value: Any) -> float | None:
    """The duration in seconds of a video input, or None if it can't be read locally.

    Probes with ffprobe only a file on this machine, and only with an ffprobe that is already
    installed; see `_probe_local_video`.
    """
    data = _probe_local_video(value, ["-show_format"])
    duration = data.get("format", {}).get("duration") if data else None
    try:
        return float(duration) if duration is not None else None
    except (TypeError, ValueError):
        return None


def _probe_local_video(value: Any, args: list[str]) -> dict[str, Any] | None:
    """ffprobe's JSON for a video input on this machine, or None.

    Only a file on this machine is probed: a remote URL would make ffprobe download it, and a data
    URI is too large to pass on its command line. ffprobe is used only if it is already installed
    (see `_installed_ffprobe`), never fetched, and is given `FFPROBE_TIMEOUT_SECONDS` to answer.
    """
    location = coerce_media_url_or_data_uri(value, kind="video")
    if not location or location.startswith("data:"):
        return None
    path = _local_path(location)
    if path is None:
        return None
    ffprobe = _installed_ffprobe()
    if ffprobe is None:
        logger.debug("ffprobe is not installed, skipping a pre-flight video check")
        return None
    cmd = [ffprobe, "-v", "error", "-print_format", "json", *args, str(path)]
    try:
        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=True, timeout=FFPROBE_TIMEOUT_SECONDS
        )
        data = json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as e:
        logger.debug("Could not probe a video for a pre-flight check, skipping it: %s", e)
        return None
    return data if isinstance(data, dict) else None


def check_count(
    count: int, *, maximum: int, parameter: str, noun: str = "items", docs_url: str | None = None
) -> NodeError | None:
    """An error when more inputs are connected than the provider accepts.

    Args:
        count: How many inputs are connected.
        maximum: The most the provider accepts.
        parameter: The parameter's name as the UI shows it.
        noun: The plural of what is counted, such as "images".
        docs_url: The provider's page documenting the limit.
    """
    if count <= maximum:
        return None
    return _error(
        f"{count} {noun} are connected to '{parameter}' and the limit is {maximum}. Remove {count - maximum}.",
        docs_url,
    )


def check_size(
    size_bytes: int | None,
    *,
    maximum_bytes: int,
    subject: str,
    fix: str = "Use a smaller file.",
    docs_url: str | None = None,
) -> NodeError | None:
    """An error when an input file is larger than the provider accepts.

    Args:
        size_bytes: The file's size, from `local_size_bytes`. None skips the check.
        maximum_bytes: The largest file the provider accepts.
        subject: The input, from `input_subject`.
        fix: What the artist can do about it.
        docs_url: The provider's page documenting the limit.
    """
    if size_bytes is None or size_bytes <= maximum_bytes:
        return None
    return _error(
        f"{subject} is {_format_past(size_bytes / BYTES_PER_MB, maximum_bytes / BYTES_PER_MB, 1)} MB and the limit "
        f"is {_format_limit(maximum_bytes / BYTES_PER_MB)} MB. {fix}",
        docs_url,
    )


def check_dimensions(
    info: ImageInfo | None,
    *,
    subject: str,
    min_side: int | None = None,
    max_side: int | None = None,
    max_megapixels: float | None = None,
    docs_url: str | None = None,
) -> NodeError | None:
    """An error when an image or video frame's pixel size is outside what the provider accepts.

    Args:
        info: The pixel size, from `local_image_info` or `local_video_info`. None skips the check.
        subject: The input, from `input_subject`.
        min_side: The fewest pixels accepted for the width and for the height.
        max_side: The most pixels accepted for the width and for the height.
        max_megapixels: The most pixels accepted in total, in millions.
        docs_url: The provider's page documenting the limit.
    """
    if info is None:
        return None
    size = f"{info.width}x{info.height}"
    if min_side is not None and min(info.width, info.height) < min_side:
        return _error(
            f"{subject} is {size} and each side must be at least {min_side} pixels. Use a larger one.", docs_url
        )
    if max_side is not None and max(info.width, info.height) > max_side:
        return _error(
            f"{subject} is {size} and each side must be at most {max_side} pixels. Resize it smaller.", docs_url
        )
    megapixels = info.width * info.height / 1_000_000
    if max_megapixels is not None and megapixels > max_megapixels:
        return _error(
            f"{subject} is {size} ({_format_past(megapixels, max_megapixels, 1)} megapixels) and the limit is "
            f"{_format_limit(max_megapixels)} megapixels. Resize it smaller.",
            docs_url,
        )
    return None


def check_aspect_ratio(
    info: ImageInfo | None, *, subject: str, allowed: Sequence[str], docs_url: str | None = None
) -> NodeError | None:
    """An error when an image or video frame doesn't match one of the provider's named aspect ratios.

    Args:
        info: The pixel size, from `local_image_info` or `local_video_info`. None skips the check.
        subject: The input, from `input_subject`.
        allowed: The ratios accepted, such as ("16:9", "9:16").
        docs_url: The provider's page documenting the limit.
    """
    if info is None or info.height == 0:
        return None
    ratio = info.width / info.height
    if any(abs(ratio - _parse_ratio(name)) <= ASPECT_RATIO_TOLERANCE for name in allowed):
        return None
    return _error(
        f"{subject} is {info.width}x{info.height}, which isn't a supported aspect ratio. "
        f"Crop it to {_join_or(allowed)}.",
        docs_url,
    )


def check_aspect_ratio_range(
    info: ImageInfo | None, *, subject: str, widest: str, tallest: str, docs_url: str | None = None
) -> NodeError | None:
    """An error when an image or video frame is wider or taller than the provider accepts.

    Args:
        info: The pixel size, from `local_image_info` or `local_video_info`. None skips the check.
        subject: The input, from `input_subject`.
        widest: The widest ratio accepted, as the provider writes it, such as "3:1".
        tallest: The tallest ratio accepted, such as "1:3".
        docs_url: The provider's page documenting the limit.
    """
    if info is None or info.height == 0:
        return None
    ratio = info.width / info.height
    if _parse_ratio(tallest) <= ratio <= _parse_ratio(widest):
        return None
    shape = "wide" if ratio > _parse_ratio(widest) else "tall"
    return _error(
        f"{subject} is {info.width}x{info.height}, which is too {shape}. The aspect ratio must be between "
        f"{tallest} and {widest}. Crop it.",
        docs_url,
    )


def check_mime_type(
    mime_type: str | None, *, subject: str, allowed: Sequence[str], docs_url: str | None = None
) -> NodeError | None:
    """An error when an input's format isn't one the provider accepts.

    Args:
        mime_type: The input's MIME type, such as from `local_image_info`. None skips the check.
        subject: The input, from `input_subject`.
        allowed: The MIME types the provider accepts, such as ("image/jpeg", "image/png").
        docs_url: The provider's page documenting the limit.
    """
    if mime_type is None or mime_type in allowed:
        return None
    formats = _join_or([name.split("/", 1)[-1].upper() for name in allowed])
    current = mime_type.split("/", 1)[-1].upper()
    return _error(f"{subject} is a {current} file. Convert it to {formats}.", docs_url)


def check_duration(  # noqa: PLR0913
    seconds: float | None,
    *,
    subject: str,
    minimum: float | None = None,
    maximum: float | None = None,
    fix: str | None = None,
    docs_url: str | None = None,
) -> NodeError | None:
    """An error when an audio or video input is shorter or longer than the provider accepts.

    Args:
        seconds: The input's duration, such as from `local_video_duration`. None skips the check.
        subject: The input, from `input_subject`.
        minimum: The shortest input accepted, in seconds.
        maximum: The longest input accepted, in seconds.
        fix: What the artist can do about it. Defaults to trimming or using a longer clip.
        docs_url: The provider's page documenting the limit.
    """
    if seconds is None:
        return None
    if maximum is not None and seconds > maximum:
        next_step = fix or "Use a Trim Video node to shorten it."
        return _error(
            f"{subject} is {_format_past(seconds, maximum, 2)} seconds long and the limit is {_format_limit(maximum)}. {next_step}",
            docs_url,
        )
    if minimum is not None and seconds < minimum:
        next_step = fix or "Use a longer clip."
        return _error(
            f"{subject} is {_format_past(seconds, minimum, 2)} seconds long and the minimum is {_format_limit(minimum, round_up=True)}. {next_step}",
            docs_url,
        )
    return None


def _error(message: str, docs_url: str | None) -> NodeError:
    links = [NodeErrorLink(label=LIMITS_LINK_LABEL, url=docs_url)] if docs_url else None
    return NodeError(message, links=links)


def _installed_ffprobe() -> str | None:
    """The path to an ffprobe that is already installed, or None.

    Prefers the static-ffmpeg copy the library uses elsewhere, but only once its download has
    finished (it writes `installed.crumb` last); asking static-ffmpeg for the path would download it
    otherwise. Falls back to an ffprobe on PATH.
    """
    try:
        platform_dir = Path(static_ffmpeg_run.get_platform_dir())
    except OSError:
        platform_dir = None
    if platform_dir is not None and (platform_dir / "installed.crumb").is_file():
        exe = platform_dir / ("ffprobe.exe" if sys.platform == "win32" else "ffprobe")
        if exe.is_file():
            return str(exe)
    return shutil.which("ffprobe")


def _local_path(location: str) -> Path | None:
    """The file on this machine a location names, or None for a remote URL or an unresolvable path."""
    try:
        resolved = File(location).resolve()
        if is_url(resolved):
            return None
        path = Path(resolved)
        return path if path.is_file() else None
    except (FileLoadError, OSError, ValueError):
        return None


def _format_limit(limit: float, *, round_up: bool = False) -> str:
    """A limit as messages show it: up to 6 significant figures, rounded toward the allowed side.

    A maximum rounds down and a minimum rounds up (`round_up`), so a value past the limit can always
    be printed past the shown limit too.
    """
    text = f"{limit:g}"
    shown = float(text)
    if shown == limit or limit <= 0 or (shown > limit) == round_up:
        return text
    scale = 10 ** max(6 - len(str(int(limit))), 0)
    rounder = math.ceil if round_up else math.floor
    return f"{rounder(limit * scale) / scale:g}"


def _format_past(value: float, limit: float, decimals: int) -> str:
    """`value` to `decimals` places, with more when fewer would read as equal to `limit` as
    `_format_limit` shows it, or on its other side.

    A file of 200,000,001 bytes against a 200,000,000-byte limit (shown as "190.734 MB") must not
    be reported as "190.735 MB" either way round.
    """
    over = value > limit
    shown_limit = float(_format_limit(limit, round_up=not over))
    for places in range(decimals, decimals + 9):
        text = f"{value:.{places}f}"
        rounded = float(text)
        if rounded != shown_limit and (rounded > shown_limit) == over:
            return text
    return repr(value)


def _parse_ratio(name: str) -> float:
    width, height = name.split(":", 1)
    return float(width) / float(height)


def _join_or(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} or {names[-1]}"
