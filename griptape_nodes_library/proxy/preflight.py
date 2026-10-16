"""Pre-flight checks for proxy node inputs.

A proxy request a provider is certain to reject still costs the artist an upload and a wait, and
sometimes credits. These helpers let a node refuse it first, from what it can read locally.

There are two halves:

- **Readers** (`local_size_bytes`, `local_image_info`, `local_video_duration`) read one property of
  a media input without a network call. They return None when the value is a remote URL, can't be
  found, or can't be parsed, and the matching check then skips. A provider limit is only enforced
  where it can be read cheaply; anything else is left for the provider to enforce.
- **Checks** (`check_count`, `check_size`, `check_dimensions`, `check_aspect_ratio`,
  `check_mime_type`, `check_duration`) compare a value with a limit and return a `NodeError` ready to
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
import logging
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, NamedTuple

from griptape_nodes.exe_types.core_types import NodeError, NodeErrorLink
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.files.path_utils import is_url
from PIL import Image, UnidentifiedImageError

from griptape_nodes_library.media import MediaKind, coerce_media_url_or_data_uri
from griptape_nodes_library.utils.ffmpeg_utils import extract_video_metadata_structured

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "ImageInfo",
    "check_aspect_ratio",
    "check_count",
    "check_dimensions",
    "check_duration",
    "check_mime_type",
    "check_size",
    "input_subject",
    "local_image_info",
    "local_size_bytes",
    "local_video_duration",
]

BYTES_PER_MB = 1024 * 1024

# How far a measured width/height ratio may be from a named one, such as 16:9, and still match it.
# Covers the rounding in sizes like 1366x768.
ASPECT_RATIO_TOLERANCE = 0.01

LIMITS_LINK_LABEL = "Provider limits"


class ImageInfo(NamedTuple):
    """An image's pixel size and MIME type, read from its header."""

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
        payload = location.split(",", 1)[-1].strip()
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
            mime_type = Image.MIME.get(image.format or "")
            return ImageInfo(width=image.width, height=image.height, mime_type=mime_type)
    except (OSError, UnidentifiedImageError, binascii.Error, ValueError) as e:
        logger.debug("Could not read image header for a pre-flight check, skipping it: %s", e)
        return None


def local_video_duration(value: Any) -> float | None:
    """The duration in seconds of a video input, or None if it can't be read locally.

    Probes the file with ffprobe through `extract_video_metadata_structured`. Only a file on this
    machine is probed: a remote URL would make ffprobe download it, and a data URI is too large to
    pass on its command line. Starts a subprocess, so call it once per input.
    """
    location = coerce_media_url_or_data_uri(value, kind="video")
    if not location or location.startswith("data:"):
        return None
    path = _local_path(location)
    if path is None or not path.is_file():
        return None
    try:
        metadata = extract_video_metadata_structured(str(path))
    except (ValueError, RuntimeError) as e:
        logger.debug("Could not probe video duration for a pre-flight check, skipping it: %s", e)
        return None
    return metadata.file_details.optional_duration


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
        f"{subject} is {size_bytes / BYTES_PER_MB:.1f} MB and the limit is {_format_mb(maximum_bytes)}. {fix}",
        docs_url,
    )


def check_dimensions(  # noqa: PLR0913
    info: ImageInfo | None,
    *,
    subject: str,
    min_width: int | None = None,
    min_height: int | None = None,
    max_width: int | None = None,
    max_height: int | None = None,
    max_megapixels: float | None = None,
    docs_url: str | None = None,
) -> NodeError | None:
    """An error when an image's pixel size is outside what the provider accepts.

    Args:
        info: The image's header, from `local_image_info`. None skips the check.
        subject: The input, from `input_subject`.
        min_width: The narrowest image accepted, in pixels.
        min_height: The shortest image accepted, in pixels.
        max_width: The widest image accepted, in pixels.
        max_height: The tallest image accepted, in pixels.
        max_megapixels: The most pixels accepted, in millions.
        docs_url: The provider's page documenting the limit.
    """
    if info is None:
        return None
    size = f"{info.width}x{info.height}"
    if (min_width is not None and info.width < min_width) or (min_height is not None and info.height < min_height):
        return _error(
            f"{subject} is {size} and the minimum is {min_width or 1}x{min_height or 1}. Use a larger image.",
            docs_url,
        )
    if (max_width is not None and info.width > max_width) or (max_height is not None and info.height > max_height):
        maximum = f"{max_width or '∞'}x{max_height or '∞'}"
        return _error(f"{subject} is {size} and the maximum is {maximum}. Resize it smaller.", docs_url)
    megapixels = info.width * info.height / 1_000_000
    if max_megapixels is not None and megapixels > max_megapixels:
        return _error(
            f"{subject} is {size} ({megapixels:.1f} megapixels) and the limit is {max_megapixels:g} megapixels. "
            "Resize it smaller.",
            docs_url,
        )
    return None


def check_aspect_ratio(  # noqa: PLR0913
    info: ImageInfo | None,
    *,
    subject: str,
    allowed: Sequence[str] = (),
    min_ratio: float | None = None,
    max_ratio: float | None = None,
    docs_url: str | None = None,
) -> NodeError | None:
    """An error when an image's shape is outside what the provider accepts.

    Args:
        info: The image's header, from `local_image_info`. None skips the check.
        subject: The input, from `input_subject`.
        allowed: Named ratios the image must match, such as ("16:9", "9:16"). Empty allows any.
        min_ratio: The smallest width/height accepted.
        max_ratio: The largest width/height accepted.
        docs_url: The provider's page documenting the limit.
    """
    if info is None or info.height == 0:
        return None
    ratio = info.width / info.height
    shape = f"{info.width}x{info.height}"
    if allowed and not any(abs(ratio - _parse_ratio(name)) <= ASPECT_RATIO_TOLERANCE for name in allowed):
        return _error(
            f"{subject} is {shape}, which isn't a supported aspect ratio. Crop it to {_join_or(allowed)}.", docs_url
        )
    if (min_ratio is not None and ratio < min_ratio) or (max_ratio is not None and ratio > max_ratio):
        low = f"{min_ratio:g}" if min_ratio is not None else "any"
        high = f"{max_ratio:g}" if max_ratio is not None else "any"
        return _error(
            f"{subject} is {shape}, a width/height ratio of {ratio:.2f}. The accepted range is {low} to {high}. "
            "Crop it to a less extreme shape.",
            docs_url,
        )
    return None


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
        return _error(f"{subject} is {seconds:.2f} seconds long and the limit is {maximum:g}. {next_step}", docs_url)
    if minimum is not None and seconds < minimum:
        next_step = fix or "Use a longer clip."
        return _error(f"{subject} is {seconds:.2f} seconds long and the minimum is {minimum:g}. {next_step}", docs_url)
    return None


def _error(message: str, docs_url: str | None) -> NodeError:
    links = [NodeErrorLink(label=LIMITS_LINK_LABEL, url=docs_url)] if docs_url else None
    return NodeError(message, links=links)


def _local_path(location: str) -> Path | None:
    """The file on this machine a location names, or None for a remote URL or an unresolvable path."""
    try:
        resolved = File(location).resolve()
    except (FileLoadError, OSError, ValueError):
        return None
    if is_url(resolved):
        return None
    path = Path(resolved)
    return path if path.is_file() else None


def _format_mb(size_bytes: int) -> str:
    return f"{size_bytes / BYTES_PER_MB:g} MB"


def _parse_ratio(name: str) -> float:
    width, height = name.split(":", 1)
    return float(width) / float(height)


def _join_or(names: Sequence[str]) -> str:
    if len(names) <= 1:
        return "".join(names)
    return f"{', '.join(names[:-1])} or {names[-1]}"
