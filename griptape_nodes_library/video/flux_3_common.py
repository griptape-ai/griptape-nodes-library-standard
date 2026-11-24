"""Shared helpers for Black Forest Labs FLUX 3 image and video proxy nodes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.utils.video_utils import find_mp4_video_input_problems

SUPPORTED_VIDEO_CONTAINER = "mp4"
# FLUX 3 Video Upscale API reference: "base64-encoded mp4 (max 50MB)"; FLUX Video Edit docs: "up to
# 15 seconds and 50 MiB". Both are applied to the data URI the request carries, so the MP4 file itself
# can be about three quarters of that.
MAX_INPUT_DATA_URI_BYTES = 50 * 1024 * 1024
BYTES_PER_MIB = 1024 * 1024
MP4_DATA_URI_PREFIX = "data:video/mp4;base64,"


def max_file_bytes_for_data_uri(max_data_uri_bytes: int) -> int:
    """Return the largest file whose MP4 data URI fits in ``max_data_uri_bytes``."""
    return (max_data_uri_bytes - len(MP4_DATA_URI_PREFIX)) // 4 * 3


def data_uri_size_for_file(file_bytes: int) -> int:
    """Return the length of the MP4 data URI a file of ``file_bytes`` encodes to."""
    return len(MP4_DATA_URI_PREFIX) + -(-file_bytes // 3) * 4


INPUT_VIDEO_SIZE_LIMIT_TEXT = (
    f"{MAX_INPUT_DATA_URI_BYTES // BYTES_PER_MIB} MiB once base64-encoded, "
    f"about {max_file_bytes_for_data_uri(MAX_INPUT_DATA_URI_BYTES) / BYTES_PER_MIB:.1f} MiB of MP4"
)


class Flux3SafetyTolerance(StrEnum):
    LEAST_RESTRICTIVE = "least restrictive"
    MODERATE = "moderate"
    MOST_RESTRICTIVE = "most restrictive"


SAFETY_TOLERANCE_OPTIONS = [
    Flux3SafetyTolerance.LEAST_RESTRICTIVE.value,
    Flux3SafetyTolerance.MODERATE.value,
    Flux3SafetyTolerance.MOST_RESTRICTIVE.value,
]


def parse_flux_3_safety_tolerance(value: str | None) -> int:
    """Convert a safety tolerance preset string to BFL's integer tolerance level."""
    if not value:
        msg = "Pick a 'safety_tolerance' value."
        raise NodeError(msg)

    match value:
        case Flux3SafetyTolerance.MOST_RESTRICTIVE:
            return 0
        case Flux3SafetyTolerance.MODERATE:
            return 2
        case Flux3SafetyTolerance.LEAST_RESTRICTIVE:
            return 4
        case _:
            msg = f"'safety_tolerance' must be one of {SAFETY_TOLERANCE_OPTIONS}. It's set to {value!r}."
            raise NodeError(msg)


def input_video_data_uri_size(video_value: str) -> int | None:
    """Return the length of the data URI the video is sent as, or None when it can't be measured.

    A data URI is measured directly. A local file's encoded length is computed from its metadata
    without reading it. A remote URL has no local size to read, so it returns None.
    """
    if video_value.startswith("data:"):
        return len(video_value)
    try:
        location = File(video_value).resolve()
    except FileLoadError:
        return None
    try:
        # A remote URL comes back from resolve() unchanged and fails the stat.
        return data_uri_size_for_file(Path(location).stat().st_size)
    except OSError:
        return None


def find_video_size_problem(
    video_input: Any, *, param_label: str, service_label: str, max_data_uri_bytes: int
) -> str | None:
    """Return an error message when the video's data URI is longer than ``max_data_uri_bytes``, else None."""
    video_value = coerce_media_url_or_data_uri(video_input, kind="video")
    if not video_value:
        return None
    size_bytes = input_video_data_uri_size(video_value)
    if size_bytes is None or size_bytes <= max_data_uri_bytes:
        return None
    return (
        f"The video connected to '{param_label}' is {size_bytes / BYTES_PER_MIB:.1f} MiB once base64-encoded. "
        f"{service_label}'s limit is {max_data_uri_bytes / BYTES_PER_MIB:g} MiB encoded, "
        f"about {max_file_bytes_for_data_uri(max_data_uri_bytes) / BYTES_PER_MIB:.1f} MiB of MP4. "
        "Use a Trim Video or Resize Video node to make it smaller."
    )


async def prepare_mp4_video_data_uri(
    video_input: Any,
    *,
    param_label: str,
    action_verb: str,
    service_label: str,
    max_duration_seconds: float,
    duration_prober: Callable[[str], float | None],
    max_data_uri_bytes: int | None = None,
) -> str:
    """Validate an MP4 input against container, duration, and size limits and return its data URI.

    The data URI's length is only checked when ``max_data_uri_bytes`` is given.
    """
    video_value = coerce_media_url_or_data_uri(video_input, kind="video")
    if not video_value:
        msg = f"Connect a video to '{param_label}'."
        raise NodeError(msg)

    problems = await asyncio.to_thread(
        find_mp4_video_input_problems,
        video_value,
        service_label=service_label,
        max_duration_seconds=max_duration_seconds,
        duration_prober=duration_prober,
        supported_container=SUPPORTED_VIDEO_CONTAINER,
    )
    if problems:
        msg = (
            f"Cannot {action_verb} this video connected to '{param_label}'. "
            f"{' '.join(problems)} Use a Trim Video node to fix this."
        )
        raise NodeError(msg)

    if max_data_uri_bytes is not None:
        size_problem = await asyncio.to_thread(
            find_video_size_problem,
            video_value,
            param_label=param_label,
            service_label=service_label,
            max_data_uri_bytes=max_data_uri_bytes,
        )
        if size_problem:
            raise NodeError(size_problem)

    if video_value.startswith("data:"):
        return video_value

    try:
        data_uri = await File(video_value).aread_data_uri(fallback_mime="video/mp4")
    except FileLoadError as e:
        msg = f"Could not load video from '{video_value}': {e}"
        raise NodeError(msg) from e

    # A remote URL's size is only known once it is downloaded, and a file's MIME type can make its
    # data URI a few bytes longer than the estimate, so the encoded result is checked as well.
    if max_data_uri_bytes is not None:
        size_problem = find_video_size_problem(
            data_uri, param_label=param_label, service_label=service_label, max_data_uri_bytes=max_data_uri_bytes
        )
        if size_problem:
            raise NodeError(size_problem)
    return data_uri
