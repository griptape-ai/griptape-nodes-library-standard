"""Shared helpers for Black Forest Labs FLUX 3 image and video proxy nodes."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.utils.video_utils import find_mp4_video_input_problems

SUPPORTED_VIDEO_CONTAINER = "mp4"


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


async def prepare_mp4_video_data_uri(
    video_input: Any,
    *,
    param_label: str,
    action_verb: str,
    service_label: str,
    max_duration_seconds: float,
    duration_prober: Callable[[str], float | None],
) -> str:
    """Validate an MP4 input against container and duration limits and return its data URI."""
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

    if video_value.startswith("data:"):
        return video_value

    try:
        return await File(video_value).aread_data_uri(fallback_mime="video/mp4")
    except FileLoadError as e:
        msg = f"Could not load video from '{video_value}': {e}"
        raise NodeError(msg) from e
