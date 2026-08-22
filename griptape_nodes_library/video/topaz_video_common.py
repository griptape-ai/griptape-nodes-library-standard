"""Shared plumbing for the Topaz video nodes.

Topaz's video API takes one request shape -- ``source`` / ``output`` / ``filters`` --
regardless of which model runs. Everything here is about *that shape* rather than about
any one model: mapping a file to Topaz's ``source.container`` enum, probing the source
with ffprobe, deriving a frame count the API will accept, and the pixel-area threshold
the per-frame billing tiers turn on.

What differs per model -- resize modes, creative controls, frame caps, output encoding --
stays in the node modules.

Everything here is a module-level pure function, following ``seedance_common``: none of it
needs node state beyond a name to put in an error message, so none of it is a mixin. The
nodes keep thin methods that delegate, which is also what lets a test monkeypatch
``SomeNode._probe_source`` and have it take effect.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.utils.ffmpeg_utils import extract_video_metadata_structured

if TYPE_CHECKING:
    from typing import Any

    from griptape_nodes_library.utils.ffmpeg_utils import VideoMetadata

__all__ = [
    "CONTAINER_BY_TOKEN",
    "TIER_1080P_MAX_PIXELS",
    "derive_container",
    "frame_count",
    "probe_source",
    "to_even",
]

# Topaz's source.container enum is exactly mp4/mov/mkv:
# https://developer.topazlabs.com/reference/api-endpoints/video/create-request.md
#
# ffprobe's format_name can't tell these apart -- mp4 and mov both report
# "mov,mp4,m4a,3gp,3g2,mj2", and mkv and webm both report "matroska,webm" (which
# isn't even a valid Topaz value) -- so container is derived from the file
# extension (or, for a data URI, the MIME subtype) instead. See derive_container.
CONTAINER_BY_TOKEN: dict[str, str] = {
    "mp4": "mp4",
    "m4v": "mp4",
    "mov": "mov",
    "qt": "mov",
    "quicktime": "mov",
    "mkv": "mkv",
    "matroska": "mkv",
    "x-matroska": "mkv",
}

# Every per-frame Topaz video model bills in two rate tiers, chosen by output pixel
# *area*, not height. A portrait 1080x1920 output bills as 1080p; 1921x1080 already
# bills as 4K. Starlight, Astra and Hyperion all share this threshold.
# https://developer.topazlabs.com/getting-started/model-pricing
TIER_1080P_MAX_PIXELS = 1920 * 1080


def probe_source(video_input: Any, *, node_name: str) -> VideoMetadata:
    """Resolve the input to a local path and probe it with ffprobe.

    Resolving through ``File`` first is what makes ``{inputs}/clip.mp4`` macro
    paths work; handing the raw value to ffprobe silently fails for those.
    """
    video_url = coerce_media_url_or_data_uri(video_input, kind="video")
    if not video_url:
        msg = f"{node_name} could not resolve the input video."
        raise ValueError(msg)

    try:
        resolved_path = File(video_url).resolve()
    except FileLoadError as e:
        msg = f"{node_name} could not resolve video path {video_url!r}: {e}"
        raise ValueError(msg) from e

    return extract_video_metadata_structured(str(resolved_path))


def derive_container(video_input: Any, *, node_name: str) -> str:
    """Map the input video to Topaz's exact ``source.container`` enum (mp4/mov/mkv).

    Deliberately independent of ``probe_source``/``VideoMetadata``: it only needs
    the raw parameter value, and calling the same cheap, pure
    ``coerce_media_url_or_data_uri`` helper here keeps this check from being
    silently bypassed by tests that stub ``probe_source`` wholesale.
    """
    video_url = coerce_media_url_or_data_uri(video_input, kind="video") or ""
    if video_url.startswith("data:"):
        header = video_url.removeprefix("data:").split(",", 1)[0]
        token = header.split(";", 1)[0].split("/", 1)[-1].lower()
    else:
        token = Path(urlsplit(video_url).path).suffix.lstrip(".").lower()

    container = CONTAINER_BY_TOKEN.get(token)
    if container is None:
        found = token or "no extension"
        msg = f"{node_name}: Topaz only accepts mp4, mov, or mkv source video, but got {found!r}."
        raise ValueError(msg)
    return container


def frame_count(metadata: VideoMetadata) -> int:
    """Derive the source frame count, which the proxy requires and never defaults.

    ffprobe omits ``nb_frames`` for plenty of ordinary MP4s, so fall back to
    duration x frame rate rather than letting the request 400 downstream.

    Returns 0 when neither is available; the caller decides what to say about it.
    """
    nb_frames = metadata.frame_details.optional_nb_frames
    if nb_frames and nb_frames > 0:
        return nb_frames

    duration = metadata.file_details.optional_duration
    frame_rate = metadata.frame_details.frame_rate
    if duration and duration > 0 and frame_rate > 0:
        return math.ceil(duration * frame_rate)

    return 0


def to_even(value: int) -> int:
    """Round down to an even number -- odd dimensions break yuv420 encoding."""
    return max(2, value - (value % 2))
