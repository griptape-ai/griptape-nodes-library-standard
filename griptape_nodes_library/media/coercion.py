"""Helpers that load or resolve image, video and audio inputs for provider nodes.

Reducing a media value (a URL, macro path, filesystem path, ``data:`` URI, artifact or serialized
artifact dict) to one string is the engine's ``coerce_media_url_or_data_uri``, re-exported here for
the nodes that import it from ``griptape_nodes_library.media``. The helpers below build on it:
``prepare_media_data_uri`` loads the input as a ``data:`` URI, and ``resolve_media_location`` turns a
location into one FFmpeg or ffprobe can open.
"""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.utils.media_values import MediaKind, coerce_media_url_or_data_uri

__all__ = ["MediaKind", "coerce_media_url_or_data_uri", "prepare_media_data_uri", "resolve_media_location"]

logger = logging.getLogger("griptape_nodes")

_FALLBACK_MIME: dict[MediaKind, str] = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/mpeg",
}


def resolve_media_location(location: str) -> str:
    """Return a location FFmpeg or ffprobe can open.

    Project macro paths (``{outputs}/clip.mp4``), relative paths, and localhost
    static-server URLs resolve to absolute local paths through ``File``. Other
    URLs come back unchanged. ``data:`` URIs skip ``File`` entirely because it
    treats them as relative file paths; FFmpeg opens them natively.

    Raises:
        FileLoadError: If macro resolution fails (e.g. no project loaded).
    """
    if location.startswith("data:"):
        return location
    return File(location).resolve()


async def prepare_media_data_uri(
    val: Any,
    *,
    kind: MediaKind,
    node_name: str | None = None,
    fallback_mime: str | None = None,
) -> str | None:
    """Coerce a media input and resolve it to a base64 ``data:`` URI.

    Returns ``None`` if the input is empty/unresolvable, or could not be
    loaded. Inputs that are already a ``data:<kind>/...`` URI are returned
    unchanged.

    ``fallback_mime`` is used by ``File.aread_data_uri`` when it cannot
    otherwise determine a MIME type from the resolved bytes; defaults to the
    canonical MIME for ``kind`` (``image/png``, ``video/mp4``, ``audio/mpeg``).
    """
    if not val:
        return None

    media_url = coerce_media_url_or_data_uri(val, kind=kind)
    if not media_url:
        return None

    mime_prefix = f"data:{kind}/"
    if media_url.startswith(mime_prefix):
        return media_url

    try:
        return await File(media_url).aread_data_uri(fallback_mime=fallback_mime or _FALLBACK_MIME[kind])
    except FileLoadError as e:
        prefix = f"{node_name} " if node_name else ""
        logger.debug("%sfailed to load %s from %s: %s", prefix, kind, media_url, e)
        return None
