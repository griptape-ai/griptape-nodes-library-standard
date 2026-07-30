from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.media import coerce_media_url_or_data_uri

__all__ = ["MAX_VIDEO_FILE_SIZE_BYTES", "PublicVideoUrlMixin"]

_LOCAL_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

# LTX nginx rejects payloads above ~23 MB encoded (~17 MB on disk). 16 MB gives a
# conservative buffer below the observed 16.58 MB success ceiling.
MAX_VIDEO_FILE_SIZE_BYTES = 16 * 1024 * 1024


class PublicVideoUrlMixin:
    def _resolve_to_public_url(self, video_input: Any) -> str | None:
        """Return a public URL for the video if one can be resolved, or None to fall back to base64.

        Any http:// or https:// URL is returned as-is. For file paths, tries
        CreateStaticFileDownloadUrlFromPathRequest (returns a real signed URL in cloud
        deployments, a localhost URL in local dev). Returns None for data: URIs and
        on any resolution failure.
        """
        coerced = coerce_media_url_or_data_uri(video_input, kind="video")
        if not coerced:
            return None
        if coerced.startswith(("http://", "https://")):
            if urlparse(coerced).hostname in _LOCAL_HOSTNAMES:
                return None
            return coerced
        if coerced.startswith("data:"):
            return None
        try:
            resolved = File(coerced).resolve()
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                url = result.url
                if urlparse(url).hostname not in _LOCAL_HOSTNAMES:
                    return url
        except Exception:
            pass
        return None
