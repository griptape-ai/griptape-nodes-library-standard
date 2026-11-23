"""Helpers for media that lives on this machine: checking a node input and showing it in a widget."""

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

__all__ = ["browser_url", "validate_url"]

logger = logging.getLogger("griptape_nodes")

BROWSER_URL_PREFIXES = ("http://", "https://", "data:")


def validate_url(url: str) -> bool:
    """Validate that the URL is safe for ffmpeg processing.

    Accepts http, https, and file URLs with a host, and absolute file paths (e.g., resolved from
    project macro paths).
    """
    try:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https", "file") and parsed.netloc:
            return True
        return Path(url).is_absolute()
    except Exception:
        return False


def browser_url(location: str, *, fallback: str | None = None) -> str:
    """Return a URL a widget in the browser can load for a file path, macro path, or URL.

    http, https, and data URLs pass through unchanged. Anything else is resolved and handed to the
    static server. When the static server gives no URL, returns `fallback`, or `location` itself
    when no fallback is given, and logs a warning rather than raising.
    """
    if not location or location.startswith(BROWSER_URL_PREFIXES):
        return location
    default = location if fallback is None else fallback
    try:
        result = GriptapeNodes.handle_request(
            CreateStaticFileDownloadUrlFromPathRequest(file_path=File(location).resolve())
        )
    except Exception as e:
        logger.warning("Could not build a browser URL for %r: %s", location, e)
        return default
    if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
        return result.url
    logger.warning("The static server would not serve %r: %s", location, result.result_details)
    return default
