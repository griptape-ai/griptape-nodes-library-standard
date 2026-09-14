"""Generated media the Griptape Cloud proxy hosts for a completed generation.

The proxy lists the media it hosts for a generation at
``GET proxy/v2/generations/{id}/artifacts``, in one shape for every model: index,
kind, content type, size, and a URL. Nodes read their media from there instead of
digging a URL or a base64 blob out of each provider's own response payload.

The proxy orders the list by index, indexes from 0 with no gaps or duplicates, and
truncates only by dropping a tail, never a middle entry. A short list is therefore
always a safe prefix to pair by position.

``fetch_hosted_artifacts`` still checks this on every call, as defense in depth: a
deployed proxy can lag this client, and artifact hosting is deployment-configurable.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "ArtifactKind",
    "HostedArtifact",
    "HostedArtifactError",
    "artifact_download_headers",
    "fetch_hosted_artifacts",
]

# Listing artifacts is a small JSON read, so it gets a short timeout rather than
# the generous one the media downloads themselves need.
ARTIFACT_LIST_TIMEOUT_SECONDS = 30

# The proxy hands out either a presigned storage URL, which carries its own
# credentials, or its own streaming route, which needs the API key. Only the
# streaming route has this path shape, and it is the only URL that gets an
# Authorization header: a presigned URL refuses a request carrying a second auth
# mechanism.
_PROXY_ARTIFACT_ROUTE = re.compile(r"/api/proxy/v2/generations/[^/]+/artifacts/\d+/?$")


# Hosted-list entries are logged at most this many characters when dropped as
# malformed: enough to identify which entry it was, not enough for a proxy bug
# that returns an oversized or deeply nested payload to flood the logs.
_MALFORMED_ENTRY_LOG_LIMIT = 200


def _bounded_repr(value: Any, limit: int = _MALFORMED_ENTRY_LOG_LIMIT) -> str:
    """repr() of value, truncated so a single log line stays bounded in size."""
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


class HostedArtifactError(RuntimeError):
    """Raised when a generation's hosted media cannot be listed or located."""


class ArtifactKind(StrEnum):
    """Coarse media category the proxy tags each hosted artifact with.

    Deliberately coarse: it says how to handle the bytes without encoding every
    provider's notion of format. Precise typing rides on ``content_type``.
    """

    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    MODEL_3D = "model_3d"
    ARCHIVE = "archive"
    OTHER = "other"


@dataclass(frozen=True)
class HostedArtifact:
    """One piece of generated media the proxy stored for a generation."""

    index: int
    kind: str
    url: str
    content_type: str | None = None
    size_bytes: int | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> HostedArtifact | None:
        """Build an artifact from one entry of the list endpoint's response.

        Returns None for an entry without an index or a URL, since neither can be
        guessed and an artifact missing either cannot be fetched. Logs when this
        happens: a caller treats a short list as truncation, so a malformed entry
        must not look identical to that in the logs.
        """
        if not isinstance(payload, dict):
            logger.warning("Hosted artifact entry was not an object, dropping it: %s", _bounded_repr(payload))
            return None

        index = payload.get("index")
        url = payload.get("url")
        if not isinstance(index, int) or not isinstance(url, str) or not url:
            logger.warning("Hosted artifact entry has no usable index or url, dropping it: %s", _bounded_repr(payload))
            return None

        return cls(
            index=index,
            kind=str(payload.get("kind") or ArtifactKind.OTHER),
            url=url,
            content_type=payload.get("content_type"),
            size_bytes=payload.get("size_bytes"),
        )


async def fetch_hosted_artifacts(proxy_base: str, generation_id: str, api_key: str) -> list[HostedArtifact]:
    """List the media the proxy hosts for a generation, in index order.

    Args:
        proxy_base: Base URL of the proxy API, ending in ``proxy/``.
        generation_id: The generation whose artifacts to list.
        api_key: Griptape Cloud credential to read the generation with.

    Returns:
        The hosted artifacts, which is empty for a generation whose media the
        proxy did not host.

    Raises:
        HostedArtifactError: If the list cannot be read, or if it fails the
            defense-in-depth check in ``_require_trustworthy_prefix``.
    """
    url = urljoin(proxy_base, f"generations/{generation_id}/artifacts")
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=ARTIFACT_LIST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as e:
        msg = f"Listing hosted artifacts for generation {generation_id} failed: HTTP {e.response.status_code}"
        raise HostedArtifactError(msg) from e
    except Exception as e:
        msg = f"Listing hosted artifacts for generation {generation_id} failed: {e}"
        raise HostedArtifactError(msg) from e

    entries = payload.get("artifacts") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        msg = f"Artifact list for generation {generation_id} was not in the expected shape: {payload!r}"
        raise HostedArtifactError(msg)

    parsed = [HostedArtifact.from_payload(entry) for entry in entries]
    dropped_count = sum(1 for artifact in parsed if artifact is None)
    artifacts = [artifact for artifact in parsed if artifact is not None]
    # Sorted defensively; does not rely on the proxy's own ordering.
    artifacts.sort(key=lambda artifact: artifact.index)
    _require_trustworthy_prefix(artifacts, dropped_count, generation_id)
    return artifacts


def _require_trustworthy_prefix(artifacts: list[HostedArtifact], dropped_count: int, generation_id: str) -> None:
    """Raise unless the list is safely a leading prefix a caller may pair by position.

    The proxy's own contract already guarantees this. This check is defense in
    depth: a deployed proxy can lag this client, and artifact hosting is
    deployment-configurable, so a dropped entry, a gap, or a duplicate index is
    rejected rather than trusted:

    - An entry was dropped (by ``HostedArtifact.from_payload``): if the dropped entry
      held the highest index, the survivors look exactly like a legitimate tail
      truncation, so a drop anywhere is treated as untrustworthy, not just a drop
      that visibly leaves a gap.
    - The surviving indices skip a value (a gap).
    - The surviving indices repeat a value (a duplicate).

    Checks contiguity rather than requiring the first index to be 0: contiguity is
    the property positional pairing actually needs, so the check asks for nothing
    more.
    """
    indices = [artifact.index for artifact in artifacts]
    has_duplicate = len(set(indices)) != len(indices)
    has_gap = len(indices) > 1 and any(b - a != 1 for a, b in zip(indices, indices[1:], strict=False))
    if dropped_count or has_gap or has_duplicate:
        msg = (
            f"Artifact list for generation {generation_id} cannot be trusted for positional "
            f"pairing: {dropped_count} entr{'y' if dropped_count == 1 else 'ies'} dropped, "
            f"surviving indices {indices}."
        )
        raise HostedArtifactError(msg)


def artifact_download_headers(url: str, api_key: str, proxy_base: str) -> dict[str, str]:
    """Headers to fetch an artifact URL with.

    Empty for a presigned storage URL, which authenticates itself and rejects a
    request that also carries a bearer token, or for any URL not on the proxy's
    own host: the bearer token is only ever sent to the proxy itself.
    """
    parsed = urlparse(url)
    # This assumes an artifact URL needing the token shares the proxy's own host,
    # which holds by default but not if a deployment points artifact hosting at a
    # different host (for example a CDN) while still requiring the token there.
    # The real fix is for the artifact envelope to state each URL's own auth
    # requirement rather than the client inferring it from the URL's shape.
    if parsed.netloc == urlparse(proxy_base).netloc and _PROXY_ARTIFACT_ROUTE.search(parsed.path):
        return {"Authorization": f"Bearer {api_key}"}
    return {}
