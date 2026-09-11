"""Generated media the Griptape Cloud proxy hosts for a completed generation.

The proxy pulls whatever media a provider produced into Griptape storage and
lists it at ``GET proxy/v2/generations/{id}/artifacts`` in one shape for every
model: index, kind, content type, size, and a URL. Nodes read their media from
there rather than each digging it out of its provider's own response payload,
which also frees them from provider URLs that expire minutes after a task
completes.

The list is ordered and an artifact's index is its stable public handle, so a
model that produces several pieces of media (a mesh plus a preview, a video plus
its last frame) always reports them in the same order.
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
        happens: a caller now treats a short list as the proxy's documented
        truncation, so a malformed entry must not look identical to that in the
        logs.
        """
        if not isinstance(payload, dict):
            logger.warning("Hosted artifact entry was not an object, dropping it: %r", payload)
            return None

        index = payload.get("index")
        url = payload.get("url")
        if not isinstance(index, int) or not isinstance(url, str) or not url:
            logger.warning("Hosted artifact entry has no usable index or url, dropping it: %r", payload)
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
        HostedArtifactError: If the list cannot be read.
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

    artifacts = [artifact for artifact in (HostedArtifact.from_payload(entry) for entry in entries) if artifact]
    artifacts.sort(key=lambda artifact: artifact.index)
    return artifacts


def artifact_download_headers(url: str, api_key: str) -> dict[str, str]:
    """Headers to fetch an artifact URL with.

    Empty for a presigned storage URL, which authenticates itself and rejects a
    request that also carries a bearer token.
    """
    if _PROXY_ARTIFACT_ROUTE.search(urlparse(url).path):
        return {"Authorization": f"Bearer {api_key}"}
    return {}
