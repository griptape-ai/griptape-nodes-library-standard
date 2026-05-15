"""Upload a local artifact to Griptape Cloud storage with an explicit Content-Type.

The default `PublicArtifactUrlParameter` / `GriptapeCloudStorageDriver.upload_file`
flow doesn't set Content-Type on the uploaded blob, so Azure serves it back as
``application/octet-stream``. RunwayML rejects non-video content types on its
``video_to_video`` and ``character_performance`` endpoints, so anything that
relies on those endpoints needs to upload with the correct Content-Type.

This helper performs an upload that mirrors the SDK's flow but injects
``x-ms-blob-content-type`` (and ``Content-Type``) into the PUT headers so the
stored blob serves with the right MIME type.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from griptape.artifacts.url_artifact import UrlArtifact
from griptape_nodes.drivers.storage.griptape_cloud_storage_driver import GriptapeCloudStorageDriver
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.files.file import File

__all__ = ["UploadResult", "upload_artifact_with_content_type"]

logger = logging.getLogger("griptape_nodes")


class UploadResult:
    """The product of an upload performed via PublicArtifactUrlParameter."""

    def __init__(self, *, public_url: str, gtc_file_path: Path | None) -> None:
        self.public_url = public_url
        self.gtc_file_path = gtc_file_path


def _coerce_to_url_or_path(parameter_value: Any) -> str | None:
    if parameter_value is None:
        return None
    if isinstance(parameter_value, UrlArtifact):
        return parameter_value.value
    if isinstance(parameter_value, str):
        return parameter_value or None
    value = getattr(parameter_value, "value", None)
    return value if isinstance(value, str) and value else None


def upload_artifact_with_content_type(
    public_artifact_url_parameter: PublicArtifactUrlParameter,
    parameter_value: Any,
    content_type: str,
) -> UploadResult:
    """Resolve a parameter to a public HTTPS URL, uploading with explicit Content-Type if needed.

    Mirrors the public-vs-upload branching of
    `PublicArtifactUrlParameter.get_public_url_for_parameter`, but routes the
    upload through this helper so the stored blob carries the requested
    Content-Type. Stores the upload's bucket path on the parameter so the
    caller can use the parameter's `delete_uploaded_artifact()` for cleanup.

    Args:
        public_artifact_url_parameter: The parameter component that owns the
            storage driver and gtc_file_path tracking.
        parameter_value: The current value of the parameter (URL artifact, str,
            or anything `getattr(value, 'value')` resolves to a string).
        content_type: MIME type to attach to the uploaded blob (e.g.
            ``"video/mp4"``).

    Returns:
        UploadResult: ``public_url`` is suitable for handing to a third-party
        service that follows the URL. ``gtc_file_path`` is set to the bucket
        path when an upload happened (so the caller / parameter can delete it
        later) and is None when the input was already a public URL.
    """
    url = _coerce_to_url_or_path(parameter_value)
    if not url:
        msg = "Cannot upload empty parameter value."
        raise ValueError(msg)

    if url.startswith(("http://", "https://")) and "localhost" not in url:
        return UploadResult(public_url=url, gtc_file_path=None)

    storage_driver: GriptapeCloudStorageDriver = public_artifact_url_parameter._storage_driver  # noqa: SLF001
    file_contents = File(url).read_bytes()
    filename = Path(urlparse(url).path).name
    gtc_file_path = Path("artifact_url_storage") / uuid4().hex / filename

    upload_response = storage_driver.create_signed_upload_url(gtc_file_path)
    headers = dict(upload_response.get("headers") or {})
    # Azure Blob Storage uses x-ms-blob-content-type to set the stored blob's
    # Content-Type. httpx's `Content-Type` header is a hint but the SDK's
    # signed-upload URL doesn't bind Content-Type into its signature, so this
    # is safe to add post-hoc.
    headers["x-ms-blob-content-type"] = content_type
    headers.setdefault("Content-Type", content_type)

    response = httpx.request(
        upload_response["method"],
        upload_response["url"],
        content=file_contents,
        headers=headers,
    )
    response.raise_for_status()

    public_url = storage_driver.create_signed_download_url(gtc_file_path)

    # Track the bucket path on the parameter so its existing
    # delete_uploaded_artifact() can clean up after the run.
    public_artifact_url_parameter.gtc_file_path = gtc_file_path

    return UploadResult(public_url=public_url, gtc_file_path=gtc_file_path)
