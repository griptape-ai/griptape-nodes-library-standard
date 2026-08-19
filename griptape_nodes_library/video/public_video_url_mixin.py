from __future__ import annotations

import logging
from contextlib import suppress
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)

from griptape_nodes_library.media import coerce_media_url_or_data_uri, is_public_https_domain_url

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger("griptape_nodes")

__all__ = ["MAX_VIDEO_DATA_URI_SIZE_BYTES", "PublicVideoUrlMixin", "decoded_data_uri_size"]

# LTX's data-URI limit is 15 MB on the *encoded* base64 string (docs.ltx.io/input-formats).
# decoded = encoded × (3/4), so the decoded cap is 15 MB × 3/4 ≈ 11.25 MB.
# The URL tier avoids this entirely by handing LTX a public URL it fetches server-side,
# where the limit is a larger 32 MB; this guard only applies to the base64 fallback.
MAX_VIDEO_DATA_URI_SIZE_BYTES = 15 * 1024 * 1024 * 3 // 4  # ≈ 11.25 MB decoded


def decoded_data_uri_size(data_uri: str) -> int:
    """Return the decoded byte size of a base64 ``data:`` URI.

    Raises ValueError on a malformed URI (missing the ``,`` separator) rather than silently
    skipping the size check on a payload we can't measure — a malformed URI would otherwise be
    sent as-is and bounce back as the 413 this guard exists to prevent.
    """
    _prefix, sep, b64 = data_uri.partition(",")
    if not sep:
        msg = "Malformed data URI: missing ',' separator."
        raise ValueError(msg)
    return (len(b64) // 4) * 3 - b64.count("=")


class PublicVideoUrlMixin:
    """Resolve a node's video input to a value for a proxy ``video_uri`` field.

    Tier 1 uploads the video to Griptape Cloud via ``PublicArtifactUrlParameter`` and returns a
    guaranteed-public HTTPS URL, which LTX fetches server-side — keeping the request body tiny
    and side-stepping the ~33% base64 inflation that triggers 413s. Tier 2 falls back to a
    base64 ``data:`` URI with a pre-flight size check for environments where the upload can't
    run (e.g. local dev without Griptape Cloud credentials).

    Consuming nodes must provide ``name``, ``get_parameter_value``, ``set_parameter_value``,
    ``remove_parameter_element_by_name`` (all on ``BaseNode``) and an async
    ``_prepare_video_data_uri_async(video_input)`` method, and must call
    ``_reset_video_uploads()`` / ``_cleanup_video_uploads()`` around processing.
    """

    _pending_video_uploads: list[tuple[PublicArtifactUrlParameter, str]]

    def _reset_video_uploads(self) -> None:
        self._pending_video_uploads = []

    def _cleanup_video_uploads(self) -> None:
        """Delete uploaded Griptape Cloud artifacts and remove their scratch parameters.

        Each scratch parameter name is unique per upload, so leaving them would accumulate
        parameters on the node across runs.
        """
        for helper, scratch_name in getattr(self, "_pending_video_uploads", []):
            with suppress(Exception):
                helper.delete_uploaded_artifact()
            with suppress(Exception):
                self.remove_parameter_element_by_name(scratch_name)  # type: ignore[attr-defined]
        self._pending_video_uploads = []

    async def _resolve_video_uri(self, video_input: Any) -> str:
        """Return the value for the proxy ``video_uri`` field (public URL or base64 data URI)."""
        public_url = self._upload_video_to_public_url(video_input)
        if public_url:
            return public_url

        # Tier 2: base64 data URI with a pre-flight size guard.
        prepare: Awaitable[str | None] = self._prepare_video_data_uri_async(video_input)  # type: ignore[attr-defined]
        data_uri = await prepare
        if not data_uri:
            msg = f"{self.name} failed to process input video."  # type: ignore[attr-defined]
            raise ValueError(msg)

        decoded_size = decoded_data_uri_size(data_uri)
        if decoded_size > MAX_VIDEO_DATA_URI_SIZE_BYTES:
            size_mb = decoded_size / 1_048_576
            limit_mb = MAX_VIDEO_DATA_URI_SIZE_BYTES // 1_048_576
            msg = (
                f"{self.name}: Source video is too large ({size_mb:.1f} MB decoded, "  # type: ignore[attr-defined]
                f"~{size_mb * 4 / 3:.1f} MB encoded). "
                f"The maximum supported encoded size is 15 MB (~{limit_mb} MB decoded). "
                "Trim the video to a shorter segment and try again."
            )
            raise ValueError(msg)
        return data_uri

    def _upload_video_to_public_url(self, video_input: Any) -> str | None:
        """Upload the video to Griptape Cloud and return a public URL, or None to fall back.

        Already-public remote https URLs pass through unchanged (LTX fetches them directly).
        Local paths, ``data:`` URIs, and localhost URLs are uploaded via a transient
        ``PublicArtifactUrlParameter``. Returns None (and logs a warning) if the upload can't
        be performed so the caller falls back to base64.
        """
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            return None
        if is_public_https_domain_url(video_value):
            return video_value

        # The scratch parameter is a transient, worker-local helper that only exists to feed the
        # upload (PublicArtifactUrlParameter reads its value locally). Its name is unique per
        # call and it is removed in _cleanup_video_uploads before the run ends.
        scratch_name = f"_video_upload_{uuid4().hex}"
        try:
            helper = PublicArtifactUrlParameter(
                node=self,  # type: ignore[arg-type]
                artifact_url_parameter=Parameter(
                    name=scratch_name,
                    input_types=["VideoUrlArtifact"],
                    type="VideoUrlArtifact",
                    default_value="",
                    tooltip="",
                    allowed_modes={ParameterMode.PROPERTY},
                    hide=True,
                    hide_property=True,
                ),
            )
            helper.add_input_parameters()
            self._pending_video_uploads.append((helper, scratch_name))
            self.set_parameter_value(scratch_name, video_value)  # type: ignore[attr-defined]
            return helper.get_public_url_for_parameter()
        except Exception as e:  # noqa: BLE001 - any upload failure should degrade to base64
            logger.warning(
                "%s: public-URL upload failed, falling back to base64: %s",
                self.name,  # type: ignore[attr-defined]
                e,
            )
            return None
