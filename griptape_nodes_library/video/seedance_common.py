"""Shared plumbing for the Seedance video generation nodes.

The Seedance 2.0 and 2.5 nodes talk to the same BytePlus endpoints through the same Griptape
Cloud proxy, so everything that is about *the provider* rather than *the model* lives here:
media coercion, audio-subtype normalization, provider response parsing, and the private-asset
(``asset://``) registration flow. What differs per model version — task/input modes, capability
tables, parameter layout, validation — stays in the node modules.

`SeedanceProxyNode` is the shared base class. It carries the private-asset flow because that
flow needs node state (the proxy base URL, the API key, the transient upload parameters it
must tear down), while the stateless helpers are module-level functions.
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
from abc import ABC
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin
from uuid import uuid4

import httpx
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    get_provider_asset_kind,
    get_provider_asset_value,
)
from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "ASSET_MAX_ATTEMPTS",
    "ASSET_MODERATION",
    "ASSET_POLL_INTERVAL",
    "ASSET_PROVIDER",
    "ASSET_STATUS_ACTIVE",
    "ASSET_STATUS_DELETED",
    "ASSET_STATUS_FAILED",
    "SEEDANCE_AUDIO_SUBTYPE_ALIASES",
    "SeedanceProxyNode",
    "coerce_video_url",
    "extract_video_url",
    "normalize_audio_data_uri_subtype",
    "parse_provider_response",
    "summarize_media_input",
]

# Provider-asset (private asset) registration via the GTC proxy. Gated to Griptape auth (not BYOK).
ASSET_PROVIDER = "byteplus_ark"
ASSET_POLL_INTERVAL = 3  # seconds
ASSET_MAX_ATTEMPTS = 60  # ~3 min cap, independent of the generation timeout
ASSET_STATUS_ACTIVE = "ACTIVE"
ASSET_STATUS_FAILED = "FAILED"
ASSET_STATUS_DELETED = "DELETED"
# Skip provider-side moderation on private-asset ingestion (content is already moderated upstream).
ASSET_MODERATION = {"Strategy": "Skip"}

# Artifact type name to wrap each asset kind in when uploading it for a public URL.
_ASSET_KIND_ARTIFACT_TYPES = {
    ASSET_KIND_IMAGE: "ImageUrlArtifact",
    ASSET_KIND_VIDEO: "VideoUrlArtifact",
    ASSET_KIND_AUDIO: "AudioUrlArtifact",
}

# Seedance only accepts audio data URIs whose subtype is exactly `wav` or `mp3` (per the BytePlus
# docs). The local File/mimetypes layer emits other subtypes for the same formats (e.g. an .mp3
# resolves to `audio/mpeg`, a .wav to `audio/x-wav`), which Seedance rejects as
# "Invalid base64 audio_url". Map those aliases back to the accepted subtypes before sending.
SEEDANCE_AUDIO_SUBTYPE_ALIASES = {
    "mpeg": "mp3",
    "mp3": "mp3",
    "x-wav": "wav",
    "wave": "wav",
    "vnd.wave": "wav",
    "wav": "wav",
}


def normalize_audio_data_uri_subtype(data_uri: str) -> str:
    """Rewrite an ``data:audio/<subtype>;base64,...`` URI to a Seedance-accepted subtype.

    Returns the URI unchanged if it is not an audio data URI or the subtype has no known alias.
    """
    prefix = "data:audio/"
    if not data_uri.startswith(prefix):
        return data_uri
    remainder = data_uri[len(prefix) :]
    subtype, separator, rest = remainder.partition(";")
    if not separator:
        return data_uri
    normalized_subtype = SEEDANCE_AUDIO_SUBTYPE_ALIASES.get(subtype.lower())
    if normalized_subtype is None or normalized_subtype == subtype:
        return data_uri
    return f"{prefix}{normalized_subtype};{rest}"


def coerce_video_url(val: Any) -> str | None:
    """Convert video input to a Seedance-supported public URL or asset ID."""
    if val is None:
        return None

    if isinstance(val, dict):
        value = val.get("value")
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith(("http://", "https://", "asset://")):
                return stripped
        return None

    if isinstance(val, str):
        v = val.strip()
        if not v:
            return None
        return v if v.startswith(("http://", "https://", "asset://")) else None

    try:
        to_dict = getattr(val, "to_dict", None)
        if callable(to_dict):
            serialized = to_dict()
            if isinstance(serialized, dict):
                coerced = coerce_video_url(serialized)
                if coerced:
                    return coerced

        v = getattr(val, "value", None)
        if isinstance(v, str):
            stripped = v.strip()
            if stripped.startswith(("http://", "https://", "asset://")):
                return stripped
    except Exception:  # noqa: S110
        pass

    return None


def extract_video_url(obj: dict[str, Any] | None) -> str | None:
    """Find the first http(s) video URL anywhere in a provider result payload."""
    if not obj:
        return None
    for key in ("url", "video_url", "output_url"):
        val = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(val, str) and val.startswith("http"):
            return val
    for key in ("result", "data", "output", "outputs", "content", "task_result"):
        nested = obj.get(key) if isinstance(obj, dict) else None
        if isinstance(nested, dict):
            url = extract_video_url(nested)
            if url:
                return url
        elif isinstance(nested, list):
            for item in nested:
                url = extract_video_url(item if isinstance(item, dict) else None)
                if url:
                    return url
    return None


def parse_provider_response(provider_response: Any) -> dict[str, Any] | None:
    """Parse provider_response if it's a JSON string."""
    if isinstance(provider_response, str):
        try:
            return _json.loads(provider_response)
        except Exception:
            return None
    if isinstance(provider_response, dict):
        return provider_response
    return None


def summarize_media_input(val: Any) -> str:
    """Describe a media input's shape for logs without dumping base64 payloads."""
    if val is None:
        return "None"

    if isinstance(val, str):
        return f"str(len={len(val)})"

    if isinstance(val, dict):
        value = val.get("value")
        value_summary = f"str(len={len(value)})" if isinstance(value, str) else type(value).__name__
        return f"dict(type={val.get('type')}, value={value_summary})"

    value_attr = getattr(val, "value", None)
    if isinstance(value_attr, str):
        return f"value=str(len={len(value_attr)})"
    if value_attr is not None:
        return f"value_type={type(value_attr).__name__}"

    return repr(val)


class SeedanceProxyNode(GriptapeProxyNode, ABC):
    """Base class for Seedance video generation nodes.

    Provides the media preparation and private-asset registration that every Seedance node
    needs; subclasses own their parameters, validation, and payload shape.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Transient (helper, scratch parameter name) pairs created while registering private
        # assets; the upload and the scratch parameter are both cleaned up in
        # _process_generation's finally block so they don't accumulate on the node across runs.
        self._pending_asset_uploads: list[tuple[PublicArtifactUrlParameter, str]] = []

    # --- Media preparation ------------------------------------------------------------------

    async def _prepare_frame_url_async(self, frame_input: Any, *, frame_label: str) -> str | None:
        """Convert frame input to a usable URL."""
        if not frame_input:
            self._log(f"{self.name} {frame_label} not provided")
            return None

        frame_url = coerce_media_url_or_data_uri(frame_input, kind="image")
        if not frame_url:
            self._log(
                f"{self.name} {frame_label} could not be converted to an image URL or data URI. "
                f"input_type={type(frame_input).__name__}, "
                f"input_module={type(frame_input).__module__}, "
                f"input_summary={summarize_media_input(frame_input)}"
            )
            return None

        if frame_url.startswith("data:image/"):
            self._log(f"{self.name} {frame_label} prepared as inline data URI")
            return frame_url

        try:
            data_uri = await File(frame_url).aread_data_uri(fallback_mime="image/jpeg")
            self._log(f"{self.name} {frame_label} loaded from file/URL into data URI")
            return data_uri
        except FileLoadError as e:
            self._log(f"{self.name} {frame_label} failed to load from {frame_url}: {e}")
            return None

    async def _prepare_audio_url_async(self, audio_input: Any, *, audio_label: str) -> str | None:
        """Convert audio input to a Seedance-accepted URL or data URI."""
        if not audio_input:
            self._log(f"{self.name} {audio_label} not provided")
            return None

        audio_url = coerce_media_url_or_data_uri(audio_input, kind="audio")
        if not audio_url:
            self._log(
                f"{self.name} {audio_label} could not be converted to an audio URL or data URI. "
                f"input_type={type(audio_input).__name__}, "
                f"input_module={type(audio_input).__module__}, "
                f"input_summary={summarize_media_input(audio_input)}"
            )
            return None

        if audio_url.startswith(("data:audio/", "http://", "https://", "asset://")):
            self._log(f"{self.name} {audio_label} prepared as direct audio URL/data URI")
            return normalize_audio_data_uri_subtype(audio_url)

        try:
            data_uri = await File(audio_url).aread_data_uri(fallback_mime="audio/wav")
            self._log(f"{self.name} {audio_label} loaded from file into data URI")
            return normalize_audio_data_uri_subtype(data_uri)
        except FileLoadError as e:
            self._log(f"{self.name} {audio_label} failed to load from {audio_url}: {e}")
            return None

    # --- Provider private-asset registration (Griptape auth only) ---------------------------

    def _proxy_headers(self) -> dict[str, str]:
        """Bearer headers for the GTC proxy (same auth as the generation requests)."""
        return {"Authorization": f"Bearer {self._validate_api_key()}", "Content-Type": "application/json"}

    async def _append_private_asset(self, ref: Any, *, expected_kind: str, label: str) -> str:
        """Resolve a private-asset reference to an `asset://{asset_id}` URL.

        Cross-checks the reference kind against the receiving input, obtains a public URL for
        the media, registers it as a provider private asset via the proxy, and polls until ACTIVE.
        """
        actual_kind = get_provider_asset_kind(ref)
        if actual_kind != expected_kind:
            msg = (
                f"{self.name}: {label} received a {actual_kind or 'unknown'} private-asset reference, "
                f"but this input requires a {expected_kind} reference. "
                f"Set the Seedance Human Reference Asset's Asset Kind to {expected_kind}."
            )
            raise ValueError(msg)

        public_url = self._resolve_public_url_for_asset(ref, asset_kind=expected_kind)
        headers = self._proxy_headers()
        asset_id = await self._create_provider_asset(public_url, expected_kind, headers)
        return f"asset://{asset_id}"

    def _resolve_public_url_for_asset(self, ref: Any, *, asset_kind: str) -> str:
        """Return a publicly fetchable URL for the reference's media.

        CreateProviderAsset requires a fetchable URL, so data URIs / unresolvable inputs raise.
        """
        media_value = get_provider_asset_value(ref)
        if not media_value:
            msg = f"{self.name}: private-asset reference has no media value to register."
            raise ValueError(msg)

        public_url = self._resolve_public_url_for_media(
            media_value, artifact_type=_ASSET_KIND_ARTIFACT_TYPES[asset_kind]
        )
        if not (public_url.startswith(("http://", "https://")) and "localhost" not in public_url):
            msg = (
                f"{self.name}: could not obtain a public URL for the {asset_kind} private asset. "
                "Provider asset registration requires a publicly fetchable URL (data URIs are not supported)."
            )
            raise RuntimeError(msg)
        return public_url

    def _resolve_public_url_for_media(self, media_value: Any, *, artifact_type: str) -> str:
        """Upload media to GTC static storage through a transient parameter and return its URL.

        Already-public http(s) URLs pass through untouched. PublicArtifactUrlParameter binds to a
        named parameter, so a hidden scratch parameter is created per call and torn down together
        with the upload by _cleanup_pending_asset_uploads.
        """
        # Media reaches this node as a URL string, an artifact, or a serialized artifact dict, and
        # any of the three can already carry a public URL that needs no upload.
        if isinstance(media_value, dict):
            url = media_value.get("value")
        else:
            url = getattr(media_value, "value", media_value)
        if not isinstance(url, str) or not url:
            msg = f"{self.name}: cannot obtain a public URL for {media_value!r}, which carries no media path or URL."
            raise ValueError(msg)
        if url.startswith(("http://", "https://")) and "localhost" not in url:
            return url

        # Adding this scratch parameter during aprocess trips the strict-mode
        # "parameter-mutation-during-aprocess" warning. That is expected and harmless here: the
        # parameter is a transient, worker-local helper that only exists to feed the upload
        # (PublicArtifactUrlParameter reads its value locally) and is removed before the run ends,
        # so there is nothing for the orchestrator to stay in sync with.
        scratch_name = f"_asset_upload_{uuid4().hex}"
        helper = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name=scratch_name,
                input_types=[artifact_type],
                type=artifact_type,
                default_value="",
                tooltip="",
                allowed_modes={ParameterMode.PROPERTY},
                hide=True,
                hide_property=True,
            ),
        )
        helper.add_input_parameters()
        self._pending_asset_uploads.append((helper, scratch_name))
        self.set_parameter_value(scratch_name, url)

        return helper.get_public_url_for_parameter()

    async def _create_provider_asset(self, public_url: str, asset_kind: str, headers: dict[str, str]) -> str:
        """POST proxy/v2/assets and poll to ACTIVE; return the provider asset id."""
        create_url = urljoin(self._proxy_base, "assets")
        payload = {
            "url": public_url,
            "provider": ASSET_PROVIDER,
            "provider_body": {"asset_type": asset_kind, "moderation": ASSET_MODERATION},
        }
        self._log(f"{self.name} registering {asset_kind} private asset via {create_url}")
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(create_url, json=payload, headers=headers, timeout=60)
                response.raise_for_status()
                response_json = response.json()
        except httpx.HTTPStatusError as e:
            msg = f"{self.name}: failed to create private asset: HTTP {e.response.status_code} - {e.response.text}"
            raise RuntimeError(msg) from e
        except Exception as e:
            msg = f"{self.name}: failed to create private asset: {e}"
            raise RuntimeError(msg) from e

        provider_asset_id = response_json.get("provider_asset_id")
        if not provider_asset_id:
            msg = f"{self.name}: CreateProviderAsset returned no provider_asset_id."
            raise RuntimeError(msg)
        return await self._poll_provider_asset(str(provider_asset_id), headers)

    async def _poll_provider_asset(self, provider_asset_id: str, headers: dict[str, str]) -> str:
        """Poll GET proxy/v2/assets/<id> until ACTIVE; return the provider asset id.

        Transient errors (a network blip, a 5xx, or the eventual-consistency 404 right after the
        asset id is minted) are logged and retried until attempts are exhausted, mirroring the
        base class's generation poll. Only a terminal status (FAILED/DELETED) fails immediately.
        """
        get_url = urljoin(self._proxy_base, f"assets/{provider_asset_id}")
        async with httpx.AsyncClient() as client:
            for attempt in range(ASSET_MAX_ATTEMPTS):
                try:
                    response = await client.get(get_url, headers=headers, timeout=60)
                    response.raise_for_status()
                    result_json = response.json()
                except Exception as e:
                    # Transient — log and retry rather than aborting an otherwise-successful run.
                    self._log(
                        f"{self.name} error polling private asset {provider_asset_id} (attempt {attempt + 1}): {e}"
                    )
                    await asyncio.sleep(ASSET_POLL_INTERVAL)
                    continue

                status = result_json.get("status", "unknown")
                self._log(f"{self.name} private asset {provider_asset_id} status: {status} (attempt {attempt + 1})")

                if status == ASSET_STATUS_ACTIVE:
                    asset_id = result_json.get("asset_id")
                    if not asset_id:
                        msg = f"{self.name}: private asset {provider_asset_id} is ACTIVE but no asset_id was returned."
                        raise RuntimeError(msg)
                    return str(asset_id)

                if status in (ASSET_STATUS_FAILED, ASSET_STATUS_DELETED):
                    detail = result_json.get("status_detail")
                    msg = f"{self.name}: private asset {provider_asset_id} ended with status {status}: {detail}"
                    raise RuntimeError(msg)

                await asyncio.sleep(ASSET_POLL_INTERVAL)

        msg = (
            f"{self.name}: private asset {provider_asset_id} did not become ACTIVE within "
            f"{ASSET_MAX_ATTEMPTS * ASSET_POLL_INTERVAL} seconds."
        )
        raise RuntimeError(msg)

    def _cleanup_pending_asset_uploads(self) -> None:
        """Delete the transient uploads and scratch parameters minted during a run.

        Provider assets are reclaimed by the backend: a submitted generation deletes its linked
        assets on terminal state, and assets we register but never submit (e.g. a build failure
        after registration) are reclaimed by the backend's orphan sweeper. The transient GTC
        static-storage upload made to feed CreateProviderAsset is ours to clean up, along with the
        scratch parameter created to perform the upload (its name is unique per call, so leaving it
        would accumulate parameters on the node).
        """
        for helper, scratch_name in self._pending_asset_uploads:
            with suppress(Exception):
                helper.delete_uploaded_artifact()
            with suppress(Exception):
                self.remove_parameter_element_by_name(scratch_name)
        self._pending_asset_uploads = []

    # --- Provider error reporting ------------------------------------------------------------

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Surface the BytePlus error object, which the proxy nests under provider_response."""
        if not response_json:
            return super()._extract_error_message(response_json)

        parsed_provider_response = parse_provider_response(response_json.get("provider_response"))
        if parsed_provider_response:
            provider_error = parsed_provider_response.get("error")
            if provider_error:
                if isinstance(provider_error, dict):
                    error_message = provider_error.get("message", "")
                    details = f"{self.name} {error_message}"
                    if error_code := provider_error.get("code"):
                        details += f"\nError Code: {error_code}"
                    if error_type := provider_error.get("type"):
                        details += f"\nError Type: {error_type}"
                    return details
                return f"{self.name} Provider error: {provider_error}"

        return super()._extract_error_message(response_json)
