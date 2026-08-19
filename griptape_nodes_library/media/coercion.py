"""Shared coercion helpers for image/video/audio inputs across provider nodes.

Provider video/image/audio nodes accept inputs in many shapes: plain strings
(URLs, project macro paths like ``{inputs}/foo.png``, filesystem paths,
``data:`` URIs), artifact objects (``ImageUrlArtifact``, ``ImageArtifact`` and
their video/audio counterparts), and serialized artifact dicts. These helpers
normalize all of those into a single string that downstream code can hand to
``File(...).aread_data_uri()``.

Pass-through rules:

* HTTP(S) URLs, ``data:<kind>/...`` URIs, project macro paths, and plain
  filesystem paths are returned unchanged (whitespace stripped). They are
  resolvable by ``File`` downstream.
* Only artifact ``.base64`` payloads, or serialized ``<Kind>Artifact`` dicts,
  are wrapped as ``data:`` URIs. Raw strings are never wrapped, even if they
  look like base64.

The serialized-dict branch was originally specific to
``Seedance20VideoGeneration._coerce_image_url_or_data_uri``; it lives here so
every caller benefits.
"""

from __future__ import annotations

import base64
import logging
from binascii import Error as BinasciiError
from typing import Any, Literal, NamedTuple

from griptape_nodes.files.file import File, FileLoadError

logger = logging.getLogger("griptape_nodes")

MediaKind = Literal["image", "video", "audio"]


class DataUriParts(NamedTuple):
    """A decoded base64 ``data:`` URI: a file extension without its dot, and the raw bytes."""

    extension: str
    content: bytes


# Subtypes whose conventional file extension differs from the MIME subtype itself.
_DATA_URI_EXTENSIONS = {
    "jpeg": "jpg",
    "mpeg": "mp3",
    "x-wav": "wav",
    "wave": "wav",
    "svg+xml": "svg",
    "quicktime": "mov",
}

_DEFAULT_MIME_SUBTYPE: dict[MediaKind, str] = {
    "image": "png",
    "video": "mp4",
    "audio": "mpeg",
}

_FALLBACK_MIME: dict[MediaKind, str] = {
    "image": "image/png",
    "video": "video/mp4",
    "audio": "audio/mpeg",
}

_ARTIFACT_NAME: dict[MediaKind, str] = {
    "image": "Image",
    "video": "Video",
    "audio": "Audio",
}


def coerce_media_url_or_data_uri(val: Any, *, kind: MediaKind) -> str | None:
    """Extract a usable string from a media input value.

    Returns one of:

    * an HTTP(S) URL,
    * a ``data:<kind>/...`` URI,
    * a project macro path like ``{inputs}/foo.png``,
    * a plain filesystem path,

    or ``None`` if the value cannot be resolved. Non-URI strings are NOT
    wrapped as raw base64; only artifact ``.base64`` payloads (or serialized
    ``<Kind>Artifact`` dicts) are wrapped as ``data:`` URIs.
    """
    if val is None:
        return None

    mime_prefix = f"data:{kind}/"
    default_subtype = _DEFAULT_MIME_SUBTYPE[kind]
    raw_artifact_type = f"{_ARTIFACT_NAME[kind]}Artifact"

    if isinstance(val, dict):
        return _coerce_from_dict(
            val,
            mime_prefix=mime_prefix,
            default_subtype=default_subtype,
            raw_artifact_type=raw_artifact_type,
        )

    if isinstance(val, str):
        v = val.strip()
        return v or None

    try:
        to_dict = getattr(val, "to_dict", None)
        if callable(to_dict):
            serialized = to_dict()
            if isinstance(serialized, dict):
                coerced = coerce_media_url_or_data_uri(serialized, kind=kind)
                if coerced:
                    return coerced

        v = getattr(val, "value", None)
        if isinstance(v, str) and v.strip():
            return v.strip()

        b64 = getattr(val, "base64", None)
        if isinstance(b64, str) and b64:
            return b64 if b64.startswith(mime_prefix) else f"{mime_prefix}{default_subtype};base64,{b64}"
    except Exception:  # noqa: BLE001 - unknown artifact shapes raise arbitrary errors; treat as unresolvable
        return None

    return None


def _coerce_from_dict(
    val: dict[str, Any],
    *,
    mime_prefix: str,
    default_subtype: str,
    raw_artifact_type: str,
) -> str | None:
    value = val.get("value")
    if not isinstance(value, str) or not value.strip():
        # Fall back to a top-level ``url`` key, used by some serialized URL artifact shapes.
        url = val.get("url")
        if isinstance(url, str) and url.strip():
            return url.strip()
        return None
    stripped = value.strip()

    # URLs / data URIs always pass through, regardless of declared artifact type.
    if stripped.startswith(("http://", "https://", mime_prefix)):
        return stripped

    # Raw <Kind>Artifact: the value is base64 bytes, optionally with a "format"
    # hint that names the MIME subtype.
    if val.get("type") == raw_artifact_type:
        subtype = str(val.get("format") or default_subtype).lower()
        return f"{mime_prefix}{subtype};base64,{stripped}"

    # Anything else (macro paths, filesystem paths, unknown artifact shapes)
    # passes through; File() resolves it downstream.
    return stripped


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


def split_data_uri(data_uri: str) -> DataUriParts | None:
    """Split a base64 ``data:`` URI into a file extension and its decoded bytes.

    Returns ``None`` for anything that is not a base64 ``data:`` URI, including a malformed one, so
    callers can treat "not a data URI" and "not usable as one" the same way.

    The extension is what makes this worth having: an uploader that derives a storage key from a
    data URI string gets the base64 payload as the name, so a caller that needs a real filename has
    to decode the URI itself.
    """
    if not data_uri.startswith("data:"):
        return None

    header, separator, payload = data_uri.partition(",")
    if not separator or not header.endswith(";base64"):
        return None

    # A header can carry parameters between the MIME type and ";base64" (";charset=utf-8"), which
    # are not part of the subtype and must not reach a filename.
    mime = header[len("data:") : -len(";base64")].partition(";")[0]
    subtype = mime.partition("/")[2].lower()

    # Strip whitespace before decoding: line-wrapped base64 is what base64.encodebytes and MIME
    # producers emit, and the driver that reads these URIs decodes it happily. Refusing it here
    # would pass the URI through to an uploader that names the stored object after the payload.
    try:
        content = base64.b64decode("".join(payload.split()), validate=True)
    except (BinasciiError, ValueError):
        return None

    extension = _DATA_URI_EXTENSIONS.get(subtype, subtype)
    if not extension.isalnum():
        # A subtype like "foo/bar" would turn the temp file into a path into a missing directory,
        # and any other punctuation ends up in the stored object's key.
        extension = "bin"
    return DataUriParts(extension=extension, content=content)
