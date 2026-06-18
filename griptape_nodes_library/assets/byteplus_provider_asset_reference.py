"""Custom artifacts marking media as a BytePlus private-asset reference.

A provider-asset reference wraps a media value (a public URL, a project macro
path, a workspace static-files path, or a ``data:`` URI) and carries the asset
*kind* (Image/Video/Audio). The `Seedance20VideoGeneration` node accepts
these types on its existing reference inputs and, at generation time, registers
the media as a BytePlus private asset via the Griptape Cloud proxy and references
it in the request as ``asset://{asset_id}``.

There is one concrete class per kind — `BytePlusImageAssetReference`,
`BytePlusVideoAssetReference`, `BytePlusAudioAssetReference` — and the class name
deliberately contains the kind word ("Image"/"Video"/"Audio"). The Griptape
Nodes UI selects a media preview by substring-matching the serialized ``type``
against those words (and renders ``src = value``), so per-kind names get the
media to preview natively without any GUI changes.

Independently of ``type``, every reference also stamps a sentinel + the kind into
``meta`` (which is ``serializable=True`` and survives the ``to_dict()`` / JSON
round trip). The Seedance node detects "this item is a private asset" from that
sentinel, so detection does not depend on the ``type`` string. The helper
functions below work whether a reference arrives as a live instance or a
serialized dict.
"""

from __future__ import annotations

from typing import Any, ClassVar

from attrs import define
from griptape.artifacts.url_artifact import UrlArtifact

ASSET_KIND_IMAGE = "Image"
ASSET_KIND_VIDEO = "Video"
ASSET_KIND_AUDIO = "Audio"
ASSET_KINDS = [ASSET_KIND_IMAGE, ASSET_KIND_VIDEO, ASSET_KIND_AUDIO]

# meta keys — both are serializable and survive the JSON boundary.
ASSET_SENTINEL_KEY = "byteplus_provider_asset"
ASSET_KIND_META_KEY = "asset_kind"

__all__ = [
    "ASSET_KINDS",
    "ASSET_KIND_AUDIO",
    "ASSET_KIND_IMAGE",
    "ASSET_KIND_VIDEO",
    "ASSET_REFERENCE_TYPE_NAMES",
    "BytePlusAudioAssetReference",
    "BytePlusImageAssetReference",
    "BytePlusVideoAssetReference",
    "create_provider_asset_reference",
    "get_provider_asset_kind",
    "get_provider_asset_value",
    "is_provider_asset_reference",
    "reference_type_name_for_kind",
]


@define
class _BytePlusAssetReferenceBase(UrlArtifact):
    """Shared behavior for the per-kind provider-asset references.

    Concrete subclasses fix the kind; prefer the module-level
    `create_provider_asset_reference(value, asset_kind)` factory or each
    subclass's `create(value)` over the attrs constructor so the kind/sentinel
    land in ``meta``.
    """

    # ClassVar so attrs treats it as a class constant, not an instance field.
    _ASSET_KIND: ClassVar[str] = ""  # overridden per concrete subclass

    @classmethod
    def create(cls, value: str, meta: dict[str, Any] | None = None) -> _BytePlusAssetReferenceBase:
        merged_meta = {
            **(meta or {}),
            ASSET_SENTINEL_KEY: True,
            ASSET_KIND_META_KEY: cls._ASSET_KIND,
        }
        return cls(value=value, meta=merged_meta)

    @property
    def asset_kind(self) -> str | None:
        return self.meta.get(ASSET_KIND_META_KEY)


@define
class BytePlusImageAssetReference(_BytePlusAssetReferenceBase):
    """An image to register as a BytePlus private asset."""

    _ASSET_KIND: ClassVar[str] = ASSET_KIND_IMAGE


@define
class BytePlusVideoAssetReference(_BytePlusAssetReferenceBase):
    """A video to register as a BytePlus private asset."""

    _ASSET_KIND: ClassVar[str] = ASSET_KIND_VIDEO


@define
class BytePlusAudioAssetReference(_BytePlusAssetReferenceBase):
    """Audio to register as a BytePlus private asset."""

    _ASSET_KIND: ClassVar[str] = ASSET_KIND_AUDIO


# kind <-> concrete class / type-name maps.
_KIND_TO_CLASS: dict[str, type[_BytePlusAssetReferenceBase]] = {
    ASSET_KIND_IMAGE: BytePlusImageAssetReference,
    ASSET_KIND_VIDEO: BytePlusVideoAssetReference,
    ASSET_KIND_AUDIO: BytePlusAudioAssetReference,
}
ASSET_REFERENCE_TYPE_NAMES: dict[str, str] = {kind: cls.__name__ for kind, cls in _KIND_TO_CLASS.items()}
_REFERENCE_TYPE_NAME_SET = set(ASSET_REFERENCE_TYPE_NAMES.values())


def reference_type_name_for_kind(asset_kind: str) -> str:
    """Return the artifact type-name string for a kind (e.g. 'BytePlusImageAssetReference')."""
    cls = _KIND_TO_CLASS.get(asset_kind)
    if cls is None:
        msg = f"Unsupported asset_kind '{asset_kind}'. Supported: {', '.join(ASSET_KINDS)}."
        raise ValueError(msg)
    return cls.__name__


def create_provider_asset_reference(value: str, asset_kind: str, meta: dict[str, Any] | None = None) -> UrlArtifact:
    """Build the per-kind provider-asset reference for ``asset_kind``."""
    cls = _KIND_TO_CLASS.get(asset_kind)
    if cls is None:
        msg = f"Unsupported asset_kind '{asset_kind}'. Supported: {', '.join(ASSET_KINDS)}."
        raise ValueError(msg)
    return cls.create(value, meta=meta)


def is_provider_asset_reference(val: Any) -> bool:
    """Return True if ``val`` is a provider-asset reference (instance or dict).

    Detection uses a dual signal so it survives serialization: the per-kind
    ``type`` name set by ``to_dict()`` and a sentinel stored in ``meta``.
    """
    if isinstance(val, _BytePlusAssetReferenceBase):
        return True
    if isinstance(val, dict):
        if val.get("type") in _REFERENCE_TYPE_NAME_SET:
            return True
        meta = val.get("meta")
        if isinstance(meta, dict) and meta.get(ASSET_SENTINEL_KEY) is True:
            return True
    return False


def get_provider_asset_kind(val: Any) -> str | None:
    """Read the asset kind (Image/Video/Audio) from an instance or serialized dict."""
    if isinstance(val, _BytePlusAssetReferenceBase):
        return val.asset_kind
    if isinstance(val, dict):
        meta = val.get("meta")
        if isinstance(meta, dict):
            kind = meta.get(ASSET_KIND_META_KEY)
            if isinstance(kind, str):
                return kind
    return None


def get_provider_asset_value(val: Any) -> str | None:
    """Read the underlying media value from an instance or serialized dict."""
    if isinstance(val, _BytePlusAssetReferenceBase):
        return val.value
    if isinstance(val, dict):
        for key in ("value", "url"):
            candidate = val.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None
