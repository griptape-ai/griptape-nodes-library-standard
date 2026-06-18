"""BytePlus private-asset support: custom artifacts + loader node."""

from griptape_nodes_library.assets.byteplus_provider_asset_reference import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    ASSET_KINDS,
    ASSET_REFERENCE_TYPE_NAMES,
    BytePlusAudioAssetReference,
    BytePlusImageAssetReference,
    BytePlusVideoAssetReference,
    create_provider_asset_reference,
    get_provider_asset_kind,
    get_provider_asset_value,
    is_provider_asset_reference,
    reference_type_name_for_kind,
)

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
