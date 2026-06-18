from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.url_artifact import UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    ASSET_KINDS,
    create_provider_asset_reference,
    reference_type_name_for_kind,
)
from griptape_nodes_library.media import MediaKind, coerce_media_url_or_data_uri
from griptape_nodes_library.proxy.provider_asset_access import (
    ProviderAssetAccess,
    ProviderAssetAccessOutcome,
    check_provider_asset_access,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["SeedanceHumanReferenceAsset"]

# asset_kind -> (media input parameter name, coercion kind)
_KIND_TO_PARAM: dict[str, tuple[str, MediaKind]] = {
    ASSET_KIND_IMAGE: ("image", "image"),
    ASSET_KIND_VIDEO: ("video", "video"),
    ASSET_KIND_AUDIO: ("audio", "audio"),
}

# Badge shown on asset_kind when the org cannot use the provider-asset feature.
_ACCESS_BADGE_TITLE = "Provider-asset access required"


class SeedanceHumanReferenceAsset(DataNode):
    """Package media as a private-asset reference for Seedance 2.0 human-reference inputs.

    Use this node for human-reference inputs (e.g. AIGC virtual-portrait images) that should be
    registered as provider private assets, rather than plugging media directly into the
    Seedance 2.0 reference inputs.

    Choose the asset kind (Image/Video/Audio) and supply the matching media. The node outputs a
    provider-asset reference carrying that media and kind; connect it into the matching reference
    input on the Seedance 2.0 node (Reference Images, Reference Video, or Reference Audio). The
    Seedance node registers the asset with the provider at generation time and references it via
    `asset://{asset_id}`. Private-asset references are only supported by the Seedance 2.0 model
    (not Seedance 2.0 Fast).

    Access to the provider-asset feature is org-gated. If the configured API key cannot use the
    provider-asset APIs, this node shows an error and asks an admin to request access from Foundry.

    Inputs:
        - asset_kind (str): "Image", "Video", or "Audio" (default: Image)
        - image/video/audio: The reference media for the selected kind

    Outputs:
        - asset_reference: The packaged provider-asset reference
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "video/seedance"
        self.description = "Package media as a provider private-asset reference for Seedance 2.0"

        self.add_parameter(
            ParameterString(
                name="asset_kind",
                default_value=ASSET_KIND_IMAGE,
                tooltip="The kind of reference media to register as a provider private asset",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Asset Kind"},
                traits={Options(choices=ASSET_KINDS)},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="image",
                default_value=None,
                tooltip="Reference image to register as a provider private asset",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Image", "clickable_file_browser": True, "expander": True},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                default_value=None,
                tooltip="Reference video to register as a provider private asset",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Video", "clickable_file_browser": True, "expander": True},
            )
        )

        self.add_parameter(
            ParameterAudio(
                name="audio",
                default_value=None,
                tooltip="Reference audio to register as a provider private asset",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Audio", "clickable_file_browser": True, "expander": True},
            )
        )

        # Output type/output_type is swapped per Asset Kind in _update_media_visibility so the
        # connection (and edge color) matches the matching Seedance 2.0 reference input.
        initial_type = reference_type_name_for_kind(ASSET_KIND_IMAGE)
        self._asset_reference_param = Parameter(
            name="asset_reference",
            type=initial_type,
            output_type=initial_type,
            default_value=None,
            tooltip="The packaged private-asset reference. Connect into a Seedance 2.0 reference input.",
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
            ui_options={"display_name": "Asset Reference", "pulse_on_run": True},
        )
        self.add_parameter(self._asset_reference_param)

        self._update_media_visibility()

        # Org-gated feature: probe access at construction so the error surfaces immediately in
        # the editor as a badge. The probe is best-effort (suppressed, short timeout, fail-closed)
        # so it never raises out of construction, though it can block for up to the probe timeout.
        # It is re-probed at run time (validate_before_node_run) because entitlement can change
        # after the graph is loaded.
        self._access: ProviderAssetAccess | None = None
        self._refresh_access()

    def _refresh_access(self) -> ProviderAssetAccess:
        """Re-probe provider-asset access, update the badge, and cache/return the result.

        Never raises: a probe failure is reported as INDETERMINATE and surfaced as a badge
        rather than breaking node construction. Only a confirmed denial (403) shows the
        error/Foundry badge; an indeterminate probe shows a non-blocking warning.
        """
        access = ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.INDETERMINATE,
            detail="Provider-asset access could not be determined.",
        )
        with suppress(Exception):
            access = check_provider_asset_access()
        self._access = access

        kind_param = self.get_parameter_by_name("asset_kind")
        if kind_param is not None:
            if access.outcome is ProviderAssetAccessOutcome.GRANTED:
                kind_param.clear_badge()
            elif access.outcome is ProviderAssetAccessOutcome.DENIED:
                kind_param.set_badge(variant="error", title=_ACCESS_BADGE_TITLE, message=access.detail)
            else:
                # Indeterminate: real but non-entitlement cause — warn, don't assert no-access.
                kind_param.set_badge(variant="warning", title="Provider-asset access unverified", message=access.detail)
        return access

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Toggle the visible media input on kind change; normalize media; publish the output."""
        if parameter.name == "asset_kind":
            self._update_media_visibility()

        if parameter.name == "image":
            artifact = normalize_artifact_input(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if artifact != value:
                self.set_parameter_value("image", artifact)
        elif parameter.name == "audio":
            artifact = normalize_artifact_input(value, AudioUrlArtifact, accepted_types=(AudioArtifact,))
            if artifact != value:
                self.set_parameter_value("audio", artifact)

        # Recompute the reference output reactively so a downstream node can preview it
        # without first running this node.
        if parameter.name in ("asset_kind", "image", "video", "audio"):
            self._publish_asset_reference()

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Block the run only on a confirmed no-access (403) result.

        An indeterminate probe (auth error, server error, connectivity) is not treated as
        no-access: the run proceeds and the real failure surfaces from the actual API call
        rather than being masked by a misleading "request access" error.
        """
        exceptions = super().validate_before_node_run() or []

        access = self._refresh_access()
        if access.is_denied:
            exceptions.append(ValueError(f"{self.name}: {access.detail}"))

        return exceptions if exceptions else None

    def _update_media_visibility(self) -> None:
        """Show only the media input matching the selected asset kind and retype the output."""
        kind = self.get_parameter_value("asset_kind") or ASSET_KIND_IMAGE
        visible_param = _KIND_TO_PARAM.get(kind, _KIND_TO_PARAM[ASSET_KIND_IMAGE])[0]
        for param_name in ("image", "video", "audio"):
            if param_name == visible_param:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

        # Retype the output so the connection (and edge color) matches the receiving
        # Seedance 2.0 reference input for this kind.
        reference_type = reference_type_name_for_kind(kind)
        self._asset_reference_param.type = reference_type
        self._asset_reference_param.output_type = reference_type

    def _build_asset_reference(self) -> UrlArtifact | None:
        """Build the reference from the current kind + media inputs (None if no media)."""
        kind = self.get_parameter_value("asset_kind") or ASSET_KIND_IMAGE
        param_name, coerce_kind = _KIND_TO_PARAM.get(kind, _KIND_TO_PARAM[ASSET_KIND_IMAGE])

        media = self.get_parameter_value(param_name)
        media_value = coerce_media_url_or_data_uri(media, kind=coerce_kind) if media is not None else None
        if not media_value:
            logger.info("%s: no %s media provided; asset_reference is None", self.name, kind)
            return None

        return create_provider_asset_reference(value=media_value, asset_kind=kind)

    def _publish_asset_reference(self) -> None:
        """Compute and publish the output so connected nodes update without a run."""
        self.publish_update_to_parameter("asset_reference", self._build_asset_reference())

    def process(self) -> None:
        self.parameter_output_values["asset_reference"] = self._build_asset_reference()
