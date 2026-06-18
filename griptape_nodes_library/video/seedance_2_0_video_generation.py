from __future__ import annotations

import asyncio
import json as _json
import logging
from contextlib import suppress
from typing import Any, ClassVar
from urllib.parse import urljoin
from uuid import uuid4

import httpx
from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input, normalize_artifact_list

from griptape_nodes_library.assets import (
    ASSET_KIND_AUDIO,
    ASSET_KIND_IMAGE,
    ASSET_KIND_VIDEO,
    ASSET_REFERENCE_TYPE_NAMES,
    get_provider_asset_kind,
    get_provider_asset_value,
    is_provider_asset_reference,
)
from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["Seedance20VideoGeneration"]

INPUT_MODE_TEXT_ONLY = "Text Only"
INPUT_MODE_FIRST_LAST_FRAME = "First/Last Frame"
INPUT_MODE_MULTIMODAL_REFERENCES = "Multimodal References"
MODEL_NAME_SEEDANCE_2_0 = "Seedance 2.0"
MODEL_NAME_SEEDANCE_2_0_FAST = "Seedance 2.0 Fast"
SEEDANCE_2_0_MODEL_ID = "dreamina-seedance-2-0-260128"
SEEDANCE_2_0_FAST_MODEL_ID = "dreamina-seedance-2-0-fast-260128"

# Provider-asset (private asset) registration via the GTC proxy. Only the Seedance 2.0 model
# supports private-asset references; Seedance 2.0 Fast does not.
ASSET_PROVIDER = "byteplus_ark"
ASSET_POLL_INTERVAL = 3  # seconds
ASSET_MAX_ATTEMPTS = 60  # ~3 min cap, independent of the generation timeout
ASSET_STATUS_ACTIVE = "ACTIVE"
ASSET_STATUS_FAILED = "FAILED"
ASSET_STATUS_DELETED = "DELETED"
# Skip provider-side moderation on private-asset ingestion (content is already moderated upstream).
ASSET_MODERATION = {"Strategy": "Skip"}


class Seedance20VideoGeneration(GriptapeProxyNode):
    """Generate a video using Seedance 2.0 models via Griptape Cloud model proxy.

    Supports three input modes:
    - Text Only: Pure text-to-video generation (default)
    - First/Last Frame: Traditional i2v with first and/or last frame images
    - Multimodal References: Up to 9 images + 3 videos + 3 audio files as references

    Inputs:
        - prompt (str): Text prompt for the video
        - model_id (str): Model to use (default: Seedance 2.0)
        - input_mode (str): "Text Only", "First/Last Frame", or "Multimodal References" (default: Text Only)
        - resolution (str): Output resolution (default: 720p, options: 480p, 720p, 1080p [Seedance 2.0 only])
        - ratio (str): Output aspect ratio (default: adaptive)
        - duration (int): Video duration in seconds (default: 5, range: 4-15 or -1 for smart)
        - generate_audio (bool): Generate audio with video (default: False)
        - first_frame/last_frame: Optional frame images (First/Last Frame mode only, last_frame requires Seedance 2.0)
        - reference_images/reference_video_1..3/reference_audio: Optional reference media (Multimodal mode only)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Verbatim response from API
        - video_url (VideoUrlArtifact): Saved static video URL
        - was_successful (bool): Whether generation succeeded
        - result_details (str): Details about the result or error
    """

    MODEL_NAME_MAP: ClassVar[dict[str, str]] = {
        MODEL_NAME_SEEDANCE_2_0_FAST: SEEDANCE_2_0_FAST_MODEL_ID,
        MODEL_NAME_SEEDANCE_2_0: SEEDANCE_2_0_MODEL_ID,
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate video via Seedance 2.0 through Griptape Cloud model proxy"

        # Transient (helper, scratch parameter name) pairs created while registering private
        # assets; the upload and the scratch parameter are both cleaned up in
        # _process_generation's finally block so they don't accumulate on the node across runs.
        self._pending_asset_uploads: list[tuple[PublicArtifactUrlParameter, str]] = []

        # Model selection
        self.add_parameter(
            ParameterString(
                name="model_id",
                default_value=MODEL_NAME_SEEDANCE_2_0,
                tooltip="Model to use for video generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Model", "hide": False},
                traits={Options(choices=[MODEL_NAME_SEEDANCE_2_0, MODEL_NAME_SEEDANCE_2_0_FAST])},
            )
        )

        # Input mode selector
        self.add_parameter(
            ParameterString(
                name="input_mode",
                default_value=INPUT_MODE_TEXT_ONLY,
                tooltip="Input mode: Text Only for pure text-to-video, First/Last Frame for i2v, or Multimodal References for images/videos/audio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Input Mode", "hide": False},
                traits={
                    Options(
                        choices=[INPUT_MODE_TEXT_ONLY, INPUT_MODE_FIRST_LAST_FRAME, INPUT_MODE_MULTIMODAL_REFERENCES]
                    )
                },
            )
        )

        # Prompt
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text prompt for the video. In Multimodal References mode, media can be referenced in the order given (e.g., [Image 1], [Image 2], [Video 1], [Video 2], [Audio 1]).",
                multiline=True,
                placeholder_text="Describe the video...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        # First/Last Frame inputs (for First/Last Frame mode)
        self.add_parameter(
            ParameterImage(
                name="first_frame",
                default_value=None,
                tooltip="Optional first frame image",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "First Frame"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="last_frame",
                default_value=None,
                tooltip="Optional last frame image (Seedance 2.0 only)",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "Last Frame"},
            )
        )

        # Multimodal reference inputs (for Multimodal mode)
        self.add_parameter(
            ParameterList(
                name="reference_images",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_IMAGE]],
                default_value=[],
                tooltip="Optional reference images (0-9 images). Connect a Seedance Human Reference Asset to register an image as a private asset (Seedance 2.0 only).",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Images", "expander": True, "hide_property": True},
                max_items=9,
            )
        )

        self._public_reference_video_parameter_1 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_1",
                input_types=["VideoUrlArtifact", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_VIDEO]],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional first reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 1"},
                hide_property=True,
            ),
            disclaimer_message="The Seedance 2.0 service utilizes this URL to access the reference video.",
        )
        self._public_reference_video_parameter_1.add_input_parameters()

        self._public_reference_video_parameter_2 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_2",
                input_types=["VideoUrlArtifact", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_VIDEO]],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional second reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 2"},
                hide_property=True,
                hide=True,
            ),
            disclaimer_message="The Seedance 2.0 service utilizes this URL to access the reference video.",
        )
        self._public_reference_video_parameter_2.add_input_parameters()
        self.hide_message_by_name("artifact_url_parameter_message_reference_video_2")

        self._public_reference_video_parameter_3 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_3",
                input_types=["VideoUrlArtifact", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_VIDEO]],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional third reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 3"},
                hide_property=True,
                hide=True,
            ),
            disclaimer_message="The Seedance 2.0 service utilizes this URL to access the reference video.",
        )
        self._public_reference_video_parameter_3.add_input_parameters()
        self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")

        self.add_parameter(
            ParameterList(
                name="reference_audio",
                input_types=["AudioArtifact", "AudioUrlArtifact", "str", ASSET_REFERENCE_TYPE_NAMES[ASSET_KIND_AUDIO]],
                default_value=[],
                tooltip="Optional reference audio (0-3 audio files, 2-15s each, max 15s total). URLs, asset:// IDs, or base64/data URIs are supported. Connect a Seedance Human Reference Asset to register audio as a private asset (Seedance 2.0 only).",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Audio", "expander": True, "hide_property": True},
                max_items=3,
            )
        )

        # Generation settings
        with ParameterGroup(name="Generation Settings") as settings_group:
            ParameterString(
                name="resolution",
                default_value="720p",
                tooltip="Output resolution (480p, 720p, or 1080p; 1080p is Seedance 2.0 only)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["480p", "720p", "1080p"])},
            )

            ParameterString(
                name="ratio",
                default_value="adaptive",
                tooltip="Output aspect ratio",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["adaptive", "21:9", "16:9", "4:3", "1:1", "3:4", "9:16"])},
            )

            ParameterInt(
                name="duration",
                default_value=5,
                tooltip="Video duration in seconds (4-15 or -1 for smart selection)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[-1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])},
            )

            ParameterBool(
                name="generate_audio",
                default_value=False,
                tooltip="Generate audio with video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

        self.add_node_element(settings_group)

        # Outputs
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="seedance_2_0_video.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        # Set initial visibility
        self._update_parameter_visibility()

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # reference_video_1/2/3 have PROPERTY mode, so the framework does not auto-clear
        # their value when an incoming connection is removed. Clear it manually so the
        # previously-connected video does not linger on the node.
        if target_parameter.name in {"reference_video_1", "reference_video_2", "reference_video_3"}:
            if target_parameter.name in self.parameter_values:
                self.remove_parameter_value(target_parameter.name)
            self._update_parameter_visibility()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes to show/hide inputs based on mode."""
        if parameter.name in {"input_mode", "model_id", "reference_video_1", "reference_video_2", "reference_video_3"}:
            self._update_parameter_visibility()

        if parameter.name in {"first_frame", "last_frame"}:
            artifact = normalize_artifact_input(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if artifact != value:
                self.set_parameter_value(parameter.name, artifact)

        # Normalize reference images
        if parameter.name == "reference_images" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("reference_images", updated_list)

        if parameter.name == "reference_audio" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, AudioUrlArtifact, accepted_types=(AudioArtifact,))
            if updated_list != value:
                self.set_parameter_value("reference_audio", updated_list)

        return super().after_value_set(parameter, value)

    def _update_parameter_visibility(self) -> None:
        """Update parameter visibility based on selected input mode."""
        input_mode = self.get_parameter_value("input_mode") or INPUT_MODE_TEXT_ONLY
        model_id = self._get_api_model_id()

        self._update_resolution_options(model_id)

        if input_mode == INPUT_MODE_MULTIMODAL_REFERENCES:
            # Show multimodal inputs, hide first/last frame
            self.hide_parameter_by_name("first_frame")
            self.hide_parameter_by_name("last_frame")
            self.show_parameter_by_name("reference_images")
            self.show_parameter_by_name("reference_audio")
            self._update_reference_video_visibility()
        elif input_mode == INPUT_MODE_FIRST_LAST_FRAME:
            # Show first/last frame, hide multimodal inputs
            self.show_parameter_by_name("first_frame")
            self.hide_parameter_by_name("reference_images")
            self.hide_parameter_by_name("reference_audio")
            self.hide_parameter_by_name(["reference_video_1", "reference_video_2", "reference_video_3"])
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_1")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_2")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")
            if self._supports_last_frame(model_id):
                self.show_parameter_by_name("last_frame")
            else:
                self.hide_parameter_by_name("last_frame")
        else:
            # Text Only mode: hide all media inputs
            self.hide_parameter_by_name("first_frame")
            self.hide_parameter_by_name("last_frame")
            self.hide_parameter_by_name("reference_images")
            self.hide_parameter_by_name("reference_audio")
            self.hide_parameter_by_name(["reference_video_1", "reference_video_2", "reference_video_3"])
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_1")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_2")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")

    def _update_reference_video_visibility(self) -> None:
        """Progressively reveal reference video inputs in multimodal mode."""
        reference_video_1 = self.get_parameter_value("reference_video_1")
        reference_video_2 = self.get_parameter_value("reference_video_2")
        reference_video_3 = self.get_parameter_value("reference_video_3")

        show_video_2 = bool(reference_video_1 or reference_video_2 or reference_video_3)
        show_video_3 = bool(reference_video_2 or reference_video_3)

        self.show_parameter_by_name("reference_video_1")
        self.show_message_by_name("artifact_url_parameter_message_reference_video_1")

        if show_video_2:
            self.show_parameter_by_name("reference_video_2")
            self.show_message_by_name("artifact_url_parameter_message_reference_video_2")
        else:
            self.hide_parameter_by_name("reference_video_2")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_2")

        if show_video_3:
            self.show_parameter_by_name("reference_video_3")
            self.show_message_by_name("artifact_url_parameter_message_reference_video_3")
        else:
            self.hide_parameter_by_name("reference_video_3")
            self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")

    def _update_resolution_options(self, model_id: str) -> None:
        """Update resolution choices based on selected model (1080p is Seedance 2.0 only)."""
        resolution_param = self.get_parameter_by_name("resolution")
        if resolution_param is None:
            return

        available_resolutions = ["480p", "720p", "1080p"] if self._supports_1080p(model_id) else ["480p", "720p"]

        existing_traits = resolution_param.find_elements_by_type(Options)
        if existing_traits:
            resolution_param.remove_trait(trait_type=existing_traits[0])
        resolution_param.add_trait(Options(choices=available_resolutions))

        current_resolution = self.get_parameter_value("resolution")
        if current_resolution not in available_resolutions:
            self.set_parameter_value("resolution", "720p")

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation."""
        raw_model_id = self.get_parameter_value("model_id") or MODEL_NAME_SEEDANCE_2_0
        return self.MODEL_NAME_MAP.get(raw_model_id, raw_model_id)

    async def _process_generation(self) -> None:
        self._pending_asset_uploads = []
        try:
            await super()._process_generation()
        finally:
            self._public_reference_video_parameter_1.delete_uploaded_artifact()
            self._public_reference_video_parameter_2.delete_uploaded_artifact()
            self._public_reference_video_parameter_3.delete_uploaded_artifact()
            # Provider assets are reclaimed by the backend: a submitted generation deletes its
            # linked assets on terminal state, and assets we register but never submit (e.g. a
            # build failure after registration) are reclaimed by the backend's orphan sweeper.
            # The transient GTC static-storage upload made to feed CreateProviderAsset is ours
            # to clean up, along with the scratch parameter created to perform the upload (its
            # name is unique per call, so leaving it would accumulate parameters on the node).
            for helper, scratch_name in self._pending_asset_uploads:
                with suppress(Exception):
                    helper.delete_uploaded_artifact()
                with suppress(Exception):
                    self.remove_parameter_element_by_name(scratch_name)
            self._pending_asset_uploads = []

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        params = self._get_parameters()

        # Validate parameter combinations
        try:
            self._validate_parameters(params)
        except ValueError as e:
            exceptions.append(e)

        return exceptions if exceptions else None

    def _get_parameters(self) -> dict[str, Any]:
        raw_model_id = self.get_parameter_value("model_id") or MODEL_NAME_SEEDANCE_2_0
        model_id = self.MODEL_NAME_MAP.get(raw_model_id, raw_model_id)
        first_frame = normalize_artifact_input(
            self.get_parameter_value("first_frame"),
            ImageUrlArtifact,
            accepted_types=(ImageArtifact,),
        )
        last_frame = normalize_artifact_input(
            self.get_parameter_value("last_frame"),
            ImageUrlArtifact,
            accepted_types=(ImageArtifact,),
        )

        # Normalize reference images
        reference_images = self.get_parameter_value("reference_images") or []
        normalized_reference_images = (
            normalize_artifact_list(reference_images, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if reference_images
            else []
        )
        reference_audio = self.get_parameter_value("reference_audio") or []
        normalized_reference_audio = (
            normalize_artifact_list(reference_audio, AudioUrlArtifact, accepted_types=(AudioArtifact,))
            if reference_audio
            else []
        )

        return {
            "prompt": self.get_parameter_value("prompt") or "",
            "model_id": model_id,
            "input_mode": self.get_parameter_value("input_mode") or INPUT_MODE_TEXT_ONLY,
            "resolution": self.get_parameter_value("resolution") or "720p",
            "ratio": self.get_parameter_value("ratio") or "adaptive",
            "duration": self.get_parameter_value("duration"),
            "generate_audio": self.get_parameter_value("generate_audio"),
            "first_frame": first_frame,
            "last_frame": last_frame,
            "reference_images": normalized_reference_images,
            "reference_video_1": self.get_parameter_value("reference_video_1"),
            "reference_video_2": self.get_parameter_value("reference_video_2"),
            "reference_video_3": self.get_parameter_value("reference_video_3"),
            "reference_audio": normalized_reference_audio,
        }

    def _validate_parameters(self, params: dict[str, Any]) -> None:
        """Validate parameter combinations before submission."""
        input_mode = params["input_mode"]

        has_first_frame = params.get("first_frame") is not None
        has_last_frame = params.get("last_frame") is not None
        has_reference_images = bool(params.get("reference_images") and len(params["reference_images"]) > 0)
        reference_video_inputs = self._get_reference_video_inputs(params)
        has_reference_videos = bool(reference_video_inputs)
        has_reference_audio = bool(params.get("reference_audio") and len(params["reference_audio"]) > 0)
        has_any_media = (
            has_first_frame or has_last_frame or has_reference_images or has_reference_videos or has_reference_audio
        )

        # Text Only mode: no media allowed
        if input_mode == INPUT_MODE_TEXT_ONLY:
            if has_any_media:
                msg = (
                    f"{self.name}: {INPUT_MODE_TEXT_ONLY} mode does not accept any media inputs. "
                    f"Switch to {INPUT_MODE_FIRST_LAST_FRAME} or {INPUT_MODE_MULTIMODAL_REFERENCES} mode, "
                    "or clear all media inputs."
                )
                raise ValueError(msg)

        # First/Last Frame mode: only first/last frame allowed
        elif input_mode == INPUT_MODE_FIRST_LAST_FRAME:
            if has_reference_images or has_reference_videos or has_reference_audio:
                msg = (
                    f"{self.name}: reference_images/reference_video_1/reference_video_2/reference_video_3/reference_audio are only used in "
                    f"{INPUT_MODE_MULTIMODAL_REFERENCES} mode. Switch input_mode to {INPUT_MODE_MULTIMODAL_REFERENCES} "
                    "or clear the multimodal reference inputs."
                )
                raise ValueError(msg)

            if params.get("last_frame") and not self._supports_last_frame(params["model_id"]):
                msg = (
                    f"{self.name}: Seedance 2.0 Fast does not support last_frame. "
                    "Use first_frame only, or switch to Seedance 2.0 for first+last frame generation."
                )
                raise ValueError(msg)

        # Multimodal References mode: only reference media allowed
        elif input_mode == INPUT_MODE_MULTIMODAL_REFERENCES:
            if has_first_frame or has_last_frame:
                msg = (
                    f"{self.name}: first_frame/last_frame inputs are only used in {INPUT_MODE_FIRST_LAST_FRAME} mode. "
                    f"Switch input_mode to {INPUT_MODE_FIRST_LAST_FRAME} or clear the frame inputs."
                )
                raise ValueError(msg)

        # Multimodal mode validation
        if input_mode == INPUT_MODE_MULTIMODAL_REFERENCES:
            # Audio requires at least one image or video
            if has_reference_audio and not (has_reference_images or has_reference_videos):
                msg = (
                    f"{self.name}: Seedance 2.0 requires at least one reference image or video when using audio. "
                    "Audio cannot be used alone."
                )
                raise ValueError(msg)

            # Validate counts
            if has_reference_images and len(params["reference_images"]) > 9:
                msg = f"{self.name}: Seedance 2.0 supports up to 9 reference images, got {len(params['reference_images'])}."
                raise ValueError(msg)

            if params.get("reference_video_2") and not params.get("reference_video_1"):
                msg = f"{self.name}: reference_video_2 requires reference_video_1 to be set first."
                raise ValueError(msg)

            if params.get("reference_video_3") and not params.get("reference_video_2"):
                msg = f"{self.name}: reference_video_3 requires reference_video_2 to be set first."
                raise ValueError(msg)

            if has_reference_audio and len(params["reference_audio"]) > 3:
                msg = f"{self.name}: Seedance 2.0 supports up to 3 reference audio files, got {len(params['reference_audio'])}."
                raise ValueError(msg)

        # Validate duration range (4-15 or -1)
        duration = params.get("duration")
        if duration is not None and duration != -1 and not (4 <= duration <= 15):
            msg = f"{self.name}: Seedance 2.0 supports duration between 4-15 seconds or -1 for smart selection, got {duration}."
            raise ValueError(msg)

        # 1080p is only supported on Seedance 2.0 (not Fast)
        if params.get("resolution") == "1080p" and not self._supports_1080p(params["model_id"]):
            msg = (
                f"{self.name}: Seedance 2.0 Fast does not support 1080p resolution. "
                "Use 480p or 720p, or switch to Seedance 2.0 for 1080p generation."
            )
            raise ValueError(msg)

        # Private-asset references require Seedance 2.0 and Griptape auth (not Fast, not BYOK).
        # Gate first so the user gets a specific message before the per-kind check.
        self._validate_private_asset_model(params)
        self._validate_private_asset_auth(params)

        # Private-asset reference kind must match the receiving input (early feedback; the
        # authoritative check happens at build time in _append_private_asset).
        if self._private_assets_active(params["model_id"]):
            self._validate_private_asset_kinds(params)

    def _iter_reference_asset_checks(self, params: dict[str, Any]) -> list[tuple[Any, str]]:
        """List (reference value, expected kind) pairs across all multimodal reference inputs."""
        checks: list[tuple[Any, str]] = []
        for item in params.get("reference_images") or []:
            checks.append((item, ASSET_KIND_IMAGE))
        for name in ("reference_video_1", "reference_video_2", "reference_video_3"):
            value = params.get(name)
            if value:
                checks.append((value, ASSET_KIND_VIDEO))
        for item in params.get("reference_audio") or []:
            checks.append((item, ASSET_KIND_AUDIO))
        return checks

    def _validate_private_asset_model(self, params: dict[str, Any]) -> None:
        """Raise if a private-asset reference is connected while the model doesn't support it."""
        if self._supports_private_assets(params["model_id"]):
            return
        for value, _ in self._iter_reference_asset_checks(params):
            if is_provider_asset_reference(value):
                msg = (
                    f"{self.name}: Seedance 2.0 Fast does not support private-asset references "
                    "(Seedance Human Reference Asset). Switch the model to Seedance 2.0, or remove the "
                    "private-asset reference inputs."
                )
                raise ValueError(msg)

    def _validate_private_asset_auth(self, params: dict[str, Any]) -> None:
        """Raise if a private-asset reference is connected while the node uses BYOK auth.

        Provider assets are registered through the Griptape Cloud proxy on the org's behalf, so
        they are unavailable when the user brings their own provider key.
        """
        if not self._is_byok_enabled():
            return
        for value, _ in self._iter_reference_asset_checks(params):
            if is_provider_asset_reference(value):
                msg = (
                    f"{self.name}: private-asset references (Seedance Human Reference Asset) require "
                    "Griptape authentication and are not available when using your own provider key. "
                    "Switch off the customer key option, or remove the private-asset reference inputs."
                )
                raise ValueError(msg)

    def _validate_private_asset_kinds(self, params: dict[str, Any]) -> None:
        """Raise if a private-asset reference's kind doesn't match its receiving input."""
        for value, expected_kind in self._iter_reference_asset_checks(params):
            if is_provider_asset_reference(value):
                actual_kind = get_provider_asset_kind(value)
                if actual_kind != expected_kind:
                    msg = (
                        f"{self.name}: a {actual_kind or 'unknown'} private-asset reference is connected to a "
                        f"{expected_kind} reference input. Set the Seedance Human Reference Asset's Asset Kind "
                        f"to {expected_kind}, or connect it to the matching reference input."
                    )
                    raise ValueError(msg)

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Seedance 2.0 API."""
        params = self._get_parameters()
        model_id = params["model_id"]
        self._log(
            f"{self.name} parameter summary: "
            f"model_id={model_id}, "
            f"input_mode={params['input_mode']}, "
            f"first_frame_present={params['first_frame'] is not None}, "
            f"first_frame_type={type(params['first_frame']).__name__ if params['first_frame'] is not None else 'None'}, "
            f"last_frame_present={params['last_frame'] is not None}, "
            f"last_frame_type={type(params['last_frame']).__name__ if params['last_frame'] is not None else 'None'}, "
            f"reference_images={len(params['reference_images'])}, "
            f"reference_videos={len(self._get_reference_video_inputs(params))}, "
            f"reference_audio={len(params['reference_audio'])}"
        )

        # Build payload with text prompt
        prompt_text = params["prompt"].strip()
        payload: dict[str, Any] = {"model": model_id}

        # Add config parameters at top level
        if params["resolution"]:
            payload["resolution"] = params["resolution"]
        if params["ratio"]:
            payload["ratio"] = params["ratio"]
        if params["duration"] is not None:
            payload["duration"] = int(params["duration"])
        if params["generate_audio"] is not None:
            payload["generate_audio"] = bool(params["generate_audio"])

        content_list = [{"type": "text", "text": prompt_text}]

        # Add media inputs based on mode
        await self._add_media_inputs_async(content_list, params)

        payload["content"] = content_list

        return payload

    async def _add_media_inputs_async(self, content_list: list[dict[str, Any]], params: dict[str, Any]) -> None:
        """Add media inputs to content list based on input mode."""
        input_mode = params["input_mode"]

        if input_mode == INPUT_MODE_MULTIMODAL_REFERENCES:
            self._log(f"{self.name} building multimodal content")
            supports_assets = self._private_assets_active(params["model_id"])
            order_log: list[str] = []

            # Reference images (Image 1..N within this list, normal or private asset).
            for idx, ref_image in enumerate(params.get("reference_images", [])[:9], start=1):
                if supports_assets and is_provider_asset_reference(ref_image):
                    asset_url = await self._append_private_asset(
                        ref_image, expected_kind=ASSET_KIND_IMAGE, label=f"reference image {idx}"
                    )
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": asset_url}, "role": "reference_image"}
                    )
                    order_log.append(f"Image {idx}: private asset")
                else:
                    ref_url = await self._prepare_frame_url_async(ref_image, frame_label="reference_image")
                    if ref_url:
                        content_list.append(
                            {"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"}
                        )
                        order_log.append(f"Image {idx}: reference")

            # Reference videos (Video 1..3 by slot).
            for idx, ref_video in enumerate(self._get_reference_video_inputs(params), start=1):
                value = ref_video["value"]
                if supports_assets and is_provider_asset_reference(value):
                    asset_url = await self._append_private_asset(
                        value, expected_kind=ASSET_KIND_VIDEO, label=f"reference video {idx}"
                    )
                    content_list.append(
                        {"type": "video_url", "video_url": {"url": asset_url}, "role": "reference_video"}
                    )
                    order_log.append(f"Video {idx}: private asset")
                else:
                    video_url = self._get_reference_video_url(ref_video["parameter_name"], value)
                    if not video_url:
                        msg = (
                            f"{self.name}: {ref_video['parameter_name']} only supports public URLs, uploaded asset URLs, "
                            "or asset:// IDs. Seedance 2.0 does not accept video base64."
                        )
                        raise ValueError(msg)
                    content_list.append(
                        {"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"}
                    )
                    order_log.append(f"Video {idx}: reference")

            # Reference audio (Audio 1..N within this list, normal or private asset).
            for idx, ref_audio in enumerate(params.get("reference_audio", [])[:3], start=1):
                if supports_assets and is_provider_asset_reference(ref_audio):
                    asset_url = await self._append_private_asset(
                        ref_audio, expected_kind=ASSET_KIND_AUDIO, label=f"reference audio {idx}"
                    )
                    content_list.append(
                        {"type": "audio_url", "audio_url": {"url": asset_url}, "role": "reference_audio"}
                    )
                    order_log.append(f"Audio {idx}: private asset")
                else:
                    audio_url = await self._prepare_audio_url_async(ref_audio, audio_label="reference_audio")
                    if audio_url:
                        content_list.append(
                            {"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"}
                        )
                        order_log.append(f"Audio {idx}: reference")

            if order_log:
                self._log(f"{self.name} resolved reference order: " + "; ".join(order_log))
        elif input_mode == INPUT_MODE_FIRST_LAST_FRAME:
            self._log(f"{self.name} building first/last-frame content")
            # First/Last Frame mode
            first_frame_url = await self._prepare_frame_url_async(params["first_frame"], frame_label="first_frame")
            if first_frame_url:
                content_list.append({"type": "image_url", "image_url": {"url": first_frame_url}, "role": "first_frame"})

            if self._supports_last_frame(params["model_id"]):
                last_frame_url = await self._prepare_frame_url_async(params["last_frame"], frame_label="last_frame")
                if last_frame_url:
                    content_list.append(
                        {"type": "image_url", "image_url": {"url": last_frame_url}, "role": "last_frame"}
                    )
        else:
            # Text Only mode: no media inputs
            self._log(f"{self.name} text-only mode, no media inputs")

    # --- Provider private-asset registration (Seedance 2.0 only) ----------------------------

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

        Already-public http(s) URLs pass through. Otherwise the media is uploaded to GTC static
        storage via a transient PublicArtifactUrlParameter (tracked for cleanup). CreateProviderAsset
        requires a fetchable URL — data URIs / unresolvable inputs raise.
        """
        media_value = get_provider_asset_value(ref)
        if not media_value:
            msg = f"{self.name}: private-asset reference has no media value to register."
            raise ValueError(msg)

        if media_value.startswith(("http://", "https://")) and "localhost" not in media_value:
            return media_value

        artifact_type = {
            ASSET_KIND_IMAGE: "ImageUrlArtifact",
            ASSET_KIND_VIDEO: "VideoUrlArtifact",
            ASSET_KIND_AUDIO: "AudioUrlArtifact",
        }[asset_kind]

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
        self.set_parameter_value(scratch_name, media_value)

        public_url = helper.get_public_url_for_parameter()
        if not (public_url.startswith(("http://", "https://")) and "localhost" not in public_url):
            msg = (
                f"{self.name}: could not obtain a public URL for the {asset_kind} private asset. "
                "Provider asset registration requires a publicly fetchable URL (data URIs are not supported)."
            )
            raise RuntimeError(msg)
        return public_url

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
                f"input_summary={self._summarize_image_input(frame_input)}"
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
        if not audio_input:
            self._log(f"{self.name} {audio_label} not provided")
            return None

        audio_url = coerce_media_url_or_data_uri(audio_input, kind="audio")
        if not audio_url:
            self._log(
                f"{self.name} {audio_label} could not be converted to an audio URL or data URI. "
                f"input_type={type(audio_input).__name__}, "
                f"input_module={type(audio_input).__module__}, "
                f"input_summary={self._summarize_image_input(audio_input)}"
            )
            return None

        if audio_url.startswith(("data:audio/", "http://", "https://", "asset://")):
            self._log(f"{self.name} {audio_label} prepared as direct audio URL/data URI")
            return audio_url

        try:
            data_uri = await File(audio_url).aread_data_uri(fallback_mime="audio/wav")
            self._log(f"{self.name} {audio_label} loaded from file into data URI")
            return data_uri
        except FileLoadError as e:
            self._log(f"{self.name} {audio_label} failed to load from {audio_url}: {e}")
            return None

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the result and set output parameters."""
        extracted_url = self._extract_video_url(result_json)
        if not extracted_url:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no video URL was found in the response.",
            )
            return

        # Download and save video
        try:
            self._log("Downloading video bytes from provider URL")
            video_bytes = await self._download_bytes_from_url(extracted_url)
        except Exception as e:
            self._log(f"Failed to download video: {e}")
            video_bytes = None

        if video_bytes:
            try:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(video_bytes)
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
                self._log(f"Saved video as {saved.name}")
                self._set_status_results(
                    was_successful=True, result_details=f"Video generated successfully and saved as {saved.name}."
                )
            except Exception as e:
                self._log(f"Failed to save video: {e}, using provider URL")
                self.parameter_output_values["video_url"] = VideoUrlArtifact(value=extracted_url)
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Video generated successfully. Using provider URL (could not save to storage: {e}).",
                )
        else:
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=extracted_url)
            self._set_status_results(
                was_successful=True,
                result_details="Video generated successfully. Using provider URL (could not download video bytes).",
            )

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from failed response."""
        if not response_json:
            return super()._extract_error_message(response_json)

        parsed_provider_response = self._parse_provider_response(response_json.get("provider_response"))
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

    def _parse_provider_response(self, provider_response: Any) -> dict[str, Any] | None:
        """Parse provider_response if it's a JSON string."""
        if isinstance(provider_response, str):
            try:
                return _json.loads(provider_response)
            except Exception:
                return None
        if isinstance(provider_response, dict):
            return provider_response
        return None

    def _set_safe_defaults(self) -> None:
        """Clear all output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None

    @staticmethod
    def _extract_video_url(obj: dict[str, Any] | None) -> str | None:
        if not obj:
            return None
        for key in ("url", "video_url", "output_url"):
            val = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(val, str) and val.startswith("http"):
                return val
        for key in ("result", "data", "output", "outputs", "content", "task_result"):
            nested = obj.get(key) if isinstance(obj, dict) else None
            if isinstance(nested, dict):
                url = Seedance20VideoGeneration._extract_video_url(nested)
                if url:
                    return url
            elif isinstance(nested, list):
                for item in nested:
                    url = Seedance20VideoGeneration._extract_video_url(item if isinstance(item, dict) else None)
                    if url:
                        return url
        return None

    @staticmethod
    def _coerce_video_url(val: Any) -> str | None:
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
                    coerced = Seedance20VideoGeneration._coerce_video_url(serialized)
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

    def _get_reference_video_inputs(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"parameter_name": parameter_name, "value": params.get(parameter_name)}
            for parameter_name in ("reference_video_1", "reference_video_2", "reference_video_3")
            if params.get(parameter_name)
        ]

    def _get_reference_video_url(self, parameter_name: str, value: Any) -> str | None:
        direct_url = self._coerce_video_url(value)
        if direct_url:
            return direct_url

        helper_map = {
            "reference_video_1": self._public_reference_video_parameter_1,
            "reference_video_2": self._public_reference_video_parameter_2,
            "reference_video_3": self._public_reference_video_parameter_3,
        }
        helper = helper_map.get(parameter_name)
        if helper is None:
            return None

        try:
            public_url = helper.get_public_url_for_parameter()
        except Exception as e:
            self._log(f"{self.name} failed to prepare public URL for {parameter_name}: {e}")
            return None

        return self._coerce_video_url(public_url)

    @staticmethod
    def _supports_last_frame(model_id: str) -> bool:
        return model_id == SEEDANCE_2_0_MODEL_ID

    @staticmethod
    def _supports_1080p(model_id: str) -> bool:
        return model_id == SEEDANCE_2_0_MODEL_ID

    @staticmethod
    def _supports_private_assets(model_id: str) -> bool:
        """Private-asset references are supported by Seedance 2.0 only (not 2.0 Fast)."""
        return model_id == SEEDANCE_2_0_MODEL_ID

    def _is_byok_enabled(self) -> bool:
        """Whether the node is configured to use the customer's own key (BYOK) instead of Griptape auth."""
        return bool(self._api_key_provider and self._api_key_provider.is_user_auth_enabled())

    def _private_assets_active(self, model_id: str) -> bool:
        """Whether private-asset registration applies for this run.

        Requires the Seedance 2.0 model AND Griptape auth: provider assets are registered through
        the Griptape Cloud proxy on the org's behalf, which does not apply when the user brings
        their own provider key (BYOK).
        """
        return self._supports_private_assets(model_id) and not self._is_byok_enabled()

    @staticmethod
    def _summarize_image_input(val: Any) -> str:
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
