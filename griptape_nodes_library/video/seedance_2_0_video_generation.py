from __future__ import annotations

import json as _json
import logging
from typing import Any, ClassVar

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
        - resolution (str): Output resolution (default: 720p, options: 480p, 720p)
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
                ui_options={"display_name": "First Frame"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="last_frame",
                default_value=None,
                tooltip="Optional last frame image (Seedance 2.0 only)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Last Frame"},
            )
        )

        # Multimodal reference inputs (for Multimodal mode)
        self.add_parameter(
            ParameterList(
                name="reference_images",
                input_types=["ImageUrlArtifact", "ImageArtifact", "str"],
                default_value=[],
                tooltip="Optional reference images (0-9 images)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Images", "expander": True},
                max_items=9,
            )
        )

        self._public_reference_video_parameter_1 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_1",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional first reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Reference Video 1"},
            ),
            disclaimer_message="The Seedance 2.0 service utilizes this URL to access the reference video.",
        )
        self._public_reference_video_parameter_1.add_input_parameters()

        self._public_reference_video_parameter_2 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_2",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional second reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Reference Video 2"},
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
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip=(
                    "Optional third reference video. Seedance only accepts public URLs or uploaded asset URLs "
                    "for videos."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Reference Video 3"},
                hide=True,
            ),
            disclaimer_message="The Seedance 2.0 service utilizes this URL to access the reference video.",
        )
        self._public_reference_video_parameter_3.add_input_parameters()
        self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")

        self.add_parameter(
            ParameterList(
                name="reference_audio",
                input_types=["AudioArtifact", "AudioUrlArtifact", "str"],
                default_value=[],
                tooltip="Optional reference audio (0-3 audio files, 2-15s each, max 15s total). URLs, asset:// IDs, or base64/data URIs are supported.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Audio", "expander": True},
                max_items=3,
            )
        )

        # Generation settings
        with ParameterGroup(name="Generation Settings") as settings_group:
            ParameterString(
                name="resolution",
                default_value="720p",
                tooltip="Output resolution (480p or 720p)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["480p", "720p"])},
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
            ParameterString(
                name="generation_id",
                tooltip="Griptape Cloud generation id",
                allowed_modes={ParameterMode.OUTPUT},
                hide=True,
            )
        )

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

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation."""
        raw_model_id = self.get_parameter_value("model_id") or MODEL_NAME_SEEDANCE_2_0
        return self.MODEL_NAME_MAP.get(raw_model_id, raw_model_id)

    async def _process_generation(self) -> None:
        try:
            await super()._process_generation()
        finally:
            self._public_reference_video_parameter_1.delete_uploaded_artifact()
            self._public_reference_video_parameter_2.delete_uploaded_artifact()
            self._public_reference_video_parameter_3.delete_uploaded_artifact()

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

        # 2.0 doesn't support 1080p
        if params.get("resolution") == "1080p":
            msg = f"{self.name}: Seedance 2.0 models do not support 1080p resolution. Use 480p or 720p."
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
            # Multimodal mode: reference images/videos/audio
            for ref_image in params.get("reference_images", [])[:9]:
                ref_url = await self._prepare_frame_url_async(ref_image, frame_label="reference_image")
                if ref_url:
                    content_list.append({"type": "image_url", "image_url": {"url": ref_url}, "role": "reference_image"})

            for ref_video in self._get_reference_video_inputs(params):
                video_url = self._get_reference_video_url(ref_video["parameter_name"], ref_video["value"])
                if not video_url:
                    msg = (
                        f"{self.name}: {ref_video['parameter_name']} only supports public URLs, uploaded asset URLs, "
                        "or asset:// IDs. Seedance 2.0 does not accept video base64."
                    )
                    raise ValueError(msg)
                content_list.append({"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"})

            for ref_audio in params.get("reference_audio", [])[:3]:
                audio_url = await self._prepare_audio_url_async(ref_audio, audio_label="reference_audio")
                if audio_url:
                    content_list.append(
                        {"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"}
                    )
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

    async def _prepare_frame_url_async(self, frame_input: Any, *, frame_label: str) -> str | None:
        """Convert frame input to a usable URL."""
        if not frame_input:
            self._log(f"{self.name} {frame_label} not provided")
            return None

        frame_url = self._coerce_image_url_or_data_uri(frame_input)
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

        audio_url = self._coerce_audio_url_or_data_uri(audio_input)
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
    def _coerce_image_url_or_data_uri(val: Any) -> str | None:
        if val is None:
            return None

        if isinstance(val, dict):
            value = val.get("value")
            artifact_type = val.get("type")
            if isinstance(value, str):
                if artifact_type == "ImageUrlArtifact":
                    return value
                if value.startswith(("http://", "https://", "data:image/")):
                    return value
                if artifact_type == "ImageArtifact":
                    image_format = str(val.get("format") or "png").lower()
                    return f"data:image/{image_format};base64,{value}"
            return None

        if isinstance(val, str):
            v = val.strip()
            if not v:
                return None
            return v if v.startswith(("http://", "https://", "data:image/")) else f"data:image/png;base64,{v}"

        try:
            to_dict = getattr(val, "to_dict", None)
            if callable(to_dict):
                serialized = to_dict()
                if isinstance(serialized, dict):
                    coerced = Seedance20VideoGeneration._coerce_image_url_or_data_uri(serialized)
                    if coerced:
                        return coerced

            v = getattr(val, "value", None)
            if isinstance(v, str):
                stripped = v.strip()
                if stripped:
                    if stripped.startswith(("http://", "https://", "data:image/")):
                        return stripped
                    return stripped
            b64 = getattr(val, "base64", None)
            if isinstance(b64, str) and b64:
                return b64 if b64.startswith("data:image/") else f"data:image/png;base64,{b64}"
        except Exception:  # noqa: S110
            pass

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
    def _coerce_audio_url_or_data_uri(val: Any) -> str | None:
        """Convert audio input to a Seedance-supported audio URL, asset ID, or data URI."""
        if val is None:
            return None

        if isinstance(val, dict):
            value = val.get("value")
            artifact_type = val.get("type")
            if isinstance(value, str):
                stripped = value.strip()
                if artifact_type == "AudioArtifact":
                    audio_format = str(val.get("format") or "wav").lower()
                    return f"data:audio/{audio_format};base64,{value}"
                if stripped.startswith(("http://", "https://", "data:audio/", "asset://")):
                    return stripped
                return stripped
            return None

        if isinstance(val, str):
            v = val.strip()
            if not v:
                return None
            return v

        try:
            to_dict = getattr(val, "to_dict", None)
            if callable(to_dict):
                serialized = to_dict()
                if isinstance(serialized, dict):
                    coerced = Seedance20VideoGeneration._coerce_audio_url_or_data_uri(serialized)
                    if coerced:
                        return coerced

            v = getattr(val, "value", None)
            if isinstance(v, str):
                stripped = v.strip()
                if stripped:
                    return stripped
            b64 = getattr(val, "base64", None)
            if isinstance(b64, str) and b64:
                return b64 if b64.startswith("data:audio/") else f"data:audio/wav;base64,{b64}"
        except Exception:  # noqa: S110
            pass

        return None

    @staticmethod
    def _supports_last_frame(model_id: str) -> bool:
        return model_id == SEEDANCE_2_0_MODEL_ID

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
