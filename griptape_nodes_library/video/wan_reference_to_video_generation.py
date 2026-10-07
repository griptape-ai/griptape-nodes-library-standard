from __future__ import annotations

import json
import logging
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.media.public_urls import adelete_uploaded_artifacts
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["WanReferenceToVideoGeneration"]

# Define constant for prompt truncation length
PROMPT_TRUNCATE_LENGTH = 100

# HTTP status code threshold for error responses
HTTP_ERROR_STATUS = 400

# Model options
MODEL_OPTIONS = [
    "wan2.6-r2v",
]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES: dict[str, str] = {
    "Wan 2.6 R2V": "wan2.6-r2v",
    "gtc_wan_2_6_r2v": "wan2.6-r2v",
}

# Size options organized by resolution tier
SIZE_OPTIONS_720P = [
    "1280*720",  # 16:9
    "720*1280",  # 9:16
    "960*960",  # 1:1
    "1088*832",  # 4:3
    "832*1088",  # 3:4
]

SIZE_OPTIONS_1080P = [
    "1920*1080",  # 16:9
    "1080*1920",  # 9:16
    "1440*1440",  # 1:1
    "1632*1248",  # 4:3
    "1248*1632",  # 3:4
]

ALL_SIZE_OPTIONS = SIZE_OPTIONS_1080P + SIZE_OPTIONS_720P

# Shot type options
SHOT_TYPE_OPTIONS = [
    "single",
    "multi",
]

# Generation status constants
STATUS_PENDING = "PENDING"
STATUS_RUNNING = "RUNNING"
STATUS_SUCCEEDED = "SUCCEEDED"
STATUS_FAILED = "FAILED"
STATUS_CANCELED = "CANCELED"
STATUS_UNKNOWN = "UNKNOWN"


class WanReferenceToVideoGeneration(GriptapeProxyNode):
    """Generate videos from reference videos using WAN models via Griptape model proxy.

    Creates a new video based on the subject and timbre of reference videos and a prompt.
    Use character1, character2, character3 in the prompt to refer to subjects in
    reference videos 1, 2, 3 respectively.

    Documentation: https://www.alibabacloud.com/help/en/model-studio/reference-to-video-api-reference

    Inputs:
        - model (str): WAN model to use (default: "wan2.6-r2v")
        - prompt (str): Text description using character1/character2/character3 to reference
            subjects in the reference videos (max 1500 characters)
        - negative_prompt (str): Description of content to avoid (max 500 characters)
        - reference_video_1 (VideoUrlArtifact): First reference video (required)
        - reference_video_2 (VideoUrlArtifact): Second reference video (optional)
        - reference_video_3 (VideoUrlArtifact): Third reference video (optional)
            Video requirements: MP4/MOV, 2-30s duration, max 100MB
        - input_audio (AudioUrlArtifact): Input audio file (optional)
            Audio requirements: WAV/MP3, 3-30s duration, max 15MB
            If audio exceeds video duration, it is truncated. If shorter, remaining video is silent.
        - size (str): Output video resolution (default: "1920*1080")
            720p tier: 1280*720, 720*1280, 960*960, 1088*832, 832*1088
            1080p tier: 1920*1080, 1080*1920, 1440*1440, 1632*1248, 1248*1632
        - duration (int): Video duration in seconds (5 or 10, default: 5)
        - shot_type (str): Shot type - "single" or "multi" (default: "single")
        - audio (bool): Auto-generate audio for video (default: True)
        - watermark (bool): Add "AI Generated" watermark (default: False)
        - randomize_seed (bool): If true, randomize the seed on each run
        - seed (int): Random seed for reproducible results (default: 42)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response from the model proxy
        - video (VideoUrlArtifact): Generated video as URL artifact
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate videos from reference videos using WAN models via Griptape model proxy"

        # Model selection
        model_param = ParameterString(
            name="model",
            default_value="wan2.6-r2v",
            tooltip="Select the WAN reference-to-video model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_OPTIONS,
            default_model="wan2.6-r2v",
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        # Prompt parameter
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description using character1/character2/character3 to reference subjects (max 1500 characters)",
                multiline=True,
                placeholder_text="character1 is happily watching a movie on the sofa...",
                allow_output=False,
                ui_options={
                    "display_name": "Prompt",
                },
            )
        )

        # Negative prompt parameter
        self.add_parameter(
            Parameter(
                name="negative_prompt",
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Description of content to avoid (max 500 characters)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "low resolution, error, worst quality...",
                    "display_name": "Negative Prompt",
                },
            )
        )

        # Reference video 1 (required) using PublicArtifactUrlParameter
        self._public_video_url_parameter_1 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_1",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip="First reference video (required). MP4/MOV, 2-30s, max 100MB. Use 'character1' in prompt to reference.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 1"},
                hide_property=True,
            ),
            disclaimer_message="The WAN Reference-to-Video service utilizes this URL to access the reference video.",
        )
        self._public_video_url_parameter_1.add_input_parameters()

        # Reference video 2 (optional) using PublicArtifactUrlParameter - hidden by default
        self._public_video_url_parameter_2 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_2",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip="Second reference video (optional). MP4/MOV, 2-30s, max 100MB. Use 'character2' in prompt to reference.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 2"},
                hide_property=True,
                hide=True,
            ),
            disclaimer_message="The WAN Reference-to-Video service utilizes this URL to access the reference video.",
        )
        self._public_video_url_parameter_2.add_input_parameters()
        # Hide the upload message for video 2 since the parameter is hidden
        self.hide_message_by_name("artifact_url_parameter_message_reference_video_2")

        # Reference video 3 (optional) using PublicArtifactUrlParameter - hidden by default
        self._public_video_url_parameter_3 = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=Parameter(
                name="reference_video_3",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                default_value="",
                tooltip="Third reference video (optional). MP4/MOV, 2-30s, max 100MB. Use 'character3' in prompt to reference.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Reference Video 3"},
                hide_property=True,
                hide=True,
            ),
            disclaimer_message="The WAN Reference-to-Video service utilizes this URL to access the reference video.",
        )
        self._public_video_url_parameter_3.add_input_parameters()
        # Hide the upload message for video 3 since the parameter is hidden
        self.hide_message_by_name("artifact_url_parameter_message_reference_video_3")

        # Audio parameter
        self.add_parameter(
            ParameterBool(
                name="audio",
                default_value=True,
                tooltip="Auto-generate audio for video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        # Input Audio (optional) using PublicArtifactUrlParameter
        # Hidden by default since audio auto-generation is enabled by default
        self._public_audio_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterAudio(
                name="input_audio",
                default_value="",
                tooltip="Input audio file (optional). WAV/MP3, 3-30s, max 15MB. Audio is used to generate video with matching sound.",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Input Audio"},
                hide_property=True,
                hide=self._should_hide_input_audio(),
            ),
            disclaimer_message="The WAN Reference-to-Video service utilizes this URL to access the audio file.",
        )
        self._public_audio_url_parameter.add_input_parameters()
        # Hide the upload message since audio auto-generation is enabled by default
        self.hide_message_by_name("artifact_url_parameter_message_input_audio")
        with ParameterGroup(name="Generation Settings") as generation_settings_group:
            # Size parameter
            ParameterString(
                name="size",
                default_value="1920*1080",
                tooltip="Output video resolution (width*height)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ALL_SIZE_OPTIONS)},
            )
            # Duration parameter
            ParameterInt(
                name="duration",
                default_value=5,
                tooltip="Video duration in seconds (5 or 10)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[5, 10])},
            )

            # Shot type parameter
            ParameterString(
                name="shot_type",
                default_value="single",
                tooltip="Shot type: 'single' for continuous shot, 'multi' for multiple changing shots",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["single", "multi"])},
            )

            # Watermark parameter
            ParameterBool(
                name="watermark",
                default_value=False,
                tooltip="Add 'AI Generated' watermark in lower-right corner",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            # Initialize SeedParameter component
            self._seed_parameter = SeedParameter(self)
            self._seed_parameter.add_input_parameters(inside_param_group=True)

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="Generated video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="wan_video.mp4",
        )
        self._output_file.add_parameter()

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []
        reference_video_1 = self.get_parameter_value("reference_video_1")
        if not reference_video_1:
            exceptions.append(ValueError("Reference Video 1 must be provided"))
        prompt = self.get_parameter_value("prompt")
        if not prompt:
            exceptions.append(ValueError("Prompt must be provided"))
        return exceptions if exceptions else None

    def _should_hide_input_audio(self) -> bool:
        """Determine if input_audio should be hidden. Hidden when audio auto-generation is enabled."""
        audio_enabled = self.get_parameter_value("audio")
        return audio_enabled is True

    def _update_input_audio_visibility(self) -> None:
        """Update input_audio parameter visibility based on audio setting."""
        if self._should_hide_input_audio():
            self.hide_parameter_by_name("input_audio")
            self.hide_message_by_name("artifact_url_parameter_message_input_audio")
        else:
            self.show_parameter_by_name("input_audio")
            self.show_message_by_name("artifact_url_parameter_message_input_audio")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        super().after_value_set(parameter, value)

        # Update input_audio visibility when audio parameter changes
        if parameter.name == "audio":
            self._update_input_audio_visibility()

    async def aprocess(self) -> None:
        await self._process_generation()

    async def _process_generation(self) -> None:
        self._seed_parameter.preprocess()
        try:
            await super()._process_generation()
        finally:
            await adelete_uploaded_artifacts(
                (
                    self._public_video_url_parameter_1,
                    self._public_video_url_parameter_2,
                    self._public_video_url_parameter_3,
                    self._public_audio_url_parameter,
                ),
                node_name=self.name,
            )

    def _get_parameters(self) -> dict[str, Any]:
        model = self.get_parameter_value("model")
        prompt = self.get_parameter_value("prompt")
        negative_prompt = self.get_parameter_value("negative_prompt") or ""
        size = self.get_parameter_value("size")
        duration = self.get_parameter_value("duration")
        shot_type = self.get_parameter_value("shot_type")
        audio = self.get_parameter_value("audio")
        watermark = self.get_parameter_value("watermark")

        # Collect reference video inputs
        reference_video_inputs: list[Any] = []

        video_1_value = self.get_parameter_value("reference_video_1")
        if video_1_value:
            reference_video_inputs.append(video_1_value)

        video_2_value = self.get_parameter_value("reference_video_2")
        if video_2_value:
            reference_video_inputs.append(video_2_value)

        video_3_value = self.get_parameter_value("reference_video_3")
        if video_3_value:
            reference_video_inputs.append(video_3_value)

        if not reference_video_inputs:
            msg = "At least one reference video URL is required"
            raise ValueError(msg)

        # Validate size
        if size not in ALL_SIZE_OPTIONS:
            msg = f"Invalid size {size}. Available sizes: {', '.join(ALL_SIZE_OPTIONS)}"
            raise ValueError(msg)

        # Validate duration
        if duration not in [5, 10]:
            msg = f"Invalid duration {duration}s. Available durations: 5, 10"
            raise ValueError(msg)

        # Validate shot_type
        if shot_type not in SHOT_TYPE_OPTIONS:
            msg = f"Invalid shot_type {shot_type}. Available options: {', '.join(SHOT_TYPE_OPTIONS)}"
            raise ValueError(msg)

        # Capture audio input if provided
        audio_input = None
        input_audio_value = self.get_parameter_value("input_audio")
        if input_audio_value:
            audio_input = input_audio_value

        return {
            "model": model,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "reference_video_inputs": reference_video_inputs,
            "audio_input": audio_input,
            "size": size,
            "duration": duration,
            "shot_type": shot_type,
            "audio": audio,
            "watermark": watermark,
            "seed": self._seed_parameter.get_seed(),
        }

    async def _build_payload(self) -> dict[str, Any]:
        params = self._get_parameters()

        reference_video_urls: list[str] = []
        for video_input in params["reference_video_inputs"]:
            video_url = await self._prepare_video_data_url_async(video_input)
            if video_url:
                reference_video_urls.append(video_url)

        if not reference_video_urls:
            msg = "At least one reference video URL is required"
            raise ValueError(msg)

        audio_url = None
        if params.get("audio_input"):
            audio_url = await self._prepare_audio_data_url_async(params["audio_input"])

        # Build payload matching proxy expected format (nested input/parameters structure)
        payload = {
            "input": {
                "prompt": params["prompt"],
                "reference_video_urls": reference_video_urls,
            },
            "parameters": {
                "size": params["size"],
                "duration": params["duration"],
                "shot_type": params["shot_type"],
                "audio": params["audio"],
                "watermark": params["watermark"],
                "seed": params["seed"],
            },
        }

        # Add negative prompt if provided
        if params["negative_prompt"]:
            payload["input"]["negative_prompt"] = params["negative_prompt"]

        # Add audio_url if provided
        if audio_url:
            payload["input"]["audio_url"] = audio_url

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        status = self._extract_status(result_json) or STATUS_UNKNOWN
        if status in {STATUS_FAILED, STATUS_CANCELED}:
            self.parameter_output_values["video"] = None
            error_details = self._provider_failure_message(
                self._extract_error_message(result_json),
                f"The generation ended as {status} and the provider gave no reason.",
            )
            self._set_status_results(was_successful=False, result_details=error_details)
            return

        await self._save_generated_media(
            generation_id,
            "video",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
        )

    async def _prepare_video_data_url_async(self, video_input: Any) -> str | None:
        return await prepare_media_data_uri(video_input, kind="video", node_name=self.name)

    async def _prepare_audio_data_url_async(self, audio_input: Any) -> str | None:
        return await prepare_media_data_uri(audio_input, kind="audio", node_name=self.name)

    def _extract_error_message(self, response_json: dict[str, Any] | None) -> str:
        """Extract the provider's reason from a failed generation response.

        Args:
            response_json: The JSON response from the API that may contain error information

        Returns:
            The provider's reason, or "" if there is none
        """
        if not response_json:
            return ""

        top_level_reason = self._error_reason(response_json.get("error"))
        parsed_provider_response = self._parse_provider_response(response_json.get("provider_response"))
        provider_reason = self._error_reason((parsed_provider_response or {}).get("error"))

        if top_level_reason and provider_reason and top_level_reason != provider_reason:
            return f"{top_level_reason}: {provider_reason}"
        return provider_reason or top_level_reason or super()._extract_error_message(response_json)

    def _parse_provider_response(self, provider_response: Any) -> dict[str, Any] | None:
        """Parse provider_response if it's a JSON string."""
        if isinstance(provider_response, str):
            try:
                return json.loads(provider_response)
            except Exception:
                return None
        if isinstance(provider_response, dict):
            return provider_response
        return None

    @staticmethod
    def _error_reason(error: Any) -> str:
        """The readable message in an error field, or "" if it has none."""
        if not error:
            return ""
        if isinstance(error, dict):
            return str(error.get("message") or error.get("error") or "")
        return str(error)

    def _set_safe_defaults(self) -> None:
        """Set safe default values for outputs."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video"] = None

    @staticmethod
    def _extract_status(response_json: dict[str, Any] | None) -> str | None:
        """Extract task status from response."""
        if not response_json:
            return None
        task_status = response_json.get("task_status")
        if isinstance(task_status, str):
            return task_status
        return None
