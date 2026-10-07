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
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["WanImageToVideoGeneration"]

# Define constant for prompt truncation length
PROMPT_TRUNCATE_LENGTH = 100

# Model options with their constraints
MODEL_OPTIONS = [
    "wan2.6-i2v",
    "wan2.5-i2v-preview",
    "wan2.2-i2v-flash",
    "wan2.2-i2v-plus",
    "wanx2.1-i2v-plus",
    "wanx2.1-i2v-turbo",
]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES: dict[str, str] = {
    "Wan 2.1 I2V Plus": "wanx2.1-i2v-plus",
    "Wan 2.1 I2V Turbo": "wanx2.1-i2v-turbo",
    "Wan 2.2 I2V Flash": "wan2.2-i2v-flash",
    "Wan 2.2 I2V Plus": "wan2.2-i2v-plus",
    "Wan 2.5 I2V Preview": "wan2.5-i2v-preview",
    "Wan 2.6 I2V": "wan2.6-i2v",
    "gtc_wan_2_2_i2v_flash": "wan2.2-i2v-flash",
    "gtc_wan_2_2_i2v_plus": "wan2.2-i2v-plus",
    "gtc_wan_2_5_i2v_preview": "wan2.5-i2v-preview",
    "gtc_wan_2_6_i2v": "wan2.6-i2v",
    "gtc_wanx_2_1_i2v_plus": "wanx2.1-i2v-plus",
    "gtc_wanx_2_1_i2v_turbo": "wanx2.1-i2v-turbo",
}

# Model-specific configurations
MODEL_CONFIGS = {
    "wan2.6-i2v": {
        "resolutions": ["720P", "1080P"],
        "durations": [5, 10, 15],
        "supports_audio": True,
        "supports_shot_type": True,
    },
    "wan2.5-i2v-preview": {
        "resolutions": ["480P", "720P", "1080P"],
        "durations": [5, 10],
        "supports_audio": True,
        "supports_shot_type": False,
    },
    "wan2.2-i2v-flash": {
        "resolutions": ["480P", "720P"],
        "durations": [5],
        "supports_audio": False,
        "supports_shot_type": False,
    },
    "wan2.2-i2v-plus": {
        "resolutions": ["480P", "1080P"],
        "durations": [5],
        "supports_audio": False,
        "supports_shot_type": False,
    },
    "wanx2.1-i2v-plus": {
        "resolutions": ["720P"],
        "durations": [5],
        "supports_audio": False,
        "supports_shot_type": False,
    },
    "wanx2.1-i2v-turbo": {
        "resolutions": ["480P", "720P"],
        "durations": [3, 4, 5],
        "supports_audio": False,
        "supports_shot_type": False,
    },
}

# Response status constants
STATUS_FAILED = "Failed"
STATUS_ERROR = "Error"
STATUS_REQUEST_MODERATED = "Request Moderated"
STATUS_CONTENT_MODERATED = "Content Moderated"


class WanImageToVideoGeneration(GriptapeProxyNode):
    """Generate videos from images using WAN models via Griptape model proxy.

    Documentation: https://www.alibabacloud.com/help/en/model-studio/image-to-video-api-reference

    Inputs:
        - model (str): WAN model to use (default: "wan2.6-i2v")
            wan2.6-i2v: Supports 720P/1080P, 5-15s duration, audio, shot_type
            wan2.5-i2v-preview: Supports 480P/720P/1080P, 5-10s duration, audio
            wan2.2-i2v-flash: Supports 480P/720P, 5s duration, 50% faster than 2.1
            wan2.2-i2v-plus: Supports 480P/1080P, 5s duration, improved stability
            wanx2.1-i2v-plus: Supports 720P, 5s duration
            wanx2.1-i2v-turbo: Supports 480P/720P, 3-5s duration
        - input_image (ImageArtifact): Input image for video generation (required)
            Supports JPG, JPEG, PNG, BMP, WEBP
            Resolution: 360-2000 pixels (width and height)
            Size: Max 10 MB
        - prompt (str): Text description of desired video elements (optional)
            Max length varies by model (800-2000 characters)
        - negative_prompt (str): Description of content to avoid (max 500 characters)
        - resolution (str): Output video resolution (model-dependent)
        - duration (int): Video duration in seconds (model-dependent)
        - audio (bool): Auto-generate audio for video (wan2.6-i2v, wan2.5-i2v-preview)
        - input_audio (AudioUrlArtifact): Input audio file (optional, wan2.6/wan2.5 only)
            Audio requirements: WAV/MP3, 3-30s duration, max 15MB
            If audio exceeds video duration, it is truncated. If shorter, remaining video is silent.
        - shot_type (str): Shot type for video (single/multi, wan2.6-i2v)
        - prompt_extend (bool): Enable intelligent prompt rewriting (default: False)
        - watermark (bool): Add "AI-generated" watermark (default: False)
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
        self.description = "Generate videos from images using WAN models via Griptape model proxy"

        # Model selection
        model_param = ParameterString(
            name="model",
            default_value="wan2.6-i2v",
            tooltip="Select the WAN image-to-video model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_OPTIONS,
            default_model="wan2.6-i2v",
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        # Prompt parameter (optional)
        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Text description of desired video elements (max 800-2000 characters depending on model)",
                multiline=True,
                placeholder_text="Describe the video elements you want...",
                allow_output=False,
                ui_options={
                    "display_name": "Prompt",
                },
            )
        )

        # Negative prompt parameter
        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                default_value="",
                tooltip="Description of content to avoid (max 500 characters)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe what you don't want in the video...",
                ui_options={
                    "display_name": "Negative Prompt",
                },
            )
        )
        # Input image parameter (required)
        self.add_parameter(
            ParameterImage(
                name="input_image",
                tooltip="Input image for video generation (JPG, PNG, BMP, WEBP; 360-2000Px; max 10MB)",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "Input Image"},
            )
        )

        # Audio auto-generation parameter (for models that support it)
        self.add_parameter(
            ParameterBool(
                name="audio",
                default_value=True,
                tooltip="Auto-generate audio for video (wan2.6-i2v, wan2.5-i2v-preview)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide_property": self._should_hide_audio()},
            )
        )
        # Hidden by default since audio auto-generation is enabled by default
        self._public_audio_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterAudio(
                name="input_audio",
                default_value="",
                tooltip="Input audio file (optional). WAV/MP3, 3-30s, max 15MB. Audio is used to generate video with matching sound. Only supported by wan2.6 and wan2.5 models.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Input Audio"},
                hide=self._should_hide_input_audio(),
            ),
            disclaimer_message="The WAN Image-to-Video service utilizes this URL to access the audio file.",
        )
        self._public_audio_url_parameter.add_input_parameters()

        # Hide the upload message since input_audio is hidden by default
        self.hide_message_by_name("artifact_url_parameter_message_input_audio")

        with ParameterGroup(name="Generation Settings") as generation_settings_group:
            # Resolution parameter
            ParameterString(
                name="resolution",
                default_value="1080P",
                tooltip="Output video resolution (available options depend on selected model)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_CONFIGS["wan2.6-i2v"]["resolutions"])},
            )

            # Duration parameter
            ParameterInt(
                name="duration",
                default_value=5,
                tooltip="Video duration in seconds (model-dependent)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_CONFIGS["wan2.6-i2v"]["durations"])},
            )

            # Shot type parameter (for models that support it)
            ParameterString(
                name="shot_type",
                default_value="single",
                tooltip="Shot type for video: single (continuous shot) or multi (multiple switched shots). Only effective when prompt_extend is true.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["single", "multi"])},
                ui_options={"hide_property": self._should_hide_shot_type()},
            )

            # Prompt extend parameter
            ParameterBool(
                name="prompt_extend",
                default_value=False,
                tooltip="Enable intelligent prompt rewriting to improve generation quality",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            # Watermark parameter
            ParameterBool(
                name="watermark",
                default_value=False,
                tooltip="Add 'AI-generated' watermark in lower-right corner",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            # Initialize SeedParameter component (at the bottom of input parameters)
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

        # Create status parameters for success/failure tracking (at the end)
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _should_hide_audio(self) -> bool:
        """Determine if audio parameter should be hidden based on model selection."""
        model = self.get_parameter_value("model")
        if not model:
            return False
        model_config = MODEL_CONFIGS.get(model)
        if not model_config:
            return False
        return not model_config.get("supports_audio", False)

    def _should_hide_input_audio(self) -> bool:
        """Determine if input_audio should be hidden. Hidden when model doesn't support audio OR when audio auto-generation is enabled."""
        if self._should_hide_audio():
            return True
        audio_enabled = self.get_parameter_value("audio")
        return audio_enabled is True

    def _should_hide_shot_type(self) -> bool:
        """Determine if shot_type parameter should be hidden based on model selection."""
        model = self.get_parameter_value("model")
        if not model:
            return False
        model_config = MODEL_CONFIGS.get(model)
        if not model_config:
            return False
        return not model_config.get("supports_shot_type", False)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes."""
        super().after_value_set(parameter, value)

        # Update parameters when model changes
        if parameter.name == "model" and value in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[value]

            # Update audio parameter visibility based on model support
            if self._should_hide_audio():
                self.hide_parameter_by_name("audio")
            else:
                self.show_parameter_by_name("audio")

            # Update input_audio visibility based on model support and audio setting
            self._update_input_audio_visibility()

            # Update shot_type parameter visibility
            if self._should_hide_shot_type():
                self.hide_parameter_by_name("shot_type")
            else:
                self.show_parameter_by_name("shot_type")

            # Update resolution choices
            current_resolution = self.get_parameter_value("resolution")
            new_resolutions = model_config["resolutions"]
            if current_resolution in new_resolutions:
                self._update_option_choices("resolution", new_resolutions, current_resolution)
            else:
                # Set to first available resolution if current is not supported
                self._update_option_choices("resolution", new_resolutions, new_resolutions[0])

            # Update duration choices
            current_duration = self.get_parameter_value("duration")
            new_durations = model_config["durations"]
            if current_duration in new_durations:
                self._update_option_choices("duration", new_durations, current_duration)
            else:
                # Set to first available duration if current is not supported
                self._update_option_choices("duration", new_durations, new_durations[0])

        # Update input_audio visibility when audio parameter changes
        if parameter.name == "audio":
            self._update_input_audio_visibility()

    def _update_input_audio_visibility(self) -> None:
        """Update input_audio parameter visibility based on audio setting."""
        if self._should_hide_input_audio():
            self.hide_parameter_by_name("input_audio")
            self.hide_message_by_name("artifact_url_parameter_message_input_audio")
        else:
            self.show_parameter_by_name("input_audio")
            self.show_message_by_name("artifact_url_parameter_message_input_audio")

    async def aprocess(self) -> None:
        await self._process_generation()

    async def _process_generation(self) -> None:
        self._seed_parameter.preprocess()
        try:
            await super()._process_generation()
        finally:
            await self._public_audio_url_parameter.adelete_uploaded_artifact()

    def _get_parameters(self) -> dict[str, Any]:
        model = self.get_parameter_value("model")
        input_image = self.get_parameter_value("input_image")
        resolution = self.get_parameter_value("resolution")
        duration = self.get_parameter_value("duration")

        # Validate input image is provided
        if not input_image:
            msg = "Input image is required for image-to-video generation"
            raise ValueError(msg)

        # Validate model-specific constraints
        model_config = MODEL_CONFIGS.get(model, MODEL_CONFIGS["wan2.6-i2v"])

        # Validate resolution
        if resolution not in model_config["resolutions"]:
            msg = f"{model} does not support resolution {resolution}. Available resolutions: {', '.join(model_config['resolutions'])}"
            raise ValueError(msg)

        # Validate duration
        if duration not in model_config["durations"]:
            msg = f"{model} does not support duration {duration}s. Available durations: {', '.join(str(d) for d in model_config['durations'])}s"
            raise ValueError(msg)

        # Get audio parameter (only for models that support it)
        # For models that don't support audio, set to None so it won't be sent
        audio = None
        if model_config.get("supports_audio", False):
            audio = self.get_parameter_value("audio")

        # Capture audio input if provided and model supports it
        audio_input = None
        if model_config.get("supports_audio", False):
            input_audio_value = self.get_parameter_value("input_audio")
            if input_audio_value:
                audio_input = input_audio_value

        # Get shot_type parameter (only for models that support it)
        # For models that don't support shot_type, set to None so it won't be sent
        shot_type = None
        if model_config.get("supports_shot_type", False):
            shot_type = self.get_parameter_value("shot_type")

        return {
            "model": model,
            "input_image": input_image,
            "prompt": self.get_parameter_value("prompt") or "",
            "negative_prompt": self.get_parameter_value("negative_prompt") or "",
            "resolution": resolution,
            "duration": duration,
            "audio": audio,
            "audio_input": audio_input,
            "shot_type": shot_type,
            "seed": self._seed_parameter.get_seed(),
            "prompt_extend": self.get_parameter_value("prompt_extend"),
            "watermark": self.get_parameter_value("watermark"),
        }

    async def _build_payload(self) -> dict[str, Any]:
        params = self._get_parameters()

        # Process input image to base64 data URI
        img_url = await self._process_input_image(params["input_image"])
        if not img_url:
            msg = "Failed to process input image"
            raise ValueError(msg)

        audio_url = None
        if params.get("audio_input"):
            audio_url = await self._prepare_audio_data_url_async(params["audio_input"])

        # Build flattened payload (all params at top level)
        payload = {
            "model": self._get_selected_model_id(),
            "img_url": img_url,
            "resolution": params["resolution"],
            "duration": params["duration"],
            "prompt_extend": params["prompt_extend"],
            "watermark": params["watermark"],
            "seed": params["seed"],
        }

        # Add optional parameters if provided
        if params["prompt"]:
            payload["prompt"] = params["prompt"]

        if params["negative_prompt"]:
            payload["negative_prompt"] = params["negative_prompt"]

        # Add model-specific parameters
        model_config = MODEL_CONFIGS.get(params["model"], {})

        # Add audio parameter (for models that support it)
        if model_config.get("supports_audio", False) and params.get("audio") is not None:
            payload["audio"] = params["audio"]

        # Add audio_url if provided (for models that support it)
        if model_config.get("supports_audio", False) and audio_url:
            payload["audio_url"] = audio_url

        # Add shot_type parameter (for models that support it, only effective when prompt_extend=true)
        if model_config.get("supports_shot_type", False) and params.get("shot_type") is not None:
            payload["shot_type"] = params["shot_type"]

        return payload

    async def _process_input_image(self, image_input: Any) -> str | None:
        """Process input image and convert to base64 data URI."""
        if not image_input:
            return None

        # Extract string value from input
        image_value = self._extract_image_value(image_input)
        if not image_value:
            return None

        try:
            return await File(image_value).aread_data_uri(fallback_mime="image/png")
        except FileLoadError:
            logger.debug("%s failed to load image value: %s", self.name, image_value)
            return None

    def _extract_image_value(self, image_input: Any) -> str | None:
        """Extract string value from various image input types."""
        if isinstance(image_input, str):
            return image_input

        try:
            # ImageUrlArtifact: .value holds URL string
            if hasattr(image_input, "value"):
                value = getattr(image_input, "value", None)
                if isinstance(value, str):
                    return value

            # ImageArtifact: .base64 holds raw or data-URI
            if hasattr(image_input, "base64"):
                b64 = getattr(image_input, "base64", None)
                if isinstance(b64, str) and b64:
                    return b64
        except Exception as e:
            logger.error("Failed to extract image value: %s", e)

        return None

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Handle WAN response and save the hosted video.

        Response shape:
        {
            "task_id": "...",
            "task_status": "SUCCEEDED",
            "submit_time": "...",
            "scheduled_time": "...",
            "end_time": "...",
            "orig_prompt": "..."
        }
        """
        # Extract task_id for generation_id
        task_id = result_json.get("task_id", "")
        self.parameter_output_values["generation_id"] = str(task_id)

        # Check task status
        task_status = result_json.get("task_status")
        if task_status != "SUCCEEDED":
            logger.error("Generation failed with task_status: %s", task_status)
            self._set_safe_defaults()
            error_details = self._provider_failure_message(
                self._extract_error_message(result_json),
                f"The generation ended as {task_status} and the provider gave no reason.",
            )
            self._set_status_results(was_successful=False, result_details=error_details)
            return

        await self._save_generated_media(
            generation_id,
            "video",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
        )

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
        if provider_reason or top_level_reason:
            return provider_reason or top_level_reason

        status = response_json.get("status")
        if status in [STATUS_REQUEST_MODERATED, STATUS_CONTENT_MODERATED]:
            return self._moderation_reason(response_json)
        if status in [STATUS_FAILED, STATUS_ERROR]:
            result = response_json.get("result")
            if isinstance(result, dict) and result.get("error"):
                return str(result["error"])
        return ""

    @staticmethod
    def _moderation_reason(response_json: dict[str, Any]) -> str:
        """Why moderation blocked the content."""
        details = response_json.get("details")
        moderation_reasons = details.get("Moderation Reasons", []) if isinstance(details, dict) else []
        if moderation_reasons:
            return f"the content was blocked by moderation ({', '.join(moderation_reasons)})"
        return "the content was blocked by safety filters"

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
