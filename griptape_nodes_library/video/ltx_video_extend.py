from __future__ import annotations

import json
import logging
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["LTXVideoExtend"]

MAX_PROMPT_LENGTH = 5000
MIN_EXTEND_DURATION = 2
MAX_EXTEND_DURATION = 20
MIN_CONTEXT_DURATION = 0
MAX_CONTEXT_DURATION = 20
# The API allows up to 505 frames (~21s at 24fps) across the combined
# context + duration window. Using seconds is a safe client-side proxy.
MAX_CONTEXT_PLUS_DURATION = 21

MODEL_MAPPING = {
    "LTX 2 Pro": "ltx-2-pro",
    "LTX 2.3 Pro": "ltx-2-3-pro",
}
DEFAULT_MODEL = "LTX 2.3 Pro"


class LTXVideoExtend(GriptapeProxyNode):
    """Extend an existing video by 2-20 seconds using LTX AI via Griptape Cloud model proxy.

    Inputs:
        - video (VideoUrlArtifact): Source video to extend (16:9 or 9:16, >=73 frames, <=4K)
        - prompt (str): Optional description of what should happen in the extended portion
        - mode (str): Whether to extend off the "start" or "end" of the source (default: "end")
        - duration (int): How many seconds of new footage to generate (2-20).
          Integer-only: the proxy bills via ceil(duration), so non-integer requests round up.
        - context (int): Optional seconds of source to use as context (0-20).
          context + duration must not exceed ~21s (505 frames at 24fps).
        - model (str): Model to use (LTX 2 Pro or LTX 2.3 Pro)

    Outputs:
        - generation_id (str): Griptape Cloud generation id
        - provider_response (dict): Response from API (latest polling response)
        - video_url (VideoUrlArtifact): Saved extended video
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    DEFAULT_MAX_ATTEMPTS = 240

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # INPUTS / PROPERTIES
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Model to use for video extension",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(MODEL_MAPPING.keys()))},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Optional description of what should happen in the extended portion (max 5000 characters)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "multiline": True,
                    "placeholder_text": "Describe what should happen in the extended portion...",
                },
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="Source video to extend (16:9 or 9:16, at least 73 frames, up to 4K)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "input video"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="mode",
                default_value="end",
                tooltip="Whether to extend off the start or end of the source video",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=["start", "end"])},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="duration",
                default_value=MIN_EXTEND_DURATION,
                tooltip=(
                    f"Seconds of new footage to generate ({MIN_EXTEND_DURATION}-{MAX_EXTEND_DURATION}). "
                    "Integer seconds only: the proxy bills via ceil(duration), so a fractional request "
                    "would round up (e.g. 2.5s would bill as 3s)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=MIN_EXTEND_DURATION, max_val=MAX_EXTEND_DURATION)},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="context",
                default_value=MIN_CONTEXT_DURATION,
                tooltip=(
                    f"Optional seconds of source video to use as context ({MIN_CONTEXT_DURATION}-"
                    f"{MAX_CONTEXT_DURATION}). context + duration must be at most "
                    f"{MAX_CONTEXT_PLUS_DURATION}s (505 frames at 24fps). Set to 0 to omit."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=MIN_CONTEXT_DURATION, max_val=MAX_CONTEXT_DURATION)},
            )
        )

        # OUTPUTS
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
                tooltip="Response from API (latest polling response)",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_url",
                tooltip="Saved extended video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="ltx_video_extend.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video extension result or any errors",
            result_details_placeholder="Extension status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _get_parameters(self) -> dict[str, Any]:
        return {
            "prompt": self.get_parameter_value("prompt") or "",
            "model": self.get_parameter_value("model") or DEFAULT_MODEL,
            "mode": self.get_parameter_value("mode") or "end",
            "duration": self.get_parameter_value("duration"),
            "context": self.get_parameter_value("context"),
        }

    def _get_api_model_id(self) -> str:
        model_name = self.get_parameter_value("model") or DEFAULT_MODEL
        model_id = MODEL_MAPPING.get(model_name, MODEL_MAPPING[DEFAULT_MODEL])
        return f"{model_id}:extend"

    async def _prepare_video_data_uri_async(self, video_input: Any) -> str | None:
        """Convert video input to a base64 data URI."""
        if not video_input:
            return None

        video_url = video_input.value if isinstance(video_input, VideoUrlArtifact) else str(video_input)
        if not video_url:
            return None

        if video_url.startswith("data:video/"):
            return video_url

        try:
            return await File(video_url).aread_data_uri(fallback_mime="video/mp4")
        except FileLoadError:
            logger.debug("%s failed to load video value: %s", self.name, video_url)
            return None

    def _validate_duration(self, duration: Any) -> str | None:
        if not isinstance(duration, int) or isinstance(duration, bool):
            return f"{self.name}: Duration must be an integer number of seconds (got {duration!r})."
        if duration < MIN_EXTEND_DURATION or duration > MAX_EXTEND_DURATION:
            return (
                f"{self.name}: Duration must be between {MIN_EXTEND_DURATION} and "
                f"{MAX_EXTEND_DURATION} seconds (got {duration})."
            )
        return None

    def _validate_context(self, context: Any) -> str | None:
        if context is None:
            return None
        if not isinstance(context, int) or isinstance(context, bool):
            return f"{self.name}: Context must be an integer number of seconds (got {context!r})."
        if context < MIN_CONTEXT_DURATION or context > MAX_CONTEXT_DURATION:
            return (
                f"{self.name}: Context must be between {MIN_CONTEXT_DURATION} and "
                f"{MAX_CONTEXT_DURATION} seconds (got {context})."
            )
        return None

    async def _build_payload(self) -> dict[str, Any]:
        params = self._get_parameters()

        video = self.get_parameter_value("video")
        if not video:
            msg = f"{self.name} requires an input video to extend."
            raise ValueError(msg)

        try:
            video_data_uri = await self._prepare_video_data_uri_async(video)
        except Exception as e:
            logger.error("%s failed to process video: %s", self.name, e)
            video_data_uri = None

        if not video_data_uri:
            msg = f"{self.name} failed to process input video."
            raise ValueError(msg)

        if len(params["prompt"]) > MAX_PROMPT_LENGTH:
            msg = (
                f"{self.name}: Prompt exceeds {MAX_PROMPT_LENGTH} characters limit "
                f"(current: {len(params['prompt'])} characters)"
            )
            raise ValueError(msg)

        if error := self._validate_duration(params["duration"]):
            raise ValueError(error)
        if error := self._validate_context(params["context"]):
            raise ValueError(error)

        duration = int(params["duration"])
        context = int(params["context"]) if params["context"] is not None else 0

        if context + duration > MAX_CONTEXT_PLUS_DURATION:
            msg = (
                f"{self.name}: context ({context}s) + duration ({duration}s) must be at most "
                f"{MAX_CONTEXT_PLUS_DURATION}s (API limit is 505 frames at 24fps)."
            )
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "video_uri": video_data_uri,
            "duration": duration,
            "mode": params["mode"],
            "model": MODEL_MAPPING.get(params["model"], MODEL_MAPPING[DEFAULT_MODEL]),
        }

        prompt = params["prompt"].strip()
        if prompt:
            payload["prompt"] = prompt

        if context > 0:
            payload["context"] = context

        return payload

    def _sanitize_video_uri_in_dict(self, data: dict[str, Any]) -> dict[str, Any]:
        """Redact base64 video data from dictionary for logging."""
        sanitized = {**data}
        if "video_uri" in sanitized and isinstance(sanitized["video_uri"], str):
            video_uri = sanitized["video_uri"]
            if video_uri.startswith("data:video/"):
                parts = video_uri.split(",", 1)
                header = parts[0] if parts else "data:video/"
                b64_len = len(parts[1]) if len(parts) > 1 else 0
                sanitized["video_uri"] = f"{header},<base64 data length={b64_len}>"
        return sanitized

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        video_bytes = result_json.get("raw_bytes")
        if not isinstance(video_bytes, (bytes, bytearray)):
            msg = f"{self.name} generation completed but no video data received."
            raise TypeError(msg)

        await self._handle_completion_async(bytes(video_bytes), generation_id)

    async def _handle_completion_async(self, video_bytes: bytes, generation_id: str) -> None:
        """Handle successful completion by saving the video to static storage."""
        if not video_bytes:
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no video data received.",
            )
            return

        try:
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(video_bytes)
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
            logger.info("%s saved video as %s", self.name, saved.name)
            self._set_status_results(
                was_successful=True,
                result_details=f"Video extension successful and saved as {saved.name}.",
            )
        except (OSError, PermissionError) as e:
            logger.error("%s failed to save to static storage: %s", self.name, e)
            self.parameter_output_values["video_url"] = None
            self._set_status_results(
                was_successful=False,
                result_details=f"Video generated but failed to save to storage: {e}",
            )

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:  # noqa: C901, PLR0912
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        status = str(response_json.get("status") or "").lower()
        status_detail = response_json.get("status_detail")
        if isinstance(status_detail, dict):
            error = status_detail.get("error", "")
            details = status_detail.get("details", "")

            if details and isinstance(details, str):
                try:
                    details_obj = json.loads(details)
                    if isinstance(details_obj, dict):
                        error_obj = details_obj.get("error")
                        if isinstance(error_obj, dict):
                            clean_message = error_obj.get("message")
                            if clean_message:
                                details = clean_message
                except (ValueError, json.JSONDecodeError):
                    pass

            if error and details:
                message = f"{error}: {details}"
            elif error:
                message = error
            elif details:
                message = details
            else:
                message = f"Generation {status or 'failed'} with no details provided"

            return f"{self.name} generation {status or 'failed'}: {message}"

        error = response_json.get("error")
        if error:
            if isinstance(error, dict):
                message = error.get("message") or error.get("type") or str(error)
                return f"{self.name} request failed: {message}"
            if isinstance(error, str):
                return f"{self.name} request failed: {error}"

        return f"{self.name} generation failed.\n\nFull API response:\n{response_json}"

    def _handle_payload_build_error(self, e: Exception) -> None:
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            return

        super()._handle_payload_build_error(e)

    def _handle_api_key_validation_error(self, e: ValueError) -> None:
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details=str(e))
        logger.error("%s API key validation failed: %s", self.name, e)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None
