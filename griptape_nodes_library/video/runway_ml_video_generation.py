from __future__ import annotations

import base64
import logging
from contextlib import suppress
from typing import Any, ClassVar

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.image_utils import shrink_image_to_size

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["RunwayMLVideoGeneration"]

# Model mapping from user-friendly names to API model IDs
MODEL_MAPPING: dict[str, str] = {
    "Gen-4.5": "gen4.5",
    "Gen-4 Turbo": "gen4_turbo",
    "Gen-3 Alpha Turbo": "gen3a_turbo",
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())
DEFAULT_MODEL = MODEL_OPTIONS[0]

# Ratio options per model
RATIO_OPTIONS: dict[str, list[str]] = {
    "gen4.5": ["1280:720", "720:1280", "1104:832", "960:960", "832:1104", "1584:672"],
    "gen4_turbo": ["1280:720", "720:1280", "1104:832", "960:960", "832:1104", "1584:672"],
    "gen3a_turbo": [
        "1280:720",
        "720:1280",
        "1104:832",
        "960:960",
        "832:1104",
        "1584:672",
        "848:480",
        "640:480",
    ],
}

# Text-only ratio options for gen4.5 (text_to_video endpoint)
TEXT_ONLY_RATIO_OPTIONS = ["1280:720", "720:1280"]

# Models that require an input image
MODELS_REQUIRING_IMAGE = {"gen4_turbo", "gen3a_turbo"}

# Duration constraints
MIN_DURATION = 2
MAX_DURATION = 10
DEFAULT_DURATION = 5

# Seed constraints
MAX_SEED = 4294967295


class RunwayMLVideoGeneration(GriptapeProxyNode):
    """Generate video using RunwayML models via Griptape Cloud model proxy.

    Supports text-to-video (Gen-4.5 only) and image-to-video generation.

    Inputs:
        - model (str): RunwayML model to use
        - prompt_text (str): Text description of the desired video (1-1000 chars)
        - prompt_image (ImageUrlArtifact): Input image for image-to-video generation
        - ratio (str): Output video resolution
        - duration (int): Video duration in seconds (2-10)
        - seed (int): Random seed for reproducibility

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from Griptape model proxy
        - video_url (VideoUrlArtifact): Generated video as URL artifact
    """

    # Ratio options vary by model
    RATIO_OPTIONS: ClassVar[dict[str, list[str]]] = RATIO_OPTIONS

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate video using RunwayML models via Griptape Cloud model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML video generation model",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt_text",
                tooltip="Text description of the desired video (1-1000 characters)",
                multiline=True,
                placeholder_text="Describe the video you want to generate...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="prompt_image",
                tooltip="Input image for image-to-video generation (required for Gen-4 Turbo and Gen-3 Alpha Turbo)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Input Image",
                    "expander": True,
                },
            )
        )

        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            ParameterString(
                name="ratio",
                default_value="1280:720",
                tooltip="Output video resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RATIO_OPTIONS["gen4.5"])},
                ui_options={"display_name": "Ratio"},
            )

            ParameterInt(
                name="duration",
                default_value=DEFAULT_DURATION,
                tooltip="Video duration in seconds",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=MIN_DURATION,
                max_val=MAX_DURATION,
                ui_options={"display_name": "Duration (seconds)"},
            )

            ParameterInt(
                name="seed",
                default_value=0,
                tooltip="Random seed for reproducibility (0 = random). Range: 0-4294967295",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=0,
                max_val=MAX_SEED,
                ui_options={"display_name": "Seed"},
            )

        self.add_node_element(gen_settings_group)

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

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
                name="video_url",
                tooltip="Generated video as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="runway_video.mp4",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the video generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Update ratio choices when the model changes."""
        super().after_value_set(parameter, value)

        if parameter.name == "model":
            api_model_id = MODEL_MAPPING.get(str(value), str(value))
            new_choices = self.RATIO_OPTIONS.get(api_model_id, RATIO_OPTIONS["gen4.5"])
            current_ratio = self.get_parameter_value("ratio")
            if current_ratio not in new_choices:
                default_ratio = "1280:720" if "1280:720" in new_choices else new_choices[0]
                self._update_option_choices("ratio", new_choices, default_ratio)
            else:
                self._update_option_choices("ratio", new_choices, current_ratio)

    def _get_api_model_id(self) -> str:
        """Map user-friendly model name to API model ID."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for RunwayML video generation.

        Determines whether to use image_to_video or text_to_video endpoint
        based on whether an input image is provided.
        """
        prompt_text = self.get_parameter_value("prompt_text") or ""
        ratio = self.get_parameter_value("ratio") or "1280:720"
        duration = self.get_parameter_value("duration") or DEFAULT_DURATION
        seed = self.get_parameter_value("seed") or 0
        prompt_image = self.get_parameter_value("prompt_image")

        api_model_id = self._get_api_model_id()

        payload: dict[str, Any] = {
            "promptText": prompt_text.strip(),
            "ratio": ratio,
            "duration": int(duration),
        }

        # Include seed only if non-zero
        if seed and int(seed) > 0:
            payload["seed"] = int(seed)

        # Determine endpoint based on image presence
        if prompt_image:
            data_uri = await self._prepare_image_data_uri(prompt_image)
            if data_uri:
                payload["promptImage"] = data_uri
                payload["_endpoint"] = "image_to_video"
            else:
                payload["_endpoint"] = "text_to_video"
        else:
            # text_to_video endpoint (gen4.5 only)
            payload["_endpoint"] = "text_to_video"

        # Validate ratio for text_to_video mode
        if payload.get("_endpoint") == "text_to_video" and api_model_id == "gen4.5":
            if ratio not in TEXT_ONLY_RATIO_OPTIONS:
                payload["ratio"] = "1280:720"

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the RunwayML result and download the video.

        The proxy client returns the output array from the RunwayML task response.
        The output is an array of signed CloudFront URLs.
        """
        # Handle binary response
        if "raw_bytes" in result_json:
            await self._handle_video_bytes(result_json["raw_bytes"])
            return

        # Look for output URLs in various possible response shapes
        output = result_json.get("output")
        if isinstance(output, list) and output:
            video_url = output[0]
        elif isinstance(output, str):
            video_url = output
        else:
            # Try direct URL fields
            video_url = result_json.get("video_url") or result_json.get("url")

        if not video_url or not isinstance(video_url, str):
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no video URL was found in the response.",
            )
            return

        await self._handle_video_url(video_url)

    async def _handle_video_url(self, video_url: str) -> None:
        """Download video from URL and save it."""
        try:
            video_bytes = await self._download_bytes_from_url(video_url)
        except Exception as e:
            self._log(f"Failed to download video: {e}")
            video_bytes = None

        if video_bytes:
            await self._handle_video_bytes(video_bytes)
        else:
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=video_url)
            self._set_status_results(
                was_successful=True,
                result_details="Video generated successfully. Using provider URL (could not download video bytes).",
            )

    async def _handle_video_bytes(self, video_bytes: bytes) -> None:
        """Save video bytes to file and set output parameters."""
        if not video_bytes:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Received empty video data from API.",
            )
            return

        try:
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(video_bytes)
            self.parameter_output_values["video_url"] = VideoUrlArtifact(value=saved.location, name=saved.name)
            self._log(f"Saved video as {saved.name}")
            self._set_status_results(
                was_successful=True,
                result_details=f"Video generated successfully and saved as {saved.name}.",
            )
        except Exception as e:
            self._log(f"Failed to save video: {e}")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"Video generation completed but failed to save: {e}",
            )

    async def _prepare_image_data_uri(self, image_input: Any) -> str | None:
        """Convert image input to a data URI, downloading from local paths if needed.

        RunwayML requires images as data URIs or publicly-accessible URLs with
        proper Content-Type headers. Data URIs are the most reliable option.
        """
        if not image_input:
            return None

        image_url = self._coerce_image_url_or_data_uri(image_input)
        if not image_url:
            return None

        if image_url.startswith("data:image/"):
            return image_url

        try:
            image_bytes = await File(image_url).aread_bytes()
        except FileLoadError as e:
            logger.debug("%s failed to load image from %s: %s", self.name, image_url, e)
            return None

        # RunwayML data URIs are limited to 5,242,880 characters.
        # The base64 prefix is ~22 chars, so raw bytes must encode to under ~3.9MB.
        max_raw_bytes = 3_900_000
        if len(image_bytes) > max_raw_bytes:
            image_bytes = shrink_image_to_size(image_bytes, max_raw_bytes, context_name=self.name)

        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    @staticmethod
    def _coerce_image_url_or_data_uri(val: Any) -> str | None:
        """Convert various image input types to a URL or data URI string."""
        if val is None:
            return None

        if isinstance(val, str):
            v = val.strip()
            if not v:
                return None
            return v if v.startswith(("http://", "https://", "data:image/")) else f"data:image/png;base64,{v}"

        # ImageArtifact: .base64 holds raw or data-URI
        b64 = getattr(val, "base64", None)
        if isinstance(b64, str) and b64:
            return b64 if b64.startswith("data:image/") else f"data:image/png;base64,{b64}"

        # ImageUrlArtifact or similar: .value holds URL or local path
        v = getattr(val, "value", None)
        if isinstance(v, str) and v.strip():
            return v.strip()

        return None

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from RunwayML error responses.

        RunwayML errors follow the structure: {"error": "...", "issues": [...]}
        """
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        # Try RunwayML-specific error format
        error = response_json.get("error")
        issues = response_json.get("issues")
        if error and isinstance(error, str):
            msg = f"{self.name}: {error}"
            if issues and isinstance(issues, list):
                issue_messages = []
                for issue in issues:
                    if isinstance(issue, dict):
                        issue_msg = issue.get("message", "")
                        issue_path = issue.get("path", [])
                        if issue_msg:
                            path_str = ".".join(str(p) for p in issue_path) if issue_path else ""
                            issue_messages.append(f"  {path_str}: {issue_msg}" if path_str else f"  {issue_msg}")
                if issue_messages:
                    msg += "\n" + "\n".join(issue_messages)
            return msg

        # Fall back to base class extraction
        return super()._extract_error_message(response_json)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before execution."""
        exceptions = super().validate_before_node_run() or []

        prompt_text = self.get_parameter_value("prompt_text") or ""
        if not prompt_text.strip():
            exceptions.append(ValueError(f"{self.name} requires a prompt to generate video."))
        elif len(prompt_text) > 1000:
            exceptions.append(
                ValueError(f"{self.name} prompt exceeds 1000 characters (got: {len(prompt_text)} characters).")
            )

        api_model_id = self._get_api_model_id()
        prompt_image = self.get_parameter_value("prompt_image")

        # gen4_turbo and gen3a_turbo require an input image
        if api_model_id in MODELS_REQUIRING_IMAGE and not prompt_image:
            exceptions.append(
                ValueError(
                    f"{self.name}: {self.get_parameter_value('model')} requires an input image for video generation."
                )
            )

        duration = self.get_parameter_value("duration")
        if duration is not None and (int(duration) < MIN_DURATION or int(duration) > MAX_DURATION):
            exceptions.append(
                ValueError(f"{self.name}: duration must be between {MIN_DURATION} and {MAX_DURATION} seconds.")
            )

        return exceptions if exceptions else None
