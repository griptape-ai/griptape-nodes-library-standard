from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["RunwayMLImageGeneration"]

# Model mapping from user-friendly names to API model IDs
MODEL_MAPPING: dict[str, str] = {
    "Gen-4 Image": "gen4_image",
    "Gen-4 Image Turbo": "gen4_image_turbo",
}
MODEL_OPTIONS = list(MODEL_MAPPING.keys())
DEFAULT_MODEL = MODEL_OPTIONS[0]

# Ratio options available for image generation
RATIO_OPTIONS = [
    "1024:1024",
    "1080:1080",
    "1168:880",
    "1360:768",
    "1440:1080",
    "1080:1440",
    "1808:768",
    "1920:1080",
    "1080:1920",
    "2112:912",
    "1280:720",
    "720:1280",
    "720:720",
    "960:720",
    "720:960",
    "1680:720",
]
DEFAULT_RATIO = "1360:768"

# Seed constraints
MAX_SEED = 4294967295


class RunwayMLImageGeneration(GriptapeProxyNode):
    """Generate images using RunwayML Gen-4 Image models via Griptape Cloud model proxy.

    Inputs:
        - model (str): RunwayML image generation model
        - prompt_text (str): Text description of the desired image (1-1000 chars)
        - reference_image (ImageUrlArtifact): Reference image for generation (required)
        - ratio (str): Output image resolution
        - seed (int): Random seed for reproducibility

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from Griptape model proxy
        - image_url (ImageUrlArtifact): Generated image as URL artifact
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using RunwayML Gen-4 Image models via Griptape Cloud model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="RunwayML image generation model",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt_text",
                tooltip="Text description of the desired image (1-1000 characters)",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="reference_image",
                tooltip="Reference image for generation (required, min 2x2 pixels)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Reference Image",
                    "expander": True,
                },
            )
        )

        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            ParameterString(
                name="ratio",
                default_value=DEFAULT_RATIO,
                tooltip="Output image resolution",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RATIO_OPTIONS)},
                ui_options={"display_name": "Ratio"},
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
            ParameterImage(
                name="image_url",
                tooltip="Generated image as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="runway_image.png",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _get_api_model_id(self) -> str:
        """Map user-friendly model name to API model ID."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for RunwayML image generation."""
        prompt_text = self.get_parameter_value("prompt_text") or ""
        ratio = self.get_parameter_value("ratio") or DEFAULT_RATIO
        seed = self.get_parameter_value("seed") or 0
        reference_image = self.get_parameter_value("reference_image")

        payload: dict[str, Any] = {
            "promptText": prompt_text.strip(),
            "ratio": ratio,
        }

        # Include seed only if non-zero
        if seed and int(seed) > 0:
            payload["seed"] = int(seed)

        # Build referenceImages array (required by the API)
        if reference_image:
            data_uri = self._image_to_data_uri(reference_image)
            if data_uri:
                payload["referenceImages"] = [{"uri": data_uri}]

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the RunwayML result and download the image.

        The proxy client returns the output array from the RunwayML task response.
        The output is an array of signed CloudFront URLs.
        """
        # Look for output URLs in various possible response shapes
        output = result_json.get("output")
        if isinstance(output, list) and output:
            image_url = output[0]
        elif isinstance(output, str):
            image_url = output
        else:
            image_url = result_json.get("image_url") or result_json.get("url")

        if not image_url or not isinstance(image_url, str):
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image URL was found in the response.",
            )
            return

        # Download and save the image
        try:
            image_bytes = await self._download_bytes_from_url(image_url)
            if image_bytes:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(image_bytes)
                self.parameter_output_values["image_url"] = ImageUrlArtifact(saved.location)
                self._log(f"Saved image as {saved.name}")
                self._set_status_results(
                    was_successful=True,
                    result_details=f"Image generated successfully and saved as {saved.name}.",
                )
            else:
                self.parameter_output_values["image_url"] = ImageUrlArtifact(value=image_url)
                self._set_status_results(
                    was_successful=True,
                    result_details="Image generated successfully. Using provider URL (could not download image bytes).",
                )
        except Exception as e:
            self._log(f"Failed to save image: {e}")
            self.parameter_output_values["image_url"] = ImageUrlArtifact(value=image_url)
            self._set_status_results(
                was_successful=True,
                result_details=f"Image generated successfully. Using provider URL (could not save to storage: {e}).",
            )

    @staticmethod
    def _image_to_data_uri(image_input: Any) -> str | None:
        """Convert an image parameter value to a data URI for RunwayML.

        RunwayML requires images as data URIs or publicly-accessible URLs with
        proper Content-Type headers. Data URIs are the most reliable option.
        """
        if image_input is None:
            return None

        # String input
        if isinstance(image_input, str):
            v = image_input.strip()
            if not v:
                return None
            if v.startswith("data:image/"):
                return v
            if v.startswith(("http://", "https://")):
                return v
            return f"data:image/png;base64,{v}"

        # ImageUrlArtifact: .value holds a URL string
        value = getattr(image_input, "value", None)
        if isinstance(value, str) and value.startswith(("http://", "https://", "data:image/")):
            return value

        # ImageArtifact: .base64 holds base64-encoded image data
        b64 = getattr(image_input, "base64", None)
        if isinstance(b64, str) and b64:
            return b64 if b64.startswith("data:image/") else f"data:image/png;base64,{b64}"

        return None

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["image_url"] = None

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
            exceptions.append(ValueError(f"{self.name} requires a prompt to generate an image."))
        elif len(prompt_text) > 1000:
            exceptions.append(
                ValueError(f"{self.name} prompt exceeds 1000 characters (got: {len(prompt_text)} characters).")
            )

        reference_image = self.get_parameter_value("reference_image")
        if not reference_image:
            exceptions.append(ValueError(f"{self.name} requires a reference image for image generation."))

        return exceptions if exceptions else None
