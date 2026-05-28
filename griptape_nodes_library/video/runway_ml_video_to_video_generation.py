from __future__ import annotations

import base64
import logging
from contextlib import suppress
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.cloud_upload import upload_artifact_with_content_type
from griptape_nodes_library.utils.image_utils import shrink_image_to_size

logger = logging.getLogger("griptape_nodes")

__all__ = ["RunwayMLVideoToVideoGeneration"]

# This node targets a single Runway video-to-video model.
API_MODEL_ID = "gen4_aleph"

# Seed constraints
MAX_SEED = 4294967295

# RunwayML data URI base64 size cap (~3.9MB raw bytes encode to <5,242,880 chars).
MAX_REFERENCE_IMAGE_BYTES = 3_900_000


class RunwayMLVideoToVideoGeneration(GriptapeProxyNode):
    """Edit/transform a video using RunwayML Gen-4 Aleph via Griptape Cloud model proxy.

    Inputs:
        - prompt_text (str): What should change in the output (1-1000 chars)
        - video (VideoUrlArtifact): Source video to transform (required, uploaded to a
          public URL via Griptape Cloud since RunwayML's video_to_video endpoint only
          accepts HTTPS URLs)
        - reference_image (ImageUrlArtifact): Optional style/content reference image
        - seed (int): Random seed for reproducibility

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from Griptape model proxy
        - video_url (VideoUrlArtifact): Generated video as URL artifact
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit or transform a video using RunwayML Gen-4 Aleph via Griptape Cloud model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="prompt_text",
                tooltip="Describe how the source video should be transformed (1-1000 characters)",
                multiline=True,
                placeholder_text="Describe how to transform the source video...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        # Source video must be reachable as an HTTPS URL by RunwayML, so route the
        # parameter through PublicArtifactUrlParameter which uploads local artifacts
        # to a Griptape Cloud bucket and returns a short-lived public URL.
        self._public_video_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="video",
                default_value="",
                tooltip="Source video to transform (required)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Source Video"},
            ),
            disclaimer_message="RunwayML uses this URL to fetch the source video for transformation.",
        )
        self._public_video_parameter.add_input_parameters()

        self.add_parameter(
            ParameterImage(
                name="reference_image",
                tooltip="Optional reference image to influence style or content",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Reference Image",
                    "expander": True,
                },
            )
        )

        with ParameterGroup(name="Generation Settings") as gen_settings_group:
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
            default_filename="runway_aleph.mp4",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the video transformation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _get_api_model_id(self) -> str:
        return API_MODEL_ID

    async def _process_generation(self) -> None:
        try:
            await super()._process_generation()
        finally:
            self._public_video_parameter.delete_uploaded_artifact()

    async def _build_payload(self) -> dict[str, Any]:
        prompt_text = self.get_parameter_value("prompt_text") or ""
        seed = self.get_parameter_value("seed") or 0
        reference_image = self.get_parameter_value("reference_image")

        try:
            upload_result = upload_artifact_with_content_type(
                self._public_video_parameter,
                self.get_parameter_value("video"),
                content_type="video/mp4",
            )
        except Exception as e:
            msg = f"{self.name} failed to upload source video for RunwayML: {e}"
            raise ValueError(msg) from e

        video_uri = upload_result.public_url
        if not video_uri:
            msg = f"{self.name} requires a source video for video-to-video generation."
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "promptText": prompt_text.strip(),
            "videoUri": video_uri,
        }

        if seed and int(seed) > 0:
            payload["seed"] = int(seed)

        if reference_image:
            data_uri = await self._prepare_image_data_uri(reference_image)
            if data_uri:
                payload["references"] = [{"type": "image", "uri": data_uri}]

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the RunwayML result and download the video.

        The proxy client returns the output array from the RunwayML task response.
        The output is an array of signed CloudFront URLs.
        """
        if "raw_bytes" in result_json:
            await self._handle_video_bytes(result_json["raw_bytes"])
            return

        output = result_json.get("output")
        if isinstance(output, list) and output:
            video_url = output[0]
        elif isinstance(output, str):
            video_url = output
        else:
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

        if len(image_bytes) > MAX_REFERENCE_IMAGE_BYTES:
            image_bytes = shrink_image_to_size(image_bytes, MAX_REFERENCE_IMAGE_BYTES, context_name=self.name)

        return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"

    @staticmethod
    def _coerce_image_url_or_data_uri(val: Any) -> str | None:
        if val is None:
            return None

        if isinstance(val, str):
            v = val.strip()
            if not v:
                return None
            return v if v.startswith(("http://", "https://", "data:image/")) else f"data:image/png;base64,{v}"

        b64 = getattr(val, "base64", None)
        if isinstance(b64, str) and b64:
            return b64 if b64.startswith("data:image/") else f"data:image/png;base64,{b64}"

        v = getattr(val, "value", None)
        if isinstance(v, str) and v.strip():
            return v.strip()

        return None

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["video_url"] = None

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message from RunwayML error responses.

        RunwayML errors follow the structure: {"error": "...", "issues": [...]}
        """
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

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

        return super()._extract_error_message(response_json)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt_text = self.get_parameter_value("prompt_text") or ""
        if not prompt_text.strip():
            exceptions.append(ValueError(f"{self.name} requires a prompt describing the desired transformation."))
        elif len(prompt_text) > 1000:
            exceptions.append(
                ValueError(f"{self.name} prompt exceeds 1000 characters (got: {len(prompt_text)} characters).")
            )

        if not self.get_parameter_value("video"):
            exceptions.append(ValueError(f"{self.name} requires a source video for video-to-video generation."))

        return exceptions if exceptions else None
