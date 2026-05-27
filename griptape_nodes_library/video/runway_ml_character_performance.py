from __future__ import annotations

import logging
import math
from contextlib import suppress
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
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
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.cloud_upload import upload_artifact_with_content_type
from griptape_nodes_library.utils.video_utils import get_video_duration_sync

logger = logging.getLogger("griptape_nodes")

__all__ = ["RunwayMLCharacterPerformance"]

# This node targets a single Runway character-performance model.
API_MODEL_ID = "act_two"

# Reference video duration constraints, per RunwayML's API.
MIN_REFERENCE_DURATION = 3
MAX_REFERENCE_DURATION = 30

# Expression intensity range, per RunwayML's API (1 = subtle, 5 = exaggerated).
MIN_EXPRESSION_INTENSITY = 1
MAX_EXPRESSION_INTENSITY = 5
DEFAULT_EXPRESSION_INTENSITY = 3

RATIO_OPTIONS = ["1280:720", "720:1280", "960:960", "1104:832", "832:1104", "1584:672"]
DEFAULT_RATIO = "1280:720"

# Seed constraints
MAX_SEED = 4294967295


class RunwayMLCharacterPerformance(GriptapeProxyNode):
    """Animate a character with a reference performance using RunwayML Act-Two.

    The character may be supplied as either an image (preserving the character's
    static environment) or a video (preserving the character's animated
    environment and partial motion). RunwayML requires HTTPS URLs for both the
    character video (when used) and the reference video, so those parameters
    upload local artifacts to a Griptape Cloud bucket via PublicArtifactUrlParameter.

    Inputs:
        - character_image (ImageUrlArtifact): Character as an image (mutually
          exclusive with character_video; exactly one is required).
        - character_video (VideoUrlArtifact): Character as a video (mutually
          exclusive with character_image).
        - reference_video (VideoUrlArtifact): Reference performance, 3-30 seconds.
        - body_control (bool): Apply body movements in addition to facial expressions.
        - expression_intensity (int): 1-5 intensity of facial expressions.
        - ratio (str): Output resolution.
        - seed (int): Random seed (0 = random).

    Outputs:
        - generation_id (str)
        - provider_response (dict)
        - video_url (VideoUrlArtifact)
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = (
            "Animate a character with a reference performance using RunwayML Act-Two via Griptape Cloud model proxy"
        )

        # --- INPUT PARAMETERS ---
        # Character: exactly one of image or video. Both routed through the
        # public-URL helper so RunwayML can fetch them via HTTPS.
        self._public_character_image_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterImage(
                name="character_image",
                default_value="",
                tooltip="Character as a still image (mutually exclusive with character_video)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Character Image"},
            ),
            disclaimer_message="RunwayML uses this URL to fetch the character image.",
        )
        self._public_character_image_parameter.add_input_parameters()

        self._public_character_video_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="character_video",
                default_value="",
                tooltip="Character as a video (mutually exclusive with character_image)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Character Video"},
            ),
            disclaimer_message="RunwayML uses this URL to fetch the character video.",
        )
        self._public_character_video_parameter.add_input_parameters()

        # Reference video drives the performance; required, 3-30 seconds.
        self._public_reference_video_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterVideo(
                name="reference_video",
                default_value="",
                tooltip="Reference performance video (3-30 seconds, required)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Reference Performance Video"},
            ),
            disclaimer_message="RunwayML uses this URL to fetch the reference performance video.",
        )
        self._public_reference_video_parameter.add_input_parameters()

        with ParameterGroup(name="Generation Settings") as gen_settings_group:
            ParameterBool(
                name="body_control",
                default_value=False,
                tooltip="Apply body movements and gestures in addition to facial expressions",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Body Control"},
            )

            ParameterInt(
                name="expression_intensity",
                default_value=DEFAULT_EXPRESSION_INTENSITY,
                tooltip="Intensity of facial expressions (1 = subtle, 5 = exaggerated)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=MIN_EXPRESSION_INTENSITY,
                max_val=MAX_EXPRESSION_INTENSITY,
                ui_options={"display_name": "Expression Intensity"},
            )

            ParameterString(
                name="ratio",
                default_value=DEFAULT_RATIO,
                tooltip="Output video resolution",
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
            default_filename="runway_act_two.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the character-performance result or any errors",
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
            self._public_character_image_parameter.delete_uploaded_artifact()
            self._public_character_video_parameter.delete_uploaded_artifact()
            self._public_reference_video_parameter.delete_uploaded_artifact()

    async def _build_payload(self) -> dict[str, Any]:
        body_control = bool(self.get_parameter_value("body_control"))
        expression_intensity = self.get_parameter_value("expression_intensity") or DEFAULT_EXPRESSION_INTENSITY
        ratio = self.get_parameter_value("ratio") or DEFAULT_RATIO
        seed = self.get_parameter_value("seed") or 0

        character_image = self.get_parameter_value("character_image")
        character_video = self.get_parameter_value("character_video")
        reference_video = self.get_parameter_value("reference_video")

        if not reference_video:
            msg = f"{self.name} requires a reference performance video."
            raise ValueError(msg)

        if bool(character_image) == bool(character_video):
            msg = f"{self.name} requires exactly one of character_image or character_video."
            raise ValueError(msg)

        if character_image:
            try:
                character_uri = upload_artifact_with_content_type(
                    self._public_character_image_parameter,
                    character_image,
                    content_type="image/png",
                ).public_url
            except Exception as e:
                msg = f"{self.name} failed to upload character image for RunwayML: {e}"
                raise ValueError(msg) from e
            character_payload = {"type": "image", "uri": character_uri}
        else:
            try:
                character_uri = upload_artifact_with_content_type(
                    self._public_character_video_parameter,
                    character_video,
                    content_type="video/mp4",
                ).public_url
            except Exception as e:
                msg = f"{self.name} failed to upload character video for RunwayML: {e}"
                raise ValueError(msg) from e
            character_payload = {"type": "video", "uri": character_uri}

        try:
            reference_uri = upload_artifact_with_content_type(
                self._public_reference_video_parameter,
                reference_video,
                content_type="video/mp4",
            ).public_url
        except Exception as e:
            msg = f"{self.name} failed to upload reference performance video for RunwayML: {e}"
            raise ValueError(msg) from e

        payload: dict[str, Any] = {
            "character": character_payload,
            "reference": {"type": "video", "uri": reference_uri},
            "ratio": ratio,
            "expressionIntensity": int(expression_intensity),
            "bodyControl": body_control,
        }

        if seed and int(seed) > 0:
            payload["seed"] = int(seed)

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
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

        character_image = self.get_parameter_value("character_image")
        character_video = self.get_parameter_value("character_video")
        if not character_image and not character_video:
            exceptions.append(
                ValueError(f"{self.name} requires a character (provide either character_image or character_video).")
            )
        elif character_image and character_video:
            exceptions.append(
                ValueError(f"{self.name} requires exactly one of character_image or character_video, not both.")
            )

        reference_video = self.get_parameter_value("reference_video")
        if not reference_video:
            exceptions.append(ValueError(f"{self.name} requires a reference performance video."))
        else:
            duration_error = self._validate_reference_duration(reference_video)
            if duration_error is not None:
                exceptions.append(duration_error)

        expression_intensity = self.get_parameter_value("expression_intensity")
        if expression_intensity is not None:
            ei = int(expression_intensity)
            if ei < MIN_EXPRESSION_INTENSITY or ei > MAX_EXPRESSION_INTENSITY:
                exceptions.append(
                    ValueError(
                        f"{self.name}: expression_intensity must be between "
                        f"{MIN_EXPRESSION_INTENSITY} and {MAX_EXPRESSION_INTENSITY} (got {ei})."
                    )
                )

        return exceptions if exceptions else None

    def _validate_reference_duration(self, reference_video: Any) -> Exception | None:
        """Best-effort validation that the reference video is 3-30 seconds.

        Skips silently if the duration can't be determined locally; RunwayML will
        reject the request server-side in that case.
        """
        url = getattr(reference_video, "value", None)
        if not isinstance(url, str) or not url:
            return None

        try:
            duration = get_video_duration_sync(url)
        except Exception as e:
            logger.debug("%s could not determine reference video duration: %s", self.name, e)
            return None

        seconds = math.ceil(duration)
        if seconds < MIN_REFERENCE_DURATION or seconds > MAX_REFERENCE_DURATION:
            return ValueError(
                f"{self.name}: reference_video must be between {MIN_REFERENCE_DURATION} "
                f"and {MAX_REFERENCE_DURATION} seconds (got ~{seconds}s)."
            )
        return None
