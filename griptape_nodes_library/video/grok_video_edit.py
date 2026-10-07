from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlsplit

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError

from griptape_nodes_library.media import coerce_media_url_or_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode
from griptape_nodes_library.utils.ffmpeg_utils import extract_video_metadata_structured

logger = logging.getLogger("griptape_nodes")

__all__ = ["GrokVideoEdit"]

# xAI's documented input limits for video editing
# (docs.x.ai/developers/model-capabilities/video/editing). xAI accepts a request that breaks
# them and then fails the job with no reason, so they are checked before submitting.
MAX_INPUT_DURATION_SECONDS = 8.7
SUPPORTED_CONTAINER = "mp4"


class GrokVideoEdit(GriptapeProxyNode):
    """Edit videos using Grok video models via Griptape model proxy.

    Inputs:
        - model (str): Grok video model to use
        - prompt (str): Prompt for video editing
        - video (VideoUrlArtifact): Input video to edit (required)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - video_url (VideoUrlArtifact): Edited video
        - was_successful (bool): Whether the edit succeeded
        - result_details (str): Details about the edit result or error
    """

    # Migrates values saved before the dropdown stored the provider's own model id.
    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = {
        "Grok Imagine Video": "grok-imagine-video",
        "gtc_grok_imagine_video": "grok-imagine-video",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit videos using Grok video models via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value="grok-imagine-video",
            tooltip="Select the Grok video model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=["grok-imagine-video"],
            default_model="grok-imagine-video",
            deprecated_values=self.LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Prompt for video editing",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the edits you want to make...",
                allow_output=False,
            )
        )

        video_param = ParameterVideo(
            name="video",
            default_value="",
            tooltip="Input video to edit",
            allowed_modes={ParameterMode.INPUT},
            hide_property=True,
            ui_options={"display_name": "Video"},
        )
        video_param.set_badge(
            variant="info",
            title="Video requirements",
            message=(
                f"- .mp4 file\n- {MAX_INPUT_DURATION_SECONDS:g} seconds or less\n\n"
                "Use a Trim Video node to convert or shorten a video."
            ),
        )
        self.add_parameter(video_param)

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
                name="video_url",
                tooltip="Edited video",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="grok_video_edit.mp4",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the video editing result or any errors",
            result_details_placeholder="Editing status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self.set_initial_node_size(height=1145)

    @staticmethod
    def _has_media_value(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "value"):
            return bool(value.value)
        return bool(value)

    @staticmethod
    def _container_token(video_value: str) -> str:
        """Return the container named by the value's extension or MIME subtype, or "" if neither."""
        if video_value.startswith("data:"):
            header = video_value.removeprefix("data:").split(",", 1)[0]
            return header.split(";", 1)[0].split("/", 1)[-1].lower()
        return Path(urlsplit(video_value).path).suffix.lstrip(".").lower()

    def _probe_duration(self, video_value: str) -> float | None:
        """Return the video's duration in seconds, or None when it cannot be determined.

        Probing is best-effort. A missing ffprobe or an unreadable remote URL should not block a
        request that xAI might accept, so failures are logged and skipped.
        """
        try:
            location = File(video_value).resolve()
            metadata = extract_video_metadata_structured(location)
        except (FileLoadError, ValueError, RuntimeError) as e:
            logger.warning("%s: could not probe input video duration, skipping the check: %s", self.name, e)
            return None
        return metadata.file_details.optional_duration

    def _find_input_problems(self, video_value: str) -> list[str]:
        """Return one short sentence pair per xAI input limit the video breaks.

        Returns an empty list when the video is fine. A missing container token carries no
        signal (a signed URL that strips the filename, say) and is let through. ffprobe can't
        take a multi-megabyte data URI as an argument, so a data URI's duration goes unchecked
        and xAI enforces it.
        """
        problems = []

        token = self._container_token(video_value)
        if token and token != SUPPORTED_CONTAINER:
            problems.append(f"It's a .{token} file. Grok needs .mp4.")

        if not video_value.startswith("data:"):
            duration = self._probe_duration(video_value)
            if duration is not None and duration > MAX_INPUT_DURATION_SECONDS:
                problems.append(
                    f"It's {duration:.2f} seconds long. Grok's limit is {MAX_INPUT_DURATION_SECONDS:g} seconds."
                )

        return problems

    async def _check_input_limits(self, video_value: str) -> None:
        """Raise a short, user-facing error if the video breaks any of xAI's input limits."""
        problems = await asyncio.to_thread(self._find_input_problems, video_value)
        if problems:
            msg = f"Grok can't edit this video. {' '.join(problems)} Use a Trim Video node to fix this."
            raise ValueError(msg)

    async def _prepare_video_data_uri(self, video_input: Any) -> str:
        """Return the video as a data URI, raising a user-facing error if it can't be sent.

        These errors are raised during execution, where the engine already prefixes the message
        with the node name several times, so they leave the name out.
        """
        video_value = coerce_media_url_or_data_uri(video_input, kind="video")
        if not video_value:
            msg = "Video input has no usable value."
            raise ValueError(msg)

        await self._check_input_limits(video_value)

        if video_value.startswith("data:"):
            return video_value

        try:
            return await File(video_value).aread_data_uri(fallback_mime="video/mp4")
        except FileLoadError as e:
            msg = f"Could not load video from '{video_value}': {e}"
            raise ValueError(msg) from e

    def _get_api_model_id(self) -> str:
        return f"{self._get_selected_model_id()}:edit"

    def _get_payload_model_id(self) -> str:
        return self._get_selected_model_id() or "grok-imagine-video"

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(ValueError("A prompt is required for video editing. Enter one in 'prompt'."))

        video_value = self.get_parameter_value("video")
        if not self._has_media_value(video_value):
            exceptions.append(ValueError("A video is required for editing. Connect one to 'Video'."))

        return exceptions if exceptions else None

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        api_model_id = self._get_payload_model_id()
        video_data_uri = await self._prepare_video_data_uri(self.get_parameter_value("video"))

        return {
            "model": api_model_id,
            "prompt": prompt,
            "video": {"url": video_data_uri},
        }

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        await self._save_generated_media(
            generation_id,
            "video_url",
            lambda v, n: VideoUrlArtifact(value=v, name=n),
            kind=ArtifactKind.VIDEO,
            action="edited",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["video_url"] = None
