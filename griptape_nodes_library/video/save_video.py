from dataclasses import dataclass
from typing import Any

from griptape_nodes.exe_types.core_types import (
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.video_utils import (
    extract_url_from_video_object,
    is_video_url_artifact,
)


@dataclass
class VideoInput:
    """Normalized video input - single source of truth."""

    data: bytes | None = None  # Video bytes (if we have them)
    source_url: str | None = None  # URL video was loaded from (for reporting)
    format_hint: str | None = None  # Format from source



class SaveVideo(SuccessFailureNode):
    """Save a video to a file."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add video input parameter
        self.add_parameter(
            ParameterVideo(
                name="video",
                allowed_modes={ParameterMode.INPUT},
                tooltip="The video to save to file",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="griptape_nodes.mp4",
        )
        self._output_file.add_parameter()

        # Add status parameters using the helper method
        self._create_status_parameters(
            result_details_tooltip="Details about the video save operation result",
            result_details_placeholder="Details on the save attempt will be presented here.",
        )

    def _extract_bytes_from_artifact(self, artifact: Any) -> bytes | None:
        """Extract bytes from various artifact types."""
        if isinstance(artifact, dict) and "value" in artifact:
            value = artifact["value"]
            if isinstance(value, bytes):
                return value
            if isinstance(value, str) and "base64," in value:
                import base64

                return base64.b64decode(value.split("base64,")[1])

        if hasattr(artifact, "value") and isinstance(artifact.value, bytes):  # type: ignore[attr-defined]
            return artifact.value  # type: ignore[attr-defined]

        return None

    def _normalize_input(self, raw_input: Any) -> VideoInput:
        """Convert ANY input to VideoInput. Single source of truth for input handling."""
        if not raw_input:
            return VideoInput()

        # Handle VideoUrlArtifact - use File to load bytes
        if is_video_url_artifact(raw_input):
            url = extract_url_from_video_object(raw_input)
            if url:
                video_bytes = File(url).read_bytes()
                return VideoInput(data=video_bytes, source_url=url)

        # Handle all other cases - try to extract bytes
        video_bytes = self._extract_bytes_from_artifact(raw_input)

        return VideoInput(data=video_bytes)

    def _report_warning(self, _message: str) -> None:
        """Report warning status."""
        result_details = "No video to save (warning)\nInput: No video input\nResult: No file created"

        self._set_status_results(was_successful=True, result_details=f"WARNING: {result_details}")

    def _report_success(self, saved_path: str, downloaded_from_url: str | None) -> None:
        """Report success with download info if applicable."""
        result_details = f"Video saved successfully\nSaved to: {saved_path}"

        # Add download info if available
        if downloaded_from_url:
            result_details = f"Downloaded from: {downloaded_from_url}\n{result_details}"

        self._set_status_results(was_successful=True, result_details=f"SUCCESS: {result_details}")
        logger.info(f"Saved video: {saved_path}")

    def _report_error(self, error_details: str, exception: Exception | None = None) -> None:
        """Report error status."""
        failure_details = f"Video save failed\nError: {error_details}"

        if exception:
            failure_details += f"\nException type: {type(exception).__name__}"
            if exception.__cause__:
                failure_details += f"\nCause: {exception.__cause__}"

        self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")
        logger.error(f"Error saving video: {error_details}")

        # Use the helper to handle exception based on connection status
        self._handle_failure_exception(RuntimeError(error_details))

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        # Validate that we have a video.
        video = self.parameter_values.get("video")
        if not video:
            exceptions.append(ValueError("Video parameter is required"))

        return exceptions if exceptions else None

    async def aprocess(self) -> None:
        """Async process method."""
        self._clear_execution_status()

        try:
            video_input = self._normalize_input(self.get_parameter_value("video"))

            if not video_input.data:
                self._report_warning("No video data available")
                return

            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(video_input.data)
            saved_path = saved.location
            self._report_success(saved_path, video_input.source_url)

        except Exception as e:
            self._report_error(str(e), e)

    def process(self) -> None:
        """Sync process method."""
        self._clear_execution_status()

        try:
            video_input = self._normalize_input(self.get_parameter_value("video"))

            if not video_input.data:
                self._report_warning("No video data available")
                return

            dest = self._output_file.build_file()
            saved = dest.write_bytes(video_input.data)
            saved_path = saved.location
            self._report_success(saved_path, video_input.source_url)

        except Exception as e:
            self._report_error(str(e), e)
