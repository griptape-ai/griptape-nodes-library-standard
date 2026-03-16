from dataclasses import dataclass
from typing import Any

from griptape_nodes.exe_types.core_types import (
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.utils.audio_utils import (
    extract_url_from_audio_object,
    is_audio_url_artifact,
)


@dataclass
class AudioInput:
    """Normalized audio input - single source of truth."""

    data: bytes | None = None  # Audio bytes (if we have them)
    source_url: str | None = None  # URL audio was loaded from (for reporting)
    format_hint: str | None = None  # Format from source



class SaveAudio(SuccessFailureNode):
    """Save a audio to a file."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add audio input parameter
        self.add_parameter(
            ParameterAudio(
                name="audio",
                allowed_modes={ParameterMode.INPUT},
                tooltip="The audio to save to file",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="griptape_nodes.mp3",
        )
        self._output_file.add_parameter()

        # Add status parameters using the helper method
        self._create_status_parameters(
            result_details_tooltip="Details about the audio save operation result",
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

    def _normalize_input(self, raw_input: Any) -> AudioInput:
        """Convert ANY input to AudioInput. Single source of truth for input handling."""
        if not raw_input:
            return AudioInput()

        # Handle AudioUrlArtifact - use File to load bytes
        if is_audio_url_artifact(raw_input):
            url = extract_url_from_audio_object(raw_input)
            if url:
                audio_bytes = File(url).read_bytes()
                return AudioInput(data=audio_bytes, source_url=url)

        # Handle all other cases - try to extract bytes
        audio_bytes = self._extract_bytes_from_artifact(raw_input)

        return AudioInput(data=audio_bytes)

    def _report_warning(self, _message: str) -> None:
        """Report warning status."""
        result_details = (
            f"No audio to save (warning)\n"
            f"Input: No audio input\n"
            f"Result: No file created"
        )

        self._set_status_results(was_successful=True, result_details=f"WARNING: {result_details}")

    def _report_success(self, saved_path: str, downloaded_from_url: str | None) -> None:
        """Report success with download info if applicable."""
        result_details = f"Audio saved successfully\nSaved to: {saved_path}"

        # Add download info if available
        if downloaded_from_url:
            result_details = f"Downloaded from: {downloaded_from_url}\n{result_details}"

        self._set_status_results(was_successful=True, result_details=f"SUCCESS: {result_details}")
        logger.info(f"Saved audio: {saved_path}")

    def _report_error(self, error_details: str, exception: Exception | None = None) -> None:
        """Report error status."""
        failure_details = f"Audio save failed\nError: {error_details}"

        if exception:
            failure_details += f"\nException type: {type(exception).__name__}"
            if exception.__cause__:
                failure_details += f"\nCause: {exception.__cause__}"

        self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")
        logger.error(f"Error saving audio: {error_details}")

        # Use the helper to handle exception based on connection status
        self._handle_failure_exception(RuntimeError(error_details))

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        # Validate that we have a audio.
        audio = self.parameter_values.get("audio")
        if not audio:
            exceptions.append(ValueError("Audio parameter is required"))

        return exceptions if exceptions else None

    async def aprocess(self) -> None:
        """Async process method."""
        self._clear_execution_status()

        try:
            audio_input = self._normalize_input(self.get_parameter_value("audio"))

            if not audio_input.data:
                self._report_warning("No audio data available")
                return

            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(audio_input.data)
            saved_path = saved.location
            self._report_success(saved_path, audio_input.source_url)

        except Exception as e:
            self._report_error(str(e), e)

    def process(self) -> None:
        """Sync process method."""
        self._clear_execution_status()

        try:
            audio_input = self._normalize_input(self.get_parameter_value("audio"))

            if not audio_input.data:
                self._report_warning("No audio data available")
                return

            dest = self._output_file.build_file()
            saved = dest.write_bytes(audio_input.data)
            saved_path = saved.location
            self._report_success(saved_path, audio_input.source_url)

        except Exception as e:
            self._report_error(str(e), e)
