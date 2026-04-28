import json
import subprocess
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# static_ffmpeg is dynamically installed by the library loader at runtime
# into the library's own virtual environment, but not available during type checking
import static_ffmpeg.run  # type: ignore[import-untyped]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File

from griptape_nodes_library.utils.file_utils import generate_filename
from griptape_nodes_library.utils.video_utils import (
    detect_video_format,
    dict_to_video_url_artifact,
    to_video_artifact,
    validate_url,
)


class BaseVideoInputNode(SuccessFailureNode, ABC):
    """Abstract base for nodes that take a video as input.

    Provides video input parameter, ffprobe utilities, URL validation, logging,
    status parameters, and temp file helpers — without assuming video output.
    Subclass this when your node produces something other than a video (images,
    audio, metadata, etc.).  For video-to-video processing, subclass
    BaseVideoProcessor instead.
    """

    DEFAULT_FRAME_RATE = 30.0
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    DEFAULT_DURATION = 0.0

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.add_parameter(
            ParameterVideo(
                name="video",
                tooltip="The video to process",
                clickable_file_browser=True,
                ui_options={
                    "expander": True,
                    "display_name": "Video or Path to Video",
                },
                converters=[self._convert_video_input],
            )
        )

        self._setup_custom_parameters()

        # Hook: lets BaseVideoProcessor insert frame-rate / speed params here,
        # between custom params and the primary output param, preserving UI order.
        self._setup_pre_output_parameters()

        self._register_primary_output_parameter()

        # Separate hook so subclasses can co-locate their output file setup with
        # _register_primary_output_parameter() if they prefer, while still keeping
        # the two steps distinct in __init__.
        self._register_output_file_parameter()

        self._setup_logging_group()

        self._create_status_parameters(
            result_details_tooltip="Details about the processing operation result",
            result_details_placeholder="Details on the processing attempt will be presented here.",
        )

    @abstractmethod
    def _setup_custom_parameters(self) -> None:
        """Setup parameters specific to this node. Override in subclasses."""

    def _setup_pre_output_parameters(self) -> None:
        """Hook called after custom params and before the primary output param.

        Override in subclasses that need to insert additional parameters at
        this position (e.g., BaseVideoProcessor adds frame rate and speed here).
        """

    @abstractmethod
    def _register_primary_output_parameter(self) -> None:
        """Register the node's primary output artifact parameter (image, audio, list, etc.)."""

    def _register_output_file_parameter(self) -> None:
        """Register the output_file project-file parameter used for saving results.

        Called immediately after _register_primary_output_parameter() so the two
        are adjacent in the UI.  Override to customise the default filename or to
        skip the parameter entirely for nodes that don't produce a saved file.
        """
        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename=self._get_output_file_default_filename(),
        )
        self._output_file.add_parameter()

    @abstractmethod
    def _get_processing_description(self) -> str:
        """Short human-readable description of what this node does."""

    @abstractmethod
    def _get_output_file_default_filename(self) -> str:
        """Return the default filename for the output_file parameter.

        Subclasses may declare OUTPUT_FILE_DEFAULT_FILENAME as a class variable
        instead of overriding this method.
        """

    def _setup_logging_group(self) -> None:
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def _get_ffmpeg_paths(self) -> tuple[str, str]:
        """Return (ffmpeg_path, ffprobe_path)."""
        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            return ffmpeg_path, ffprobe_path  # noqa: TRY300
        except Exception as e:
            error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
            raise ValueError(error_msg) from e

    def _detect_video_properties(self, input_url: str, ffprobe_path: str) -> tuple[float, tuple[int, int], float]:
        """Detect (frame_rate, (width, height), duration) from a video."""
        try:
            cmd = [
                ffprobe_path,
                "-v", "quiet",
                "-print_format", "json",
                "-show_streams",
                "-select_streams", "v:0",
                input_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            streams_data = json.loads(result.stdout)

            if streams_data.get("streams") and len(streams_data["streams"]) > 0:
                video_stream = streams_data["streams"][0]

                fps_str = video_stream.get("r_frame_rate", "30/1")
                if "/" in fps_str:
                    num, den = map(int, fps_str.split("/"))
                    frame_rate = num / den
                else:
                    frame_rate = float(fps_str)

                width = int(video_stream.get("width", self.DEFAULT_WIDTH))
                height = int(video_stream.get("height", self.DEFAULT_HEIGHT))

                duration_str = video_stream.get("duration", "0")
                duration = float(duration_str) if duration_str != "N/A" else self.DEFAULT_DURATION

                return frame_rate, (width, height), duration
            return self.DEFAULT_FRAME_RATE, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), self.DEFAULT_DURATION  # noqa: TRY300

        except Exception as e:
            self.append_value_to_parameter("logs", f"Warning: Could not detect video properties, using defaults: {e}\n")
            return self.DEFAULT_FRAME_RATE, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), self.DEFAULT_DURATION

    def _detect_audio_stream(self, input_url: str, ffprobe_path: str) -> bool:
        """Return True if the video contains an audio stream."""
        try:
            cmd = [
                ffprobe_path,
                "-v", "quiet",
                "-select_streams", "a",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                input_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            return "audio" in result.stdout.strip()
        except subprocess.CalledProcessError:
            return False
        except Exception:
            return False

    def _convert_video_input(self, value: Any) -> Any:
        """Convert dict inputs to VideoUrlArtifact (string paths handled by ParameterVideo)."""
        if isinstance(value, dict):
            return dict_to_video_url_artifact(value)
        return value

    def _validate_video_input(self) -> list[Exception] | None:
        exceptions = []
        video = self.parameter_values.get("video")
        if not video:
            msg = f"{self.name}: Video parameter is required"
            exceptions.append(ValueError(msg))
        if not isinstance(video, VideoUrlArtifact):
            msg = f"{self.name}: Video parameter must be a VideoUrlArtifact"
            exceptions.append(ValueError(msg))
        if hasattr(video, "value") and not video.value:  # type: ignore  # noqa: PGH003
            msg = f"{self.name}: Video parameter must have a value"
            exceptions.append(ValueError(msg))
        return exceptions if exceptions else None

    def _validate_url_safety(self, url: str) -> None:
        if not validate_url(url):
            msg = f"{self.name}: Invalid or unsafe URL provided: {url}"
            raise ValueError(msg)

    def _get_video_input_data(self) -> tuple[str, str]:
        """Return (resolved_url, detected_format) for the video input."""
        video = self.parameter_values.get("video")
        video_artifact = to_video_artifact(video)
        input_url = File(video_artifact.value).resolve()
        detected_format = detect_video_format(video) or "mp4"
        return input_url, detected_format

    def _create_temp_output_file(self, format_extension: str) -> tuple[str, Path]:
        with tempfile.NamedTemporaryFile(suffix=f".{format_extension}", delete=False) as output_file:
            output_path = Path(output_file.name)
        return str(output_path), output_path

    def _run_ffmpeg_command(self, cmd: list[str], timeout: int = 300) -> None:
        self.append_value_to_parameter("logs", f"Running ffmpeg command: {' '.join(cmd)}\n")
        try:
            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=timeout
            )
            self.append_value_to_parameter("logs", f"FFmpeg stdout: {result.stdout}\n")
        except subprocess.TimeoutExpired as e:
            error_msg = f"FFmpeg process timed out after {timeout} seconds"
            self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
            raise ValueError(error_msg) from e
        except subprocess.CalledProcessError as e:
            error_msg = f"FFmpeg error: {e.stderr}"
            self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
            raise ValueError(error_msg) from e

    def _cleanup_temp_file(self, file_path: Path) -> None:
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            self.append_value_to_parameter("logs", f"Warning: Failed to clean up temporary file: {e}\n")

    def _log_video_properties(self, frame_rate: float, resolution: tuple[int, int], duration: float) -> None:
        self.append_value_to_parameter(
            "logs", f"Detected video: {resolution[0]}x{resolution[1]} @ {frame_rate}fps, duration: {duration}s\n"
        )

    def _log_format_detection(self, detected_format: str) -> None:
        self.append_value_to_parameter("logs", f"Detected video format: {detected_format}\n")

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Override in subclasses to add custom validation."""
        return None

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []
        base_exceptions = self._validate_video_input()
        if base_exceptions:
            exceptions.extend(base_exceptions)
        custom_exceptions = self._validate_custom_parameters()
        if custom_exceptions:
            exceptions.extend(custom_exceptions)
        return exceptions if exceptions else None

    def _generate_filename(self, suffix: str = "", extension: str = "") -> str:
        return generate_filename(node_name=self.name, suffix=suffix, extension=extension)
