from abc import abstractmethod
from typing import Any, ClassVar

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.options import Options

from griptape_nodes_library.video.base_video_input_node import BaseVideoInputNode


class BaseVideoProcessor(BaseVideoInputNode):
    """Base class for video-to-video processing nodes.

    Extends BaseVideoInputNode with output frame-rate control, processing-speed
    control, the video output parameter, and the standard async processing
    pipeline (_process_video → _process → process).

    For nodes that produce something other than a video (images, audio, etc.),
    subclass BaseVideoInputNode directly instead.
    """

    # Common frame rate options for different platforms
    FRAME_RATE_OPTIONS: ClassVar[dict[str, str]] = {
        "auto": "Auto (use input frame rate)",
        "24": "24 fps (Film)",
        "25": "25 fps (PAL)",
        "29.97": "29.97 fps (NTSC)",
        "30": "30 fps (YouTube, Web)",
        "50": "50 fps (PAL HD)",
        "59.94": "59.94 fps (NTSC HD)",
        "60": "60 fps (YouTube, Web HD)",
    }

    # Frame rate tolerance for comparison (in fps)
    FRAME_RATE_TOLERANCE: ClassVar[float] = 0.01

    def _setup_pre_output_parameters(self) -> None:
        """Insert frame-rate and processing-speed controls before the output param."""
        # Add frame rate parameter
        frame_rate_param = ParameterString(
            name="output_frame_rate",
            default_value="auto",
            tooltip="Output frame rate. Choose 'auto' to preserve input frame rate, or select a specific rate for your target platform.",
        )
        frame_rate_param.add_trait(Options(choices=list(self.FRAME_RATE_OPTIONS.keys())))
        self.add_parameter(frame_rate_param)

        # Add processing speed parameter
        speed_param = ParameterString(
            name="processing_speed",
            default_value="balanced",
            tooltip="Processing speed vs quality trade-off",
        )
        speed_param.add_trait(Options(choices=["fast", "balanced", "quality"]))
        self.add_parameter(speed_param)

    def _get_output_file_default_filename(self) -> str:
        candidate = getattr(type(self), "OUTPUT_FILE_DEFAULT_FILENAME", None)
        if isinstance(candidate, str) and candidate:
            return candidate
        return "output.mp4"

    def _register_primary_output_parameter(self) -> None:
        self.add_parameter(
            ParameterVideo(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The processed video",
                ui_options={"pulse_on_run": True, "expander": True, "display_name": "Processed Video"},
            )
        )

    @abstractmethod
    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:
        """Build the FFmpeg command for this processor. Override in subclasses."""

    def _get_processing_speed_settings(self) -> tuple[str, str, int]:
        """Return (preset, pixel_format, crf) based on processing_speed parameter."""
        speed = self.get_parameter_value("processing_speed") or "balanced"

        if speed == "fast":
            return "ultrafast", "yuv420p", 30  # Fastest encoding, lower quality
        if speed == "quality":
            return "slow", "yuv420p", 18  # Slowest encoding, highest quality
        # balanced
        return "medium", "yuv420p", 23  # Balanced speed and quality

    def _get_frame_rate_filter(self, input_frame_rate: float) -> str:
        """Return an ffmpeg fps filter string, or empty string if no conversion needed."""
        output_frame_rate = self.get_parameter_value("output_frame_rate") or "auto"

        if output_frame_rate == "auto":
            return ""  # No frame rate conversion needed

        # Convert string to float
        target_fps = float(output_frame_rate)

        # If target is same as input (within tolerance), no conversion needed
        if abs(target_fps - input_frame_rate) < self.FRAME_RATE_TOLERANCE:
            return ""

        # Return fps filter for frame rate conversion
        return f"fps=fps={target_fps}:round=up"

    def _combine_video_filters(self, custom_filter: str, input_frame_rate: float) -> str:
        """Combine a custom vf filter with the frame-rate filter."""
        frame_rate_filter = self._get_frame_rate_filter(input_frame_rate)

        if not frame_rate_filter:
            return custom_filter

        if not custom_filter or custom_filter == "null":
            return frame_rate_filter

        # Combine filters with comma separator
        return f"{custom_filter},{frame_rate_filter}"

    def _save_video_artifact(self, video_bytes: bytes, format_extension: str, suffix: str = "") -> VideoUrlArtifact:
        dest = self._output_file.build_file()
        saved = dest.write_bytes(video_bytes)
        return VideoUrlArtifact(saved.location)

    def _process_video(self, input_url: str, output_path: str, **kwargs) -> None:
        """Process video using the custom FFmpeg command from subclasses."""
        try:
            # Validate URL before using in subprocess
            self._validate_url_safety(input_url)

            # Get FFmpeg paths
            _ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()

            # Detect input video properties for frame rate handling
            input_frame_rate, _, _ = self._detect_video_properties(input_url, ffprobe_path)

            # Build the FFmpeg command using the subclass implementation
            cmd = self._build_ffmpeg_command(input_url, output_path, input_frame_rate=input_frame_rate, **kwargs)

            # Use base class method to run FFmpeg command
            self._run_ffmpeg_command(cmd, timeout=300)
        except Exception as e:
            error_msg = f"Error during video processing: {e!s}"
            self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
            raise ValueError(error_msg) from e

    def _process(self, input_url: str, detected_format: str, **kwargs) -> None:
        """Common processing wrapper."""
        # Create temporary output file using base class method
        output_path, output_path_obj = self._create_temp_output_file(detected_format)

        try:
            self.append_value_to_parameter("logs", f"{self._get_processing_description()}\n")

            # Process video using the custom implementation
            self._process_video(input_url, output_path, **kwargs)

            # Read processed video
            with output_path_obj.open("rb") as f:
                output_bytes = f.read()

            # Get output suffix from subclass
            suffix = self._get_output_suffix(**kwargs)

            # Save processed video artifact
            output_artifact = self._save_video_artifact(output_bytes, detected_format, suffix)

            self.append_value_to_parameter(
                "logs", f"Successfully processed video with suffix: {suffix}.{detected_format}\n"
            )

            # Save to parameter
            self.parameter_output_values["output"] = output_artifact
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error processing video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e
        finally:
            # Clean up temporary file using base class method
            self._cleanup_temp_file(output_path_obj)

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        return ""

    def _get_custom_parameters(self) -> dict[str, Any]:
        return {}

    def process(self) -> AsyncResult[None]:
        """Common async processing entry point."""
        # Clear execution status at start
        self._clear_execution_status()

        # Get video input data
        input_url, detected_format = self._get_video_input_data()
        self._log_format_detection(detected_format)

        # Get custom parameters from subclasses
        custom_params = self._get_custom_parameters()

        # Initialize logs
        self.append_value_to_parameter("logs", f"[Processing {self._get_processing_description()}..]\n")

        try:
            # Run the video processing asynchronously
            self.append_value_to_parameter("logs", "[Started video processing..]\n")
            yield lambda: self._process(input_url, detected_format, **custom_params)
            self.append_value_to_parameter("logs", "[Finished video processing.]\n")

            # Report success
            result_details = f"Successfully processed video: {self._get_processing_description()}"
            self._set_status_results(was_successful=True, result_details=result_details)

        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error processing video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")

            # Report failure
            failure_details = f"Video processing failed: {error_message}"
            self._set_status_results(was_successful=False, result_details=failure_details)

            # Handle failure exception (raises if no failure output connected)
            self._handle_failure_exception(ValueError(msg))
