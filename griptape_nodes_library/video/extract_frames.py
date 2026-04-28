"""Video Frame Extractor node — frame-accurate scrubbing and extraction."""

import json
import logging
import pathlib
import typing
import subprocess


from griptape_nodes_library.utils.video_utils import seconds_to_ts
from griptape_nodes_library.video.base_video_input_node import BaseVideoInputNode


import griptape_nodes.exe_types.core_types as core_types
import griptape_nodes.exe_types.node_types as node_types
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
import griptape_nodes.traits.widget as widget
from griptape_nodes.retained_mode import griptape_nodes
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)

logger = logging.getLogger(__name__)


# TODO:
# 1. Display selected frames in widget
# 2. Frame extraction
# 3. Output extracted frames as ImageArtifacts with staticfiles-backed URLs and paths


"""
Algo:
- get extraction type
- get output formats
    - file format (png, jpg)
    - rename?
    - padding
    - prefix
- if a list, iterate over this list, if every Nth, iterate over generator
- for each frame to extract, seek to frame, read frame, convert to image artifact, save to staticfiles dir, output path/url/artifact

"""


class VideoFrameExtractor(BaseVideoInputNode):
    """Extract specific frames from a video and save them as images."""

    def __init__(self, name: str, metadata: dict[str, typing.Any] | None = None, **kwargs) -> None:
        node_metadata = {
            "category": "video",
            "description": "Import video, inspect frame-by-frame, and extract selected frames as images.",
        }
        if metadata:
            node_metadata.update(metadata)
        super().__init__(name=name, metadata=node_metadata, **kwargs)
        self.set_initial_node_size(width=980, height=880)

    def _get_output_file_default_filename(self) -> str:
        return "output_##.png"

    def _register_primary_output_parameter(self) -> None:
        self.add_parameter(
            core_types.Parameter(
                name="output_paths",
                type="list[ImageArtifact]|list[str|pathlib.Path]",
                allowed_modes={core_types.ParameterMode.OUTPUT},
                tooltip="Paths to the extracted frame images",
                ui_options={"pulse_on_run": True},
            )
        )

    def _setup_custom_parameters(self) -> None:
        # Input parameters

        self.add_parameter(
            core_types.Parameter(
                name="input_video",
                input_types=["VideoUrlArtifact", "VideoArtifact", "str"],
                type="VideoUrlArtifact",
                output_type="VideoUrlArtifact",
                default_value=None,
                allowed_modes={core_types.ParameterMode.INPUT},
                ui_options={"display_name": "Video Input", "hide_property": True},
                tooltip="Connect a video source here.",
            )
        )

        self.add_parameter(
            core_types.Parameter(
                name="video_player",
                type="str",
                output_type="str",
                default_value="",
                allowed_modes={core_types.ParameterMode.PROPERTY},
                tooltip="Video player for precise frame selection.",
                traits={widget.Widget(name="VideoPlayerFrameSelector", library="Griptape Nodes Library")},
            )
        )

        self.add_parameter(
            core_types.Parameter(
                name="input_frame_numbers",
                output_type="str",
                tooltip="Comma-separated list of frame numbers or ranges to extract from the video.",
                allowed_modes={core_types.ParameterMode.INPUT},
            )
        )

        # Output parameters

        self.add_parameter(
            core_types.Parameter(
                name="extracted_frames",
                output_type="list[ImageArtifact]",
                tooltip="Extracted frame as ImageArtifact.",
                allowed_modes={core_types.ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
            )
        )

        self.add_parameter(
            core_types.Parameter(
                name="extracted_frame_paths",
                output_type="list[str]",
                tooltip="Paths to the extracted frame images.",
                allowed_modes={core_types.ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
            )
        )

        self.add_parameter(
            core_types.ParameterList(
                name="extraction_output_formats",
                input_types=["List[str]", "str"],
                output_type="dict",
                tooltip="Output format options for extracted frames, including file format (png, jpg), renaming pattern, padding, and prefix.",
                allowed_modes={core_types.ParameterMode.PROPERTY, core_types.ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            core_types.Parameter(
                name="frames",
                type="list",
                tooltip="1-based frame numbers to extract (e.g. [1, 10, 50])",
                ui_options={"placeholder_text": "[1, 10, 50]"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_dir",
                default_value="",
                tooltip="Directory to save extracted frames. Defaults to a temp directory.",
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_prefix",
                default_value="frame",
                tooltip="Filename prefix for each saved frame (e.g. 'frame' → 'frame000001.png')",
            )
        )
        self.add_parameter(
            core_types.Parameter(
                name="frame_padding",
                type="int",
                default_value=6,
                tooltip="Zero-padding width for the frame number in the output filename",
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_format",
                default_value="png",
                tooltip="Image format for extracted frames: png, jpg, or webp",
            )
        )

    def _get_processing_description(self) -> str:
        return "extracting frames from video"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:
        # Single-frame extraction; timestamp is passed via kwargs
        ffmpeg_path, _ = self._get_ffmpeg_paths()
        timestamp: str = kwargs.get("timestamp", "00:00:00.000")
        return [
            ffmpeg_path,
            "-ss",
            timestamp,
            "-i",
            input_url,
            "-vframes",
            "1",
            "-y",
            output_path,
        ]

    def _resolve_video_url(self, raw_value: typing.Any) -> str | None:
        file_path = str(raw_value) if raw_value is not None else ""
        if not file_path:
            return None
        if file_path.startswith(("http://", "https://", "blob:", "data:")):
            return file_path
        try:
            result = griptape_nodes.GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=file_path)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                logger.info("Resolved video URL: %s → %s", file_path, result.url)
                return result.url
        except Exception:
            logger.warning("Failed to resolve video URL for: %s", file_path, exc_info=True)
        return None

    def _get_output_suffix(self, **kwargs) -> str:
        frame_number: int = kwargs.get("frame_number", 0)
        frame_padding: int = kwargs.get("frame_padding", 6)
        return f"_{str(frame_number).zfill(frame_padding)}"

    def _validate_custom_parameters(self) -> list[Exception] | None:
        frames = self.get_parameter_value("frames")
        if not frames:
            return [ValueError(f"{self.name}: 'frames' must be a non-empty list of frame numbers")]
        if not all(isinstance(f, int) and f >= 1 for f in frames):
            return [ValueError(f"{self.name}: all frame numbers must be integers >= 1")]
        return None

    def _detect_frame_rate(self, input_url: str) -> float:
        """Detect frame rate from the video; falls back to 30.0 on failure."""
        _, ffprobe_path = self._get_ffmpeg_paths()
        cmd = [
            ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            input_url,
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            streams_data = json.loads(result.stdout)
            fps_str = streams_data["streams"][0].get("r_frame_rate", "30/1")
            num, den = map(int, fps_str.split("/"))
            return num / den
        except Exception as e:
            self.append_value_to_parameter("logs", f"Warning: could not detect frame rate, defaulting to 30fps: {e}\n")
            return 30.0

    def _extract_all_frames(self, input_url: str) -> list[pathlib.Path]:
        """Extract each requested frame and return their saved paths."""
        self._validate_url_safety(input_url)

        frames: list[int] = self.get_parameter_value("frames")
        output_format: str = self.get_parameter_value("output_format") or "png"
        output_prefix: str = self.get_parameter_value("output_prefix") or "frame"
        frame_padding: int = self.get_parameter_value("frame_padding") or 6
        output_dir_str: str = self.get_parameter_value("output_dir") or ""

        output_dir = pathlib.Path(output_dir_str) if output_dir_str else self._output_file.build_file().parent
        output_dir.mkdir(parents=True, exist_ok=True)

        frame_rate = self._detect_frame_rate(input_url)
        self.append_value_to_parameter("logs", f"Detected frame rate: {frame_rate:.3f} fps\n")

        saved_paths: list[pathlib.Path] = []
        for frame_number in frames:
            zero_based = max(frame_number - 1, 0)
            timestamp = seconds_to_ts(zero_based / frame_rate)

            padded = str(frame_number).zfill(frame_padding)
            output_path = output_dir / f"{output_prefix}{padded}.{output_format}"

            cmd = self._build_ffmpeg_command(
                input_url,
                str(output_path),
                frame_rate,
                timestamp=timestamp,
                frame_number=frame_number,
                frame_padding=frame_padding,
            )

            self.append_value_to_parameter("logs", f"Extracting frame {frame_number} at {timestamp}...\n")
            self._run_ffmpeg_command(cmd, timeout=60)

            if not output_path.exists() or output_path.stat().st_size == 0:
                msg = f"FFmpeg did not produce output for frame {frame_number}"
                raise ValueError(msg)

            saved_paths.append(output_path)
            self.append_value_to_parameter("logs", f"Saved frame {frame_number} → {output_path}\n")

        return saved_paths

    def after_value_set(self, parameter: core_types.Parameter, value: typing.Any) -> None:
        """Automatically update video_player URL when input_video changes."""
        if parameter.name == "input_video":
            url = self._resolve_video_url(value)
            current = self.parameter_values.get("video_player", "")
            current_base = str(current).split("?")[0] if current else ""
            if url:
                new_base = url.split("?")[0]
                if new_base != current_base:
                    self.set_parameter_value("video_player", url)
            elif current_base:
                self.set_parameter_value("video_player", "")
        return super().after_value_set(parameter, value)

    def process(self) -> node_types.AsyncResult[None]:
        self._clear_execution_status()
        input_url, detected_format = self._get_video_input_data()
        self._log_format_detection(detected_format)
        self.append_value_to_parameter("logs", "[Started frame extraction..]\n")

        try:
            yield lambda: self._run_extraction(input_url)
            self.append_value_to_parameter("logs", "[Finished frame extraction.]\n")
            self._set_status_results(was_successful=True, result_details="Frames extracted successfully")
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error extracting frames: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            self._set_status_results(was_successful=False, result_details=f"Frame extraction failed: {error_message}")
            self._handle_failure_exception(ValueError(msg))

    def _run_extraction(self, input_url: str) -> None:
        saved_paths = self._extract_all_frames(input_url)
        self.parameter_output_values["output_paths"] = [str(p) for p in saved_paths]
