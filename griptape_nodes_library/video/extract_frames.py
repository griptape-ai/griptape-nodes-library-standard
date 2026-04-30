"""Video Frame Extractor node — frame-accurate scrubbing and extraction."""

import logging
import pathlib
import tempfile
import typing

from griptape.artifacts import ImageArtifact
from PIL import Image

import griptape_nodes.exe_types.core_types as core_types
import griptape_nodes.exe_types.node_types as node_types
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode import griptape_nodes
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.traits import options
from griptape_nodes.traits.file_system_picker import FileSystemPicker
import griptape_nodes.traits.widget as widget

from griptape_nodes_library.utils.video_utils import seconds_to_ts
from griptape_nodes_library.video.base_video_input_node import BaseVideoInputNode

logger = logging.getLogger(__name__)


def parse_frame_string(frame_str: str) -> list[int]:
    """Parse a comma-separated frame specification into a sorted, deduplicated list.

    Supports individual frames and inclusive ranges: ``"1,4,5-9,11"`` → ``[1, 4, 5, 6, 7, 8, 9, 11]``.
    Values < 1 are silently discarded.
    """
    if not frame_str or not frame_str.strip():
        return []
    result: set[int] = set()
    for token in frame_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            try:
                start, end = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if start > end or start < 1:
                continue
            result.update(range(start, end + 1))
        else:
            try:
                n = int(token)
            except ValueError:
                continue
            if n >= 1:
                result.add(n)
    return sorted(result)


class VideoFrameExtractor(BaseVideoInputNode):
    """Extract specific frames from a video and save them as images."""

    DEFAULT_OUTPUT_PREFIX = "frame"
    DEFAULT_FRAME_PADDING = 4
    DEFAULT_OUTPUT_FILE_FORMAT = "png"

    def __init__(self, name: str, metadata: dict[str, typing.Any] | None = None, **kwargs) -> None:
        node_metadata = {
            "category": "video",
            "description": "Import video, inspect frame-by-frame, and extract selected frames as images.",
        }
        if metadata:
            node_metadata.update(metadata)
        super().__init__(name=name, metadata=node_metadata, **kwargs)
        self.remove_parameter_element_by_name("video")
        # self.set_initial_node_size(width=980, height=480)
        self.hide_parameter_by_name("every_n")

    # ── Parameter setup ────────────────────────────────────────────────────────

    def _get_output_file_default_filename(self) -> str:
        return f"{self.DEFAULT_OUTPUT_PREFIX}.{'#' * self.DEFAULT_FRAME_PADDING}.{self.DEFAULT_OUTPUT_FILE_FORMAT}"

    def _get_processing_description(self) -> str:
        return "extracting frames from video"

    def _register_primary_output_parameter(self) -> None:
        self.add_parameter(
            core_types.Parameter(
                name="output_paths",
                type="list[str]",
                allowed_modes={core_types.ParameterMode.OUTPUT},
                tooltip="Paths to the extracted frame images",
                ui_options={"pulse_on_run": True},
            )
        )
        self.add_parameter(
            core_types.Parameter(
                name="extracted_frames",
                output_type="list[ImageArtifact]",
                tooltip="Extracted frames as ImageArtifacts.",
                allowed_modes={core_types.ParameterMode.OUTPUT},
                ui_options={"hide_property": True},
            )
        )

    def _setup_custom_parameters(self) -> None:
        self.add_parameter(
            core_types.Parameter(
                name="input_video",
                type="str",
                output_type="str",
                input_types=["VideoUrlArtifact", "VideoArtifact", "str"],
                default_value="",
                allowed_modes={core_types.ParameterMode.INPUT},
                tooltip="Video player for precise frame selection. Connect a video source here.",
                ui_options={"display_name": "Video Input"},
                traits={widget.Widget(name="VideoPlayerFrameSelector", library="Griptape Nodes Library")},
            )
        )
        self.add_parameter(
            core_types.Parameter(
                name="frame_selection_mode",
                type="str",
                default_value="list",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
                traits={options.Options(choices=["list", "every_Nth"])},
                tooltip=(
                    "Mode for selecting frames to extract. 'list' for specific frame numbers, "
                    "'every_Nth' for regular intervals. "
                    "For list, use comma-separated integers and ranges: `1,4,5-9,11`"
                ),
            )
        )
        self.add_parameter(
            core_types.Parameter(
                name="input_frame_numbers",
                output_type="str",
                tooltip=(
                    "Frames to extract, as a comma-separated list of numbers or ranges (e.g. 1,4,5-9,11).\n\n"
                    "In the video player above:\n"
                    "• Double-click the marker zone to add a single-frame marker\n"
                    "• Double-click and drag to create a range\n"
                    "• Drag an existing marker triangle to move it\n"
                    "• Alt+click a marker to remove it"
                ),
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            core_types.Parameter(
                name="every_n",
                type="int",
                default_value=1,
                tooltip="Interval for extracting frames when 'every_Nth' mode is selected.",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
            )
        )

        with core_types.ParameterGroup(
            name="settings", ui_options={"collapsed": True, "display_name": "Output format settings"}
        ) as settings_group:
            ParameterString(
                name="output_format",
                default_value=self.DEFAULT_OUTPUT_FILE_FORMAT,
                input_types=["List[str]", "str"],
                output_type="str",
                tooltip="Output image format for extracted frames.",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
                traits={options.Options(choices=["png", "jpg", "exr"])},
            )
            ParameterString(
                name="output_dir",
                default_value="",
                tooltip="Directory to save extracted frames. Defaults to the project output directory.",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
                traits={FileSystemPicker(allow_files=False, allow_directories=True, multiple=False)},
            )
            ParameterString(
                name="output_prefix",
                default_value=self.DEFAULT_OUTPUT_PREFIX,
                tooltip="Filename prefix for each saved frame (e.g. 'frame' → 'frame.0001.png')",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
            )
            core_types.Parameter(
                name="frame_padding",
                type="int",
                default_value=self.DEFAULT_FRAME_PADDING,
                tooltip="Zero-padding width for the frame number in the output filename.",
                allowed_modes={core_types.ParameterMode.INPUT, core_types.ParameterMode.PROPERTY},
            )
        self.add_node_element(settings_group)

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        if not str(self.get_parameter_value("input_video") or "").strip():
            exceptions.append(ValueError(f"{self.name}: 'input_video' is required"))
        custom = self._validate_custom_parameters()
        if custom:
            exceptions.extend(custom)
        return exceptions or None

    def _validate_custom_parameters(self) -> list[Exception] | None:
        mode = self.get_parameter_value("frame_selection_mode") or "list"
        if mode == "list":
            if not parse_frame_string(self.get_parameter_value("input_frame_numbers") or ""):
                return [ValueError(
                    f"{self.name}: 'input_frame_numbers' must specify at least one frame (e.g. '1,4,5-9')"
                )]
        elif mode == "every_Nth":
            if (self.get_parameter_value("every_n") or 1) < 1:
                return [ValueError(f"{self.name}: 'every_n' must be >= 1")]
        return None

    # ── URL resolution ─────────────────────────────────────────────────────────

    def _resolve_video_url(self, raw_value: typing.Any) -> str | None:
        """Resolve a video artifact or path to a browser-accessible presigned URL.

        Follows the AdjustMaskSize pattern — accesses .value from artifacts,
        then resolves via CreateStaticFileDownloadUrlFromPathRequest.
        """
        if not raw_value:
            return None
        file_path = raw_value.value if hasattr(raw_value, "value") else str(raw_value)
        if not file_path:
            return None
        if isinstance(file_path, str) and file_path.startswith(("http://", "https://")):
            return file_path
        try:
            result = griptape_nodes.GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=file_path)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            logger.warning("Failed to resolve video URL for: %s", file_path, exc_info=True)
        return None

    def _get_video_input_data(self) -> tuple[str, str]:
        """Return (resolved_url, format) for the video input.

        Format is always the user-chosen output format since the source format
        is irrelevant to the extraction process.
        """
        url = self.get_parameter_value("input_video") or ""
        if not url:
            raise ValueError(f"{self.name}: No video loaded in 'input_video'")
        if not str(url).startswith(("http://", "https://", "blob:", "data:")):
            resolved = self._resolve_video_url(url)
            if not resolved:
                raise ValueError(f"{self.name}: Could not resolve video URL: {url}")
            url = resolved
        output_format = self.get_parameter_value("output_format") or self.DEFAULT_OUTPUT_FILE_FORMAT
        return url, output_format

    # ── Output filename display ────────────────────────────────────────────────

    def _build_output_filename_pattern(self) -> str:
        prefix = self.get_parameter_value("output_prefix") or self.DEFAULT_OUTPUT_PREFIX
        padding = self.get_parameter_value("frame_padding") or self.DEFAULT_FRAME_PADDING
        fmt = self.get_parameter_value("output_format") or self.DEFAULT_OUTPUT_FILE_FORMAT
        return f"{prefix}.{'#' * padding}.{fmt}"

    def _sync_output_file_display(self) -> None:
        filename = self._build_output_filename_pattern()
        self.set_parameter_value("output_file", filename)
        self.publish_update_to_parameter("output_file", filename)

    # ── Lifecycle hooks ────────────────────────────────────────────────────────

    def after_value_set(self, parameter: core_types.Parameter, value: typing.Any) -> None:
        if parameter.name == "input_video":
            url = self._resolve_video_url(value)
            current = self.parameter_values.get("input_video", "")
            current_base = str(current).split("?")[0] if current else ""
            if url:
                if url.split("?")[0] != current_base:
                    self.set_parameter_value("input_video", url)
            elif current_base:
                self.set_parameter_value("input_video", "")

        elif parameter.name == "frame_selection_mode":
            mode = self.parameter_values.get("frame_selection_mode")
            if mode == "every_Nth":
                self.hide_parameter_by_name("input_frame_numbers")
                self.show_parameter_by_name("every_n")
            elif mode == "list":
                self.show_parameter_by_name("input_frame_numbers")
                self.hide_parameter_by_name("every_n")

        elif parameter.name in ("output_format", "output_prefix", "frame_padding"):
            self._sync_output_file_display()

        return super().after_value_set(parameter, value)

    # ── Execution ──────────────────────────────────────────────────────────────

    def process(self) -> node_types.AsyncResult[None]:
        self._clear_execution_status()
        input_url, output_format = self._get_video_input_data()
        self._log_format_detection(output_format)
        self.append_value_to_parameter("logs", "[Started frame extraction]\n")
        try:
            yield lambda: self._run_extraction(input_url, output_format)
            self.append_value_to_parameter("logs", "[Finished frame extraction]\n")
            self._set_status_results(was_successful=True, result_details="Frames extracted successfully")
        except Exception as e:
            msg = f"{self.name}: Error extracting frames: {e}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(ValueError(msg))

    def _run_extraction(self, input_url: str, output_format: str) -> None:
        saved_paths = self._extract_all_frames(input_url, output_format)

        self.parameter_output_values["output_paths"] = [str(p) for p in saved_paths]
        self.parameter_output_values["extracted_frames"] = [
            self._path_to_image_artifact(p, output_format) for p in saved_paths
        ]

    def _extract_all_frames(self, input_url: str, output_format: str) -> list[pathlib.Path]:
        """Extract each requested frame via ffmpeg and return the saved paths."""
        self._validate_url_safety(input_url)

        output_prefix = self.get_parameter_value("output_prefix") or self.DEFAULT_OUTPUT_PREFIX
        frame_padding = self.get_parameter_value("frame_padding") or self.DEFAULT_FRAME_PADDING
        output_dir_str = self.get_parameter_value("output_dir") or ""

        output_dir = pathlib.Path(output_dir_str) if output_dir_str else pathlib.Path(tempfile.mkdtemp())
        output_dir.mkdir(parents=True, exist_ok=True)

        ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        frame_rate, _, vid_duration = self._detect_video_properties(input_url, ffprobe_path)
        self.append_value_to_parameter("logs", f"Detected frame rate: {frame_rate:.3f} fps\n")

        frames = self._build_frame_list(frame_rate, vid_duration)
        self.append_value_to_parameter("logs", f"Extracting {len(frames)} frame(s)...\n")

        saved_paths: list[pathlib.Path] = []
        for frame_number in frames:
            timestamp = seconds_to_ts(max(frame_number - 1, 0) / frame_rate)
            output_path = output_dir / f"{output_prefix}{str(frame_number).zfill(frame_padding)}.{output_format}"

            cmd = self._build_ffmpeg_command(ffmpeg_path, input_url, str(output_path), timestamp)
            self.append_value_to_parameter("logs", f"Extracting frame {frame_number} at {timestamp}...\n")
            self._run_ffmpeg_command(cmd, timeout=60)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise ValueError(f"FFmpeg did not produce output for frame {frame_number}")

            saved_paths.append(output_path)
            self.append_value_to_parameter("logs", f"Saved frame {frame_number} → {output_path}\n")

        return saved_paths

    def _build_frame_list(self, frame_rate: float, vid_duration: float) -> list[int]:
        """Return the ordered list of 1-based frame numbers to extract."""
        mode = self.get_parameter_value("frame_selection_mode") or "list"
        if mode == "every_Nth":
            every_n = max(1, self.get_parameter_value("every_n") or 1)
            total_frames = max(1, round(vid_duration * frame_rate))
            return list(range(1, total_frames + 1, every_n))
        frame_str = self.get_parameter_value("input_frame_numbers") or ""
        return parse_frame_string(frame_str)

    def _build_ffmpeg_command(
        self, ffmpeg_path: str, input_url: str, output_path: str, timestamp: str
    ) -> list[str]:
        """Build the ffmpeg command to extract a single frame at the given timestamp."""
        return [ffmpeg_path, "-ss", timestamp, "-i", input_url, "-vframes", "1", "-y", output_path]

    @staticmethod
    def _path_to_image_artifact(path: pathlib.Path, fmt: str) -> ImageArtifact:
        """Read a saved frame file and wrap it as an ImageArtifact."""
        with Image.open(path) as img:
            w, h = img.size
        return ImageArtifact(value=path.read_bytes(), format=fmt, width=w, height=h)
