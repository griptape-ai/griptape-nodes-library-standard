from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.common.macro_parser import MacroSyntaxError, ParsedMacro
from griptape_nodes.exe_types.core_types import (
    BadgeData,
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode, NodeResolutionState, SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileDestination, FileDestinationProvider, FileLoadError
from griptape_nodes.files.project_file import SITUATION_TO_FILE_POLICY, ProjectFileDestination
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.os_events import ExistingFilePolicy
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultFailure,
    GetSituationRequest,
    GetSituationResultSuccess,
    MacroPath,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

logger = logging.getLogger("griptape_nodes")

__all__ = ["ExtractFrames"]

# Project situation for writes: versioned run folder ({_index}) + per-item names (e.g. frame_number).
OUTPUT_FILE_SITUATION = "save_node_output_group"

# Constants
EXTRACTION_MODES = ["All", "Step", "List"]
FRAME_NUMBERING_OPTIONS = ["Keep original frame numbers", "Renumber sequentially"]
FORMAT_OPTIONS = ["jpg", "png", "webp"]
# User path (extension comes from format): hashes set frame padding (#### -> four digits).
DEFAULT_OUTPUT_FILE_SPEC = "extracted_frames/frames.####"
DEFAULT_PAD_DIGITS = 4
DEFAULT_BATCH_INDEX_PAD_DIGITS = 3
DEFAULT_STEP = 2
MIN_FRAME_NUMBER = 0
DEFAULT_END_FRAME = 100
FRAME_INDEX_MAX_UI = 1000
RANGE_PARTS_LENGTH = 2
# {_index…}/ is the versioned run folder (OUTPUT_FILE_SITUATION / batch _index). Other {_index} tokens → frame.
_LEGACY_INDEX_FRAME_PATTERN = re.compile(r"\{_index(?:\?\:(\d+)|\:(\d+))?\}")
_BATCH_INDEX_BEFORE_SLASH = re.compile(r"\{_index(?:\?\:\d+|\:\d+)?\}/")  # run folder token before next path segment
_BATCH_INDEX_PAD_IN_FOLDER_RE = re.compile(r"\{_index(?:\?\:(\d+)|\:(\d+))?\}/")
_VERSIONED_OUTPUT_SUBDIR_RE = re.compile(r"\{outputs\}/(?:.*/)?([^/{]+)\.\{_index(?:\?\:\d+|\:\d+)?\}/")


def _batch_index_macro_segment(pad_width: int) -> str:
    """Run / batch folder segment e.g. {_index?:03} — aligns with per-save _index versioning."""
    w = max(1, min(pad_width, 99))
    token = f"0{w}" if w < 10 else str(w)
    return f"{{_index?:{token}}}"


def _frame_number_macro_segment(pad_width: int) -> str:
    """Per-frame sequence in the filename e.g. {frame_number?:04} (from #### in friendly paths)."""
    w = max(1, min(pad_width, 99))
    token = f"0{w}" if w < 10 else str(w)
    return f"{{frame_number?:{token}}}"


class ExtractFrames(SuccessFailureNode):
    """Extract frames from a video to image files on disk.

    Inputs:
        - video (VideoUrlArtifact): Input video to extract frames from (required)
        - extraction_frame_range (group): start_frame and end_frame with optional reset-to-video buttons
        - extraction_mode (str): All, Step, or List — how frames are chosen between start_frame and end_frame
        - start_frame (int): First frame index (inclusive) of the extraction window
        - end_frame (int): Last frame index (inclusive) of the extraction window
        - frame_list (str): Comma/space-separated frame numbers or ranges (e.g., "1, 2, 3, 5-8, 14 27")
        - step (int): Extract every Nth frame (default: 2)
        - output_file (str): Human path like extracted_frames/frames.#### (each # is one digit of zero-padding).
          Writes under a versioned folder matching save _index semantics: {outputs}/extracted_frames.{_index}/frames.{frame_number}.jpg.
          Extension comes from format. Macros: {_index} = run/batch folder, {frame_number} = per-frame sequence.
        - format (str): File format written to disk (jpg/png/webp); overrides the extension in output_file for the actual files
        - overwrite_files (bool): If true, clears and reuses the latest numbered run folder (e.g. .../extracted_frames.003);
          if false, creates the next folder (004, ...).
        - frame_numbering (str): "Keep original frame numbers" or "Renumber sequentially" (how frame_number is chosen)
        - remove_previous_frames (bool): Remove previously generated frames before extracting

    Outputs:
        - frame_paths (list[str]): Absolute filesystem paths to each extracted frame file
        - was_successful (bool): Whether the extraction succeeded
        - result_details (str): Details about the extraction result or error
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_synced_video_identity: str | None = None

        # INPUTS / PROPERTIES

        # Video input parameter
        self.add_parameter(
            Parameter(
                name="video",
                input_types=["VideoUrlArtifact"],
                type="VideoUrlArtifact",
                tooltip="Input video to extract frames from",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "input video"},
            )
        )

        # Video information parameters
        with ParameterGroup(name="Video Information") as video_info_group:
            ParameterString(
                name="frame_rate",
                default_value="",
                tooltip="Video frame rate (FPS) - set automatically based on connected video",
                allow_input=False,
                allow_property=False,
                placeholder_text="FPS will appear here...",
            )

            ParameterInt(
                name="frame_count",
                default_value=0,
                tooltip="Total number of frames in the video - set automatically based on connected video",
                allow_input=False,
                allow_property=False,
            )

            ParameterButton(
                name="refresh_video_info",
                label="Refresh Video Info",
                variant="secondary",
                icon="refresh-cw",
                on_click=self._refresh_video_info,
            )

        self.add_node_element(video_info_group)

        with ParameterGroup(name="extraction_frame_range") as extraction_frame_range_group:
            ParameterInt(
                name="start_frame",
                default_value=MIN_FRAME_NUMBER,
                tooltip=(
                    "Inclusive start frame index for the extraction window. extraction_mode controls how frames "
                    "are chosen only within this window and end_frame. Use the reset control to set to the first "
                    "frame of the video."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=MIN_FRAME_NUMBER,
                max_val=FRAME_INDEX_MAX_UI,
                step=1,
                traits={
                    Button(
                        icon="rotate-ccw",
                        size="icon",
                        tooltip="Set start frame to the first frame of the video (0)",
                        on_click=self._reset_start_frame_to_video,
                    ),
                },
            )
            ParameterInt(
                name="end_frame",
                default_value=DEFAULT_END_FRAME,
                tooltip=(
                    "Inclusive end frame index for the extraction window. extraction_mode controls how frames "
                    "are chosen only within this window and start_frame. Use the reset control to set to the last "
                    "frame index from the video."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=MIN_FRAME_NUMBER,
                max_val=FRAME_INDEX_MAX_UI,
                step=1,
                traits={
                    Button(
                        icon="rotate-ccw",
                        size="icon",
                        tooltip="Set end frame to the last frame index of the video (requires readable frame count)",
                        on_click=self._reset_end_frame_to_video,
                    ),
                },
            )

        self.add_node_element(extraction_frame_range_group)

        with ParameterGroup(name="Extraction Options") as extraction_options_group:
            ParameterString(
                name="extraction_mode",
                default_value=EXTRACTION_MODES[0],
                tooltip="How frames are picked between start frame and end frame (see help badge for All, Step, and List).",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=EXTRACTION_MODES)},
                badge=BadgeData(
                    variant="help",
                    title="Extraction modes",
                    message=(
                        "Determines how frames are picked between `start frame` and `end frame`.\n\n"
                        "**All** — Saves every frame.\n\n"
                        "**Step** — Saves every Nth frame in that range. For example, step **3** means “every 3rd "
                        "frame”: keep one frame, skip two, repeat.\n\n"
                        "**List** — Saves only the frame numbers you enter. Mix single frames and ranges; separate "
                        "with commas or spaces, e.g. `1, 3, 4-7, 22`."
                    ),
                ),
            )

            # Frame list string field (shown when mode is "List")
            ParameterString(
                name="frame_list",
                default_value="",
                tooltip='Comma or space-separated frame numbers or ranges (e.g., "1, 2, 3, 5-8, 14 27")',
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="1, 2, 3, 5-8, 14 27",
                hide=True,
            )

            # Step integer field (shown when mode is "Step")
            ParameterInt(
                name="step",
                default_value=DEFAULT_STEP,
                tooltip="Extract every Nth frame (e.g., 2 = every 2nd frame)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=True,
            )

            # Frame numbering option
            ParameterString(
                name="frame_numbering",
                default_value=FRAME_NUMBERING_OPTIONS[0],
                tooltip="Keep original frame numbers from video or renumber sequentially starting from 1",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=FRAME_NUMBERING_OPTIONS)},
                badge=BadgeData(
                    variant="help",
                    title="Frame numbering",
                    message=(
                        "Determines how the frame numbers will be applied to the output files.\n\n"
                        "**Keep original** — Uses the frame numbers from the video. For example, if extracting every 4 frames, files will be saved with `filename.0001.jpg`, `filename.0005.jpg`, etc.\n\n"
                        "**Renumber sequentially** — Starts from 1 and increments by 1 for each frame. For example, if extracting every 4 frames, files will be saved with `filename.0001.jpg`, `filename.0002.jpg`, etc."
                    ),
                ),
            )
        self.add_node_element(extraction_options_group)

        with ParameterGroup(name="Overwrite Options") as overwrite_options_group:
            # Overwrite files option
            ParameterBool(
                name="overwrite_files",
                default_value=True,
                tooltip=(
                    "When on, reuses the highest existing numbered output folder after clearing it (same batch path as "
                    "last run). When off, writes into a new numbered folder (extracted_frames.001, .002, …)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                badge=BadgeData(
                    variant="help",
                    title="Run folder vs new folder",
                    message=(
                        "Outputs go under a versioned subfolder such as `extracted_frames.001`.\n\n"
                        "**On** — Deletes that folder’s contents from the **latest** number in use, then writes there again.\n\n"
                        "**Off** — Picks the **next** number so each run keeps earlier batches.\n\n"
                        "FFmpeg still needs distinct filenames inside the folder; see also `remove_previous_frames`."
                    ),
                ),
            )

            # Remove previous frames option
            ParameterBool(
                name="remove_previous_frames",
                default_value=False,
                tooltip=(
                    "Before saving, delete images in this run’s output folder that match this export’s frame filename "
                    "pattern and format (jpg/png/webp)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                badge=BadgeData(
                    variant="help",
                    title="Remove old stills in this folder first",
                    message=(
                        "Targets only the **current run folder** (the numbered directory like `extracted_frames.003`), "
                        "not earlier numbered folders.\n\n"
                        "Deletes files matching the same stem pattern and format, e.g. `frames.0001.jpg` …\n\n"
                        "Use when iterating on one batch without touching older runs."
                    ),
                ),
            )

        self.add_node_element(overwrite_options_group)

        self.add_parameter(
            ParameterString(
                name="format",
                default_value=FORMAT_OPTIONS[0],
                tooltip=(
                    "Image format written to disk. The extension here is used for output files even if output_file shows "
                    "a different suffix (e.g. .jpg in the path is only a hint)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=FORMAT_OPTIONS)},
                badge=BadgeData(
                    variant="help",
                    title="Image format",
                    message="The extension used for output files even if `output_file` shows a different suffix (e.g. .jpg in the path is only a hint).",
                ),
            )
        )
        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename=DEFAULT_OUTPUT_FILE_SPEC,
            situation=OUTPUT_FILE_SITUATION,
            ui_options={
                "tooltip": (
                    "Folder and frame pattern under project outputs, e.g. extracted_frames/frames.#### "
                    "(one # per digit). Frames are written to extracted_frames.001/…, .002/… unless Overwrite reuses "
                    "the latest. Expert macros: {_index} = run folder, {frame_number} = frame sequence (legacy {_index} in filenames is normalized to frame_number)."
                ),
            },
        )
        self._output_file.add_parameter()
        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="frame_paths",
                output_type="list",
                type="list",
                tooltip="Absolute paths to each extracted frame file",
                allowed_modes={ParameterMode.OUTPUT},
                default_value=[],
            )
        )

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the frame extraction result or any errors",
            result_details_placeholder="Extraction status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name == "video":
            # Target value may not be committed yet; read from the upstream port like load_video.
            incoming_video = source_node.get_parameter_value(source_parameter.name)
            self._sync_frame_range_after_video_connected(incoming_video)
        super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes to update dependent parameters."""

        if parameter.name == "extraction_mode":
            if value == "All":
                self.hide_parameter_by_name("frame_list")
                self.hide_parameter_by_name("step")
            elif value == "List":
                self.show_parameter_by_name("frame_list")
                self.hide_parameter_by_name("step")
            elif value == "Step":
                self.hide_parameter_by_name("frame_list")
                self.show_parameter_by_name("step")
            else:
                self.hide_parameter_by_name("frame_list")
                self.hide_parameter_by_name("step")

        # Refresh fps / frame_count when the video source changes (not on every re-application at run).
        # start/end are updated when the video input is wired (see after_incoming_connection), not here.
        if parameter.name == "video" and self.state != NodeResolutionState.RESOLVING:
            new_identity = self._stable_video_identity(value)
            if new_identity != self._last_synced_video_identity:
                self._last_synced_video_identity = new_identity
                self._update_video_info(value)

        super().after_value_set(parameter, value)

    def _sync_frame_range_after_video_connected(self, video: Any | None = None) -> None:
        """Set start/end to the full clip and refresh video metadata after input video is wired."""
        if video is None:
            video = self.get_parameter_value("video")
        if not video:
            return
        self._last_synced_video_identity = self._stable_video_identity(video)
        self._update_video_info(video)
        fc_raw = self.get_parameter_value("frame_count")
        try:
            fc = int(fc_raw) if fc_raw is not None else 0
        except (TypeError, ValueError):
            return
        if fc <= 0:
            return
        last = fc - 1
        self.set_parameter_value("start_frame", MIN_FRAME_NUMBER)
        self.set_parameter_value("end_frame", last)

    def _extract_video_url(self, video_input: Any) -> str | None:
        """Extract video URL from video input."""
        if isinstance(video_input, VideoUrlArtifact):
            return video_input.value
        return str(video_input) if video_input else None

    def _stable_video_identity(self, video_input: Any) -> str | None:
        """Comparable id for the same on-disk / URL source (macro vs absolute path).

        Avoids treating a re-resolved path on each run as a new video.
        """
        url = self._extract_video_url(video_input)
        if not url or not str(url).strip():
            return None
        s = str(url).strip()
        if s.startswith(("http://", "https://", "file://")):
            return s
        try:
            return str(File(s).resolve())
        except (ValueError, FileLoadError, OSError):
            return s

    def _resolve_video_input_for_local_tools(self, location: str) -> str:
        """Resolve project macro paths (e.g. {inputs}/file.mp4) for ffmpeg/ffprobe.

        Leaves http(s) and file:// URLs unchanged.
        """
        if not location or not str(location).strip():
            return location
        s = str(location).strip()
        if s.startswith(("http://", "https://", "file://")):
            return s
        try:
            return File(s).resolve()
        except FileLoadError as e:
            msg = f"{self.name}: Could not resolve video path for local processing: {e.result_details}"
            raise ValueError(msg) from e

    def _get_video_fps(self, video_url: str) -> float | None:
        """Get video frame rate (FPS) using ffprobe."""
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                video_url,
            ]

            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            stream_data = json.loads(result.stdout)
            streams = stream_data.get("streams", [])
            if not streams:
                return None

            video_stream = streams[0]
            r_frame_rate_str = video_stream.get("r_frame_rate", "30/1")

            if "/" in r_frame_rate_str:
                num, den = map(int, r_frame_rate_str.split("/"))
                return num / den if den != 0 else None
            return float(r_frame_rate_str)

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ):
            return None

    def _update_video_info(self, video: Any | None = None) -> None:
        """Update frame_rate and frame_count from the connected video (read-only for the window)."""
        if video is None:
            video = self.get_parameter_value("video")
        if not video:
            self.set_parameter_value("frame_rate", "")
            self.set_parameter_value("frame_count", 0)
            return

        video_url = self._extract_video_url(video)
        if not video_url:
            self.set_parameter_value("frame_rate", "")
            self.set_parameter_value("frame_count", 0)
            return

        try:
            video_url = self._resolve_video_input_for_local_tools(video_url)
        except ValueError:
            self.set_parameter_value("frame_rate", "")
            self.set_parameter_value("frame_count", 0)
            return

        # Get video information
        frame_count = self._get_video_frame_count(video_url)
        fps = self._get_video_fps(video_url)

        # Update frame_rate
        if fps is not None:
            self.set_parameter_value("frame_rate", f"{fps:.2f}")
        else:
            self.set_parameter_value("frame_rate", "")

        if frame_count is not None:
            self.set_parameter_value("frame_count", frame_count)
        else:
            self.set_parameter_value("frame_count", 0)

    def _refresh_video_info(self, _button: Button, _details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        """Refresh video information when button is clicked."""
        self._update_video_info()
        return NodeMessageResult(
            success=True,
            details="Video information refreshed",
            response=None,
            altered_workflow_state=False,
        )

    def _last_frame_index_from_current_video(self) -> int | None:
        video = self.get_parameter_value("video")
        if not video:
            return None
        video_url = self._extract_video_url(video)
        if not video_url:
            return None
        try:
            video_url = self._resolve_video_input_for_local_tools(video_url)
        except ValueError:
            return None
        frame_count = self._get_video_frame_count(video_url)
        if frame_count is None or frame_count <= 0:
            return None
        return frame_count - 1

    def _reset_start_frame_to_video(self, _button: Button, _details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        self.set_parameter_value("start_frame", MIN_FRAME_NUMBER)
        return NodeMessageResult(
            success=True,
            details="start_frame set to first frame (0)",
            response=None,
            altered_workflow_state=True,
        )

    def _reset_end_frame_to_video(self, _button: Button, _details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        last = self._last_frame_index_from_current_video()
        if last is None:
            return NodeMessageResult(
                success=False,
                details=(
                    f"{self.name}: Could not determine last frame index. Connect a video and use "
                    "Refresh Video Info if needed."
                ),
                response=None,
                altered_workflow_state=False,
            )
        self.set_parameter_value("end_frame", last)
        return NodeMessageResult(
            success=True,
            details=f"end_frame set to last frame index ({last})",
            response=None,
            altered_workflow_state=True,
        )

    def _get_video_frame_count(self, video_url: str) -> int | None:
        """Extract video frame count using ffprobe.

        Tries multiple methods in order of accuracy:
        1. nb_frames from stream (most accurate)
        2. Count frames using -count_frames (accurate but slower)
        3. Calculate from duration * frame_rate (least accurate, fallback)
        """
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()

            # First try: Get nb_frames from stream metadata
            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-select_streams",
                "v:0",
                video_url,
            ]

            result = subprocess.run(  # noqa: S603
                cmd, capture_output=True, text=True, check=True, timeout=30
            )

            stream_data = json.loads(result.stdout)
            streams = stream_data.get("streams", [])
            if not streams:
                return None

            video_stream = streams[0]

            # Try to get nb_frames directly (most accurate)
            nb_frames_str = video_stream.get("nb_frames")
            if nb_frames_str:
                try:
                    frame_count = int(nb_frames_str)
                    logger.debug("%s got frame count from nb_frames: %d", self.name, frame_count)
                    return frame_count
                except (ValueError, TypeError):
                    pass

            # Second try: Count frames explicitly (accurate but slower)
            try:
                count_cmd = [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=nokey=1:noprint_wrapper=1",
                    video_url,
                ]
                count_result = subprocess.run(  # noqa: S603
                    count_cmd, capture_output=True, text=True, check=True, timeout=60
                )
                frame_count_str = count_result.stdout.strip()
                if frame_count_str and frame_count_str.isdigit():
                    frame_count = int(frame_count_str)
                    logger.debug("%s got frame count from -count_frames: %d", self.name, frame_count)
                    return frame_count
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError, TypeError):
                logger.debug("%s -count_frames method failed, trying duration calculation", self.name)

            # Third try: Calculate from duration and frame rate (fallback, less accurate)
            duration_str = video_stream.get("duration")
            r_frame_rate_str = video_stream.get("r_frame_rate", "30/1")

            if duration_str and r_frame_rate_str:
                try:
                    duration = float(duration_str)
                    if "/" in r_frame_rate_str:
                        num, den = map(int, r_frame_rate_str.split("/"))
                        frame_rate = num / den if den != 0 else 30.0
                    else:
                        frame_rate = float(r_frame_rate_str)

                    frame_count = int(duration * frame_rate)
                    logger.warning(
                        "%s calculated frame count from duration*FPS: %d (may be inaccurate). Duration: %.2f, FPS: %.2f",
                        self.name,
                        frame_count,
                        duration,
                        frame_rate,
                    )
                    return frame_count
                except (ValueError, TypeError, ZeroDivisionError):
                    pass

            return None

        except (
            subprocess.TimeoutExpired,
            subprocess.CalledProcessError,
            json.JSONDecodeError,
            ValueError,
            KeyError,
        ) as e:
            logger.debug("%s ffprobe failed to extract frame count: %s", self.name, e)
            return None

    def process(self) -> AsyncResult[None]:
        """Process method to extract frames from video."""
        self._clear_execution_status()
        logger.info("%s starting frame extraction", self.name)

        # Validate video input
        video = self.get_parameter_value("video")
        if not video:
            self._set_safe_defaults()
            error_msg = f"{self.name} requires an input video for frame extraction."
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s validation failed: no video input", self.name)
            return

        video_url = self._extract_video_url(video)
        if not video_url:
            self._set_safe_defaults()
            error_msg = f"{self.name} could not extract video URL from input."
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s validation failed: invalid video URL", self.name)
            return

        try:
            video_url = self._resolve_video_input_for_local_tools(video_url)
        except ValueError as e:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            logger.error("%s validation failed: %s", self.name, e)
            return

        # Get and validate parameters
        params = self._get_and_validate_parameters(video_url)
        if params is None:
            return

        # Extract frames asynchronously (subprocess calls are blocking)
        try:
            yield lambda: self._perform_extraction_async(video_url, params)
        except Exception as e:
            self._set_safe_defaults()
            error_msg = f"{self.name} failed to extract frames: {e}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s extraction failed: %s", self.name, e)
            self._handle_failure_exception(RuntimeError(error_msg))

    def _perform_extraction_async(self, video_url: str, params: dict[str, Any]) -> None:
        """Perform the actual frame extraction (blocking operation)."""
        run_version, output_template = self._prepare_output_run(params["overwrite_files"])
        if params["remove_previous_frames"]:
            first_num = self._first_output_frame_number(params["frames_to_extract"], params["frame_numbering"])
            first_path = Path(
                self._resolve_frame_output_path(first_num, params["format_type"], run_version, output_template)
            )
            output_dir = first_path.parent
            stem_hint = self._output_stem_glob_hint()
            self._remove_previous_frames_matching_stem(output_dir, params["format_type"], stem_hint)

        frame_paths = self._extract_frames(
            video_url=video_url,
            frame_numbers=params["frames_to_extract"],
            format_type=params["format_type"],
            frame_numbering=params["frame_numbering"],
            overwrite_files=params["overwrite_files"],
            run_version=run_version,
            output_template=output_template,
        )

        # Set results after extraction completes
        self.parameter_output_values["frame_paths"] = frame_paths
        out_spec = self.get_parameter_value("output_file")
        dest_summary = out_spec if isinstance(out_spec, str) and out_spec.strip() else DEFAULT_OUTPUT_FILE_SPEC
        result_details = f"Successfully extracted {len(frame_paths)} frames ({dest_summary})"
        self._set_status_results(was_successful=True, result_details=result_details)
        logger.info("%s extracted %d frames successfully", self.name, len(frame_paths))

    def _get_and_validate_parameters(self, video_url: str | None = None) -> dict[str, Any] | None:
        """Get and validate all parameters."""
        extraction_mode = self.get_parameter_value("extraction_mode") or EXTRACTION_MODES[0]

        start_frame_raw = self.get_parameter_value("start_frame")
        end_frame_raw = self.get_parameter_value("end_frame")
        start_frame = int(MIN_FRAME_NUMBER if start_frame_raw is None else start_frame_raw)
        end_frame = int(DEFAULT_END_FRAME if end_frame_raw is None else end_frame_raw)
        logger.debug("%s frame window: start=%d end=%d", self.name, start_frame, end_frame)

        frame_list_str = self.get_parameter_value("frame_list") or ""
        step = self.get_parameter_value("step") or DEFAULT_STEP
        format_type = self.get_parameter_value("format") or FORMAT_OPTIONS[0]
        overwrite_files = self.get_parameter_value("overwrite_files") or False
        frame_numbering = self.get_parameter_value("frame_numbering") or FRAME_NUMBERING_OPTIONS[0]
        remove_previous_frames = self.get_parameter_value("remove_previous_frames") or False

        if start_frame < 0 or end_frame < start_frame:
            self._set_safe_defaults()
            error_msg = (
                f"{self.name}: Invalid frame window: start_frame must be >= 0 and end_frame must be >= start_frame"
            )
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s validation failed: invalid start_frame/end_frame", self.name)
            return None

        # Clamp end_frame to video length when frame count is known
        if video_url:
            video_frame_count = self._get_video_frame_count(video_url)
            if video_frame_count is not None:
                max_frame = video_frame_count - 1
                if end_frame > max_frame:
                    logger.warning(
                        "%s end_frame (%d) exceeds maximum frame index %d for this video; clamping to %d",
                        self.name,
                        end_frame,
                        max_frame,
                        max_frame,
                    )
                    end_frame = max_frame
                    if start_frame > end_frame:
                        start_frame = 0

        logger.info(
            "%s extracting frames in window [%d, %d] with mode '%s'",
            self.name,
            start_frame,
            end_frame,
            extraction_mode,
        )

        # Determine which frames to extract
        frames_to_extract = self._determine_frames_to_extract(
            extraction_mode, frame_list_str, step, start_frame, end_frame
        )

        if not frames_to_extract:
            self._set_safe_defaults()
            error_msg = f"{self.name}: No frames to extract based on current settings"
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s validation failed: no frames to extract", self.name)
            return None

        # Validate all frames are within the range
        invalid_frames = [f for f in frames_to_extract if f < start_frame or f > end_frame]
        if invalid_frames:
            logger.warning(
                "%s found frames outside range [%d, %d]: %s, filtering them out",
                self.name,
                start_frame,
                end_frame,
                invalid_frames[:10],
            )
            frames_to_extract = [f for f in frames_to_extract if start_frame <= f <= end_frame]

        max_frames_to_show = 10
        frames_preview = (
            frames_to_extract[:max_frames_to_show] if len(frames_to_extract) > max_frames_to_show else frames_to_extract
        )
        logger.info("%s will extract %d frames: %s", self.name, len(frames_to_extract), frames_preview)

        return {
            "extraction_mode": extraction_mode,
            "frames_to_extract": frames_to_extract,
            "format_type": format_type,
            "overwrite_files": overwrite_files,
            "frame_numbering": frame_numbering,
            "remove_previous_frames": remove_previous_frames,
        }

    def _determine_frames_to_extract(
        self, extraction_mode: str, frame_list_str: str, step: int, start_frame: int, end_frame: int
    ) -> list[int]:
        """Determine which frame numbers to extract based on mode."""
        if extraction_mode == "All":
            return list(range(start_frame, end_frame + 1))

        if extraction_mode == "List":
            if not frame_list_str:
                return []

            parsed_frames = self._parse_frame_list(frame_list_str)
            # Filter to only include frames within range
            filtered_frames = [f for f in parsed_frames if start_frame <= f <= end_frame]
            return sorted(set(filtered_frames))

        if extraction_mode == "Step":
            frames = []
            current = start_frame
            while current <= end_frame:
                frames.append(current)
                current += step
            return frames

        return []

    def _parse_frame_list(self, frame_list_str: str) -> list[int]:
        """Parse frame list string with ranges and separators.

        Handles formats like: "1, 2, 3, 5-8, 14 27"
        """
        frames = []
        # Replace spaces with commas for easier parsing
        normalized = re.sub(r"\s+", ",", frame_list_str.strip())
        parts = [p.strip() for p in normalized.split(",") if p.strip()]

        for part in parts:
            if "-" in part:
                # Handle range like "5-8"
                range_parts = part.split("-", 1)
                if len(range_parts) == RANGE_PARTS_LENGTH:
                    try:
                        start = int(range_parts[0].strip())
                        end = int(range_parts[1].strip())
                        frames.extend(range(start, end + 1))
                    except (ValueError, TypeError):
                        logger.warning("%s could not parse frame range: %s", self.name, part)
                        continue
            else:
                # Handle single frame number
                try:
                    frame_num = int(part.strip())
                    frames.append(frame_num)
                except (ValueError, TypeError):
                    logger.warning("%s could not parse frame number: %s", self.name, part)
                    continue

        return frames

    def _first_output_frame_number(self, frame_numbers: list[int], frame_numbering: str) -> int:
        if not frame_numbers:
            return MIN_FRAME_NUMBER
        if frame_numbering == FRAME_NUMBERING_OPTIONS[1]:
            return 1
        return frame_numbers[0]

    def _parse_user_output_path(self, spec: str) -> tuple[str, str, int]:
        """Parse paths like extracted_frames/frames.#### into subdirs, stem, digit width.

        Avoids Path.stem for the basename: a lone .#### is treated as extension by pathlib and would drop padding.
        """
        s = spec.strip().replace("\\", "/")
        parent, sep, name = s.rpartition("/")
        sub_dirs = parent if sep else ""
        m_hash = re.search(r"#+", name)
        if m_hash:
            pad = len(m_hash.group(0))
            stem_base = name[: m_hash.start()].rstrip("._") or "frame"
        else:
            pad = DEFAULT_PAD_DIGITS
            stem_base = Path(name).stem.strip("._") or "frame"
        return sub_dirs, stem_base, pad

    def _infer_file_name_base_for_expert_macro(self) -> str:
        """file_name_base for templates that reference it; derive from literal prefix before first brace."""
        raw = self.get_parameter_value("output_file")
        if not isinstance(raw, str) or not raw.strip():
            return "frame"
        lit = raw.split("{", 1)[0]
        leaf = Path(lit.rstrip("/")).name
        stem = Path(leaf).stem if leaf else Path(lit).stem
        return re.sub(r"#+", "", stem).strip("._") or "frame"

    def _normalize_output_macro_template(self, template: str) -> str:
        """Map legacy filename {_index} to {frame_number}; preserve {_index…}/ run-folder tokens."""
        protected: list[str] = []

        def shield(m: re.Match[str]) -> str:
            protected.append(m.group(0))
            return f"__GTN_BATCH_INDEX_{len(protected) - 1}__"

        shielded = _BATCH_INDEX_BEFORE_SLASH.sub(shield, template)

        def repl(m: re.Match[str]) -> str:
            w = m.group(1) or m.group(2)
            pad = int(w) if w else DEFAULT_PAD_DIGITS
            return _frame_number_macro_segment(pad)

        normalized = _LEGACY_INDEX_FRAME_PATTERN.sub(repl, shielded)
        for i, token in enumerate(protected):
            normalized = normalized.replace(f"__GTN_BATCH_INDEX_{i}__", token)
        return re.sub(r"\}\{frame_number", "}.{frame_number", normalized)

    def _output_stem_glob_hint(self) -> str:
        incoming = self._incoming_output_file_destination()
        if incoming is not None:
            template = incoming.location
        else:
            raw = self.get_parameter_value("output_file")
            template = raw if isinstance(raw, str) and raw.strip() else DEFAULT_OUTPUT_FILE_SPEC
        if "{" in template:
            segment = template.rsplit("/", 1)[-1]
            stem_part = segment.split("{", 1)[0].rstrip(".")
            return stem_part if stem_part else "*"
        _, stem_base, _ = self._parse_user_output_path(template)
        return stem_base if stem_base else "*"

    def _variables_for_output_macro(
        self, template: str, output_frame_num: int, format_type: str
    ) -> dict[str, str | int]:
        try:
            parsed = ParsedMacro(template)
        except MacroSyntaxError as e:
            msg = f"{self.name}: Invalid output path macro: {e}"
            raise ValueError(msg) from e
        variables: dict[str, str | int] = {
            "node_name": self.name,
            "frame_number": output_frame_num,
            "file_extension": format_type,
        }
        for v in parsed.get_variables():
            if v.name == "file_name_base":
                variables["file_name_base"] = self._infer_file_name_base_for_expert_macro()
        return variables

    def _project_destination_from_template(
        self, template: str, variables: dict[str, str | int]
    ) -> ProjectFileDestination:
        result = GriptapeNodes.handle_request(
            GetSituationRequest(situation_name=OUTPUT_FILE_SITUATION)
        )
        if isinstance(result, GetSituationResultSuccess):
            on_collision = result.situation.policy.on_collision
            existing_file_policy = SITUATION_TO_FILE_POLICY.get(on_collision, ExistingFilePolicy.OVERWRITE)
            create_dirs = result.situation.policy.create_dirs
        else:
            existing_file_policy = ExistingFilePolicy.OVERWRITE
            create_dirs = True
        try:
            macro_path = MacroPath(ParsedMacro(template), variables)
        except MacroSyntaxError as e:
            msg = f"{self.name}: Invalid output path macro: {e}"
            raise ValueError(msg) from e
        return ProjectFileDestination(
            macro_path,
            existing_file_policy=existing_file_policy,
            create_parents=create_dirs,
        )

    def _outputs_root_dir(self) -> Path:
        result = GriptapeNodes.handle_request(
            GetPathForMacroRequest(parsed_macro=ParsedMacro("{outputs}"), variables={})
        )
        if isinstance(result, GetPathForMacroResultFailure):
            msg = f"{self.name}: Could not resolve project outputs directory: {result.result_details}"
            raise ValueError(msg)
        abs_path = getattr(result, "absolute_path", None)
        if abs_path is None:
            msg = f"{self.name}: Could not resolve project outputs directory."
            raise ValueError(msg)
        return Path(abs_path)

    def _template_uses_versioned_output_folder(self, template: str) -> bool:
        return _VERSIONED_OUTPUT_SUBDIR_RE.search(template) is not None

    def _batch_index_padding_from_template(self, template: str) -> int:
        m = _BATCH_INDEX_PAD_IN_FOLDER_RE.search(template)
        if m:
            w = m.group(1) or m.group(2)
            return max(1, min(int(w), 99))
        return DEFAULT_BATCH_INDEX_PAD_DIGITS

    def _inject_batch_index_into_expert_template(self, template: str) -> str:
        if _BATCH_INDEX_BEFORE_SLASH.search(template):
            return template
        m = re.match(r"^(\{outputs\}/)([^{}/]+)/(.+)$", template)
        if m:
            batch_seg = _batch_index_macro_segment(DEFAULT_BATCH_INDEX_PAD_DIGITS)
            return f"{m.group(1)}{m.group(2)}.{batch_seg}/{m.group(3)}"
        return template

    def _folder_base_for_versioned_output(self, template: str) -> str:
        m = _VERSIONED_OUTPUT_SUBDIR_RE.search(template)
        if m:
            return m.group(1)
        safe = re.sub(r"[^\w\-]+", "_", self.name).strip("_")
        return safe or "extract_frames"

    def _friendly_versioned_macro_template(self, sub_dirs: str, stem: str, pad: int) -> str:
        frame_seg = _frame_number_macro_segment(pad)
        batch_seg = _batch_index_macro_segment(DEFAULT_BATCH_INDEX_PAD_DIGITS)
        if sub_dirs:
            if "/" in sub_dirs:
                parent, leaf = sub_dirs.rsplit("/", 1)
                middle = f"{parent}/{leaf}.{batch_seg}"
            else:
                middle = f"{sub_dirs}.{batch_seg}"
        else:
            middle = f"{stem}.{batch_seg}"
        return f"{{outputs}}/{middle}/{stem}.{frame_seg}.{{file_extension}}"

    def _output_template_and_folder_base(self) -> tuple[str, str]:
        incoming = self._incoming_output_file_destination()
        if incoming is not None:
            template = self._normalize_output_macro_template(incoming.location)
            if "{" not in template:
                msg = (
                    f"{self.name}: Connected file output must be a macro path that includes frame placeholders "
                    f"(e.g. {{frame_number?:04}}), got: {template!r}"
                )
                raise ValueError(msg)
            template = self._inject_batch_index_into_expert_template(template)
            folder_base = self._folder_base_for_versioned_output(template)
            return template, folder_base

        raw = self.get_parameter_value("output_file")
        spec = raw.strip() if isinstance(raw, str) and raw.strip() else DEFAULT_OUTPUT_FILE_SPEC
        if "{" in spec:
            template = self._normalize_output_macro_template(spec)
            template = self._inject_batch_index_into_expert_template(template)
            folder_base = self._folder_base_for_versioned_output(template)
            return template, folder_base

        sub_dirs, stem, pad = self._parse_user_output_path(spec)
        template = self._friendly_versioned_macro_template(sub_dirs, stem, pad)
        folder_base = stem if not sub_dirs else sub_dirs.rsplit("/", 1)[-1]
        return template, folder_base

    def _allocate_output_run_version(
        self, outputs_root: Path, folder_base: str, overwrite: bool, pad: int
    ) -> int:
        width = max(1, min(pad, 99))
        prefix = f"{folder_base}."
        max_n = 0
        found_any = False
        if outputs_root.is_dir():
            for p in outputs_root.iterdir():
                if not p.is_dir():
                    continue
                name = p.name
                if name.startswith(prefix):
                    suffix = name[len(prefix) :]
                    if suffix.isdigit():
                        found_any = True
                        max_n = max(max_n, int(suffix))
        if overwrite:
            n = max_n if found_any else 1
            target = outputs_root / f"{folder_base}.{n:0{width}d}"
            if target.exists():
                shutil.rmtree(target)
            return n
        return max_n + 1 if found_any else 1

    def _prepare_output_run(self, overwrite_files: bool) -> tuple[int, str]:
        template, folder_base = self._output_template_and_folder_base()
        if not self._template_uses_versioned_output_folder(template):
            return 1, template
        run_pad = self._batch_index_padding_from_template(template)
        outputs_root = self._outputs_root_dir()
        run_n = self._allocate_output_run_version(outputs_root, folder_base, overwrite_files, run_pad)
        return run_n, template

    def _resolve_frame_output_path(
        self, output_frame_num: int, format_type: str, run_version: int, template: str
    ) -> str:
        variables = self._variables_for_output_macro(template, output_frame_num, format_type)
        if self._template_uses_versioned_output_folder(template):
            variables["_index"] = run_version
        return self._project_destination_from_template(template, variables).resolve()

    def _incoming_output_file_destination(self) -> FileDestination | None:
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(result, ListConnectionsForNodeResultSuccess):
            for conn in result.incoming_connections:
                if conn.target_parameter_name == "output_file":
                    source_node = GriptapeNodes.ObjectManager().attempt_get_object_by_name(conn.source_node_name)
                    if isinstance(source_node, FileDestinationProvider):
                        fd = source_node.file_destination
                        if fd is not None:
                            return fd
        return None

    def _resolved_output_path_string(self, path: Path) -> str:
        return str(path.resolve())

    def _remove_previous_frames_matching_stem(self, output_dir: Path, format_type: str, stem_hint: str) -> None:
        """Remove files from a prior run in the same output folder."""
        if stem_hint == "*":
            glob_pattern = f"*.{format_type}"
        else:
            glob_pattern = f"{stem_hint}*.{format_type}"
        if output_dir.exists() and not output_dir.is_dir():
            error_msg = f"{self.name} output path is a file, not a directory: {output_dir}"
            raise ValueError(error_msg)
        output_dir.mkdir(parents=True, exist_ok=True)
        for file_path in output_dir.glob(glob_pattern):
            if file_path.is_file():
                try:
                    file_path.unlink()
                    logger.debug("%s removed previous frame: %s", self.name, file_path)
                except OSError as e:
                    logger.warning("%s failed to remove previous frame %s: %s", self.name, file_path, e)

    def _extract_frames(
        self,
        video_url: str,
        frame_numbers: list[int],
        format_type: str,
        frame_numbering: str,
        *,
        overwrite_files: bool,
        run_version: int,
        output_template: str,
    ) -> list[str]:
        """Extract frames from video to still image files."""
        if not frame_numbers:
            return []

        try:
            ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        except Exception as e:
            error_msg = f"Video export tools are not available on this system. {e}"
            raise ValueError(error_msg) from e

        frame_paths = []
        renumber = frame_numbering == FRAME_NUMBERING_OPTIONS[1]

        # Safety check: ensure all frame numbers are valid
        if not frame_numbers:
            logger.warning("%s no frame numbers provided for extraction", self.name)
            return []

        max_frame_num = max(frame_numbers)
        min_frame_num = min(frame_numbers)
        logger.debug(
            "%s extracting frames: min=%d, max=%d, total=%d",
            self.name,
            min_frame_num,
            max_frame_num,
            len(frame_numbers),
        )

        for idx, frame_num in enumerate(frame_numbers):
            if renumber:
                output_frame_num = idx + 1
            else:
                output_frame_num = frame_num

            output_path = Path(
                self._resolve_frame_output_path(output_frame_num, format_type, run_version, output_template)
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Skip if file exists and overwrite is disabled
            if output_path.exists() and not overwrite_files:
                logger.debug("%s skipping existing frame: %s", self.name, output_path)
                frame_paths.append(self._resolved_output_path_string(output_path))
                continue

            # Build ffmpeg command to extract specific frame
            # Use select filter to extract frame at specific position
            cmd = [
                ffmpeg_path,
                "-i",
                video_url,
                "-vf",
                f"select='eq(n\\,{frame_num})'",
                "-vsync",
                "0",
                "-frames:v",
                "1",
                "-y" if overwrite_files else "-n",
                str(output_path),
            ]

            try:
                subprocess.run(  # noqa: S603
                    cmd, capture_output=True, text=True, check=True, timeout=60
                )
                frame_paths.append(self._resolved_output_path_string(output_path))
                logger.debug("%s extracted frame %d to %s", self.name, frame_num, output_path)

            except subprocess.TimeoutExpired as e:
                error_msg = (
                    f"Saving frame {frame_num} took too long and was stopped. "
                    f"Try a shorter clip, fewer frames, or check disk and permissions. ({e})"
                )
                raise RuntimeError(error_msg) from e
            except subprocess.CalledProcessError as e:
                detail = (e.stderr or str(e)).strip()
                if len(detail) > 400:
                    detail = detail[:400] + "..."
                error_msg = (
                    f"Could not export frame {frame_num}. "
                    f"The file or video may be unreadable, or the output location may not be writable."
                )
                if detail:
                    error_msg = f"{error_msg} Detail: {detail}"
                raise RuntimeError(error_msg) from e

        return frame_paths

    def _set_safe_defaults(self) -> None:
        """Set safe default output values on failure."""
        self.parameter_output_values["frame_paths"] = []
