"""Extract specific frames from a video and save them as images."""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from enum import StrEnum
from typing import Any

from griptape_nodes.common.sequences import MissingItemPolicy, Sequence
from griptape_nodes.exe_types.core_types import (
    NodeError,
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.file_format import FileFormat, FileFormatKind
from griptape_nodes.traits.options import Options
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.utils.versioned_output_directory import (
    build_directory_parameter,
    clear_directory_value,
    configure_directory_with_settings_node,
    normalize_directory_value,
    resolve_output_dir,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["ExtractFrames"]

FORMAT_OPTIONS = ["png", "jpg", "webp"]
PADDING_OPTIONS = ["1", "2", "3", "4", "5", "6", "7", "8"]

DEFAULT_OUTPUT_PREFIX = "frames"
DEFAULT_FRAME_PADDING = 4
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_EVERY_N = 2
DEFAULT_DIRECTORY = "{outputs}/frames_v{###}"


class FrameSelectionMode(StrEnum):
    LIST = "list"
    EVERY_NTH = "every_Nth"


class ExtractFrames(SuccessFailureNode):
    """Extract specific frames from a video and save them as images."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            ParameterVideo(
                name="input_video",
                default_value="",
                edit_video=True,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Video player for precise frame selection. Connect a video source here.",
                ui_options={"display_name": "Video Input"},
            )
        )

        self.add_parameter(
            Parameter(
                name="_video_fps",
                type="float",
                default_value=0.0,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Native frame rate detected from the video (set automatically).",
                ui_options={"hide_property": True},
            )
        )

        with ParameterGroup(
            name="frame_selection", ui_options={"display_name": "Frame Selection"}
        ) as frame_selection_group:
            ParameterString(
                name="frame_selection_mode",
                default_value=FrameSelectionMode.LIST,
                tooltip=(
                    "How frames are selected. "
                    "'Specific Frames' uses markers placed in the video player above. "
                    "'Every N Frames' extracts one frame every N frames across the whole clip."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Selection",
                    "data": [
                        {"name": "list", "label": "Specific Frames"},
                        {"name": "every_Nth", "label": "Every N Frames"},
                    ],
                },
                traits={Options(choices=list(FrameSelectionMode))},
            )

            ParameterString(
                name="input_frame_numbers",
                default_value="",
                placeholder_text="e.g. 1,4,5-9,11",
                ui_options={"display_name": "Specific Frames", "frame_selector": True},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip=(
                    "Frames to extract as a comma-separated list of numbers or ranges (e.g. 1,4,5-9,11).\n\n"
                    "Use the video player above to add markers visually:\n"
                    "• Hover above the timeline to reveal +, click to add a marker\n"
                    "• Click and drag the + to create a range\n"
                    "• Drag an existing marker triangle to move it\n"
                    "• Hover a marker and click the trash icon to remove it"
                ),
            )

            ParameterInt(
                name="every_n",
                default_value=DEFAULT_EVERY_N,
                tooltip="Extract one frame every N frames across the full clip.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Every N Frames"},
                hide=True,
            )

        self.add_node_element(frame_selection_group)

        with ParameterGroup(
            name="output_settings", ui_options={"display_name": "Output Settings"}
        ) as output_settings_group:
            build_directory_parameter(
                default_directory=DEFAULT_DIRECTORY,
                tooltip=(
                    "Directory where frames are saved. Use {###} for auto-incrementing versions "
                    "(e.g. {outputs}/frames_v{###} → frames_v001, frames_v002, …). "
                    "Click the gear to connect a FileOutputSettings node."
                ),
                on_configure_clicked=self._on_configure_output_dir_clicked,
                on_connection_removed=self._on_output_dir_connection_removed,
            )

            ParameterString(
                name="file_prefix",
                default_value=DEFAULT_OUTPUT_PREFIX,
                tooltip="Filename prefix for each saved frame (e.g. 'frames' → 'frames.0001.png')",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            ParameterString(
                name="frame_padding",
                default_value=str(DEFAULT_FRAME_PADDING),
                tooltip="Number of digits used to zero-pad frame numbers in output filenames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=PADDING_OPTIONS)},
            )

            ParameterString(
                name="file_format",
                default_value=DEFAULT_OUTPUT_FORMAT,
                tooltip="Image format for extracted frames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=FORMAT_OPTIONS), FileFormat(kind=FileFormatKind.IMAGE)},
            )

        self.add_node_element(output_settings_group)

        self.add_parameter(
            Parameter(
                name="output_frames",
                type="list[str]",
                output_type="list[str]",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Absolute paths to the extracted frame images.",
                ui_options={"pulse_on_run": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="sequence",
                type="Sequence",
                output_type="Sequence",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip=(
                    "The extracted frames as a sequence, with frame numbers, paths, and the "
                    "filename pattern in one object. Wire into Inspect Sequence or any other "
                    "node that takes a sequence."
                ),
                ui_options={"display_name": "Sequence"},
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_directory",
                type="str",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Directory where frames were saved.",
                placeholder_text="This will be automatically set to the output directory",
            )
        )
        self._fps_probed_source: str | None = None
        self.progress_component = ProgressBarComponent(self)
        self.progress_component.add_property_parameters()
        self._create_status_parameters(
            result_details_tooltip="Details about the frame extraction result or any errors",
            result_details_placeholder="Extraction status will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # ── Lifecycle hooks ─────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        match parameter.name:
            case "input_video":
                self._update_video_fps(value)
            case "directory":
                normalize_directory_value(self, value)
            case "frame_selection_mode":
                match value:
                    case FrameSelectionMode.LIST:
                        self.show_parameter_by_name("input_frame_numbers")
                        self.hide_parameter_by_name("every_n")
                    case FrameSelectionMode.EVERY_NTH:
                        self.hide_parameter_by_name("input_frame_numbers")
                        self.show_parameter_by_name("every_n")
                    case _:
                        msg = f"Unknown frame_selection_mode: {value!r}"
                        raise ValueError(msg)
        super().after_value_set(parameter, value)

    # ── Validation ──────────────────────────────────────────────────────────────

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        if not str(self.get_parameter_value("input_video") or "").strip():
            exceptions.append(ValueError("A video is required. Connect one to 'Video Input'."))
        mode = self.get_parameter_value("frame_selection_mode") or FrameSelectionMode.LIST
        match mode:
            case FrameSelectionMode.LIST:
                if not _parse_frame_string(self.get_parameter_value("input_frame_numbers") or ""):
                    exceptions.append(ValueError("'Specific Frames' must list at least one frame, e.g. '1,4,5-9'."))
            case FrameSelectionMode.EVERY_NTH:
                if (self.get_parameter_value("every_n") or 1) < 1:
                    exceptions.append(ValueError("'Every N Frames' must be 1 or more."))
            case _:
                exceptions.append(ValueError(f"Unknown 'Selection' mode: {mode!r}"))
        return exceptions or None

    # ── Video URL resolution ────────────────────────────────────────────────────

    def _resolve_video_for_ffmpeg(self, raw_video: Any) -> str:
        """Resolve the video input to a path or URL ffmpeg can read."""
        file_path = raw_video.value if hasattr(raw_video, "value") else str(raw_video or "")
        if not file_path:
            msg = "No video is loaded in 'Video Input'."
            raise ValueError(msg)
        s = str(file_path).strip()
        if s.startswith(("http://", "https://", "blob:", "data:")):
            return s
        try:
            return File(s).resolve()
        except Exception as e:
            msg = f"Could not resolve the video path: {e}"
            raise ValueError(msg) from e

    # ── FFprobe helpers ─────────────────────────────────────────────────────────

    def _probe_video(self, video_url: str) -> tuple[float, float]:
        """Return (fps, duration_seconds) for the first video stream.

        Falls back to format-level duration when the stream doesn't have one
        (common in Matroska/WebM containers). Returns (30.0, 0.0) on failure.
        """
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                "-show_format",
                "-select_streams",
                "v:0",
                video_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            data = json.loads(result.stdout)
            streams = data.get("streams", [])
            if not streams:
                return 30.0, 0.0
            stream = streams[0]

            r_frame_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = map(int, r_frame_rate.split("/"))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(r_frame_rate)

            # stream.duration is absent or "N/A" for Matroska and some other containers
            for duration_str in (stream.get("duration"), data.get("format", {}).get("duration")):
                if duration_str and duration_str not in ("N/A", ""):
                    return fps, float(duration_str)

            return fps, 0.0
        except Exception:
            return 30.0, 0.0

    def _get_video_fps(self, video_url: str) -> float | None:
        fps, _ = self._probe_video(video_url)
        return fps if fps else None

    def _update_video_fps(self, raw_video: Any) -> None:
        """Detect the video's frame rate for the frame selector, once per video."""
        if not raw_video:
            return
        try:
            video_source = self._resolve_video_for_ffmpeg(raw_video)
            # Upstream nodes send the same video again on every run.
            if video_source == self._fps_probed_source:
                return
            self._fps_probed_source = video_source
            fps = self._get_video_fps(video_source)
            if fps:
                self.set_parameter_value("_video_fps", fps)
                self.publish_update_to_parameter("_video_fps", fps)
        except Exception:
            logger.warning("%s: could not detect video FPS", self.name, exc_info=True)

    # ── Frame list building ─────────────────────────────────────────────────────

    def _build_frame_list(self, video_url: str) -> list[int]:
        """Return the ordered list of 1-based frame numbers to extract."""
        mode = self.get_parameter_value("frame_selection_mode") or FrameSelectionMode.LIST
        match mode:
            case FrameSelectionMode.LIST:
                return _parse_frame_string(self.get_parameter_value("input_frame_numbers") or "")
            case FrameSelectionMode.EVERY_NTH:
                every_n = max(1, self.get_parameter_value("every_n") or DEFAULT_EVERY_N)
                fps, duration = self._probe_video(video_url)
                total_frames = max(1, round(duration * fps))
                return list(range(1, total_frames + 1, every_n))
            case _:
                msg = f"Unknown frame_selection_mode: {mode!r}"
                raise ValueError(msg)

    # ── Output directory ────────────────────────────────────────────────────────

    def _on_configure_output_dir_clicked(
        self, _button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult:
        return configure_directory_with_settings_node(self, button_details)

    def _on_output_dir_connection_removed(self, *_: object) -> None:
        clear_directory_value(self)

    def _resolve_output_dir(self) -> pathlib.Path:
        return resolve_output_dir(self, DEFAULT_DIRECTORY)

    # ── Extraction ──────────────────────────────────────────────────────────────

    def _extract_frames(
        self,
        video_url: str,
        frame_numbers: list[int],
        output_dir: pathlib.Path,
        prefix: str,
        padding: int,
        fmt: str,
    ) -> list[pathlib.Path]:
        try:
            ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        except Exception as e:
            msg = f"FFmpeg is not available: {e}"
            raise ValueError(msg) from e

        # One ffmpeg call selects all requested frames (0-based indices).
        # Outputs are sequentially numbered (__ext_tmp1.png, __ext_tmp2.png, …);
        # we rename them to final names afterward and check each was written.
        select_expr = "+".join(f"eq(n\\,{fn - 1})" for fn in frame_numbers)
        tmp_stem = "__ext_tmp"
        tmp_pattern = output_dir / f"{tmp_stem}%d.{fmt}"
        timeout = max(120, 30 + 2 * len(frame_numbers))

        cmd = [
            ffmpeg_path,
            "-i",
            video_url,
            "-vf",
            f"select='{select_expr}'",
            "-vsync",
            "0",
            "-y",
            str(tmp_pattern),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)  # noqa: S603
        except subprocess.TimeoutExpired as e:
            msg = "Timed out extracting frames. Try fewer frames or a shorter clip."
            raise RuntimeError(msg) from e
        except subprocess.CalledProcessError as e:
            detail = (e.stderr or str(e)).strip()[:400]
            msg = f"Could not extract frames. {detail}"
            raise RuntimeError(msg) from e

        self.progress_component.initialize(len(frame_numbers))
        saved_paths: list[pathlib.Path] = []
        missing: list[int] = []
        for i, frame_num in enumerate(frame_numbers, start=1):
            tmp_path = output_dir / f"{tmp_stem}{i}.{fmt}"
            final_path = output_dir / f"{prefix}.{str(frame_num).zfill(padding)}.{fmt}"
            if tmp_path.exists():
                tmp_path.rename(final_path)
                saved_paths.append(final_path)
            else:
                missing.append(frame_num)
            self.progress_component.increment()

        if missing:
            skipped = ", ".join(str(n) for n in missing[:10])
            suffix = f" … ({len(missing)} total)" if len(missing) > 10 else ""
            logger.warning("%s: frame(s) past end of clip, skipped: %s%s", self.name, skipped, suffix)

        return saved_paths

    # ── Process ─────────────────────────────────────────────────────────────────

    def process(self) -> AsyncResult[None]:
        self._clear_execution_status()
        self.progress_component.reset()

        raw_video = self.get_parameter_value("input_video")
        try:
            video_url = self._resolve_video_for_ffmpeg(raw_video)
        except ValueError as e:
            self._fail(e)
            return

        try:
            frame_numbers = self._build_frame_list(video_url)
        except ValueError as e:
            self._fail(e)
            return

        if not frame_numbers:
            self._fail(ValueError("No frames to extract with the current settings."))
            return

        # Resolved before extraction so a settings failure goes through _fail like the checks above;
        # an exception raised inside the yielded callable skips the except around the yield.
        try:
            output_dir = self._resolve_output_dir()
        except (NodeError, ValueError, FileLoadError) as e:
            self._fail(e)
            return

        try:
            yield lambda: self._perform_extraction(video_url, frame_numbers, output_dir)
        except Exception as e:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=f"Frame extraction failed: {e}")
            self._handle_failure_exception(e)

    def _fail(self, error: Exception) -> None:
        """Clear outputs, record the failure, and raise unless Failed is wired."""
        self._set_safe_defaults()
        self._set_status_results(was_successful=False, result_details=str(error))
        self._handle_failure_exception(error)

    def _perform_extraction(self, video_url: str, frame_numbers: list[int], output_dir: pathlib.Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.get_parameter_value("file_prefix") or DEFAULT_OUTPUT_PREFIX
        padding = int(self.get_parameter_value("frame_padding") or DEFAULT_FRAME_PADDING)
        fmt = self.get_parameter_value("file_format") or DEFAULT_OUTPUT_FORMAT

        saved_paths = self._extract_frames(
            video_url=video_url,
            frame_numbers=frame_numbers,
            output_dir=output_dir,
            prefix=prefix,
            padding=padding,
            fmt=fmt,
        )

        out_dir_str = str(output_dir)
        details = f"Extracted {len(saved_paths)} frame(s) to {out_dir_str}"

        sequence = self._scan_output_sequence(output_dir, prefix, padding, fmt, frame_numbers)
        if sequence is None:
            details += " (could not build the sequence output)"

        self.parameter_output_values["output_directory"] = out_dir_str
        self.parameter_output_values["output_frames"] = [str(p.resolve()) for p in saved_paths]
        self.parameter_output_values["sequence"] = sequence
        self._set_status_results(was_successful=True, result_details=details)

    def _scan_output_sequence(
        self,
        output_dir: pathlib.Path,
        prefix: str,
        padding: int,
        fmt: str,
        frame_numbers: list[int],
    ) -> Sequence | None:
        """Scan the frames just written into a Sequence, bounded to the requested frame range."""
        result = GriptapeNodes.handle_request(
            ScanSequencesRequest(
                path=str(output_dir / f"{prefix}.{'#' * padding}.{fmt}"),
                policy=MissingItemPolicy.SKIP,
                start_number=min(frame_numbers),
                end_number=max(frame_numbers),
            )
        )
        if isinstance(result, ScanSequencesResultSuccess) and result.sequences:
            return result.sequences[0]
        logger.warning("%s: could not build sequence output: %s", self.name, result.result_details)
        return None

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["output_directory"] = ""
        self.parameter_output_values["output_frames"] = []
        self.parameter_output_values["sequence"] = None


# ── Module-level frame string parser ───────────────────────────────────────────


def _parse_frame_string(frame_str: str) -> list[int]:
    """Parse comma-separated frame spec into a sorted deduplicated list (1-based).

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
