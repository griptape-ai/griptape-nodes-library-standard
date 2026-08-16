"""Extract specific frames from a video and save them as images."""

from __future__ import annotations

import json
import logging
import pathlib
import subprocess
from enum import StrEnum
from typing import Any

import griptape_nodes.traits.widget as widget_trait
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileDestinationProvider
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.os_events import OpenAssociatedFileRequest, OpenAssociatedFileResultSuccess
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.retained_mode import RetainedMode
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from static_ffmpeg import run  # type: ignore[import-untyped]

from griptape_nodes_library.utils.macro_path_utils import resolve_to_macro_path

logger = logging.getLogger("griptape_nodes")

__all__ = ["ExtractFrames"]

FORMAT_OPTIONS = ["png", "jpg", "webp"]
PADDING_OPTIONS = ["1", "2", "3", "4", "5", "6"]

DEFAULT_OUTPUT_PREFIX = "frames"
DEFAULT_FRAME_PADDING = 4
DEFAULT_OUTPUT_FORMAT = "png"
DEFAULT_EVERY_N = 1


class FrameSelectionMode(StrEnum):
    LIST = "list"
    EVERY_NTH = "every_Nth"


class ExtractFrames(SuccessFailureNode):
    """Extract specific frames from a video and save them as images."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._last_output_dir: pathlib.Path | None = None

        self.add_parameter(
            Parameter(
                name="input_video",
                type="str",
                output_type="str",
                input_types=["VideoUrlArtifact", "VideoArtifact", "str"],
                default_value="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Video player for precise frame selection. Connect a video source here.",
                ui_options={"display_name": "Video Input"},
                traits={
                    widget_trait.Widget(
                        name="VideoPlayerFrameSelector",
                        library="Griptape Nodes Library",
                    )
                },
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

        self.add_parameter(
            ParameterString(
                name="frame_selection_mode",
                default_value=FrameSelectionMode.LIST,
                tooltip=(
                    "How frames are selected. "
                    "'list' uses markers placed in the video player above. "
                    "'every_Nth' extracts one frame every N frames across the whole clip."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(FrameSelectionMode))},
            )
        )

        self.add_parameter(
            Parameter(
                name="input_frame_numbers",
                type="str",
                output_type="str",
                default_value="",
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
        )

        self.add_parameter(
            ParameterInt(
                name="every_n",
                default_value=DEFAULT_EVERY_N,
                tooltip="Extract one frame every N frames across the full clip.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                hide=True,
            )
        )

        output_dir_param = Parameter(
            name="output_directory",
            type="str",
            default_value="",
            input_types=["str"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip=(
                "Directory where frames are saved. Supports project macros (e.g. {outputs}/my_frames). "
                "Click the gear to connect a FileOutputSettings node."
            ),
            ui_options={"placeholder_text": "{outputs}/frames"},
            traits={
                Button(
                    icon="cog",
                    size="icon",
                    variant="secondary",
                    tooltip="Create and connect a FileOutputSettings node",
                    on_click=self._on_configure_output_dir_clicked,
                ),
                FileSystemPicker(allow_files=False, allow_directories=True, multiple=False),
            },
        )
        output_dir_param.on_incoming_connection_removed.append(self._on_output_dir_connection_removed)
        self.add_parameter(output_dir_param)

        self.add_parameter(
            ParameterString(
                name="output_file_prefix",
                default_value=DEFAULT_OUTPUT_PREFIX,
                tooltip="Filename prefix for each saved frame (e.g. 'frames' → 'frames.0001.png')",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_frame_padding",
                default_value=str(DEFAULT_FRAME_PADDING),
                tooltip="Number of digits used to zero-pad frame numbers in output filenames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=PADDING_OPTIONS)},
            )
        )
        self.add_parameter(
            ParameterString(
                name="output_file_format",
                default_value=DEFAULT_OUTPUT_FORMAT,
                tooltip="Image format for extracted frames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=FORMAT_OPTIONS)},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_location",
                type="str",
                output_type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Directory where frames were saved.",
            )
        )
        self.add_parameter(
            ParameterButton(
                name="open_output_folder",
                label="Open Output Folder",
                variant="secondary",
                icon="folder-open",
                on_click=self._on_open_output_folder,
                ui_options={"disabled": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="output_paths",
                type="list[str]",
                output_type="list[str]",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Absolute paths to the extracted frame images.",
                ui_options={"pulse_on_run": True},
            )
        )
        self._create_status_parameters(
            result_details_tooltip="Details about the frame extraction result or any errors",
            result_details_placeholder="Extraction status will appear here.",
            parameter_group_initially_collapsed=True,
        )

    # ── Lifecycle hooks ─────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        match parameter.name:
            case "input_video":
                url = self._resolve_video_url(value)
                current = self.parameter_values.get("input_video", "")
                current_base = str(current).split("?")[0] if current else ""
                if url:
                    if url.split("?")[0] != current_base:
                        self.set_parameter_value("input_video", url)
                        try:
                            fps = self._get_video_fps(url)
                            if fps:
                                self.set_parameter_value("_video_fps", fps)
                                self.publish_update_to_parameter("_video_fps", fps)
                        except Exception:
                            logger.warning("%s: could not detect video FPS", self.name, exc_info=True)
                elif current_base:
                    self.set_parameter_value("input_video", "")
            case "output_directory":
                if value and not str(value).startswith("{"):
                    result = resolve_to_macro_path(str(value))
                    if not result.is_external and result.resolved_path != str(value):
                        self.set_parameter_value("output_directory", result.resolved_path)
                        self.publish_update_to_parameter("output_directory", result.resolved_path)
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
            exceptions.append(ValueError(f"{self.name}: 'input_video' is required"))
        mode = self.get_parameter_value("frame_selection_mode") or FrameSelectionMode.LIST
        match mode:
            case FrameSelectionMode.LIST:
                if not _parse_frame_string(self.get_parameter_value("input_frame_numbers") or ""):
                    exceptions.append(
                        ValueError(
                            f"{self.name}: 'input_frame_numbers' must specify at least one frame (e.g. '1,4,5-9')"
                        )
                    )
            case FrameSelectionMode.EVERY_NTH:
                if (self.get_parameter_value("every_n") or 1) < 1:
                    exceptions.append(ValueError(f"{self.name}: 'every_n' must be >= 1"))
            case _:
                exceptions.append(ValueError(f"{self.name}: Unknown frame_selection_mode: {mode!r}"))
        return exceptions or None

    # ── Video URL resolution ────────────────────────────────────────────────────

    def _resolve_video_url(self, raw_value: Any) -> str | None:
        """Resolve a video input value to an HTTP URL the widget can play."""
        if not raw_value:
            return None
        file_path = raw_value.value if hasattr(raw_value, "value") else str(raw_value)
        if not file_path:
            return None
        if str(file_path).startswith(("http://", "https://", "blob:", "data:")):
            return str(file_path)
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=str(file_path))
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            logger.warning("%s: failed to resolve video URL for %s", self.name, file_path, exc_info=True)
        return None

    def _resolve_video_for_ffmpeg(self, raw_video: Any) -> str:
        """Resolve the video input to a path or URL ffmpeg can read."""
        file_path = raw_video.value if hasattr(raw_video, "value") else str(raw_video or "")
        if not file_path:
            msg = f"{self.name}: No video loaded in 'input_video'"
            raise ValueError(msg)
        s = str(file_path).strip()
        if s.startswith(("http://", "https://", "blob:", "data:")):
            return s
        try:
            return File(s).resolve()
        except Exception as e:
            msg = f"{self.name}: Could not resolve video path: {e}"
            raise ValueError(msg) from e

    # ── FFprobe helpers ─────────────────────────────────────────────────────────

    def _get_video_fps(self, video_url: str) -> float | None:
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
            cmd = [
                ffprobe_path, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", video_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            streams = json.loads(result.stdout).get("streams", [])
            if not streams:
                return None
            r_frame_rate = streams[0].get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = map(int, r_frame_rate.split("/"))
                return num / den if den != 0 else None
            return float(r_frame_rate)
        except Exception:
            return None

    def _get_video_fps_and_duration(self, video_url: str) -> tuple[float, float]:
        """Return (fps, duration_seconds). Used for every_Nth frame calculation."""
        try:
            _, ffprobe_path = run.get_or_fetch_platform_executables_else_raise()
            cmd = [
                ffprobe_path, "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "v:0", video_url,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            streams = json.loads(result.stdout).get("streams", [])
            if not streams:
                return 30.0, 0.0
            stream = streams[0]
            r_frame_rate = stream.get("r_frame_rate", "30/1")
            if "/" in r_frame_rate:
                num, den = map(int, r_frame_rate.split("/"))
                fps = num / den if den != 0 else 30.0
            else:
                fps = float(r_frame_rate)
            duration_str = stream.get("duration", "0")
            duration = float(duration_str) if duration_str not in ("N/A", "", None) else 0.0
            return fps, duration
        except Exception:
            return 30.0, 0.0

    # ── Frame list building ─────────────────────────────────────────────────────

    def _build_frame_list(self, video_url: str) -> list[int]:
        """Return the ordered list of 1-based frame numbers to extract."""
        mode = self.get_parameter_value("frame_selection_mode") or FrameSelectionMode.LIST
        match mode:
            case FrameSelectionMode.LIST:
                return _parse_frame_string(self.get_parameter_value("input_frame_numbers") or "")
            case FrameSelectionMode.EVERY_NTH:
                every_n = max(1, self.get_parameter_value("every_n") or DEFAULT_EVERY_N)
                fps, duration = self._get_video_fps_and_duration(video_url)
                total_frames = max(1, round(duration * fps))
                return list(range(1, total_frames + 1, every_n))
            case _:
                msg = f"Unknown frame_selection_mode: {mode!r}"
                raise ValueError(msg)

    # ── Output directory ────────────────────────────────────────────────────────

    def _on_configure_output_dir_clicked(
        self, _button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult:
        conn_result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(conn_result, ListConnectionsForNodeResultSuccess):
            if any(c.target_parameter_name == "output_directory" for c in conn_result.incoming_connections):
                return NodeMessageResult(
                    success=False,
                    details=f"{self.name}: output_directory already has an incoming connection",
                    response=button_details,
                    altered_workflow_state=False,
                )

        create_result = RetainedMode.create_node_relative_to(
            reference_node_name=self.name,
            new_node_type="FileOutputSettings",
            offset_side="left",
            offset_x=-750,
            offset_y=0,
            lock=False,
        )
        if not isinstance(create_result, str):
            return NodeMessageResult(
                success=False,
                details=f"{self.name}: Failed to create FileOutputSettings node",
                response=button_details,
                altered_workflow_state=False,
            )

        configure_node_name = create_result
        configure_node = GriptapeNodes.ObjectManager().attempt_get_object_by_name(configure_node_name)
        if configure_node is not None:
            configure_node.set_parameter_value("situation", "save_node_output")
            configure_node.publish_update_to_parameter("situation", "save_node_output")
            current_dir = self.get_parameter_value("output_directory") or ""
            if current_dir:
                configure_node.set_parameter_value("filename", current_dir)
                configure_node.publish_update_to_parameter("filename", current_dir)

        connection_result = RetainedMode.connect(
            source=f"{configure_node_name}.file_destination",
            destination=f"{self.name}.output_directory",
        )
        if not connection_result.succeeded():
            return NodeMessageResult(
                success=False,
                details=f"{self.name}: Failed to connect {configure_node_name}.file_destination to output_directory",
                response=button_details,
                altered_workflow_state=True,
            )
        return NodeMessageResult(
            success=True,
            details=f"{self.name}: Created and connected {configure_node_name}",
            response=button_details,
            altered_workflow_state=True,
        )

    def _on_output_dir_connection_removed(self, *_: object) -> None:
        self.set_parameter_value("output_directory", "")
        self.publish_update_to_parameter("output_directory", "")

    def _resolve_output_dir(self) -> pathlib.Path:
        conn_result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(conn_result, ListConnectionsForNodeResultSuccess):
            for conn in conn_result.incoming_connections:
                if conn.target_parameter_name == "output_directory":
                    source = GriptapeNodes.ObjectManager().attempt_get_object_by_name(conn.source_node_name)
                    if isinstance(source, FileDestinationProvider):
                        fd = source.file_destination
                        if fd is not None:
                            return pathlib.Path(fd.resolve()).parent
        value = (self.get_parameter_value("output_directory") or "").strip().rstrip("/")
        if not value:
            return pathlib.Path(File(f"{{outputs}}/{self.name}").resolve())
        # Treat bare relative paths (no leading / or macro {) as relative to {outputs}
        if not value.startswith(("/", "{")):
            value = f"{{outputs}}/{value}"
        return pathlib.Path(File(value).resolve())

    def _on_open_output_folder(
        self, _button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult:
        if not self._last_output_dir or not self._last_output_dir.exists():
            return NodeMessageResult(
                success=False,
                details="No output folder yet — run the node first.",
                response=button_details,
                altered_workflow_state=False,
            )
        result = GriptapeNodes.handle_request(OpenAssociatedFileRequest(path_to_file=str(self._last_output_dir)))
        success = isinstance(result, OpenAssociatedFileResultSuccess)
        return NodeMessageResult(
            success=success,
            details=str(self._last_output_dir) if success else f"Could not open folder: {result}",
            response=button_details,
            altered_workflow_state=False,
        )

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

        saved_paths: list[pathlib.Path] = []
        for frame_num in frame_numbers:
            filename = f"{prefix}.{str(frame_num).zfill(padding)}.{fmt}"
            output_path = output_dir / filename

            # select filter uses 0-based index; user-facing frame numbers are 1-based
            cmd = [
                ffmpeg_path,
                "-i", video_url,
                "-vf", f"select='eq(n\\,{frame_num - 1})'",
                "-vsync", "0",
                "-frames:v", "1",
                "-y",
                str(output_path),
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)  # noqa: S603
            except subprocess.TimeoutExpired as e:
                msg = f"Timed out extracting frame {frame_num}. Try fewer frames or a shorter clip."
                raise RuntimeError(msg) from e
            except subprocess.CalledProcessError as e:
                detail = (e.stderr or str(e)).strip()[:400]
                msg = f"Could not extract frame {frame_num}. {detail}"
                raise RuntimeError(msg) from e

            saved_paths.append(output_path)

        return saved_paths

    # ── Process ─────────────────────────────────────────────────────────────────

    def process(self) -> AsyncResult[None]:
        self._clear_execution_status()

        raw_video = self.get_parameter_value("input_video")
        try:
            video_url = self._resolve_video_for_ffmpeg(raw_video)
        except ValueError as e:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            return

        try:
            frame_numbers = self._build_frame_list(video_url)
        except ValueError as e:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            return

        if not frame_numbers:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: No frames to extract with current settings",
            )
            return

        try:
            yield lambda: self._perform_extraction(video_url, frame_numbers)
        except Exception as e:
            self._set_safe_defaults()
            error_msg = f"{self.name}: Frame extraction failed: {e}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            self._handle_failure_exception(RuntimeError(error_msg))

    def _perform_extraction(self, video_url: str, frame_numbers: list[int]) -> None:
        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)

        prefix = self.get_parameter_value("output_file_prefix") or DEFAULT_OUTPUT_PREFIX
        padding = int(self.get_parameter_value("output_frame_padding") or str(DEFAULT_FRAME_PADDING))
        fmt = self.get_parameter_value("output_file_format") or DEFAULT_OUTPUT_FORMAT

        saved_paths = self._extract_frames(
            video_url=video_url,
            frame_numbers=frame_numbers,
            output_dir=output_dir,
            prefix=prefix,
            padding=padding,
            fmt=fmt,
        )

        self._last_output_dir = output_dir
        out_dir_str = str(output_dir)

        self.parameter_output_values["output_location"] = out_dir_str
        self.parameter_output_values["output_paths"] = [str(p.resolve()) for p in saved_paths]
        self.set_parameter_value("open_output_folder", {"disabled": False})
        self.publish_update_to_parameter("open_output_folder", {"disabled": False})
        self._set_status_results(
            was_successful=True,
            result_details=f"Extracted {len(saved_paths)} frame(s) to {out_dir_str}",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["output_location"] = ""
        self.parameter_output_values["output_paths"] = []


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
