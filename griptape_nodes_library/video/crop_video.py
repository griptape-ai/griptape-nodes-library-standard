import subprocess
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.video_utils import to_video_artifact
from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor

RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "480p (854x480)": (854, 480),
    "720p (1280x720)": (1280, 720),
    "1080p (1920x1080)": (1920, 1080),
    "1440p (2560x1440)": (2560, 1440),
    "2160p (3840x2160)": (3840, 2160),
    "Square (1080x1080)": (1080, 1080),
    "Portrait (1080x1920)": (1080, 1920),
    "4:3 (1440x1080)": (1440, 1080),
}


def parse_size_string(size_str: str) -> tuple[int, int] | None:
    if not size_str:
        return None
    size_str = size_str.strip()
    for preset_name, dimensions in RESOLUTION_PRESETS.items():
        if preset_name.lower() == size_str.lower():
            return dimensions
    if "x" in size_str:
        parts = size_str.split("x")
        if len(parts) == 2:
            try:
                w, h = int(parts[0].strip()), int(parts[1].strip())
                if w > 0 and h > 0:
                    return (w, h)
            except ValueError:
                pass
    return None


class CropVideo(BaseVideoProcessor):
    """Crop a video using size/position presets or free-form drag with the interactive editor."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self._syncing_to_widget = False
        self._syncing_to_params = False

    def _setup_custom_parameters(self) -> None:
        self.add_parameter(
            ParameterDict(
                name="crop_editor",
                default_value={
                    "video_url": "",
                    "video_path": "",
                    "video_width": 0,
                    "video_height": 0,
                    "left": 0,
                    "top": 0,
                    "width": 0,
                    "height": 0,
                    "total_frames": 0,
                    "locked": [],
                },
                tooltip="Interactive crop editor — drag to set crop area, or use the preset controls below",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="CropVideoEditor", library="Griptape Nodes Library")},
            )
        )

        with ParameterGroup(name="crop_coordinates", ui_options={"collapsed": True}) as crop_coordinates:
            size_param = ParameterString(
                name="crop_size",
                default_value="Custom",
                tooltip="Target crop size — choose a preset or use Custom to set dimensions freely",
            )
            size_param.add_trait(Options(choices=[*list(RESOLUTION_PRESETS.keys()), "Custom"]))

            ParameterInt(name="custom_width", default_value=0, tooltip="Crop width in pixels")
            ParameterInt(name="custom_height", default_value=0, tooltip="Crop height in pixels")

            position_param = ParameterString(
                name="crop_position",
                default_value="center",
                tooltip="Where to position the crop area — use Custom to set position freely",
            )
            position_param.add_trait(
                Options(
                    choices=[
                        "center",
                        "top-left",
                        "top-right",
                        "bottom-left",
                        "bottom-right",
                        "top-center",
                        "bottom-center",
                        "left-center",
                        "right-center",
                        "Custom",
                    ]
                )
            )

            ParameterInt(name="custom_left", default_value=0, tooltip="Left edge of crop area in pixels")
            ParameterInt(name="custom_top", default_value=0, tooltip="Top edge of crop area in pixels")

        self.add_node_element(crop_coordinates)

        # Hide custom fields unless their respective preset is set to "Custom"
        self.hide_parameter_by_name("custom_left")
        self.hide_parameter_by_name("custom_top")

        video_param = self.get_parameter_by_name("video")
        if video_param:
            video_param.hide_property = True

    def _get_processing_description(self) -> str:
        return "cropping video"

    # ── Preset helpers ─────────────────────────────────────────────────────────

    def _get_crop_dimensions(self) -> tuple[int, int] | None:
        crop_size = self.get_parameter_value("crop_size") or "Custom"
        if crop_size == "Custom":
            w = self.get_parameter_value("custom_width") or 0
            h = self.get_parameter_value("custom_height") or 0
            if w <= 0 or h <= 0:
                return None
            return (w, h)
        if crop_size in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[crop_size]
        return parse_size_string(crop_size)

    def _calculate_crop_coordinates(
        self, video_width: int, video_height: int, crop_width: int, crop_height: int
    ) -> tuple[int, int]:
        position = self.get_parameter_value("crop_position") or "center"
        if position == "Custom":
            x = self.get_parameter_value("custom_left") or 0
            y = self.get_parameter_value("custom_top") or 0
            return (max(0, min(x, video_width - crop_width)), max(0, min(y, video_height - crop_height)))
        positions = {
            "center": ((video_width - crop_width) // 2, (video_height - crop_height) // 2),
            "top-left": (0, 0),
            "top-right": (video_width - crop_width, 0),
            "bottom-left": (0, video_height - crop_height),
            "bottom-right": (video_width - crop_width, video_height - crop_height),
            "top-center": ((video_width - crop_width) // 2, 0),
            "bottom-center": ((video_width - crop_width) // 2, video_height - crop_height),
            "left-center": (0, (video_height - crop_height) // 2),
            "right-center": (video_width - crop_width, (video_height - crop_height) // 2),
        }
        x, y = positions.get(position, ((video_width - crop_width) // 2, (video_height - crop_height) // 2))
        return (max(0, min(x, video_width - crop_width)), max(0, min(y, video_height - crop_height)))

    def _get_video_dimensions(self) -> tuple[int, int] | None:
        video = self.parameter_values.get("video")
        if not video:
            return None
        try:
            artifact = to_video_artifact(video)
        except (ValueError, TypeError, AttributeError):
            return None
        if not artifact or not getattr(artifact, "value", None):
            return None
        input_url = File(artifact.value).resolve()
        try:
            self._validate_url_safety(input_url)
            _, ffprobe_path = self._get_ffmpeg_paths()
            _, (w, h), _ = self._detect_video_properties(input_url, ffprobe_path)
            return (w, h)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
            return None

    # ── Lock detection ─────────────────────────────────────────────────────────

    _LOCKABLE_PARAMS = frozenset(
        {"crop_size", "crop_position", "custom_width", "custom_height", "custom_left", "custom_top"}
    )

    def _get_locked_params(self) -> list[str]:
        """Return widget lock keys for any connected wires on crop-affecting params."""
        try:
            result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
            if not isinstance(result, ListConnectionsForNodeResultSuccess):
                return []
            connected = {c.target_parameter_name for c in result.incoming_connections}
        except Exception:
            return []
        locked: set[str] = set()
        if "crop_size" in connected:
            locked |= {"width", "height"}
        if "custom_width" in connected:
            locked.add("width")
        if "custom_height" in connected:
            locked.add("height")
        if "crop_position" in connected or "custom_left" in connected:
            locked.add("left")
        if "crop_position" in connected or "custom_top" in connected:
            locked.add("top")
        return list(locked)

    def _refresh_locked_in_widget(self) -> None:
        existing = self.get_parameter_value("crop_editor") or {}
        new_locked = self._get_locked_params()
        if set(existing.get("locked") or []) == set(new_locked):
            return
        self._syncing_to_widget = True
        try:
            updated = {**existing, "locked": new_locked}
            self.set_parameter_value("crop_editor", updated)
            self.publish_update_to_parameter("crop_editor", updated)
        finally:
            self._syncing_to_widget = False

    # ── Widget sync ────────────────────────────────────────────────────────────

    def _resolve_video_url(self, artifact: Any) -> str:
        raw = getattr(artifact, "value", "") or ""
        if not raw:
            return ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw
        try:
            resolved = File(raw).resolve()
        except Exception:
            return raw
        try:
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return raw

    def _push_to_widget(self) -> None:
        """Compute pixel crop coords from current settings and push to the interactive widget."""
        video = self.parameter_values.get("video")
        if not video:
            return
        try:
            artifact = to_video_artifact(video)
        except Exception:
            return

        video_path = getattr(artifact, "value", "") or ""
        if not video_path:
            return

        dims = self._get_video_dimensions()
        if not dims:
            return
        vw, vh = dims

        crop_dims = self._get_crop_dimensions()
        w, h = (min(crop_dims[0], vw), min(crop_dims[1], vh)) if crop_dims else (vw, vh)
        left, top = self._calculate_crop_coordinates(vw, vh, w, h)

        url = self._resolve_video_url(artifact)

        total_frames = 0
        with suppress(Exception):
            _, ffprobe_path = self._get_ffmpeg_paths()
            resolved = File(video_path).resolve()
            frame_rate, _, duration = self._detect_video_properties(resolved, ffprobe_path)
            total_frames = max(0, int(duration * frame_rate))

        widget_dict = {
            "video_url": url,
            "video_path": video_path,
            "video_width": vw,
            "video_height": vh,
            "left": left,
            "top": top,
            "width": w,
            "height": h,
            "total_frames": total_frames,
            "locked": self._get_locked_params(),
        }
        self._syncing_to_widget = True
        try:
            self.set_parameter_value("crop_editor", widget_dict)
            self.publish_update_to_parameter("crop_editor", widget_dict)
        finally:
            self._syncing_to_widget = False

    def _sync_from_widget(self, widget_dict: dict) -> None:
        """User dragged the widget — switch both dropdowns to Custom and store pixel values."""
        w = widget_dict.get("width") or 0
        h = widget_dict.get("height") or 0
        left = widget_dict.get("left") or 0
        top = widget_dict.get("top") or 0

        # Pre-publish crop_editor so the frontend's stored value is authoritative
        # before individual param publishes arrive (mirrors CropImage pattern).
        self.publish_update_to_parameter("crop_editor", widget_dict)

        self._syncing_to_params = True
        try:
            # Set pixel values before the dropdowns so after_value_set show/hide
            # reveals fields that already have the correct values (no stale flash).
            self.set_parameter_value("custom_width", w)
            self.set_parameter_value("custom_height", h)
            self.set_parameter_value("custom_left", left)
            self.set_parameter_value("custom_top", top)
            self.set_parameter_value("crop_size", "Custom")
            self.set_parameter_value("crop_position", "Custom")

            self.show_parameter_by_name("custom_width")
            self.show_parameter_by_name("custom_height")
            self.show_parameter_by_name("custom_left")
            self.show_parameter_by_name("custom_top")

            # Publish values before dropdowns for the same reason.
            for pname in ("custom_width", "custom_height", "custom_left", "custom_top", "crop_size", "crop_position"):
                self.publish_update_to_parameter(pname, self.get_parameter_value(pname))
        finally:
            self._syncing_to_params = False

    # ── after_value_set ────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "crop_size":
            if value == "Custom":
                self.show_parameter_by_name("custom_width")
                self.show_parameter_by_name("custom_height")
            else:
                self.hide_parameter_by_name("custom_width")
                self.hide_parameter_by_name("custom_height")
            if not self._syncing_to_params:
                self._push_to_widget()

        elif parameter.name == "crop_position":
            if value == "Custom":
                self.show_parameter_by_name("custom_left")
                self.show_parameter_by_name("custom_top")
            else:
                self.hide_parameter_by_name("custom_left")
                self.hide_parameter_by_name("custom_top")
            if not self._syncing_to_params:
                self._push_to_widget()

        elif parameter.name in ("custom_width", "custom_height", "custom_left", "custom_top"):
            if not self._syncing_to_params:
                self._push_to_widget()

        elif parameter.name == "video":
            self._push_to_widget()

        elif parameter.name == "crop_editor" and not self._syncing_to_widget:
            if isinstance(value, dict):
                self._sync_from_widget(value)

        return super().after_value_set(parameter, value)

    # ── Incoming connection ────────────────────────────────────────────────────

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name == "video":
            val = source_node.get_parameter_value(source_parameter.name)
            if not val:
                val = self.get_parameter_value("video")
            if val:
                self.after_value_set(target_parameter, val)
            else:
                # Try to reinitialize from path saved in the widget dict (workflow reload)
                saved_path = (self.get_parameter_value("crop_editor") or {}).get("video_path") or ""
                if saved_path:
                    self._reinitialize_from_path(saved_path)
        elif target_parameter.name in self._LOCKABLE_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in self._LOCKABLE_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def _reinitialize_from_path(self, path: str) -> None:
        existing = self.get_parameter_value("crop_editor") or {}
        try:
            _, ffprobe_path = self._get_ffmpeg_paths()
            resolved = File(path).resolve()
            frame_rate, (vw, vh), duration = self._detect_video_properties(resolved, ffprobe_path)
            total_frames = max(0, int(duration * frame_rate))
        except Exception as e:
            logger.warning("%s: could not probe %r on reload: %s", self.name, path, e)
            vw = existing.get("video_width") or 0
            vh = existing.get("video_height") or 0
            total_frames = existing.get("total_frames") or 0
        try:
            resolved = File(path).resolve()
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            url = result.url if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess) else ""
        except Exception:
            url = ""
        new_dict = {
            **existing,
            "video_url": url,
            "video_path": path,
            "video_width": vw,
            "video_height": vh,
            "total_frames": total_frames,
            "locked": self._get_locked_params(),
        }
        self._syncing_to_widget = True
        try:
            self.set_parameter_value("crop_editor", new_dict)
            self.publish_update_to_parameter("crop_editor", new_dict)
        finally:
            self._syncing_to_widget = False

    # ── FFmpeg command ─────────────────────────────────────────────────────────

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        _, (video_width, video_height), _ = self._detect_video_properties(input_url, ffprobe_path)

        crop_dims = self._get_crop_dimensions()
        w = min(crop_dims[0], video_width) if crop_dims else video_width
        h = min(crop_dims[1], video_height) if crop_dims else video_height
        left, top = self._calculate_crop_coordinates(video_width, video_height, w, h)

        left = max(0, min(left, video_width - 1))
        top = max(0, min(top, video_height - 1))
        w = min(w, video_width - left)
        h = min(h, video_height - top)
        w = max(2, (w // 2) * 2)
        h = max(2, (h // 2) * 2)

        crop_filter = f"crop={w}:{h}:{left}:{top}"
        video_filter = self._combine_video_filters(crop_filter, input_frame_rate)

        has_audio = self._detect_audio_stream(input_url, ffprobe_path)
        cmd = [ffmpeg_path, "-y", "-i", input_url, "-vf", video_filter]
        cmd.extend(["-c:a", "copy"] if has_audio else ["-an"])
        preset, pixel_format, crf = self._get_processing_speed_settings()
        cmd.extend(["-preset", preset, "-pix_fmt", pixel_format, "-crf", str(crf), output_path])
        return cmd

    def _get_custom_parameters(self) -> dict[str, Any]:
        return {}

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        crop_dims = self._get_crop_dimensions()
        return f"_crop_{crop_dims[0]}x{crop_dims[1]}" if crop_dims else "_crop"

    # ── Process ────────────────────────────────────────────────────────────────

    def process(self) -> AsyncResult[None]:
        self._clear_execution_status()
        input_url, detected_format = self._get_video_input_data()
        self._log_format_detection(detected_format)
        self.append_value_to_parameter("logs", "[Processing video crop..]\n")
        try:
            self.append_value_to_parameter("logs", "[Started video cropping..]\n")
            yield lambda: self._process(input_url, detected_format)
            self.append_value_to_parameter("logs", "[Finished video cropping.]\n")
            self._set_status_results(was_successful=True, result_details="Successfully cropped video")
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error cropping video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            self._set_status_results(was_successful=False, result_details=f"Video cropping failed: {error_message}")
            self._handle_failure_exception(ValueError(msg))
