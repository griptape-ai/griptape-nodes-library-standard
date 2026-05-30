from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
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
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor

NO_ZOOM = 100.0


class CropVideoInteractive(BaseVideoProcessor):
    """Crop a video to a pixel-precise area using an interactive drag-handle editor."""

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
                    "video_width": 0,
                    "video_height": 0,
                    "left": 0,
                    "top": 0,
                    "width": 0,
                    "height": 0,
                    "total_frames": 0,
                    "locked": [],
                },
                tooltip="Interactive crop editor — drag to set crop area",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="CropVideoEditor", library="Griptape Nodes Library")},
            )
        )

        with ParameterGroup(name="crop_coordinates", ui_options={"collapsed": True}) as crop_coordinates:
            ParameterInt(
                name="left",
                default_value=0,
                tooltip="Left edge of crop area in pixels",
                traits={Slider(min_val=0, max_val=7680)},
            )
            ParameterInt(
                name="top",
                default_value=0,
                tooltip="Top edge of crop area in pixels",
                traits={Slider(min_val=0, max_val=4320)},
            )
            ParameterInt(
                name="width",
                default_value=0,
                tooltip="Width of crop area in pixels (0 = full width)",
                traits={Slider(min_val=0, max_val=7680)},
            )
            ParameterInt(
                name="height",
                default_value=0,
                tooltip="Height of crop area in pixels (0 = full height)",
                traits={Slider(min_val=0, max_val=4320)},
            )
        self.add_node_element(crop_coordinates)

        # hide the input video
        video_param = self.get_parameter_by_name("video")
        if video_param:
            video_param.ui_options["hide_parameter"] = True

    def _get_processing_description(self) -> str:
        return "cropping video"

    # ── Connection detection ───────────────────────────────────────────────────

    _CROP_PARAMS = frozenset({"left", "top", "width", "height"})

    def _get_locked_params(self) -> list[str]:
        try:
            result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
            if isinstance(result, ListConnectionsForNodeResultSuccess):
                return [c.target_parameter_name for c in result.incoming_connections
                        if c.target_parameter_name in self._CROP_PARAMS]
        except Exception:
            pass
        return []

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in self._CROP_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in self._CROP_PARAMS:
            self._refresh_locked_in_widget()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # ── Video URL resolution ───────────────────────────────────────────────────

    def _extract_video_path(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        try:
            if hasattr(value, "value"):
                v = getattr(value, "value", None)
                if isinstance(v, str):
                    return v
        except Exception:
            pass
        return None

    def _resolve_video_url(self, value: Any) -> str:
        raw = value if isinstance(value, str) else (getattr(value, "value", "") or "")
        if not raw:
            return ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw
        try:
            resolved = File(raw).resolve()
        except Exception:
            resolved = str(raw)
        try:
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return raw

    # ── Widget ↔ parameter sync ────────────────────────────────────────────────

    def _build_widget_dict(self, video_url: str = "", video_width: int = 0, video_height: int = 0, total_frames: int = 0) -> dict:
        return {
            "video_url": video_url,
            "video_width": video_width,
            "video_height": video_height,
            "left": self.get_parameter_value("left") or 0,
            "top": self.get_parameter_value("top") or 0,
            "width": self.get_parameter_value("width") or 0,
            "height": self.get_parameter_value("height") or 0,
            "total_frames": total_frames,
            "locked": self._get_locked_params(),
        }

    def _push_widget(self, new_dict: dict) -> None:
        self.set_parameter_value("crop_editor", new_dict)
        self.publish_update_to_parameter("crop_editor", new_dict)

    def _refresh_locked_in_widget(self) -> None:
        existing = self.get_parameter_value("crop_editor") or {}
        new_locked = self._get_locked_params()
        if set(existing.get("locked") or []) == set(new_locked):
            return
        self._syncing_to_widget = True
        try:
            self._push_widget({**existing, "locked": new_locked})
        finally:
            self._syncing_to_widget = False

    def _update_widget_coords(self) -> None:
        existing = self.get_parameter_value("crop_editor") or {}
        new_left = self.get_parameter_value("left") or 0
        new_top = self.get_parameter_value("top") or 0
        new_width = self.get_parameter_value("width") or 0
        new_height = self.get_parameter_value("height") or 0
        if (existing.get("left") == new_left and existing.get("top") == new_top
                and existing.get("width") == new_width and existing.get("height") == new_height):
            return
        new_dict = {**existing, "left": new_left, "top": new_top,
                    "width": new_width, "height": new_height, "locked": self._get_locked_params()}
        self._syncing_to_widget = True
        try:
            self._push_widget(new_dict)
        finally:
            self._syncing_to_widget = False

    def _sync_params_from_widget(self, widget_dict: dict) -> None:
        locked = set(widget_dict.get("locked", []))
        mapping = {"left": "left", "top": "top", "width": "width", "height": "height"}
        self.publish_update_to_parameter("crop_editor", widget_dict)
        self._syncing_to_params = True
        try:
            for wkey, pkey in mapping.items():
                if wkey in widget_dict and pkey not in locked:
                    val = widget_dict[wkey]
                    self.set_parameter_value(pkey, val)
                    self.publish_update_to_parameter(pkey, val)
        finally:
            self._syncing_to_params = False

    # ── after_value_set ────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "video":
            if not value:
                return super().after_value_set(parameter, value)
            path = self._extract_video_path(value)
            if not path:
                return super().after_value_set(parameter, value)
            try:
                _, ffprobe_path = self._get_ffmpeg_paths()
                resolved = File(path).resolve()
                frame_rate, (vw, vh), duration = self._detect_video_properties(resolved, ffprobe_path)
                total_frames = max(0, int(duration * frame_rate))
            except Exception as e:
                logger.error("%s: Could not read video properties: %s", self.name, e)
                vw, vh, total_frames = 0, 0, 0  # widget fills dims from <video> loadedmetadata

            # Update slider max values to match video dimensions
            for pname, max_val in [("left", vw), ("top", vh), ("width", vw), ("height", vh)]:
                param = self.get_parameter_by_name(pname)
                if param:
                    param.update_ui_options({"slider": {"max_val": max_val}})

            url = self._resolve_video_url(value)
            new_dict = self._build_widget_dict(video_url=url, video_width=vw, video_height=vh, total_frames=total_frames)
            self._syncing_to_widget = True
            try:
                self._push_widget(new_dict)
            finally:
                self._syncing_to_widget = False

        elif parameter.name == "crop_editor" and not self._syncing_to_widget:
            if isinstance(value, dict):
                self._sync_params_from_widget(value)

        elif not self._syncing_to_params and parameter.name in self._CROP_PARAMS:
            self._update_widget_coords()

        return super().after_value_set(parameter, value)

    # ── FFmpeg command ─────────────────────────────────────────────────────────

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        _, (video_width, video_height), _ = self._detect_video_properties(input_url, ffprobe_path)

        left = self.get_parameter_value("left") or 0
        top = self.get_parameter_value("top") or 0
        w = self.get_parameter_value("width") or video_width
        h = self.get_parameter_value("height") or video_height

        # Clamp to video bounds
        left = max(0, min(left, video_width - 1))
        top = max(0, min(top, video_height - 1))
        w = min(w, video_width - left)
        h = min(h, video_height - top)

        # Round to even dimensions for codec compatibility
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
        w = self.get_parameter_value("width") or 0
        h = self.get_parameter_value("height") or 0
        return f"_crop_{w}x{h}" if w and h else "_crop"

    # ── Validation ─────────────────────────────────────────────────────────────

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []
        video = self.get_parameter_value("video")
        if not video:
            exceptions.append(Exception(f"{self.name} - Input video is required"))
        for pname in ("left", "top", "width", "height"):
            val = self.get_parameter_value(pname)
            if val is not None and val < 0:
                exceptions.append(Exception(f"{self.name} - {pname} must be non-negative, got {val}"))
        return exceptions or None

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
