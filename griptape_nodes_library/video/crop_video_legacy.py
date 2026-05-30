import os
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import DeprecationMessage, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from PIL import Image, ImageDraw

from griptape_nodes_library.utils.image_utils import image_to_bytes
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

PREVIEW_BG_COLOR = (128, 128, 128)
PREVIEW_BG_OUTLINE = (255, 255, 255)
PREVIEW_CROP_OUTLINE = (0, 100, 255)
PREVIEW_MAX_WIDTH = 800
PREVIEW_MAX_HEIGHT = 600
PREVIEW_WEBP_QUALITY = 60


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
    """Legacy preset-based crop node. Use CropVideo for interactive pixel-level control."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self._cached_first_frame: Image.Image | None = None
        self._cached_video_url: str | None = None

    def _setup_custom_parameters(self) -> None:
        msg = DeprecationMessage(
            value="This node uses legacy preset-based cropping. Use the new Crop Video node for interactive pixel-level control.",
            button_text="Create New Crop Video",
            migrate_function=self._migrate_to_new_crop_video,
        )
        self.add_node_element(msg)

        size_param = ParameterString(
            name="crop_size",
            default_value="Custom",
            tooltip="Target crop size. Choose a preset or enter custom dimensions (e.g., '1280x1024')",
        )
        size_param.add_trait(Options(choices=[*list(RESOLUTION_PRESETS.keys()), "Custom"]))
        self.add_parameter(size_param)

        self.add_parameter(ParameterInt(name="custom_width", default_value=800, tooltip="Custom crop width in pixels"))
        self.add_parameter(ParameterInt(name="custom_height", default_value=800, tooltip="Custom crop height in pixels"))

        position_param = ParameterString(
            name="crop_position",
            default_value="center",
            tooltip="Where to position the crop area",
        )
        position_param.add_trait(
            Options(choices=["center", "top-left", "top-right", "bottom-left", "bottom-right",
                              "top-center", "bottom-center", "left-center", "right-center"])
        )
        self.add_parameter(position_param)

        self.add_parameter(
            ParameterImage(
                name="preview",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Preview image showing the crop area overlaid on the first frame of the video",
                ui_options={"expander": True},
            )
        )

    def _migrate_to_new_crop_video(self, button: Any, details: Any) -> None:
        pos = self.metadata.get("position", {"x": 0, "y": 0})
        result = GriptapeNodes.handle_request(
            CreateNodeRequest(
                node_type="CropVideoInteractive",
                specific_library_name="Griptape Nodes Library",
                metadata={"position": {"x": pos.get("x", 0), "y": pos.get("y", 0) + 350}},
            )
        )
        if not isinstance(result, CreateNodeResultSuccess):
            return
        new_node = result.node_name

        # Re-wire the video input if one exists on this node
        connections = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if isinstance(connections, ListConnectionsForNodeResultSuccess):
            for conn in connections.incoming_connections:
                if conn.target_parameter_name == "video":
                    GriptapeNodes.handle_request(
                        CreateConnectionRequest(
                            source_node_name=conn.source_node_name,
                            source_parameter_name=conn.source_parameter_name,
                            target_node_name=new_node,
                            target_parameter_name="video",
                        )
                    )
                    break

        # Convert crop_size → width/height
        crop_dims = self._get_crop_dimensions()
        w, h = crop_dims if crop_dims else (0, 0)

        # Convert crop_position → left/top using video dimensions
        left, top = 0, 0
        if w and h:
            video_dims = self._get_video_dimensions_for_preview()
            if video_dims:
                vw, vh = video_dims
                cw = min(w, vw)
                ch = min(h, vh)
                left, top = self._calculate_crop_coordinates(vw, vh, cw, ch)

        if w and h:
            GriptapeNodes.handle_request(
                SetParameterValueRequest(parameter_name="width", value=w, node_name=new_node)
            )
            GriptapeNodes.handle_request(
                SetParameterValueRequest(parameter_name="height", value=h, node_name=new_node)
            )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(parameter_name="left", value=left, node_name=new_node)
        )
        GriptapeNodes.handle_request(
            SetParameterValueRequest(parameter_name="top", value=top, node_name=new_node)
        )

    def _get_processing_description(self) -> str:
        return "cropping video"

    def _extract_first_frame(self, input_url: str) -> Image.Image | None:
        try:
            self._validate_url_safety(input_url)
            ffmpeg_path, _ = self._get_ffmpeg_paths()
            fd, temp_path_str = tempfile.mkstemp(suffix=".webp")
            os.close(fd)
            temp_path = Path(temp_path_str)
            try:
                scale_filter = f"scale=min({PREVIEW_MAX_WIDTH}\\,iw):min({PREVIEW_MAX_HEIGHT}\\,ih):force_original_aspect_ratio=decrease"
                cmd = [ffmpeg_path, "-y", "-ss", "0", "-i", input_url, "-vframes", "1",
                       "-vf", scale_filter, "-f", "webp", "-quality", str(PREVIEW_WEBP_QUALITY), str(temp_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)  # noqa: S603
                time.sleep(0.1)
                if temp_path.exists() and temp_path.stat().st_size > 0:
                    return Image.open(temp_path)
                self.append_value_to_parameter("logs", f"Warning: FFmpeg did not create output file. stderr: {result.stderr}\n")
            except subprocess.CalledProcessError as e:
                self.append_value_to_parameter("logs", f"Warning: Could not extract first frame: {e.stderr}\n")
            finally:
                with suppress(Exception):
                    temp_path.unlink(missing_ok=True)
        except (ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            self.append_value_to_parameter("logs", f"Warning: Could not extract first frame for preview: {e}\n")
        return None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "crop_size":
            if value == "Custom":
                self.show_parameter_by_name("custom_width")
                self.show_parameter_by_name("custom_height")
            else:
                self.hide_parameter_by_name("custom_width")
                self.hide_parameter_by_name("custom_height")
            self._generate_preview()
        if parameter.name in ["custom_width", "custom_height", "crop_position"]:
            self._generate_preview()
        if parameter.name == "video":
            if value:
                try:
                    video_artifact = to_video_artifact(value)
                    if video_artifact and hasattr(video_artifact, "value") and video_artifact.value:
                        input_url = File(video_artifact.value).resolve()
                        if input_url != self._cached_video_url:
                            self._cached_first_frame = self._extract_first_frame(input_url)
                            self._cached_video_url = input_url
                    else:
                        self._cached_first_frame = None
                        self._cached_video_url = None
                except (ValueError, TypeError, AttributeError) as e:
                    self._cached_first_frame = None
                    self._cached_video_url = None
                    self.append_value_to_parameter("logs", f"Warning: Could not process video for preview: {e}\n")
            else:
                self._cached_first_frame = None
                self._cached_video_url = None
            self._generate_preview()
        return super().after_value_set(parameter, value)

    def _get_crop_dimensions(self) -> tuple[int, int] | None:
        crop_size = self.get_parameter_value("crop_size") or "Custom"
        if crop_size == "Custom":
            w = self.get_parameter_value("custom_width") or 1024
            h = self.get_parameter_value("custom_height") or 1024
            if w <= 0 or h <= 0:
                return None
            return (w, h)
        if crop_size in RESOLUTION_PRESETS:
            return RESOLUTION_PRESETS[crop_size]
        return parse_size_string(crop_size)

    def _calculate_crop_coordinates(self, video_width: int, video_height: int, crop_width: int, crop_height: int) -> tuple[int, int]:
        position = self.get_parameter_value("crop_position") or "center"
        positions = {
            "center":        ((video_width - crop_width) // 2,  (video_height - crop_height) // 2),
            "top-left":      (0,                                 0),
            "top-right":     (video_width - crop_width,          0),
            "bottom-left":   (0,                                 video_height - crop_height),
            "bottom-right":  (video_width - crop_width,          video_height - crop_height),
            "top-center":    ((video_width - crop_width) // 2,   0),
            "bottom-center": ((video_width - crop_width) // 2,   video_height - crop_height),
            "left-center":   (0,                                 (video_height - crop_height) // 2),
            "right-center":  (video_width - crop_width,          (video_height - crop_height) // 2),
        }
        x, y = positions.get(position, ((video_width - crop_width) // 2, (video_height - crop_height) // 2))
        return (max(0, min(x, video_width - crop_width)), max(0, min(y, video_height - crop_height)))

    def _get_video_dimensions_for_preview(self) -> tuple[int, int] | None:
        video = self.parameter_values.get("video")
        if not video:
            return None
        try:
            video_artifact = to_video_artifact(video)
        except (ValueError, TypeError, AttributeError):
            return None
        if not video_artifact or not hasattr(video_artifact, "value") or not video_artifact.value:
            return None
        input_url = File(video_artifact.value).resolve()
        try:
            self._validate_url_safety(input_url)
        except ValueError as e:
            self.append_value_to_parameter("logs", f"Warning: Invalid video URL for preview: {e}\n")
            return None
        try:
            _ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        except ValueError as e:
            self.append_value_to_parameter("logs", f"Warning: FFmpeg not available for preview: {e}\n")
            return None
        try:
            _, (video_width, video_height), _ = self._detect_video_properties(input_url, ffprobe_path)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError) as e:
            self.append_value_to_parameter("logs", f"Warning: Could not get video dimensions for preview: {e}\n")
            return None
        return (video_width, video_height)

    def _create_preview_image_with_overlay(self, preview_size, crop_rect, scale_factor):
        preview_width, preview_height = preview_size
        preview_crop_x, preview_crop_y, preview_crop_width, preview_crop_height = crop_rect
        if self._cached_first_frame:
            preview_image = self._cached_first_frame.copy().resize((preview_width, preview_height), Image.Resampling.LANCZOS)
        else:
            preview_image = Image.new("RGB", (preview_width, preview_height), PREVIEW_BG_COLOR)
        draw = ImageDraw.Draw(preview_image)
        outline_width = max(1, int(2 * scale_factor))
        draw.rectangle([(0, 0), (preview_width - 1, preview_height - 1)], outline=PREVIEW_BG_OUTLINE, width=outline_width)
        overlay = Image.new("RGBA", (preview_width, preview_height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([(0, 0), (preview_width, preview_height)], fill=(0, 0, 0, 128))
        overlay_draw.rectangle(
            [(preview_crop_x, preview_crop_y), (preview_crop_x + preview_crop_width, preview_crop_y + preview_crop_height)],
            fill=(0, 0, 0, 0),
        )
        preview_image = Image.alpha_composite(preview_image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(preview_image)
        crop_outline_width = max(2, int(3 * scale_factor))
        draw.rectangle(
            [(preview_crop_x, preview_crop_y), (preview_crop_x + preview_crop_width, preview_crop_y + preview_crop_height)],
            outline=PREVIEW_CROP_OUTLINE,
            width=crop_outline_width,
        )
        return preview_image

    def _generate_preview(self) -> None:
        video_dims = self._get_video_dimensions_for_preview()
        if not video_dims:
            return
        video_width, video_height = video_dims
        crop_dims = self._get_crop_dimensions()
        if not crop_dims:
            return
        crop_width, crop_height = min(crop_dims[0], video_width), min(crop_dims[1], video_height)
        crop_x, crop_y = self._calculate_crop_coordinates(video_width, video_height, crop_width, crop_height)
        scale_factor = min(1920 / video_width, 1080 / video_height, 1.0)
        preview_size = (int(video_width * scale_factor), int(video_height * scale_factor))
        crop_rect = (int(crop_x * scale_factor), int(crop_y * scale_factor),
                     int(crop_width * scale_factor), int(crop_height * scale_factor))
        preview_image = self._create_preview_image_with_overlay(preview_size, crop_rect, scale_factor)
        try:
            preview_bytes = image_to_bytes(preview_image, "PNG")
            safe_name = self.name.replace(" ", "_").replace("/", "_")
            dest = ProjectFileDestination.from_situation(
                filename="preview.png", situation="save_griptape_nodes_preview",
                source_file_name=f"crop_video_preview_{safe_name}", preview_format="png",
            )
            preview_saved = dest.write_bytes(preview_bytes)
            self.parameter_output_values["preview"] = ImageUrlArtifact(preview_saved.location)
        except (ValueError, OSError) as e:
            self.append_value_to_parameter("logs", f"Warning: Could not save preview image: {e}\n")

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:  # noqa: ARG002
        ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        _, (video_width, video_height), _ = self._detect_video_properties(input_url, ffprobe_path)
        crop_dims = self._get_crop_dimensions()
        if not crop_dims:
            msg = f"{self.name}: Invalid crop dimensions"
            raise ValueError(msg)
        crop_width = min(crop_dims[0], video_width)
        crop_height = min(crop_dims[1], video_height)
        crop_x, crop_y = self._calculate_crop_coordinates(video_width, video_height, crop_width, crop_height)
        crop_x = max(0, min(crop_x, video_width - crop_width))
        crop_y = max(0, min(crop_y, video_height - crop_height))
        crop_width = (crop_width // 2) * 2
        crop_height = (crop_height // 2) * 2
        if crop_width <= 0 or crop_height <= 0:
            msg = f"{self.name}: Crop dimensions too small after rounding to even numbers"
            raise ValueError(msg)
        crop_filter = f"crop={crop_width}:{crop_height}:{crop_x}:{crop_y}"
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

    def process(self) -> AsyncResult[None]:
        self._clear_execution_status()
        self._generate_preview()
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
