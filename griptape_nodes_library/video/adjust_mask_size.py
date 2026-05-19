import subprocess
import tempfile
from pathlib import Path
from typing import Any

import static_ffmpeg.run  # type: ignore[import-untyped]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget


class AdjustMaskSize(DataNode):
    """Adjust mask size in a video by dilating (expanding) or eroding (shrinking) the mask.

    Positive values dilate (expand) the mask, negative values erode (shrink) it.
    Processes each frame individually while maintaining video properties.
    """

    MIN_ADJUSTMENT = -25
    MAX_ADJUSTMENT = 25
    DEFAULT_ADJUSTMENT = 0

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Original video input (for preview widget)
        self.add_parameter(
            ParameterVideo(
                name="original_video",
                tooltip="Original video for mask preview overlay (optional)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={
                    "display_name": "Original Video",
                    "hide_property": True,
                },
            )
        )

        # Input mask video parameter
        self.add_parameter(
            ParameterVideo(
                name="mask_video",
                tooltip="Input mask video to adjust",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "display_name": "Mask Video",
                    "hide_property": True,
                },
            )
        )

        # Preview widget parameter
        preview_param = Parameter(
            name="preview",
            type="dict",
            default_value={
                "original_video_url": "",
                "mask_video_url": "",
                "adjustment": 0,
                "current_frame": 0,
                "total_frames": 0,
            },
            tooltip="Interactive preview of mask adjustment",
            allowed_modes={ParameterMode.PROPERTY},
            ui_options={"display_name": "Preview"},
        )
        preview_param.add_trait(Widget(name="MaskAdjustmentPreview", library="Griptape Nodes Library"))
        self.add_parameter(preview_param)

        # Output video parameter
        self.add_parameter(
            ParameterVideo(
                name="output_mask",
                default_value=None,
                tooltip="Adjusted mask video",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True, "expander": True, "display_name": "Adjusted Mask Video"},
            )
        )

        # Create progress bar component
        self.progress_component = ProgressBarComponent(self)
        self.progress_component.add_property_parameters()

        # Output file parameter (controls save location)
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="adjusted_mask.mp4")
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> Any:
        """Called after a parameter value is set.

        Updates the preview widget when video inputs or adjustment changes.

        Args:
            parameter: The parameter that changed
            value: The new value
        """
        # Update preview when relevant parameters change
        if parameter.name in ["original_video", "mask_video"]:
            self._update_preview()

        return super().after_value_set(parameter, value)

    def _resolve_video_url(self, video_artifact: Any) -> str:
        """Resolve a video artifact to a browser-accessible presigned URL.

        Uses the same CreateStaticFileDownloadUrlFromPathRequest that the editor
        uses to convert file paths and macro paths to presigned HTTP URLs.

        Args:
            video_artifact: A VideoUrlArtifact or similar artifact with a .value path/URL

        Returns:
            Presigned HTTP URL string, or empty string on failure
        """
        if not video_artifact:
            return ""

        value = video_artifact.value
        if not value:
            return ""

        # Already a browser-accessible URL — do not re-resolve
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            return value

        try:
            from griptape_nodes.retained_mode.events.static_file_events import (
                CreateStaticFileDownloadUrlFromPathRequest,
                CreateStaticFileDownloadUrlFromPathResultSuccess,
            )

            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=value))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass

        return ""

    def _update_preview(self) -> None:
        """Update preview widget with current video URLs, preserving widget-owned state.

        Skips the update entirely when the resolved URLs haven't changed, which
        prevents unnecessary widget rebuilds during processing (each rebuild
        creates new <video> elements, exhausting Chrome's WebMediaPlayer limit).
        """
        original_video = self.get_parameter_value("original_video")
        mask_video = self.get_parameter_value("mask_video")

        original_video_url = self._resolve_video_url(original_video)
        mask_video_url = self._resolve_video_url(mask_video)

        preview = self.get_parameter_value("preview") or {}
        current_frame = preview.get("current_frame", 0)
        total_frames = preview.get("total_frames", 0)
        adjustment = preview.get("adjustment", 0)

        # Skip if URLs haven't changed — avoids unnecessary widget rebuild
        if preview.get("original_video_url") == original_video_url and preview.get("mask_video_url") == mask_video_url:
            return

        self.set_parameter_value(
            "preview",
            {
                "original_video_url": original_video_url,
                "mask_video_url": mask_video_url,
                "adjustment": adjustment,
                "current_frame": current_frame,
                "total_frames": total_frames,
            },
        )

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        # Validate mask video input
        mask_video = self.get_parameter_value("mask_video")
        if not mask_video:
            msg = f"{self.name}: Mask video is required"
            exceptions.append(ValueError(msg))

        # Validate adjustment value from preview widget
        preview = self.get_parameter_value("preview") or {}
        adjustment = preview.get("adjustment", 0)
        if adjustment < self.MIN_ADJUSTMENT or adjustment > self.MAX_ADJUSTMENT:
            msg = f"{self.name}: Adjustment must be between {self.MIN_ADJUSTMENT} and {self.MAX_ADJUSTMENT}, got {adjustment}"
            exceptions.append(ValueError(msg))

        return exceptions or None

    def process(self) -> AsyncResult[None]:
        """Process video asynchronously."""
        # Reset progress and output
        self.progress_component.reset()
        self.parameter_output_values["output_mask"] = None

        mask_video = self.get_parameter_value("mask_video")
        preview = self.get_parameter_value("preview") or {}
        adjustment = preview.get("adjustment", 0)

        if not mask_video:
            return

        # If adjustment is 0, just pass through the input
        if adjustment == 0:
            self.parameter_output_values["output_mask"] = mask_video
            return

        try:
            yield lambda: self._process_mask_video(mask_video, adjustment)
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error adjusting mask video: {error_message}"
            raise ValueError(msg) from e

    def _process_mask_video(self, mask_video: Any, adjustment: int) -> None:
        """Process mask video using FFmpeg's native morphological filters.

        Chains dilation or erosion N times (once per pixel of adjustment),
        replacing the Python frame-extract → process → reassemble pipeline
        with a single FFmpeg invocation.
        """
        output_video: Path | None = None
        try:
            ffmpeg_path, _ = self._get_ffmpeg_paths()
            video_url = File(mask_video.value).resolve()
            self._validate_url_safety(video_url)

            # Chain dilation or erosion N times for N-pixel radius effect
            steps = abs(adjustment)
            filter_name = "dilation" if adjustment > 0 else "erosion"
            vf = "format=gray," + ",".join([filter_name] * steps)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_video = Path(tmp.name)

            self.progress_component.initialize(2)

            cmd = [
                ffmpeg_path,
                "-i",
                video_url,
                "-vf",
                vf,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-preset",
                "medium",
                "-crf",
                "18",
                "-y",
                str(output_video),
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)  # noqa: S603
            except subprocess.CalledProcessError as e:
                msg = f"{self.name}: FFmpeg morphological filter failed: {e.stderr}"
                raise ValueError(msg) from e
            except subprocess.TimeoutExpired as e:
                msg = f"{self.name}: FFmpeg morphological filter timed out"
                raise ValueError(msg) from e

            self.progress_component.increment()

            with output_video.open("rb") as f:
                video_bytes = f.read()

            dest = self._output_file.build_file()
            saved = dest.write_bytes(video_bytes)
            self.parameter_output_values["output_mask"] = VideoUrlArtifact(saved.location)

            self.progress_component.increment()

        finally:
            if output_video and output_video.exists():
                output_video.unlink(missing_ok=True)

    def _get_ffmpeg_paths(self) -> tuple[str, str]:
        """Get FFmpeg and FFprobe executable paths."""
        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            return ffmpeg_path, ffprobe_path  # noqa: TRY300
        except Exception as e:
            error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
            raise ValueError(error_msg) from e

    def _validate_url_safety(self, url: str) -> None:
        """Validate that the URL is safe for ffmpeg processing."""
        from griptape_nodes_library.utils.video_utils import validate_url

        if not validate_url(url):
            msg = f"{self.name}: Invalid or unsafe URL provided: {url}"
            raise ValueError(msg)
