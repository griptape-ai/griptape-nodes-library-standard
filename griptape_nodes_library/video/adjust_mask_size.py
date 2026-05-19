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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Original video input — optional, used for mask overlay preview
        self.add_parameter(
            ParameterVideo(
                name="original_video",
                tooltip="Original video for mask preview overlay (optional)",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Original Video", "hide_property": True},
            )
        )

        # Input mask video
        self.add_parameter(
            ParameterVideo(
                name="mask_video",
                tooltip="Input mask video to adjust",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Mask Video", "hide_property": True},
            )
        )

        # Adjustment slider with floating preview tooltip.
        # Value is a dict so the widget can carry video URLs for the tooltip preview.
        # The node reads adjustment["value"] for actual processing.
        self.add_parameter(
            Parameter(
                name="adjustment",
                type="dict",
                default_value={"value": 0, "mask_video_url": "", "original_video_url": ""},
                tooltip="Drag to preview mask dilation (+) or erosion (−). Range: −25 to +25 pixels.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Widget(name="MaskAdjustmentPreview", library="Griptape Nodes Library")},
                ui_options={"display_name": "Mask Adjustment"},
            )
        )

        # Adjusted mask video output
        self.add_parameter(
            ParameterVideo(
                name="output_mask",
                default_value=None,
                tooltip="Adjusted mask video",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True, "expander": True, "display_name": "Adjusted Mask Video"},
            )
        )

        self.progress_component = ProgressBarComponent(self)
        self.progress_component.add_property_parameters()

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="adjusted_mask.mp4")
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> Any:
        if parameter.name in ["original_video", "mask_video"]:
            self._update_adjustment_urls()
        return super().after_value_set(parameter, value)

    def _resolve_video_url(self, video_artifact: Any) -> str:
        """Return a browser-accessible URL for a video artifact, or empty string."""
        if not video_artifact:
            return ""
        val = video_artifact.value
        if not val:
            return ""
        if isinstance(val, str) and val.startswith(("http://", "https://")):
            return val
        try:
            from griptape_nodes.retained_mode.events.static_file_events import (
                CreateStaticFileDownloadUrlFromPathRequest,
                CreateStaticFileDownloadUrlFromPathResultSuccess,
            )

            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=val))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return ""

    def _url_base(self, url: str) -> str:
        """Strip query params from a URL for stable comparison."""
        return url.split("?")[0] if url else ""

    def _update_adjustment_urls(self) -> None:
        """Push current video URLs into the adjustment widget value.

        Skips the update when the underlying file paths haven't changed
        (presigned URLs include a changing timestamp in the query string).
        """
        orig_url = self._resolve_video_url(self.get_parameter_value("original_video"))
        mask_url = self._resolve_video_url(self.get_parameter_value("mask_video"))

        adj = self.get_parameter_value("adjustment") or {}
        adj_value = adj.get("value", 0) if isinstance(adj, dict) else int(adj) if isinstance(adj, (int, float)) else 0

        stored_orig = self._url_base(adj.get("original_video_url", "") if isinstance(adj, dict) else "")
        stored_mask = self._url_base(adj.get("mask_video_url", "") if isinstance(adj, dict) else "")
        new_orig = self._url_base(orig_url)
        new_mask = self._url_base(mask_url)

        # Always update on first call (both empty) to initialise the widget;
        # skip only when non-empty URLs are unchanged.
        urls_unchanged = stored_orig == new_orig and stored_mask == new_mask
        any_url_set = stored_orig or stored_mask or new_orig or new_mask
        if urls_unchanged and any_url_set:
            return

        self.set_parameter_value(
            "adjustment",
            {"value": adj_value, "mask_video_url": mask_url, "original_video_url": orig_url},
        )

    def _get_adjustment_value(self) -> int:
        """Extract the integer adjustment from the widget parameter value."""
        adj = self.get_parameter_value("adjustment")
        if isinstance(adj, dict):
            return int(adj.get("value", 0))
        if isinstance(adj, (int, float)):
            return int(adj)
        return 0

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        if not self.get_parameter_value("mask_video"):
            exceptions.append(ValueError(f"{self.name}: Mask video is required"))

        adjustment = self._get_adjustment_value()
        if not (self.MIN_ADJUSTMENT <= adjustment <= self.MAX_ADJUSTMENT):
            exceptions.append(
                ValueError(
                    f"{self.name}: Adjustment must be between {self.MIN_ADJUSTMENT} and {self.MAX_ADJUSTMENT}, got {adjustment}"
                )
            )

        return exceptions or None

    def process(self) -> AsyncResult[None]:
        """Process video asynchronously."""
        self.progress_component.reset()
        self.parameter_output_values["output_mask"] = None

        mask_video = self.get_parameter_value("mask_video")
        adjustment = self._get_adjustment_value()

        if not mask_video:
            return

        if adjustment == 0:
            self.parameter_output_values["output_mask"] = mask_video
            return

        try:
            yield lambda: self._process_mask_video(mask_video, adjustment)
        except Exception as e:
            msg = f"{self.name}: Error adjusting mask video: {e}"
            raise ValueError(msg) from e

    def _process_mask_video(self, mask_video: Any, adjustment: int) -> None:
        """Process mask video using FFmpeg's native morphological filters."""
        output_video: Path | None = None
        try:
            ffmpeg_path, _ = self._get_ffmpeg_paths()
            video_url = File(mask_video.value).resolve()
            self._validate_url_safety(video_url)

            steps = abs(adjustment)
            filter_name = "dilation" if adjustment > 0 else "erosion"
            vf = "format=gray," + ",".join([filter_name] * steps)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                output_video = Path(tmp.name)

            self.progress_component.initialize(2)

            cmd = [
                ffmpeg_path, "-i", video_url,
                "-vf", vf,
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "18",
                "-y", str(output_video),
            ]

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=600)  # noqa: S603
            except subprocess.CalledProcessError as e:
                raise ValueError(f"{self.name}: FFmpeg morphological filter failed: {e.stderr}") from e
            except subprocess.TimeoutExpired as e:
                raise ValueError(f"{self.name}: FFmpeg morphological filter timed out") from e

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
        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            return ffmpeg_path, ffprobe_path  # noqa: TRY300
        except Exception as e:
            raise ValueError(f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}") from e

    def _validate_url_safety(self, url: str) -> None:
        from griptape_nodes_library.utils.video_utils import validate_url

        if not validate_url(url):
            raise ValueError(f"{self.name}: Invalid or unsafe URL provided: {url}")
