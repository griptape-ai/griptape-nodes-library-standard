import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import static_ffmpeg.run  # type: ignore[import-untyped]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget
from PIL import Image, ImageFilter


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

        # Logging group
        self._setup_logging_group()

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
                self.append_value_to_parameter("logs", f"Resolved video URL: {result.url}\n")
                return result.url

            self.append_value_to_parameter(
                "logs", f"Failed to resolve video URL for '{value}': {result.result_details}\n"
            )
        except Exception as e:
            self.append_value_to_parameter("logs", f"Failed to resolve video URL for '{value}': {e}\n")

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

    def _setup_logging_group(self) -> None:
        """Setup the common logging parameter group."""
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

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
        self.append_value_to_parameter("logs", "[Starting mask adjustment..]\n")

        mask_video = self.get_parameter_value("mask_video")
        preview = self.get_parameter_value("preview") or {}
        adjustment = preview.get("adjustment", 0)

        if not mask_video:
            return

        # If adjustment is 0, just pass through the input
        if adjustment == 0:
            self.append_value_to_parameter("logs", "Adjustment is 0 - returning input video unchanged\n")
            self.parameter_output_values["output_mask"] = mask_video
            return

        try:
            yield lambda: self._process_mask_video(mask_video, adjustment)
            self.append_value_to_parameter("logs", "[Finished mask adjustment.]\n")
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error adjusting mask video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e

    def _process_mask_video(self, mask_video: Any, adjustment: int) -> None:
        """Process mask video by adjusting each frame."""
        temp_dirs: list[Path] = []
        temp_files: list[Path] = []

        try:
            # Get FFmpeg paths
            ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()

            # Get video URL
            video_url = File(mask_video.value).resolve()
            self._validate_url_safety(video_url)

            # Get video properties
            props = self._get_video_properties(video_url, ffprobe_path)
            self.append_value_to_parameter(
                "logs",
                f"Video: {props['width']}x{props['height']}, {props['fps']:.2f} fps, {props['frame_count']} frames\n",
            )

            # Extract frames
            self.append_value_to_parameter("logs", "Extracting frames from video...\n")
            frames_dir = Path(tempfile.mkdtemp())
            temp_dirs.append(frames_dir)
            self._extract_frames(video_url, frames_dir, ffmpeg_path)

            # Adjust frames
            self.append_value_to_parameter(
                "logs",
                f"Adjusting mask size by {adjustment} pixels ({'dilation' if adjustment > 0 else 'erosion'})...\n",
            )
            adjusted_frames_dir = Path(tempfile.mkdtemp())
            temp_dirs.append(adjusted_frames_dir)
            last_progress = self._adjust_frames(frames_dir, adjusted_frames_dir, props["frame_count"], adjustment)

            # Reassemble video
            self.append_value_to_parameter("logs", "Reassembling video from adjusted frames...\n")
            output_video = self._reassemble_video(adjusted_frames_dir, props, ffmpeg_path, last_progress)
            temp_files.append(output_video)

            # Read video bytes
            with output_video.open("rb") as f:
                video_bytes = f.read()

            # Save output
            dest = self._output_file.build_file()
            saved = dest.write_bytes(video_bytes)
            output_artifact = VideoUrlArtifact(saved.location)

            self.parameter_output_values["output_mask"] = output_artifact
            self.append_value_to_parameter("logs", f"Successfully adjusted mask video: {saved.location}\n")

        finally:
            # Cleanup temp directories and files
            for temp_dir in temp_dirs:
                try:
                    import shutil

                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    self.append_value_to_parameter("logs", f"Warning: Failed to cleanup temp dir: {e}\n")

            for temp_file in temp_files:
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception as e:
                    self.append_value_to_parameter("logs", f"Warning: Failed to cleanup temp file: {e}\n")

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

    def _get_video_properties(self, video_url: str, ffprobe_path: str) -> dict[str, Any]:
        """Get video properties (resolution, fps, frame count)."""
        try:
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

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            streams_data = json.loads(result.stdout)

            if not streams_data.get("streams") or len(streams_data["streams"]) == 0:
                msg = f"{self.name}: No video stream found in {video_url}"
                raise ValueError(msg)

            video_stream = streams_data["streams"][0]

            # Get frame rate
            fps_str = video_stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = map(int, fps_str.split("/"))
                fps = num / den
            else:
                fps = float(fps_str)

            # Get resolution
            width = int(video_stream.get("width", 0))
            height = int(video_stream.get("height", 0))

            # Get frame count
            nb_frames = video_stream.get("nb_frames")
            if nb_frames:
                frame_count = int(nb_frames)
            else:
                # Calculate from duration and fps
                duration = float(video_stream.get("duration", 0))
                frame_count = int(duration * fps)

            return {"width": width, "height": height, "fps": fps, "frame_count": frame_count}

        except Exception as e:
            msg = f"{self.name}: Failed to get video properties: {e}"
            raise ValueError(msg) from e

    def _extract_frames(self, video_url: str, output_dir: Path, ffmpeg_path: str) -> None:
        """Extract all frames from video to directory."""
        output_pattern = str(output_dir / "frame_%06d.png")

        cmd = [
            ffmpeg_path,
            "-i",
            video_url,
            "-vf",
            "format=gray",  # Convert to grayscale
            "-y",
            output_pattern,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # noqa: S603
        except subprocess.CalledProcessError as e:
            msg = f"{self.name}: FFmpeg frame extraction failed: {e.stderr}"
            raise ValueError(msg) from e
        except subprocess.TimeoutExpired as e:
            msg = f"{self.name}: FFmpeg frame extraction timed out"
            raise ValueError(msg) from e

    def _adjust_frames(self, input_dir: Path, output_dir: Path, frame_count: int, adjustment: int) -> int:
        """Adjust mask size in each frame using dilation or erosion.

        Args:
            input_dir: Directory containing input frames
            output_dir: Directory to save adjusted frames
            frame_count: Number of frames to process
            adjustment: Adjustment amount (positive for dilation, negative for erosion)

        Returns:
            The last progress step (should be 9 after frame processing)
        """
        # 10 steps for frame processing + 1 step for reassembly = 11 total
        self.progress_component.initialize(11)
        last_step = 0

        # Create structuring element for morphological operations
        # Use a circular/elliptical kernel for more natural-looking results
        kernel_size = abs(adjustment)
        y, x = np.ogrid[-kernel_size : kernel_size + 1, -kernel_size : kernel_size + 1]
        kernel = x**2 + y**2 <= kernel_size**2
        kernel = kernel.astype(np.uint8)

        for frame_idx in range(1, frame_count + 1):
            frame_filename = f"frame_{frame_idx:06d}.png"
            input_path = input_dir / frame_filename
            output_path = output_dir / frame_filename

            if not input_path.exists():
                msg = f"{self.name}: Missing frame {frame_filename}"
                raise ValueError(msg)

            # Load frame
            frame_img = Image.open(input_path).convert("L")  # Ensure grayscale
            frame_array = np.array(frame_img)

            # Apply morphological operation
            if adjustment > 0:
                # Dilation (expand mask)
                adjusted_array = self._dilate_binary(frame_array > 127, kernel).astype(np.uint8) * 255
            else:
                # Erosion (shrink mask)
                adjusted_array = self._erode_binary(frame_array > 127, kernel).astype(np.uint8) * 255

            # Save adjusted frame
            adjusted_img = Image.fromarray(adjusted_array, mode="L")
            adjusted_img.save(output_path, "PNG")

            # Step progress at each 10% boundary (10 steps total for frame processing)
            current_step = int((frame_idx / frame_count) * 10)
            if current_step > last_step:
                for _ in range(current_step - last_step):
                    self.progress_component.increment()
                last_step = current_step
                pct = current_step * 10
                self.append_value_to_parameter("logs", f"Adjusted {pct}% ({frame_idx}/{frame_count} frames)\n")

        return last_step

    def _dilate_binary(self, binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Perform binary dilation on a mask using a structuring element.

        Uses PIL's MaxFilter to expand the mask. Dilation sets a pixel to True if any
        pixel in the kernel neighborhood is True.

        Args:
            binary_mask: Boolean array representing the binary mask
            kernel: Boolean or uint8 array representing the structuring element

        Returns:
            Boolean array with dilated mask
        """
        # Convert binary mask to uint8 image (0 or 255)
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")

        # Apply MaxFilter with kernel size
        dilated_img = mask_img.filter(ImageFilter.MaxFilter(size=kernel.shape[0]))

        # Convert back to boolean array
        dilated_array = np.array(dilated_img) > 127
        return dilated_array

    def _erode_binary(self, binary_mask: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Perform binary erosion on a mask using a structuring element.

        Uses PIL's MinFilter to shrink the mask. Erosion sets a pixel to False if any
        pixel in the kernel neighborhood is False.

        Args:
            binary_mask: Boolean array representing the binary mask
            kernel: Boolean or uint8 array representing the structuring element

        Returns:
            Boolean array with eroded mask
        """
        # Convert binary mask to uint8 image (0 or 255)
        mask_img = Image.fromarray((binary_mask * 255).astype(np.uint8), mode="L")

        # Apply MinFilter with kernel size
        eroded_img = mask_img.filter(ImageFilter.MinFilter(size=kernel.shape[0]))

        # Convert back to boolean array
        eroded_array = np.array(eroded_img) > 127
        return eroded_array

    def _reassemble_video(self, frames_dir: Path, props: dict[str, Any], ffmpeg_path: str, last_progress: int) -> Path:
        """Reassemble video from adjusted frames.

        Args:
            frames_dir: Directory containing adjusted frames
            props: Video properties (fps, width, height)
            ffmpeg_path: Path to FFmpeg executable
            last_progress: Current progress value from frame processing

        Returns:
            Path to reassembled video file
        """
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_video = Path(temp_file.name)
        input_pattern = str(frames_dir / "frame_%06d.png")

        # Ensure frame processing steps are complete before reassembly
        while last_progress < 10:
            self.progress_component.increment()
            last_progress += 1

        cmd = [
            ffmpeg_path,
            "-framerate",
            str(props["fps"]),
            "-i",
            input_pattern,
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

            # Final step for reassembly complete
            self.progress_component.increment()

        except subprocess.CalledProcessError as e:
            msg = f"{self.name}: FFmpeg video reassembly failed: {e.stderr}"
            raise ValueError(msg) from e
        except subprocess.TimeoutExpired as e:
            msg = f"{self.name}: FFmpeg video reassembly timed out"
            raise ValueError(msg) from e

        return output_video
