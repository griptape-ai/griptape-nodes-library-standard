import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import static_ffmpeg.run  # type: ignore[import-untyped]
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, DataNode
from griptape_nodes.exe_types.param_components.progress_bar_component import ProgressBarComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from PIL import Image, ImageChops


class CombineMasksVideo(DataNode):
    """Combine a list of mask videos into a single consolidated mask video (pixel-wise max per frame).

    Videos must have matching resolution but can have different durations.
    Shorter videos are automatically padded with black frames to match the longest video.
    """

    _ALPHA_NEAR_OPAQUE_MIN = 200
    _ALPHA_NEAR_OPAQUE_RANGE = 32
    _EXTREMA_TUPLE_LEN = 2

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Input parameter: list of mask videos
        self.add_parameter(
            ParameterList(
                name="mask_videos",
                input_types=[
                    "VideoArtifact",
                    "VideoUrlArtifact",
                    "list",
                    "list[VideoArtifact]",
                    "list[VideoUrlArtifact]",
                ],
                default_value=[],
                tooltip="List of mask videos to combine into a single mask video (union/max per frame). Videos must have the same resolution but can have different durations - shorter videos will be padded with black frames.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Mask Videos"},
            )
        )

        # Output video parameter
        self.add_parameter(
            ParameterVideo(
                name="output_mask",
                default_value=None,
                tooltip="Combined mask video.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True, "expander": True, "display_name": "Combined Mask Video"},
            )
        )

        # Create progress bar component
        self.progress_component = ProgressBarComponent(self)
        self.progress_component.add_property_parameters()

        # Output file parameter (controls save location)
        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="combined_mask.mp4")
        self._output_file.add_parameter()

        # Logging group
        self._setup_logging_group()

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

        mask_videos = self.get_parameter_list_value("mask_videos")
        if not mask_videos:
            msg = f"{self.name}: At least one mask video is required"
            exceptions.append(ValueError(msg))
            return exceptions

        # Validate all are video artifacts (duck typing - check for .value attribute)
        for idx, mask_value in enumerate(mask_videos):
            # Accept VideoArtifact, VideoUrlArtifact, or any object with a .value attribute
            if not hasattr(mask_value, "value"):
                msg = f"{self.name}: mask_videos[{idx}] must be a video artifact with a .value attribute, got {type(mask_value)}."
                exceptions.append(ValueError(msg))

        return exceptions or None

    def process(self) -> AsyncResult[None]:
        """Process videos asynchronously."""
        # Reset progress and output
        self.progress_component.reset()
        self.parameter_output_values["output_mask"] = None
        self.append_value_to_parameter("logs", "[Starting mask video combination..]\n")

        mask_videos = self.get_parameter_list_value("mask_videos")
        if not mask_videos:
            return

        try:
            yield lambda: self._process_mask_videos(mask_videos)
            self.append_value_to_parameter("logs", "[Finished mask video combination.]\n")
        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error combining mask videos: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e

    def _process_mask_videos(self, mask_videos: list) -> None:
        """Process mask videos by combining them frame-by-frame."""
        temp_dirs: list[Path] = []
        temp_files: list[Path] = []

        try:
            # Get FFmpeg paths
            ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()

            # Extract frames from all videos
            self.append_value_to_parameter("logs", f"Extracting frames from {len(mask_videos)} videos...\n")
            frame_dirs = []
            video_properties = []

            for idx, mask_video in enumerate(mask_videos):
                video_url = File(mask_video.value).resolve()
                self._validate_url_safety(video_url)

                # Create temp directory for this video's frames
                temp_dir = Path(tempfile.mkdtemp())
                temp_dirs.append(temp_dir)

                # Get video properties (resolution, fps, frame count)
                props = self._get_video_properties(video_url, ffprobe_path)
                video_properties.append(props)

                # Extract frames
                self._extract_frames(video_url, temp_dir, ffmpeg_path)
                frame_dirs.append(temp_dir)

                self.append_value_to_parameter(
                    "logs",
                    f"Video {idx + 1}: {props['width']}x{props['height']}, {props['fps']:.2f} fps, {props['frame_count']} frames\n",
                )

            # Validate all videos have same properties (resolution only, frame counts can differ)
            self._validate_video_properties(video_properties)

            # Get reference properties from first video and find max frame count
            ref_props = video_properties[0]
            max_frame_count = max(props["frame_count"] for props in video_properties)

            # Log frame count information
            if any(props["frame_count"] != max_frame_count for props in video_properties):
                self.append_value_to_parameter(
                    "logs",
                    f"Videos have different durations - will pad shorter videos with black frames to {max_frame_count} frames\n",
                )

            self.append_value_to_parameter(
                "logs",
                f"Combining {max_frame_count} frames at {ref_props['width']}x{ref_props['height']}...\n",
            )

            # Combine frames (pass video_properties for frame count info)
            output_frames_dir = Path(tempfile.mkdtemp())
            temp_dirs.append(output_frames_dir)

            last_progress = self._combine_frames(
                frame_dirs, output_frames_dir, max_frame_count, video_properties, ref_props
            )

            # Reassemble video (last 10% of progress)
            self.append_value_to_parameter("logs", "Reassembling video from combined frames...\n")
            # Increment to 90% if not already there
            while last_progress < 90:
                self.progress_component.increment()
                last_progress += 1

            output_video = self._reassemble_video(output_frames_dir, ref_props, ffmpeg_path)
            temp_files.append(output_video)

            # Mark reassembly complete (100%) - increment remaining 10%
            while last_progress < 100:
                self.progress_component.increment()
                last_progress += 1

            # Read video bytes
            with output_video.open("rb") as f:
                video_bytes = f.read()

            # Save output
            dest = self._output_file.build_file()
            saved = dest.write_bytes(video_bytes)
            output_artifact = VideoUrlArtifact(saved.location)

            self.set_parameter_value("output_mask", output_artifact)
            self.publish_update_to_parameter("output_mask", output_artifact)
            self.parameter_output_values["output_mask"] = output_artifact

            self.append_value_to_parameter("logs", f"Successfully combined mask videos: {saved.location}\n")

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

    def _validate_video_properties(self, video_properties: list[dict[str, Any]]) -> None:
        """Validate that all videos have the same resolution.

        Note: Frame counts can differ - shorter videos will be padded with black frames.
        """
        if not video_properties:
            return

        ref_props = video_properties[0]
        ref_width = ref_props["width"]
        ref_height = ref_props["height"]

        for idx, props in enumerate(video_properties[1:], start=1):
            if props["width"] != ref_width or props["height"] != ref_height:
                msg = (
                    f"{self.name}: All mask videos must have the same resolution. "
                    f"Expected {ref_width}x{ref_height}, got {props['width']}x{props['height']} at index {idx}."
                )
                raise ValueError(msg)

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

    def _combine_frames(
        self,
        frame_dirs: list[Path],
        output_dir: Path,
        max_frame_count: int,
        video_properties: list[dict[str, Any]],
        ref_props: dict[str, Any],
    ) -> int:
        """Combine corresponding frames from all videos using pixel-wise max.

        Shorter videos are padded with black frames to match the longest video.

        Args:
            frame_dirs: List of directories containing extracted frames
            output_dir: Directory to save combined frames
            max_frame_count: Maximum frame count across all videos
            video_properties: List of video property dicts with frame counts
            ref_props: Reference properties (width, height) for creating black frames

        Returns:
            The last progress value (should be ~90 for 90% complete after frame processing)
        """
        # Initialize progress bar with 100 (representing percentage points)
        # 90 points for frame processing, 10 points for video reassembly
        self.progress_component.initialize(100)
        last_progress = 0

        # Create a black frame for padding (grayscale L mode)
        black_frame = Image.new("L", (ref_props["width"], ref_props["height"]), 0)

        for frame_idx in range(1, max_frame_count + 1):
            frame_filename = f"frame_{frame_idx:06d}.png"

            # Load all corresponding frames (or use black frame if video is shorter)
            frames: list[Image.Image] = []
            for video_idx, frame_dir in enumerate(frame_dirs):
                frame_path = frame_dir / frame_filename

                # Check if this video has this frame
                if frame_idx <= video_properties[video_idx]["frame_count"]:
                    # Frame exists - load it
                    if not frame_path.exists():
                        msg = f"{self.name}: Missing frame {frame_filename} in {frame_dir}"
                        raise ValueError(msg)

                    frame_img = Image.open(frame_path)
                    frame_l = self._mask_to_l(frame_img)
                    frames.append(frame_l)
                else:
                    # Video is shorter - use black frame for padding
                    frames.append(black_frame)

            # Combine frames using pixel-wise max (ImageChops.lighter)
            combined = frames[0]
            for frame in frames[1:]:
                combined = ImageChops.lighter(combined, frame)

            # Save combined frame
            output_path = output_dir / frame_filename
            combined.save(output_path, "PNG")

            # Update progress (0-90% range)
            current_progress = int((frame_idx / max_frame_count) * 90)
            increments_needed = current_progress - last_progress
            for _ in range(increments_needed):
                self.progress_component.increment()
            last_progress = current_progress

            # Log progress every 100 frames
            if frame_idx % 100 == 0:
                self.append_value_to_parameter("logs", f"Combined {frame_idx}/{max_frame_count} frames\n")

        return last_progress

    def _reassemble_video(self, frames_dir: Path, props: dict[str, Any], ffmpeg_path: str) -> Path:
        """Reassemble video from combined frames."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_video = Path(temp_file.name)
        input_pattern = str(frames_dir / "frame_%06d.png")

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
        except subprocess.CalledProcessError as e:
            msg = f"{self.name}: FFmpeg video reassembly failed: {e.stderr}"
            raise ValueError(msg) from e
        except subprocess.TimeoutExpired as e:
            msg = f"{self.name}: FFmpeg video reassembly timed out"
            raise ValueError(msg) from e

        return output_video

    def _mask_to_l(self, mask_pil: Image.Image) -> Image.Image:
        """Convert mask image to single-channel (L) grayscale."""
        result = mask_pil.convert("L")
        if mask_pil.mode in {"RGBA", "LA"}:
            alpha = mask_pil.getchannel("A")
            alpha_extrema = alpha.getextrema()
            use_alpha = True
            if isinstance(alpha_extrema, tuple) and len(alpha_extrema) == self._EXTREMA_TUPLE_LEN:
                alpha_min, alpha_max = alpha_extrema
                if isinstance(alpha_min, (int, float)) and isinstance(alpha_max, (int, float)):
                    alpha_range = alpha_max - alpha_min
                    if alpha_min >= self._ALPHA_NEAR_OPAQUE_MIN and alpha_range <= self._ALPHA_NEAR_OPAQUE_RANGE:
                        use_alpha = False
            if use_alpha:
                result = alpha
            elif mask_pil.mode == "RGBA":
                result = mask_pil.getchannel("R")
            else:
                result = mask_pil.getchannel("L")
        elif mask_pil.mode == "L":
            result = mask_pil
        elif mask_pil.mode == "RGB":
            result = mask_pil.convert("L")
        return result
