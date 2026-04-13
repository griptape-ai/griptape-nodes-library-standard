"""VideoColorMatch node for transferring color characteristics from an image to a video.

This node transfers color characteristics from a reference image to a target video
frame-by-frame using the color-matcher library. It supports multiple color transfer
algorithms and preserves the original video's format, aspect ratio, and frame rate.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import static_ffmpeg.run  # type: ignore[import-untyped]
from color_matcher import ColorMatcher  # type: ignore[reportMissingImports]
from griptape.artifacts import ImageUrlArtifact, VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image

from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_pil_from_url,
)
from griptape_nodes_library.utils.video_utils import (
    detect_video_format,
    dict_to_video_url_artifact,
    to_video_artifact,
    validate_url,
)


class VideoColorMatch(SuccessFailureNode):
    """Transfer color characteristics from a reference image to a video frame-by-frame.

    This node uses the color-matcher library to perform color transfer from a reference
    image to each frame of a video, preserving the video's format, aspect ratio, and
    frame rate. Useful for color grading videos to match a specific look or aesthetic.

    Features:
    - Multiple color transfer algorithms (MKL, Histogram Matching, Reinhard, MVGD)
    - Adjustable transfer strength for blending
    - Preserves original video properties (format, resolution, frame rate, audio)
    """

    # Available color transfer methods
    COLOR_MATCH_METHODS: ClassVar[list[str]] = [
        "mkl",  # Monge-Kantorovich Linearization (fast, good quality)
        "hm",  # Histogram Matching
        "reinhard",  # Reinhard et al. color transfer
        "mvgd",  # Multi-Variate Gaussian Distribution
        "hm-mvgd-hm",  # Compound method (best quality)
        "hm-mkl-hm",  # Alternative compound method
    ]

    # Strength constants
    MIN_STRENGTH = 0.0
    MAX_STRENGTH = 10.0
    DEFAULT_STRENGTH = 1.0

    # Default video properties
    DEFAULT_FRAME_RATE = 30.0
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._output_file = ProjectFileParameter(
            node=self, name="output_file", default_filename="video_colormatch.mp4"
        )
        self._output_file.add_parameter()

        # Reference image input (source of the color palette)
        self.add_parameter(
            ParameterImage(
                name="reference_image",
                tooltip="Reference image - the source of the color palette to transfer to the video",
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                },
            )
        )

        # Target video input (video to modify)
        self.add_parameter(
            ParameterVideo(
                name="target_video",
                tooltip="Target video - the video to apply the color transfer to",
                clickable_file_browser=True,
                ui_options={
                    "expander": True,
                },
                converters=[self._convert_video_input],
            )
        )

        # Color match settings
        with ParameterGroup(name="color_match_settings", ui_options={"collapsed": False}) as settings_group:
            # Method selection
            method_param = ParameterString(
                name="method",
                default_value="mkl",
                tooltip=(
                    "Color transfer algorithm:\n"
                    "• mkl: Monge-Kantorovich Linearization (fast, default)\n"
                    "• hm: Histogram Matching\n"
                    "• reinhard: Reinhard et al. color transfer\n"
                    "• mvgd: Multi-Variate Gaussian Distribution\n"
                    "• hm-mvgd-hm: Compound method (best quality)\n"
                    "• hm-mkl-hm: Alternative compound method"
                ),
            )
            method_param.add_trait(Options(choices=self.COLOR_MATCH_METHODS))
            self.add_parameter(method_param)

            # Strength parameter
            strength_param = ParameterFloat(
                name="strength",
                default_value=self.DEFAULT_STRENGTH,
                tooltip=(
                    f"Blending strength ({self.MIN_STRENGTH}-{self.MAX_STRENGTH}):\n"
                    "0.0 = no change, 1.0 = full color transfer\n"
                    "Values > 1.0 exaggerate the effect"
                ),
            )
            strength_param.add_trait(Slider(min_val=self.MIN_STRENGTH, max_val=self.MAX_STRENGTH))
            self.add_parameter(strength_param)

        self.add_node_element(settings_group)

        # Output video parameter
        self.add_parameter(
            ParameterVideo(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The color-matched output video",
                ui_options={"pulse_on_run": True, "expander": True},
            )
        )

        # Add status parameters
        self._create_status_parameters(
            result_details_tooltip="Details about the color matching result",
            result_details_placeholder="Details on the color matching will be presented here.",
            parameter_group_initially_collapsed=True,
        )

    def _convert_video_input(self, value: Any) -> Any:
        """Convert video input (dict or VideoUrlArtifact) to VideoUrlArtifact."""
        if isinstance(value, dict):
            return dict_to_video_url_artifact(value)
        return value

    def _get_ffmpeg_paths(self) -> tuple[str, str]:
        """Get FFmpeg and FFprobe executable paths."""
        try:
            ffmpeg_path, ffprobe_path = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()
            return ffmpeg_path, ffprobe_path  # noqa: TRY300
        except Exception as e:
            error_msg = f"FFmpeg not found. Please ensure static-ffmpeg is properly installed. Error: {e!s}"
            raise ValueError(error_msg) from e

    def _detect_video_properties(self, input_url: str, ffprobe_path: str) -> tuple[float, tuple[int, int], bool]:
        """Detect video frame rate, resolution, and audio presence.

        Returns:
            Tuple of (frame_rate, (width, height), has_audio)
        """
        # Validate URL before using in subprocess
        if not validate_url(input_url):
            msg = f"{self.name}: Invalid or unsafe URL provided: {input_url}"
            raise ValueError(msg)

        try:
            cmd = [
                ffprobe_path,
                "-v",
                "quiet",
                "-print_format",
                "json",
                "-show_streams",
                input_url,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)  # noqa: S603
            import json

            streams_data = json.loads(result.stdout)

            # Get video stream properties
            frame_rate = self.DEFAULT_FRAME_RATE
            width = self.DEFAULT_WIDTH
            height = self.DEFAULT_HEIGHT
            has_audio = False

            if streams_data.get("streams"):
                for stream in streams_data["streams"]:
                    if stream.get("codec_type") == "video":
                        # Get frame rate
                        fps_str = stream.get("r_frame_rate", "30/1")
                        if "/" in fps_str:
                            num, den = map(int, fps_str.split("/"))
                            frame_rate = num / den
                        else:
                            frame_rate = float(fps_str)

                        # Get resolution
                        width = int(stream.get("width", self.DEFAULT_WIDTH))
                        height = int(stream.get("height", self.DEFAULT_HEIGHT))

                    elif stream.get("codec_type") == "audio":
                        has_audio = True

            return frame_rate, (width, height), has_audio
        except Exception as e:
            logger.warning(f"{self.name}: Could not detect video properties, using defaults: {e}")
            return self.DEFAULT_FRAME_RATE, (self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), False

    def _apply_color_match_to_frame(
        self, frame_pil: Image.Image, ref_pil: Image.Image, method: str, strength: float
    ) -> Image.Image:
        """Apply color matching to a single frame.

        Args:
            frame_pil: The video frame to process
            ref_pil: The reference image
            method: Color matching method
            strength: Blending strength

        Returns:
            The color-matched frame
        """
        # Skip processing if strength is 0
        if strength == 0:
            return frame_pil

        # Convert images to RGB if necessary
        if frame_pil.mode != "RGB":
            frame_pil = frame_pil.convert("RGB")
        if ref_pil.mode != "RGB":
            ref_pil = ref_pil.convert("RGB")

        # Convert to float32 numpy arrays in 0-1 range
        frame_np = np.array(frame_pil, dtype=np.float32) / 255.0
        ref_np = np.array(ref_pil, dtype=np.float32) / 255.0

        # Ensure C-contiguous arrays
        if not frame_np.flags["C_CONTIGUOUS"]:
            frame_np = np.ascontiguousarray(frame_np)
        if not ref_np.flags["C_CONTIGUOUS"]:
            ref_np = np.ascontiguousarray(ref_np)

        # Apply color matching
        cm = ColorMatcher()
        result = cm.transfer(src=frame_np, ref=ref_np, method=method)

        # Apply strength blending if not 1.0
        if strength != 1.0:
            result = frame_np + strength * (result - frame_np)

        # Clamp values to valid range
        result = np.clip(result, 0, 1)

        # Convert back to PIL Image
        result_uint8 = (result * 255).astype(np.uint8)
        return Image.fromarray(result_uint8, mode="RGB")

    def _process_video(
        self,
        input_url: str,
        ref_pil: Image.Image,
        output_path: str,
        method: str,
        strength: float,
        frame_rate: float,
        has_audio: bool,
    ) -> None:
        """Process video by extracting frames, applying color matching, and reassembling.

        Args:
            input_url: Input video URL/path
            ref_pil: Reference image for color matching
            output_path: Output video path
            method: Color matching method
            strength: Blending strength
            frame_rate: Original video frame rate
            has_audio: Whether the video has audio
        """
        ffmpeg_path, _ = self._get_ffmpeg_paths()

        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            frames_dir = temp_path / "frames"
            frames_dir.mkdir()
            processed_dir = temp_path / "processed"
            processed_dir.mkdir()

            # Extract frames from video
            logger.debug(f"{self.name}: Extracting frames from video")
            extract_cmd = [
                ffmpeg_path,
                "-i",
                input_url,
                "-qscale:v",
                "1",  # High quality
                str(frames_dir / "frame_%06d.png"),
            ]
            try:
                subprocess.run(extract_cmd, capture_output=True, check=True, timeout=600)  # noqa: S603
            except subprocess.TimeoutExpired as e:
                error_msg = f"{self.name}: Frame extraction timed out after 600 seconds"
                raise ValueError(error_msg) from e
            except subprocess.CalledProcessError as e:
                error_msg = f"{self.name}: FFmpeg frame extraction failed: {e.stderr}"
                raise ValueError(error_msg) from e

            # Get list of extracted frames
            frame_files = sorted(frames_dir.glob("frame_*.png"))
            total_frames = len(frame_files)

            if total_frames == 0:
                msg = f"{self.name}: No frames extracted from video"
                raise ValueError(msg)

            logger.debug(f"{self.name}: Processing {total_frames} frames")

            # Process each frame
            for i, frame_file in enumerate(frame_files, 1):
                frame_pil = Image.open(frame_file)
                processed_frame = self._apply_color_match_to_frame(frame_pil, ref_pil, method, strength)

                # Save processed frame
                output_frame_path = processed_dir / frame_file.name
                processed_frame.save(output_frame_path, "PNG")

                if i % 30 == 0 or i == total_frames:
                    logger.debug(f"{self.name}: Processed {i}/{total_frames} frames")

            # Reassemble video from processed frames
            logger.debug(f"{self.name}: Reassembling video")

            # Build FFmpeg command for reassembly
            reassemble_cmd = [
                ffmpeg_path,
                "-framerate",
                str(frame_rate),
                "-i",
                str(processed_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ]

            # Add audio if present
            if has_audio:
                reassemble_cmd.extend(["-i", input_url, "-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "1:a:0"])

            reassemble_cmd.extend(["-y", output_path])

            try:
                subprocess.run(reassemble_cmd, capture_output=True, check=True, timeout=600)  # noqa: S603
            except subprocess.TimeoutExpired as e:
                error_msg = f"{self.name}: Video reassembly timed out after 600 seconds"
                raise ValueError(error_msg) from e
            except subprocess.CalledProcessError as e:
                error_msg = f"{self.name}: FFmpeg video reassembly failed: {e.stderr}"
                raise ValueError(error_msg) from e

    def _set_safe_defaults(self) -> None:
        """Set safe default values for output parameters."""
        self.parameter_output_values["output"] = None

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate parameters before running."""
        exceptions = []

        # Set safe defaults before validation
        self._set_safe_defaults()

        # Check that both inputs are provided
        target_video = self.get_parameter_value("target_video")
        ref_image = self.get_parameter_value("reference_image")

        if target_video is None:
            exceptions.append(ValueError(f"{self.name} - Target video is required"))
        if ref_image is None:
            exceptions.append(ValueError(f"{self.name} - Reference image is required"))

        # Validate strength
        strength = self.get_parameter_value("strength")
        if strength is not None and (strength < self.MIN_STRENGTH or strength > self.MAX_STRENGTH):
            msg = f"{self.name} - Strength must be between {self.MIN_STRENGTH} and {self.MAX_STRENGTH}, got {strength}"
            exceptions.append(ValueError(msg))

        # Validate method
        method = self.get_parameter_value("method")
        if method is not None and method not in self.COLOR_MATCH_METHODS:
            msg = f"{self.name} - Invalid method '{method}'. Must be one of: {', '.join(self.COLOR_MATCH_METHODS)}"
            exceptions.append(ValueError(msg))

        # Set failure status if there are validation errors
        if exceptions:
            error_messages = [str(e) for e in exceptions]
            error_details = f"Validation failed: {'; '.join(error_messages)}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")

        return exceptions if exceptions else None

    def process(self) -> AsyncResult[None]:
        """Main workflow execution method."""
        # Reset execution state and clear status
        self._clear_execution_status()

        # Get input parameters
        target_video = self.get_parameter_value("target_video")
        ref_image = self.get_parameter_value("reference_image")

        if target_video is None or ref_image is None:
            return

        method = self.get_parameter_value("method") or "mkl"
        strength = self.get_parameter_value("strength")
        if strength is None:
            strength = self.DEFAULT_STRENGTH

        try:
            # Convert inputs to artifacts if needed
            if isinstance(target_video, dict):
                target_video = dict_to_video_url_artifact(target_video)

            if isinstance(ref_image, dict):
                ref_image = dict_to_image_url_artifact(ref_image)

            # Get video artifact and resolve path
            video_artifact = to_video_artifact(target_video)
            input_url = File(video_artifact.value).resolve()

            # Validate URL
            if not validate_url(input_url):
                msg = f"{self.name}: Invalid or unsafe video URL provided"
                raise ValueError(msg)

            # Load reference image
            ref_pil = load_pil_from_url(ref_image.value)

            # Detect video format
            detected_format = detect_video_format(target_video)
            if not detected_format:
                detected_format = "mp4"

            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=f".{detected_format}", delete=False) as output_file:
                output_path = Path(output_file.name)

            try:
                # Get FFmpeg paths and detect video properties
                _, ffprobe_path = self._get_ffmpeg_paths()
                frame_rate, resolution, has_audio = self._detect_video_properties(input_url, ffprobe_path)

                logger.debug(
                    f"{self.name}: Processing video - {resolution[0]}x{resolution[1]} @ {frame_rate}fps, "
                    f"audio={has_audio}, method={method}, strength={strength}"
                )

                # Process video asynchronously
                yield lambda: self._process_video(
                    input_url, ref_pil, str(output_path), method, strength, frame_rate, has_audio
                )

                # Read processed video
                with output_path.open("rb") as f:
                    output_bytes = f.read()

                # Save video artifact
                dest = self._output_file.build_file()
                saved = dest.write_bytes(output_bytes)
                output_artifact = VideoUrlArtifact(saved.location)

                self.parameter_output_values["output"] = output_artifact

                # Set success status
                success_details = (
                    f"Successfully applied color transfer\n"
                    f"Method: {method}, Strength: {strength}\n"
                    f"Video: {resolution[0]}x{resolution[1]} @ {frame_rate}fps\n"
                    f"Reference: {ref_pil.width}x{ref_pil.height}"
                )
                self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")

            finally:
                # Clean up temporary file
                output_path.unlink(missing_ok=True)

        except Exception as e:
            error_message = str(e)
            logger.error(f"{self.name}: Processing failed: {error_message}")

            # Set failure status
            failure_details = f"Color matching failed\nError: {error_message}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {failure_details}")

            # Handle failure
            self._handle_failure_exception(ValueError(error_message))
            raise
