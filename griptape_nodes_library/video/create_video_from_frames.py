from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import httpx
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.artifacts.image_url_artifact import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.common.sequences import Sequence
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import _resolve_file_path
from PIL import Image

# static_ffmpeg is dynamically installed by the library loader at runtime
from static_ffmpeg import run  # type: ignore[import-untyped]

logger = logging.getLogger("griptape_nodes")

__all__ = ["CreateVideoFromFrames"]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_FRAME_RATE = 30.0

# Token pattern that matches ####, @@@@, %04d, %d, etc.
_SEQUENCE_TOKEN_RE = re.compile(r"(#+|@+|%\d*d)")


ORDERING_MODES = ["Sequential", "Respect frame numbers"]
VIDEO_FORMATS = ["mp4", "mov", "gif"]


def _is_sequence_pattern(s: str) -> bool:
    """Return True if string contains a sequence token (####, @@@@, %04d, etc.)."""
    return bool(_SEQUENCE_TOKEN_RE.search(s))


class CreateVideoFromFrames(SuccessFailureNode):
    """Combine image frames into a video using ffmpeg.

    Inputs:
        - frames_input: Sequence (from Scan Sequence), sequence pattern (frame.####.png),
          directory path, list of file paths / ImageUrlArtifacts, or single ImageUrlArtifact
        - frame_rate (float): Output frame rate in fps (default: 30.0)
        - ordering_mode (str): For list/directory inputs only — Sequential or Respect frame numbers
        - format (str): Output format — mp4, mov, or gif
        - audio (optional): Audio to add (mp4/mov only)

    Outputs:
        - video (VideoUrlArtifact): Combined video (mp4/mov)
        - image (ImageUrlArtifact): Combined GIF (hidden unless format=gif)
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="frames_input",
                type="str",
                input_types=["Sequence", "list", "str", "ImageUrlArtifact"],
                tooltip=(
                    "Frame source. Use the file browser to pick a folder or image sequence "
                    "(frame.####.png), or wire in a Sequence from Scan Sequence, a list of "
                    "image paths / ImageUrlArtifacts, or a single ImageUrlArtifact."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={FileSystemPicker(allow_files=True, allow_directories=True, allow_sequences=True)},
                display_name="Frames or Sequence",
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="frame_rate",
                default_value=DEFAULT_FRAME_RATE,
                tooltip="Output frame rate in fps",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterString(
                name="ordering_mode",
                default_value=ORDERING_MODES[0],
                tooltip=(
                    "How to order frames when the input is a list or folder. "
                    "Sequential: filename order. "
                    "Respect frame numbers: gaps in numbering hold the last frame. "
                    "Not applied when the input is a Sequence or sequence pattern — "
                    "those already encode ordering."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=ORDERING_MODES)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="format",
                default_value=VIDEO_FORMATS[0],
                tooltip="Output video format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=VIDEO_FORMATS)},
            )
        )

        processing_speed_param = ParameterString(
            name="processing_speed",
            default_value="balanced",
            tooltip="Encoding speed vs quality trade-off (mp4/mov only)",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        processing_speed_param.add_trait(Options(choices=["fast", "balanced", "quality"]))
        self.add_parameter(processing_speed_param)

        self.add_parameter(
            ParameterAudio(
                name="audio",
                default_value="",
                tooltip="Optional audio to add to the video (mp4/mov only, not GIF)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Combined video output (mp4/mov)",
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="image",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Combined GIF output",
                hide=True,
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_video_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="combined.mp4",
        )
        self._output_video_file.add_parameter()

        self._output_image_file = ProjectFileParameter(
            node=self,
            name="output_gif_file",
            default_filename="combined.gif",
            ui_options={"hide": True},
        )
        self._output_image_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the frame combination result or any errors",
            result_details_placeholder="Combination status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)

        if parameter.name == "format":
            if value == "gif":
                self.show_parameter_by_name("image")
                self.hide_parameter_by_name("video")
                self.show_parameter_by_name("output_gif_file")
                self.hide_parameter_by_name("output_file")
                self.hide_parameter_by_name("processing_speed")
            else:
                self.hide_parameter_by_name("image")
                self.show_parameter_by_name("video")
                self.hide_parameter_by_name("output_gif_file")
                self.show_parameter_by_name("output_file")
                self.show_parameter_by_name("processing_speed")

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        # Clear the stale connection value so the FileSystemPicker doesn't try to
        # open a serialized Sequence dict as a path. A connection overwrites any
        # previously-picked path anyway, so an empty field is the correct end state.
        if target_parameter.name == "frames_input":
            self.set_parameter_value("frames_input", "")
        super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    async def aprocess(self) -> None:
        self._clear_execution_status()
        logger.info("%s starting frame combination", self.name)

        frames_input = self.get_parameter_value("frames_input")
        frame_rate = self.get_parameter_value("frame_rate") or DEFAULT_FRAME_RATE
        ordering_mode = self.get_parameter_value("ordering_mode") or ORDERING_MODES[0]
        format_type = self.get_parameter_value("format") or VIDEO_FORMATS[0]

        if not frames_input:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} requires frames input (Sequence, pattern, list, or directory).",
            )
            return

        if frame_rate <= 0:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name}: frame rate must be > 0 (got {frame_rate})",
            )
            return

        frame_paths = self._get_frame_paths(frames_input)
        if not frame_paths:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False, result_details=f"{self.name}: no valid frame files found in input"
            )
            return

        # Fill gaps for all input types — a Sequence with SKIP policy still has gaps
        if ordering_mode == "Respect frame numbers":
            frame_paths = self._process_respect_frame_numbers(frame_paths)

        if not frame_paths:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False, result_details=f"{self.name}: no frames remaining after ordering"
            )
            return

        audio_input = self.get_parameter_value("audio")

        try:
            if format_type == "gif":
                image_artifact = await asyncio.to_thread(self._combine_frames_to_gif, frame_paths, frame_rate)
                self.parameter_output_values["image"] = image_artifact
                self.parameter_output_values["video"] = None
                result_details = f"Successfully combined {len(frame_paths)} frames into GIF at {frame_rate} fps"
            else:
                video_artifact = await asyncio.to_thread(
                    self._combine_frames_to_video, frame_paths, frame_rate, format_type, audio_input
                )
                self.parameter_output_values["video"] = video_artifact
                self.parameter_output_values["image"] = None
                audio_note = " with audio" if audio_input else ""
                result_details = (
                    f"Successfully combined {len(frame_paths)} frames into "
                    f"{format_type} video at {frame_rate} fps{audio_note}"
                )

            self._set_status_results(was_successful=True, result_details=result_details)
            logger.info("%s combined %d frames successfully", self.name, len(frame_paths))

        except Exception as e:
            self._set_safe_defaults()
            error_msg = f"{self.name} failed to combine frames: {e}"
            self._set_status_results(was_successful=False, result_details=error_msg)
            logger.error("%s combination failed: %s", self.name, e)
            self._handle_failure_exception(RuntimeError(error_msg))

    @staticmethod
    def _is_sequence(value: Any) -> bool:
        """Return True if value is a Sequence instance or a dict with Sequence shape."""
        return isinstance(value, Sequence) or (isinstance(value, dict) and "entries" in value)

    def _get_frame_paths(self, frames_input: Any) -> list[Path]:
        """Resolve frames_input to an ordered list of validated image Paths."""
        # --- Sequence object (from ScanSequenceNode) ---
        if self._is_sequence(frames_input):
            sequence = Sequence.model_validate(frames_input) if isinstance(frames_input, dict) else frames_input
            paths = []
            for entry in sequence.entries:
                path = Path(entry.path)
                if path.exists() and path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    if self._validate_image_file(path):
                        paths.append(path)
            return paths  # Already ordered by ScanSequenceNode — do not sort

        # --- Serialized list string ---
        if isinstance(frames_input, str):
            stripped = frames_input.strip()
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = json.loads(frames_input)
                    frames_input = parsed if isinstance(parsed, list) else ast.literal_eval(frames_input)
                except (json.JSONDecodeError, ValueError, SyntaxError):
                    try:
                        frames_input = ast.literal_eval(frames_input)
                    except (ValueError, SyntaxError) as e:
                        logger.warning("%s failed to parse list string: %s", self.name, e)

        # --- List of paths / ImageUrlArtifacts ---
        if isinstance(frames_input, list):
            paths = []
            for item in frames_input:
                path = self._extract_path_from_item(item)
                if path and path.exists() and path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    if self._validate_image_file(path):
                        paths.append(path)
            paths.sort(key=lambda p: p.name)
            return paths

        # --- String: sequence pattern or directory/file path ---
        if isinstance(frames_input, str):
            if _is_sequence_pattern(frames_input):
                return self._expand_sequence_pattern(frames_input)

            input_path = Path(frames_input)
            if not input_path.exists():
                return []

            paths = []
            if input_path.is_file():
                if input_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS and self._validate_image_file(input_path):
                    paths.append(input_path)
            elif input_path.is_dir():
                for ext in SUPPORTED_IMAGE_EXTENSIONS:
                    for found in [*input_path.glob(f"*{ext}"), *input_path.glob(f"*{ext.upper()}")]:
                        if self._validate_image_file(found):
                            paths.append(found)
                paths.sort(key=lambda p: p.name)
            return paths

        # --- Single ImageUrlArtifact ---
        if isinstance(frames_input, ImageUrlArtifact):
            path = self._extract_path_from_item(frames_input)
            if path and path.exists() and path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                if self._validate_image_file(path):
                    return [path]

        return []

    def _expand_sequence_pattern(self, pattern: str) -> list[Path]:
        """Expand a sequence pattern (frame.####.png) to a sorted list of file paths."""
        path = Path(pattern)
        directory = path.parent
        if not directory.exists():
            logger.warning("%s sequence pattern directory does not exist: %s", self.name, directory)
            return []

        # Build a regex from the filename template by splitting on tokens,
        # escaping literal parts, and replacing tokens with (\d+)
        parts = _SEQUENCE_TOKEN_RE.split(path.name)
        regex_parts = [r"(\d+)" if _SEQUENCE_TOKEN_RE.fullmatch(p) else re.escape(p) for p in parts]
        try:
            regex = re.compile("".join(regex_parts))
        except re.error as e:
            logger.warning("%s failed to compile sequence pattern regex for %s: %s", self.name, pattern, e)
            return []

        found: list[tuple[int, Path]] = []
        for f in directory.iterdir():
            if not f.is_file() or f.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            m = regex.match(f.name)
            if m:
                try:
                    frame_num = int(m.group(1))
                except (IndexError, ValueError):
                    continue
                if self._validate_image_file(f):
                    found.append((frame_num, f))

        found.sort(key=lambda x: x[0])
        return [f for _, f in found]

    def _extract_path_from_item(self, item: Any) -> Path | None:
        """Extract a file path from a str, Path, or ImageUrlArtifact."""
        if isinstance(item, Path):
            return item

        url_or_path = None
        if isinstance(item, str):
            url_or_path = item
        elif isinstance(item, ImageUrlArtifact):
            url_or_path = item.value
            if not url_or_path:
                return None
            if not isinstance(url_or_path, str):
                return Path(str(url_or_path))
        else:
            return None

        if isinstance(url_or_path, str) and url_or_path.startswith(("http://", "https://")):
            if not url_or_path.startswith(("http://localhost:", "https://localhost:")):
                return self._download_url_to_temp_file(url_or_path)

        resolved = _resolve_file_path(url_or_path)
        return resolved if resolved else (Path(url_or_path) if url_or_path else None)

    def _download_url_to_temp_file(self, url: str) -> Path | None:
        """Download an image from a remote URL to a temporary file."""
        try:
            ext = ".jpg"
            for supported_ext in SUPPORTED_IMAGE_EXTENSIONS:
                if supported_ext in url.lower():
                    ext = supported_ext
                    break

            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                temp_path = Path(tmp.name)

            with httpx.Client(timeout=60) as client:
                response = client.get(url)
                response.raise_for_status()
                temp_path.write_bytes(response.content)

            logger.debug("%s downloaded image from URL to temp file: %s", self.name, temp_path)
            return temp_path
        except Exception as e:
            logger.warning("%s failed to download image from URL %s: %s", self.name, url, e)
            return None

    def _validate_image_file(self, image_path: Path) -> bool:
        """Return True if the file exists, is non-empty, and is a valid image."""
        if not image_path.exists() or not image_path.is_file():
            return False
        if image_path.stat().st_size == 0:
            logger.warning("%s skipping empty image file: %s", self.name, image_path)
            return False
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            logger.warning("%s skipping invalid/corrupted image file %s: %s", self.name, image_path, e)
            return False

    def _detect_image_format(self, image_path: Path) -> tuple[str, str]:
        """Detect the actual image format; returns (format_name, extension)."""
        if not image_path.exists():
            return ("PNG", ".png")
        try:
            with Image.open(image_path) as img:
                actual_format = img.format
                if actual_format:
                    format_to_ext = {
                        "PNG": ".png",
                        "JPEG": ".jpg",
                        "JPG": ".jpg",
                        "WEBP": ".webp",
                        "GIF": ".gif",
                        "BMP": ".bmp",
                        "TIFF": ".tiff",
                        "TIF": ".tiff",
                    }
                    ext = format_to_ext.get(actual_format.upper(), ".png")
                    return (actual_format.upper(), ext)
        except Exception as e:
            logger.warning("%s failed to detect format for %s: %s", self.name, image_path, e)
        return ("PNG", ".png")

    def _normalize_image_for_ffmpeg(self, image_path: Path, output_path: Path, target_format: str) -> None:
        """Convert image to RGB and save in target_format for FFmpeg compatibility."""
        with Image.open(image_path) as img:
            if img.mode in ("RGBA", "LA", "P"):
                rgb_img = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "P":
                    img = img.convert("RGBA")
                rgb_img.paste(img, mask=img.split()[-1] if img.mode in ("RGBA", "LA") else None)
                img = rgb_img
            elif img.mode != "RGB":
                img = img.convert("RGB")

            save_kwargs: dict[str, Any] = {}
            if target_format in ("JPEG", "WEBP"):
                save_kwargs["quality"] = 95
            img.save(output_path, target_format, **save_kwargs)

    def _process_respect_frame_numbers(self, frame_paths: list[Path]) -> list[Path]:
        """Duplicate frames to fill gaps in frame numbering."""
        if not frame_paths:
            return []

        frame_data = []
        for frame_path in frame_paths:
            frame_num = self._extract_frame_number_from_filename(frame_path.name)
            if frame_num is not None:
                frame_data.append((frame_num, frame_path))

        if not frame_data:
            return frame_paths  # No frame numbers found — return as-is

        frame_data.sort(key=lambda x: x[0])

        processed: list[Path] = []
        for idx, (frame_num, frame_path) in enumerate(frame_data):
            processed.append(frame_path)
            if idx < len(frame_data) - 1:
                gap = frame_data[idx + 1][0] - frame_num
                if gap > 1:
                    processed.extend([frame_path] * (gap - 1))

        return processed

    def _extract_frame_number_from_filename(self, filename: str) -> int | None:
        """Extract a frame number from a filename (e.g. frame.0001.png → 1)."""
        match = re.search(r"(\d{4,})", filename)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
        match = re.search(r"(\d+)", filename)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, TypeError):
                pass
        return None

    def _combine_frames_to_gif(self, frame_paths: list[Path], frame_rate: float) -> ImageUrlArtifact:
        """Combine frames into a GIF using ffmpeg."""
        if not frame_paths:
            raise ValueError("No frames to combine")

        try:
            ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        except Exception as e:
            raise ValueError(f"FFmpeg not found: {e}") from e

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            detected_format, detected_ext = self._detect_image_format(frame_paths[0])

            for idx, frame_path in enumerate(frame_paths):
                seq_path = temp_path / f"frame_{idx:04d}{detected_ext}"
                try:
                    self._normalize_image_for_ffmpeg(frame_path, seq_path, detected_format)
                except Exception as e:
                    raise RuntimeError(f"Failed to normalize frame {frame_path}: {e}") from e

            input_pattern = str(temp_path / f"frame_%04d{detected_ext}")
            output_path = temp_path / "output.gif"
            cmd = self._build_ffmpeg_command(ffmpeg_path, input_pattern, output_path, frame_rate, "gif")

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # noqa: S603
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"FFmpeg timed out combining frames: {e}") from e
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg failed to combine frames: {e.stderr}") from e

            if not output_path.exists():
                raise RuntimeError("FFmpeg did not create output GIF file")

            gif_bytes = output_path.read_bytes()
            dest = self._output_image_file.build_file()
            saved = dest.write_bytes(gif_bytes)
            return ImageUrlArtifact(value=saved.location)

    def _combine_frames_to_video(
        self,
        frame_paths: list[Path],
        frame_rate: float,
        format_type: str,
        audio_input: Any = None,
    ) -> VideoUrlArtifact:
        """Combine frames into a video using ffmpeg."""
        if not frame_paths:
            raise ValueError("No frames to combine")

        try:
            ffmpeg_path, _ = run.get_or_fetch_platform_executables_else_raise()
        except Exception as e:
            raise ValueError(f"FFmpeg not found: {e}") from e

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            detected_format, detected_ext = self._detect_image_format(frame_paths[0])

            for idx, frame_path in enumerate(frame_paths):
                seq_path = temp_path / f"frame_{idx:04d}{detected_ext}"
                try:
                    self._normalize_image_for_ffmpeg(frame_path, seq_path, detected_format)
                except Exception as e:
                    raise RuntimeError(f"Failed to normalize frame {frame_path}: {e}") from e

            input_pattern = str(temp_path / f"frame_%04d{detected_ext}")
            output_path = temp_path / f"output.{format_type}"
            audio_url = self._extract_audio_url(audio_input)
            cmd = self._build_ffmpeg_command(
                ffmpeg_path, input_pattern, output_path, frame_rate, format_type, audio_url
            )

            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=300)  # noqa: S603
            except subprocess.TimeoutExpired as e:
                raise RuntimeError(f"FFmpeg timed out combining frames: {e}") from e
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"FFmpeg failed to combine frames: {e.stderr}") from e

            if not output_path.exists():
                raise RuntimeError("FFmpeg did not create output video file")

            video_bytes = output_path.read_bytes()
            dest = self._output_video_file.build_file()
            saved = dest.write_bytes(video_bytes)
            return VideoUrlArtifact(saved.location)

    def _extract_audio_url(self, audio_input: Any) -> str | None:
        """Extract a resolvable file path from various audio input types."""
        if not audio_input:
            return None

        audio_url: str | None = None

        if isinstance(audio_input, AudioUrlArtifact):
            audio_url = audio_input.value if isinstance(audio_input.value, str) else None
        elif isinstance(audio_input, str):
            audio_url = audio_input
        elif isinstance(audio_input, dict) and "value" in audio_input:
            audio_url = str(audio_input["value"])
        elif not isinstance(audio_input, dict) and hasattr(audio_input, "value"):
            try:
                val = audio_input.value
                if val:
                    audio_url = str(val)
            except (AttributeError, TypeError):
                pass

        if not audio_url:
            return None

        try:
            return File(audio_url).resolve()
        except Exception:
            resolved = _resolve_file_path(audio_url)
            return str(resolved) if resolved else audio_url

    def _build_ffmpeg_command(
        self,
        ffmpeg_path: str,
        input_pattern: str,
        output_path: Path,
        frame_rate: float,
        format_type: str,
        audio_url: str | None = None,
    ) -> list[str]:
        """Build the ffmpeg command list for combining frames."""
        cmd = [ffmpeg_path, "-f", "image2", "-framerate", str(frame_rate), "-i", input_pattern]

        if audio_url and format_type in ("mp4", "mov"):
            cmd.extend(["-i", audio_url])

        match format_type:
            case "gif":
                palette_path = output_path.parent / "palette.png"
                fps_filter = f"fps={frame_rate}"
                palette_cmd = [
                    ffmpeg_path,
                    "-f",
                    "image2",
                    "-framerate",
                    str(frame_rate),
                    "-i",
                    input_pattern,
                    "-vf",
                    f"{fps_filter},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,palettegen",
                    "-y",
                    str(palette_path),
                ]
                try:
                    subprocess.run(palette_cmd, capture_output=True, text=True, check=True, timeout=60)  # noqa: S603
                except subprocess.CalledProcessError as e:
                    logger.warning("%s palette generation failed, falling back to simple GIF: %s", self.name, e.stderr)
                    cmd.extend(["-vf", fps_filter])
                else:
                    cmd = [
                        ffmpeg_path,
                        "-f",
                        "image2",
                        "-framerate",
                        str(frame_rate),
                        "-i",
                        input_pattern,
                        "-i",
                        str(palette_path),
                        "-lavfi",
                        f"{fps_filter},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos[x];[x][1:v]paletteuse",
                    ]

            case "mp4" | "mov":
                preset, pix_fmt, crf = self._get_processing_speed_settings()
                cmd.extend(
                    [
                        "-c:v",
                        "libx264",
                        "-preset",
                        preset,
                        "-crf",
                        str(crf),
                        "-pix_fmt",
                        pix_fmt,
                        "-movflags",
                        "+faststart",
                    ]
                )
                if audio_url:
                    cmd.extend(["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "192k", "-shortest"])
                else:
                    cmd.append("-an")

            case _:
                msg = f"Unsupported format: {format_type!r}"
                raise ValueError(msg)

        cmd.extend(["-y", str(output_path)])
        return cmd

    def _get_processing_speed_settings(self) -> tuple[str, str, int]:
        """Return (preset, pix_fmt, crf) for the current processing_speed value."""
        speed = self.get_parameter_value("processing_speed") or "balanced"
        match speed:
            case "fast":
                return "ultrafast", "yuv420p", 30
            case "quality":
                return "slow", "yuv420p", 18
            case _:
                return "medium", "yuv420p", 23

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["video"] = None
        self.parameter_output_values["image"] = None
