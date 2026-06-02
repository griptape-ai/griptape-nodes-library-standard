import tempfile
from pathlib import Path
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.ffmpeg_utils import (
    build_video_segment_cmd,
    detect_video_properties,
    get_ffmpeg_paths,
    run_ffmpeg_cmd,
)
from griptape_nodes_library.utils.video_utils import (
    MIN_VIDEO_FILE_SIZE,
    VIDEO_DURATION_BUFFER,
    smpte_to_seconds,
    to_video_artifact,
    validate_url,
)


class TrimVideo(ControlNode):
    """Trim a video to a specific start and end point using ffmpeg."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterVideo(
                name="video",
                allowed_modes={ParameterMode.INPUT},
                tooltip="The video to trim",
            )
        )

        trim_by_parameter = Parameter(
            name="trim_by",
            tooltip="Choose whether to trim by timecodes or frame range",
            input_types=["str"],
            allowed_modes={ParameterMode.PROPERTY},
            default_value="timecode",
        )
        trim_by_parameter.add_trait(Options(choices=["timecode", "frame range"]))
        self.add_parameter(trim_by_parameter)

        # Timecode parameters (visible by default)
        self.add_parameter(
            Parameter(
                name="start_timecode",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="00:00:00:00",
                tooltip="Start timecode in SMPTE format (HH:MM:SS:FF)",
                ui_options={"placeholder_text": "00:00:00:00"},
            )
        )
        self.add_parameter(
            Parameter(
                name="end_timecode",
                input_types=["str"],
                type="str",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="00:00:01:00",
                tooltip="End timecode in SMPTE format (HH:MM:SS:FF)",
                ui_options={"placeholder_text": "00:00:01:00"},
            )
        )

        # Frame range parameters (hidden by default)
        self.add_parameter(
            Parameter(
                name="start_frame",
                input_types=["int"],
                type="int",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=0,
                tooltip="Start frame number (0-indexed)",
                ui_options={"hide": True},
            )
        )
        self.add_parameter(
            Parameter(
                name="end_frame",
                input_types=["int"],
                type="int",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=100,
                tooltip="End frame number (exclusive)",
                ui_options={"hide": True},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="video_out",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The trimmed video",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="trimmed.mp4",
        )
        self._output_file.add_parameter()

        with ParameterGroup(name="Logs") as logs_group:
            ParameterString(
                name="logs",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}
        self.add_node_element(logs_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "trim_by":
            match value:
                case "timecode":
                    self.show_parameter_by_name("start_timecode")
                    self.show_parameter_by_name("end_timecode")
                    self.hide_parameter_by_name("start_frame")
                    self.hide_parameter_by_name("end_frame")
                case "frame range":
                    self.hide_parameter_by_name("start_timecode")
                    self.hide_parameter_by_name("end_timecode")
                    self.show_parameter_by_name("start_frame")
                    self.show_parameter_by_name("end_frame")
                case _:
                    logger.warning("%s: unrecognised trim_by value %r — defaulting to timecode", self.name, value)
                    self.show_parameter_by_name("start_timecode")
                    self.show_parameter_by_name("end_timecode")
                    self.hide_parameter_by_name("start_frame")
                    self.hide_parameter_by_name("end_frame")
        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        video = self.parameter_values.get("video")
        if not video:
            exceptions.append(ValueError(f"{self.name}: Video parameter is required"))
        elif not isinstance(video, VideoUrlArtifact):
            exceptions.append(ValueError(f"{self.name}: Video parameter must be a VideoUrlArtifact"))
        elif hasattr(video, "value") and not video.value:  # type: ignore  # noqa: PGH003
            exceptions.append(ValueError(f"{self.name}: Video parameter must have a value"))

        trim_by = self.get_parameter_value("trim_by") or "timecode"
        if trim_by == "frame range":
            start_frame = self.get_parameter_value("start_frame")
            end_frame = self.get_parameter_value("end_frame")
            if start_frame is None:
                exceptions.append(ValueError(f"{self.name}: Start frame is required"))
            if end_frame is None:
                exceptions.append(ValueError(f"{self.name}: End frame is required"))
            if start_frame is not None and end_frame is not None and end_frame <= start_frame:
                exceptions.append(ValueError(f"{self.name}: End frame must be greater than start frame"))
        else:
            if not self.get_parameter_value("start_timecode"):
                exceptions.append(ValueError(f"{self.name}: Start timecode is required"))
            if not self.get_parameter_value("end_timecode"):
                exceptions.append(ValueError(f"{self.name}: End timecode is required"))

        return exceptions if exceptions else None

    def _process(self, input_url: str, start_sec: float, end_sec: float) -> None:
        """Run ffmpeg and save the trimmed video to the output parameter."""
        ffmpeg_path, _ = get_ffmpeg_paths()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "trimmed.mp4"
            cmd = build_video_segment_cmd(ffmpeg_path, input_url, start_sec, end_sec, str(out_path))
            run_ffmpeg_cmd(cmd, log=lambda msg: self.append_value_to_parameter("logs", msg))

            if not out_path.exists():
                raise ValueError(f"{self.name}: expected output file not found: {out_path}")

            file_size = out_path.stat().st_size
            self.append_value_to_parameter("logs", f"Output file size: {file_size} bytes\n")
            if file_size < MIN_VIDEO_FILE_SIZE:
                raise ValueError(f"{self.name}: output too small ({file_size} bytes) — likely empty or invalid")

            video_bytes = out_path.read_bytes()

        dest = self._output_file.build_file()
        saved = dest.write_bytes(video_bytes)
        video_artifact = VideoUrlArtifact(saved.location)
        self.set_parameter_value("video_out", video_artifact)
        self.publish_update_to_parameter("video_out", video_artifact)
        self.parameter_output_values["video_out"] = video_artifact
        self.append_value_to_parameter("logs", f"Saved trimmed video: {saved.name}\n")

    def process(self) -> AsyncResult[None]:
        """Executes the main logic of the node asynchronously."""
        video = self.get_parameter_value("video")
        trim_by = self.get_parameter_value("trim_by") or "timecode"

        self.append_value_to_parameter("logs", "[Processing video trim..]\n")

        try:
            video_artifact = to_video_artifact(video)
            input_url = File(video_artifact.value).resolve()

            if not validate_url(input_url):
                raise ValueError(f"Invalid or unsafe URL: {input_url}")  # noqa: TRY301

            self.append_value_to_parameter("logs", "Detecting video properties...\n")
            _, ffprobe_path = get_ffmpeg_paths()
            frame_rate, drop_frame, video_duration = detect_video_properties(
                input_url, ffprobe_path, log=lambda msg: self.append_value_to_parameter("logs", msg)
            )
            self.append_value_to_parameter("logs", f"Frame rate: {frame_rate} fps, Duration: {video_duration:.2f}s\n")

            if trim_by == "frame range":
                start_frame_val = self.get_parameter_value("start_frame")
                end_frame_val = self.get_parameter_value("end_frame")
                start_frame = int(start_frame_val) if start_frame_val is not None else 0
                end_frame = int(end_frame_val) if end_frame_val is not None else 100
                start_sec = start_frame / frame_rate
                end_sec = end_frame / frame_rate
                self.append_value_to_parameter(
                    "logs", f"Frame range {start_frame}-{end_frame} → {start_sec:.3f}s-{end_sec:.3f}s\n"
                )
            else:
                start_tc = self.get_parameter_value("start_timecode") or "00:00:00:00"
                end_tc = self.get_parameter_value("end_timecode") or "00:00:01:00"
                start_sec = smpte_to_seconds(start_tc, frame_rate, drop_frame=drop_frame)
                end_sec = smpte_to_seconds(end_tc, frame_rate, drop_frame=drop_frame)
                self.append_value_to_parameter(
                    "logs", f"Timecodes {start_tc}-{end_tc} → {start_sec:.3f}s-{end_sec:.3f}s\n"
                )

            if end_sec <= start_sec:
                raise ValueError(f"End ({end_sec:.3f}s) must be after start ({start_sec:.3f}s)")  # noqa: TRY301

            # Clamp to video duration
            if video_duration > 0:
                start_sec = max(0.0, start_sec)
                if end_sec > video_duration:
                    end_sec = video_duration - VIDEO_DURATION_BUFFER
                    self.append_value_to_parameter("logs", f"Clamped end to {end_sec:.3f}s\n")
                if end_sec <= start_sec:
                    raise ValueError("After clamping to video duration, end is not after start")  # noqa: TRY301

            self.append_value_to_parameter("logs", f"Trimming {start_sec:.3f}s → {end_sec:.3f}s\n")
            yield lambda: self._process(input_url, start_sec, end_sec)
            self.append_value_to_parameter("logs", "[Finished video trim.]\n")

        except Exception as e:
            msg = f"{self.name}: Error trimming video: {e!s}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e
