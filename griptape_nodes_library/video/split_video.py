import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape.drivers.prompt.griptape_cloud_prompt_driver import GriptapeCloudPromptDriver
from griptape.structures import Agent as GriptapeAgent
from griptape.tasks import PromptTask
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
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
    detect_video_format,
    sanitize_filename,
    smpte_to_seconds,
    to_video_artifact,
    validate_url,
)

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"
MODEL = "gpt-4.1-mini"

TIMECODE_SEGMENT_PARTS = 2


@dataclass
class Segment:
    start_sec: float
    end_sec: float
    title: str
    raw_id: str | None = None


def build_ffmpeg_cmd(input_path: str, seg: Segment, outdir: str) -> list[str]:
    """Return a single ffmpeg command as a list for the given segment."""
    Path(outdir).mkdir(parents=True, exist_ok=True)
    out_path = Path(outdir) / f"{sanitize_filename(seg.title)}.mp4"
    return build_video_segment_cmd("ffmpeg", input_path, seg.start_sec, seg.end_sec, str(out_path))


class SplitVideo(ControlNode):
    """Split a video into multiple parts using ffmpeg."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add video input parameter
        self.add_parameter(
            ParameterVideo(
                name="video",
                allowed_modes={ParameterMode.INPUT},
                tooltip="The video to split",
            )
        )

        # Add split_by dropdown
        split_by_parameter = Parameter(
            name="split_by",
            tooltip="Choose whether to split by timecodes or frame ranges",
            input_types=["str"],
            allowed_modes={ParameterMode.PROPERTY},
            default_value="timecode",
        )
        split_by_parameter.add_trait(Options(choices=["timecode", "frame range"]))
        self.add_parameter(split_by_parameter)

        # Add timecodes parameter
        timecodes_parameter = Parameter(
            name="timecodes",
            input_types=["str", "json"],
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="00:00:00:00-00:01:00:00",
            tooltip="Timecodes to split the video at. Can be JSON format or simple timecode ranges.",
            ui_options={"multiline": True, "placeholder_text": "Enter timecodes or JSON..."},
        )
        self.add_parameter(timecodes_parameter)

        # Add frame_ranges parameter (hidden by default since timecode is the default mode)
        frame_ranges_parameter = Parameter(
            name="frame_ranges",
            input_types=["str"],
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value="0-100",
            tooltip="Frame ranges to split the video at. Format: start-end or start-end|Title, one per line.",
            ui_options={
                "multiline": True,
                "placeholder_text": "Enter frame ranges, e.g.\n0-100|Intro\n100-250|Main Content",
                "hide": True,
            },
        )
        self.add_parameter(frame_ranges_parameter)

        # Add output videos parameter list
        self.split_videos_list = ParameterList(
            name="split_videos",
            type="VideoUrlArtifact",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The split video segments",
        )
        self.add_parameter(self.split_videos_list)

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="segment.mp4",
        )
        self._output_file.add_parameter()
        # Group for logging information
        with ParameterGroup(name="Logs") as logs_group:
            ParameterString(
                name="logs",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}  # Hide the logs group by default

        self.add_node_element(logs_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "split_by":
            match value:
                case "timecode":
                    self.show_parameter_by_name("timecodes")
                    self.hide_parameter_by_name("frame_ranges")
                case "frame range":
                    self.hide_parameter_by_name("timecodes")
                    self.show_parameter_by_name("frame_ranges")
                case _:
                    logger.warning("%s: unrecognised split_by value %r — UI state unchanged", self.name, value)
        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        # Validate that we have a video
        video = self.parameter_values.get("video")
        if not video:
            msg = f"{self.name}: Video parameter is required"
            exceptions.append(ValueError(msg))

        # Make sure it's a video artifact
        if not isinstance(video, VideoUrlArtifact):
            msg = f"{self.name}: Video parameter must be a VideoUrlArtifact"
            exceptions.append(ValueError(msg))

        # Make sure it has a value
        if hasattr(video, "value") and not video.value:  # type: ignore  # noqa: PGH003
            msg = f"{self.name}: Video parameter must have a value"
            exceptions.append(ValueError(msg))

        split_by = self.get_parameter_value("split_by") or "timecode"
        if split_by == "timecode":
            timecodes = self.get_parameter_value("timecodes")
            if not timecodes:
                msg = f"{self.name}: Timecodes parameter is required"
                exceptions.append(ValueError(msg))
        else:
            frame_ranges = self.get_parameter_value("frame_ranges")
            if not frame_ranges:
                msg = f"{self.name}: Frame Ranges parameter is required"
                exceptions.append(ValueError(msg))

        return exceptions

    def _clear_list(self) -> None:
        """Clear the parameter list."""
        self.split_videos_list.clear_list()

    def _parse_timecodes_with_agent(self, timecodes_str: str) -> str:
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            error_msg = f"No API key found for {SERVICE}. Please set {API_KEY_ENV_VAR} environment variable."
            raise ValueError(error_msg)

        prompt_driver = GriptapeCloudPromptDriver(
            model=MODEL, api_key=api_key, stream=True, structured_output_strategy="tool"
        )
        agent = GriptapeAgent()
        agent.add_task(PromptTask(prompt_driver=prompt_driver))
        msg = f"""
Please parse the timecodes from the following string:
{timecodes_str}

IMPORTANT: Return ONLY the exact segments provided, with no additional segments or gap-filling.
Each line MUST include the pipe character (|) followed by "Segment X:" where X is the segment number.

Return in this EXACT format with no commentary or other text:

00:00:00:00-00:00:04:07|Segment 1:
00:00:04:08-00:00:08:15|Segment 2:

If no title is provided, just use "Segment X:" format.
"""
        try:
            response = agent.run(msg)
            self.append_value_to_parameter("logs", f"Agent response: {response}\n")
            self.append_value_to_parameter("logs", f"Agent output: {agent.output}\n")

            # The agent.output should contain the actual response text
            if hasattr(agent, "output") and agent.output:
                return str(agent.output)
            if hasattr(response, "output") and hasattr(response.output, "value"):
                return response.output.value
            error_msg = f"Unexpected agent response format: {response}"
            raise ValueError(error_msg)  # noqa: TRY301

        except Exception as e:
            error_msg = f"Agent failed to parse timecodes: {e!s}"
            self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
            raise ValueError(error_msg) from e

    def _parse_agent_response(self, agent_response: str, frame_rate: float, *, drop_frame: bool) -> list[Segment]:
        """Parse the agent's response string into segments."""
        segments = []
        for line_raw in agent_response.strip().split("\n"):
            line = line_raw.strip()
            if not line:
                continue

            # Try to parse format: HH:MM:SS:FF-HH:MM:SS:FF|Title
            parts = line.split("|", 1)
            if len(parts) == TIMECODE_SEGMENT_PARTS:
                time_range, title = parts
            else:
                # Fallback: if no pipe, treat the whole line as time range and generate title
                time_range = line
                title = f"Segment {len(segments) + 1}:"

            # Parse time range: HH:MM:SS:FF-HH:MM:SS:FF
            time_parts = time_range.split("-")
            if len(time_parts) != TIMECODE_SEGMENT_PARTS:
                continue

            start_tc, end_tc = time_parts

            try:
                start_sec = smpte_to_seconds(start_tc, frame_rate, drop_frame=drop_frame)
                end_sec = smpte_to_seconds(end_tc, frame_rate, drop_frame=drop_frame)

                if end_sec > start_sec:
                    segments.append(Segment(start_sec, end_sec, title.strip()))
            except Exception as e:
                self.append_value_to_parameter("logs", f"Warning: Could not parse line '{line}': {e}\n")
                continue

        return segments

    def _trim_segments_to_duration(self, segments: list[Segment], video_duration: float) -> list[Segment]:
        """Trim segments that exceed the video duration."""
        if video_duration <= 0:
            # If we can't determine duration, return segments as-is
            return segments

        trimmed_segments = []
        for i, segment in enumerate(segments):
            # If segment starts after video ends, skip it
            if segment.start_sec >= video_duration:
                self.append_value_to_parameter(
                    "logs",
                    f"Skipping segment {i + 1} '{segment.title}' - starts after video ends ({segment.start_sec:.2f}s >= {video_duration:.2f}s)\n",
                )
                continue

            # Create a copy of the segment to avoid modifying the original
            trimmed_segment = Segment(
                start_sec=segment.start_sec, end_sec=segment.end_sec, title=segment.title, raw_id=segment.raw_id
            )

            # If segment ends after video ends, trim it with a small buffer
            if trimmed_segment.end_sec > video_duration:
                original_end = trimmed_segment.end_sec
                # Use a small buffer to avoid keyframe boundary issues
                trimmed_segment.end_sec = video_duration - VIDEO_DURATION_BUFFER
                self.append_value_to_parameter(
                    "logs",
                    f"Trimming segment {i + 1} '{trimmed_segment.title}' - end time {original_end:.2f}s exceeds video duration {video_duration:.2f}s, trimming to {trimmed_segment.end_sec:.2f}s\n",
                )

            # If segment duration is too short after trimming, skip it
            if trimmed_segment.end_sec <= trimmed_segment.start_sec:
                self.append_value_to_parameter(
                    "logs",
                    f"Skipping segment {i + 1} '{trimmed_segment.title}' - duration too short after trimming ({trimmed_segment.end_sec:.2f}s <= {trimmed_segment.start_sec:.2f}s)\n",
                )
                continue

            trimmed_segments.append(trimmed_segment)

        return trimmed_segments

    def _validate_segment_bounds(self, segment: Segment, video_duration: float) -> Segment:
        """Ensure segment bounds are within video duration with safety buffer."""
        if video_duration <= 0:
            return segment

        # Create a copy to avoid modifying the original
        validated_segment = Segment(
            start_sec=segment.start_sec, end_sec=segment.end_sec, title=segment.title, raw_id=segment.raw_id
        )

        # Log original segment for debugging
        self.append_value_to_parameter(
            "logs",
            f"Validating segment '{segment.title}': {segment.start_sec:.3f}s - {segment.end_sec:.3f}s (video duration: {video_duration:.3f}s)\n",
        )

        # Ensure start time is not negative
        validated_segment.start_sec = max(0.0, validated_segment.start_sec)

        # Only trim if segment actually exceeds video duration
        if validated_segment.end_sec > video_duration:
            # Apply a small buffer only when necessary
            max_end_time = video_duration - VIDEO_DURATION_BUFFER
            self.append_value_to_parameter(
                "logs",
                f"Trimming end time from {validated_segment.end_sec:.3f}s to {max_end_time:.3f}s (exceeds video duration {video_duration:.3f}s)\n",
            )
            validated_segment.end_sec = max_end_time

        # Ensure we have a minimum duration
        if validated_segment.end_sec <= validated_segment.start_sec:
            self.append_value_to_parameter(
                "logs",
                f"Segment too short, extending end time from {validated_segment.end_sec:.3f}s to {validated_segment.start_sec + 0.1:.3f}s\n",
            )
            validated_segment.end_sec = validated_segment.start_sec + 0.1

        # Log final segment
        self.append_value_to_parameter(
            "logs",
            f"Final validated segment '{validated_segment.title}': {validated_segment.start_sec:.3f}s - {validated_segment.end_sec:.3f}s\n",
        )

        return validated_segment

    def _final_validate_segments(self, segments: list[Segment], video_duration: float) -> list[Segment]:
        """Final validation of all segments to ensure they're within bounds."""
        self.append_value_to_parameter("logs", "Final validation of segment bounds...\n")
        validated_segments = []
        for i, segment in enumerate(segments):
            validated_segment = self._validate_segment_bounds(segment, video_duration)
            if validated_segment.end_sec > validated_segment.start_sec:
                validated_segments.append(validated_segment)
                if validated_segment.start_sec != segment.start_sec or validated_segment.end_sec != segment.end_sec:
                    self.append_value_to_parameter(
                        "logs",
                        f"Adjusted segment {i + 1} bounds: {segment.start_sec:.2f}s-{segment.end_sec:.2f}s -> {validated_segment.start_sec:.2f}s-{validated_segment.end_sec:.2f}s\n",
                    )
            else:
                self.append_value_to_parameter(
                    "logs", f"Skipping segment {i + 1} after final validation - duration too short\n"
                )

        return validated_segments

    def _parse_timecodes(self, timecodes_str: str, frame_rate: float, *, drop_frame: bool) -> list[Segment]:
        """Parse timecodes using agent-based parsing."""
        try:
            # Use agent to parse timecodes
            agent_response = self._parse_timecodes_with_agent(timecodes_str)
            self.append_value_to_parameter("logs", f"Agent response: {agent_response}\n")

            # Parse agent response into segments
            segments = self._parse_agent_response(agent_response, frame_rate, drop_frame=drop_frame)

            if not segments:
                error_msg = "No valid segments found in agent response"
                raise ValueError(error_msg)  # noqa: TRY301
            return segments  # noqa: TRY300

        except Exception as e:
            error_msg = f"Error parsing timecodes with agent: {e!s}"
            raise ValueError(error_msg) from e

    def _parse_frame_ranges(self, frame_ranges_str: str, frame_rate: float) -> list[Segment]:
        """Parse frame ranges into segments. Format: start-end or start-end|Title, one per line."""
        segments = []
        for line_raw in frame_ranges_str.strip().split("\n"):
            line = line_raw.strip()
            if not line:
                continue

            parts = line.split("|", 1)
            if len(parts) == TIMECODE_SEGMENT_PARTS:
                range_part, title = parts
            else:
                range_part = line
                title = f"Segment {len(segments) + 1}"

            range_parts = range_part.strip().split("-")
            if len(range_parts) != TIMECODE_SEGMENT_PARTS:
                self.append_value_to_parameter("logs", f"Warning: Could not parse frame range '{line}', skipping\n")
                continue

            try:
                start_frame = int(range_parts[0].strip())
                end_frame = int(range_parts[1].strip())
            except ValueError:
                self.append_value_to_parameter("logs", f"Warning: Invalid frame numbers in '{line}', skipping\n")
                continue

            if end_frame <= start_frame:
                self.append_value_to_parameter(
                    "logs",
                    f"Warning: End frame {end_frame} must be greater than start frame {start_frame} in '{line}', skipping\n",
                )
                continue

            start_sec = start_frame / frame_rate
            end_sec = end_frame / frame_rate
            segments.append(Segment(start_sec=start_sec, end_sec=end_sec, title=title.strip()))

        return segments

    def _process_segment(self, segment: Segment, input_url: str, temp_dir: str, ffmpeg_path: str) -> str:
        """Process a single video segment and return the output file path."""
        duration = segment.end_sec - segment.start_sec
        self.append_value_to_parameter("logs", f"Processing segment: {segment.title} (duration: {duration:.2f}s)\n")

        cmd = build_ffmpeg_cmd(input_url, segment, temp_dir)
        cmd[0] = ffmpeg_path
        try:
            run_ffmpeg_cmd(cmd, log=lambda msg: self.append_value_to_parameter("logs", msg))
        except ValueError as e:
            raise ValueError(f"{self.name}: segment '{segment.title}': {e}") from e

        output_path = Path(temp_dir) / f"{sanitize_filename(segment.title)}.mp4"
        if not output_path.exists():
            raise ValueError(f"Expected output file not found: {output_path}")

        file_size = output_path.stat().st_size
        self.append_value_to_parameter("logs", f"Created segment: {output_path} ({file_size} bytes)\n")
        if file_size < MIN_VIDEO_FILE_SIZE:
            self.append_value_to_parameter("logs", f"WARNING: output is suspiciously small ({file_size} bytes)\n")

        return str(output_path)

    def _split_video_with_ffmpeg(self, input_url: str, segments: list[Segment]) -> list[bytes]:
        """Split video into segments using ffmpeg, returning raw bytes for each."""
        if not validate_url(input_url):
            raise ValueError(f"{self.name}: Invalid or unsafe URL provided: {input_url}")

        try:
            ffmpeg_path, _ = get_ffmpeg_paths()

            with tempfile.TemporaryDirectory() as temp_dir:
                output_files = []
                for i, segment in enumerate(segments):
                    self.append_value_to_parameter(
                        "logs", f"Processing segment {i + 1}/{len(segments)}: {segment.title}\n"
                    )
                    output_file = self._process_segment(segment, input_url, temp_dir, ffmpeg_path)
                    with Path(output_file).open("rb") as f:
                        output_files.append(f.read())

                return output_files

        except Exception as e:
            error_msg = f"Error during video splitting: {e!s}"
            self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
            raise ValueError(error_msg) from e

    def _process(self, input_url: str, segments: list[Segment]) -> None:
        """Performs the synchronous video splitting operation."""
        try:
            self.append_value_to_parameter("logs", f"Splitting video into {len(segments)} segments\n")

            output_files = self._split_video_with_ffmpeg(input_url, segments)

            # Convert output files to artifacts
            split_video_artifacts = []

            for i, video_bytes in enumerate(output_files):
                # Save to project storage
                dest = self._output_file.build_file(_index=i + 1)
                saved = dest.write_bytes(video_bytes)

                # Create output artifact
                video_artifact = VideoUrlArtifact(saved.location)
                split_video_artifacts.append(video_artifact)

                self.append_value_to_parameter("logs", f"Saved segment {i + 1}: {saved.name}\n")

            # Save all artifacts to parameter list
            logger.info(f"Saving {len(split_video_artifacts)} split video artifacts")
            for i, item in enumerate(split_video_artifacts):
                if i < len(self.split_videos_list):
                    current_parameter = self.split_videos_list[i]
                    self.set_parameter_value(current_parameter.name, item)
                    # Using to ensure updates are being propagated
                    self.publish_update_to_parameter(current_parameter.name, item)
                    self.parameter_output_values[current_parameter.name] = item
                    continue
                new_child = self.split_videos_list.add_child_parameter()
                # Set the parameter value
                self.set_parameter_value(new_child.name, item)

        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error splitting video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e

    def process(self) -> AsyncResult[None]:
        """Executes the main logic of the node asynchronously."""
        # Clear the parameter list
        self._clear_list()

        # Get the video and split mode
        video = self.get_parameter_value("video")
        split_by = self.get_parameter_value("split_by") or "timecode"
        timecodes = self.get_parameter_value("timecodes") or ""
        frame_ranges = self.get_parameter_value("frame_ranges") or ""

        # Initialize logs
        self.append_value_to_parameter("logs", "[Processing video split..]\n")

        try:
            # Convert to video artifact
            video_artifact = to_video_artifact(video)

            # Get the video URL directly
            input_url = File(video_artifact.value).resolve()

            # Always detect video properties for best results
            self.append_value_to_parameter("logs", "Detecting video properties...\n")
            _, ffprobe_path = get_ffmpeg_paths()
            frame_rate, drop_frame, video_duration = detect_video_properties(
                input_url, ffprobe_path, log=lambda msg: self.append_value_to_parameter("logs", msg)
            )

            self.append_value_to_parameter("logs", f"Detected frame rate: {frame_rate} fps\n")
            self.append_value_to_parameter("logs", f"Detected drop frame: {drop_frame}\n")
            self.append_value_to_parameter("logs", f"Detected video duration: {video_duration:.2f} seconds\n")

            # Parse segments based on split mode
            if split_by == "frame range":
                self.append_value_to_parameter("logs", "Parsing frame ranges...\n")
                segments = self._parse_frame_ranges(frame_ranges, frame_rate)
            else:
                self.append_value_to_parameter("logs", "Parsing timecodes...\n")
                segments = self._parse_timecodes(timecodes, frame_rate, drop_frame=drop_frame)
            self.append_value_to_parameter("logs", f"Parsed {len(segments)} segments\n")

            # Trim segments that exceed video duration
            if video_duration > 0:
                self.append_value_to_parameter("logs", "Trimming segments to video duration...\n")
                original_count = len(segments)
                # Log original segments before trimming
                for i, seg in enumerate(segments):
                    self.append_value_to_parameter(
                        "logs", f"Original segment {i + 1}: {seg.start_sec:.2f}s - {seg.end_sec:.2f}s ({seg.title})\n"
                    )

                segments = self._trim_segments_to_duration(segments, video_duration)

                # Log final segments after trimming
                for i, seg in enumerate(segments):
                    self.append_value_to_parameter(
                        "logs", f"Final segment {i + 1}: {seg.start_sec:.2f}s - {seg.end_sec:.2f}s ({seg.title})\n"
                    )

                if len(segments) != original_count:
                    self.append_value_to_parameter(
                        "logs", f"Trimmed from {original_count} to {len(segments)} segments\n"
                    )

            # Check if we have any valid segments after trimming
            if not segments:
                error_msg = "No valid segments found after parsing and trimming timecodes"
                self.append_value_to_parameter("logs", f"ERROR: {error_msg}\n")
                raise ValueError(error_msg)  # noqa: TRY301

            # Final validation of all segments to ensure they're within bounds
            segments = self._final_validate_segments(segments, video_duration)

            # Detect video format for output filename
            detected_format = detect_video_format(video)
            if not detected_format:
                detected_format = "mp4"  # default fallback

            self.append_value_to_parameter("logs", f"Detected video format: {detected_format}\n")

            # Run the video processing asynchronously
            self.append_value_to_parameter("logs", "[Started video processing..]\n")
            yield lambda: self._process(input_url, segments)
            self.append_value_to_parameter("logs", "[Finished video processing.]\n")

        except Exception as e:
            error_message = str(e)
            msg = f"{self.name}: Error splitting video: {error_message}"
            self.append_value_to_parameter("logs", f"ERROR: {msg}\n")
            raise ValueError(msg) from e
