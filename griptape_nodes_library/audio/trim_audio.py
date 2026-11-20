import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.media import resolve_media_location
from griptape_nodes_library.utils.audio_utils import to_audio_artifact
from griptape_nodes_library.utils.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path, run_ffmpeg_cmd
from griptape_nodes_library.utils.video_utils import validate_url

# The editor widget reads and writes these directly. An end time of 0 means the end of the clip.
TRIM_PARAMETERS = ["start_time", "end_time"]

DEFAULT_OUTPUT_EXTENSION = ".wav"


def build_audio_trim_cmd(
    ffmpeg_path: str, input_path: str, start_sec: float, end_sec: float, output_path: str
) -> list[str]:
    """Build an ffmpeg command that keeps start_sec to end_sec. An end of 0 or less keeps the rest of the clip."""
    # Post-input -ss decodes up to the start, so the cut is sample-accurate.
    cmd = [ffmpeg_path, "-hide_banner", "-y", "-i", input_path, "-ss", f"{start_sec:.3f}"]
    if end_sec > 0:
        cmd += ["-to", f"{end_sec:.3f}"]
    return [*cmd, "-vn", output_path]


def probe_audio_duration(ffprobe_path: str, input_path: str) -> float | None:
    """Return the clip length in seconds, or None when ffprobe cannot tell."""
    cmd = [ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", input_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)  # noqa: S603
        return float(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


class TrimAudio(SuccessFailureNode):
    """Trim audio to a start and end time, picked by dragging in or out points on the waveform."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterAudio(
                name="audio",
                allow_output=False,
                hide_property=True,
                tooltip="The audio to trim",
            )
        )

        self.add_parameter(
            ParameterDict(
                name="trim_editor",
                default_value={"audio_url": ""},
                tooltip="Drag the in and out handles on the waveform to choose what to keep",
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Widget(
                        name="TrimAudioEditor",
                        library="Griptape Nodes Library",
                        synced_parameters=TRIM_PARAMETERS,
                    )
                },
            )
        )

        with ParameterGroup(name="trim_points") as trim_points:
            ParameterFloat(name="start_time", default_value=0.0, min_val=0.0, tooltip="Start of the clip in seconds")
            ParameterFloat(
                name="end_time",
                default_value=0.0,
                min_val=0.0,
                tooltip="End of the clip in seconds. 0 keeps everything after the start.",
            )
        self.add_node_element(trim_points)

        self.add_parameter(
            ParameterAudio(
                name="audio_out",
                allow_input=False,
                allow_property=False,
                tooltip="The trimmed audio",
                pulse_on_run=True,
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="trimmed.wav")
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the audio trim operation result",
            result_details_placeholder="Details on the trim attempt will be presented here.",
            parameter_group_initially_collapsed=True,
        )

    def _push_audio_to_widget(self, audio: Any) -> None:
        """Give the widget a URL to play. The trim points reach it from their own parameters."""
        widget_value = {"audio_url": self._resolve_audio_url(audio)}
        self.set_parameter_value("trim_editor", widget_value)
        self.publish_update_to_parameter("trim_editor", widget_value)

    def _resolve_audio_url(self, audio: Any) -> str:
        """Return a URL the browser can play, or "" when there is no playable audio."""
        if not audio:
            return ""
        try:
            raw = getattr(to_audio_artifact(audio), "value", None)
            if not isinstance(raw, str) or not raw:
                return ""
            if raw.startswith(("http://", "https://", "data:")):
                return raw
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=File(raw).resolve())
            )
        except Exception as e:
            logger.warning("%s: could not build a playable URL for the audio: %s", self.name, e)
            return ""
        return result.url if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess) else ""

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "audio":
            self._push_audio_to_widget(value)
        return super().after_value_set(parameter, value)

    def after_incoming_connection(
        self, source_node: Any, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        # A loaded workflow stores values without calling after_value_set, so push the connected audio here.
        if target_parameter.name == "audio":
            self._push_audio_to_widget(source_node.get_parameter_value(source_parameter.name))
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self, source_node: Any, source_parameter: Parameter, target_parameter: Parameter
    ) -> None:
        if target_parameter.name == "audio":
            self._push_audio_to_widget(None)
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        audio = self.get_parameter_value("audio")
        audio_value = audio.get("value") if isinstance(audio, dict) else getattr(audio, "value", None)
        if not audio_value:
            exceptions.append(NodeError("Connect audio to 'Audio'."))
        elif not isinstance(audio_value, str):
            exceptions.append(
                NodeError(
                    "The audio connected to 'Audio' is raw audio data with no file or URL to trim. "
                    "Save it to a file first, then connect the saved audio."
                )
            )
        start = self.get_parameter_value("start_time") or 0.0
        end = self.get_parameter_value("end_time") or 0.0
        if 0 < end <= start:
            exceptions.append(
                NodeError(
                    f"'End Time' must be greater than 'Start Time'. 'End Time' is {end:g} and 'Start Time' is "
                    f"{start:g}. Set 'End Time' after the start, or to 0 to keep the rest of the clip."
                )
            )
        return exceptions or None

    def _process(self, input_path: str, start_sec: float, end_sec: float) -> None:
        dest = self._output_file.build_file()
        extension = Path(dest.location).suffix or DEFAULT_OUTPUT_EXTENSION

        duration = probe_audio_duration(get_ffprobe_path(), input_path)
        if duration is not None and start_sec >= duration:
            msg = (
                f"'Start Time' is {start_sec:g}s, but the audio is only {duration:.3f}s long. "
                "Set 'Start Time' before the end of the clip."
            )
            raise NodeError(msg)

        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / f"trimmed{extension}"
            run_ffmpeg_cmd(build_audio_trim_cmd(get_ffmpeg_path(), input_path, start_sec, end_sec, str(out_path)))
            if not out_path.exists() or out_path.stat().st_size == 0:
                msg = "The trim produced no audio. Check 'Start Time' and 'End Time', then run the node again."
                raise NodeError(msg)
            audio_bytes = out_path.read_bytes()

        saved = dest.write_bytes(audio_bytes)
        artifact = AudioUrlArtifact(saved.location)
        self.set_parameter_value("audio_out", artifact)
        self.publish_update_to_parameter("audio_out", artifact)
        self.parameter_output_values["audio_out"] = artifact

    async def aprocess(self) -> None:
        self._clear_execution_status()
        start_sec = self.get_parameter_value("start_time") or 0.0
        end_sec = self.get_parameter_value("end_time") or 0.0
        try:
            input_path = resolve_media_location(to_audio_artifact(self.get_parameter_value("audio")).value)
            if not input_path.startswith("data:") and not validate_url(input_path):
                msg = f"The audio connected to 'Audio' can't be opened: {input_path}. Connect audio from a file or an http(s) URL."
                raise NodeError(msg)  # noqa: TRY301
            await asyncio.to_thread(self._process, input_path, start_sec, end_sec)
        except Exception as e:
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return
        end_label = f"{end_sec:.3f}s" if end_sec > 0 else "the end"
        self._set_status_results(
            was_successful=True, result_details=f"Successfully trimmed audio: {start_sec:.3f}s to {end_label}"
        )
