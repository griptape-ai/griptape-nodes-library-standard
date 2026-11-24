import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from griptape.artifacts import AudioArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.griptape_nodes import logger

from griptape_nodes_library.media import resolve_media_location
from griptape_nodes_library.utils.ffmpeg_utils import describe_ffmpeg_failure, get_ffprobe_path


class AudioDetails(DataNode):
    """Extract detailed information from an audio file including duration, format, bitrate, and sample rate."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add input parameter for the audio
        self.add_parameter(
            ParameterAudio(
                name="audio",
                default_value=value,
                tooltip="The audio file to analyze",
                allow_output=False,
                hide_property=True,
            )
        )

        # Basic Info group (default open)
        with ParameterGroup(name="Basic Info") as basic_group:
            self._duration_parameter = ParameterFloat(
                name="duration",
                default_value=0.0,
                tooltip="Audio duration in seconds",
                allow_input=False,
                allow_property=False,
                settable=False,
            )

            self._format_parameter = ParameterString(
                name="format",
                default_value="UNKNOWN",
                tooltip="Audio format (e.g., 'mp3', 'wav', 'ogg')",
                allow_input=False,
                allow_property=False,
                settable=False,
            )

            self._file_size_parameter = ParameterInt(
                name="file_size",
                default_value=0,
                tooltip="File size in bytes",
                allow_input=False,
                allow_property=False,
                settable=False,
            )
        self.add_node_element(basic_group)

        # Audio Properties group (collapsed by default)
        with ParameterGroup(name="Audio Properties", ui_options={"collapsed": True}) as properties_group:
            self._sample_rate_parameter = ParameterInt(
                name="sample_rate",
                default_value=0,
                tooltip="Sample rate in Hz",
                allow_input=False,
                allow_property=False,
                settable=False,
            )

            self._channels_parameter = ParameterInt(
                name="channels",
                default_value=0,
                tooltip="Number of audio channels",
                allow_input=False,
                allow_property=False,
                settable=False,
            )

            self._bitrate_parameter = ParameterInt(
                name="bitrate",
                default_value=0,
                tooltip="Audio bitrate in bits per second",
                allow_input=False,
                allow_property=False,
                settable=False,
            )

            self._codec_parameter = ParameterString(
                name="codec",
                default_value="UNKNOWN",
                tooltip="Audio codec (e.g., 'mp3', 'aac', 'pcm')",
                allow_input=False,
                allow_property=False,
                settable=False,
            )
        self.add_node_element(properties_group)

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        """Override to update outputs immediately when audio parameter is set."""
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )

        if param_name == "audio":
            # Setting a value must not raise: the engine would skip pushing the value downstream.
            # A failure here is logged, and process() reports it when the node runs.
            try:
                self._update_audio_details(value)
            except Exception as e:
                logger.error(f"{self.name}: {e}")
                self._set_default_values()

    def _update_audio_details(self, audio: AudioUrlArtifact | AudioArtifact | None) -> None:
        """Update all output values based on the audio.

        Raises ValueError or RuntimeError when the audio can't be read, rather than
        reporting defaults that look like a real zero-length clip.
        """
        audio_url = self._get_audio_url(audio)
        if not audio_url:
            self._set_default_values()
            return

        audio_info = self._analyze_audio_with_ffprobe(audio_url)

        self.parameter_output_values["duration"] = audio_info["duration"]
        self.parameter_output_values["format"] = audio_info["format"]
        self.parameter_output_values["file_size"] = audio_info["file_size"]
        self.parameter_output_values["sample_rate"] = audio_info["sample_rate"]
        self.parameter_output_values["channels"] = audio_info["channels"]
        self.parameter_output_values["bitrate"] = audio_info["bitrate"]
        self.parameter_output_values["codec"] = audio_info["codec"]

    def process(self) -> None:
        """Extract and output all audio details.

        The extraction also runs in set_parameter_value when the audio parameter is
        set; running it here refreshes the outputs and fails the node with the reason
        if the audio can't be read.
        """
        audio = self.get_parameter_value("audio")
        self._update_audio_details(audio)

    def _get_audio_url(self, audio: AudioUrlArtifact | AudioArtifact | dict | str | None) -> str | None:
        """Extract URL from audio artifact or save bytes to temp file.

        Returns None when there is no audio. Raises ValueError when a location or
        bytes are present but can't be turned into something ffprobe can open.
        """
        # FAILURE CASES FIRST
        if not audio:
            return None

        # A serialized artifact dict or a bare string can arrive here because
        # set_parameter_value calls this before the ParameterAudio converter runs.
        location = None
        if isinstance(audio, AudioUrlArtifact):
            location = audio.value
        elif isinstance(audio, dict) and isinstance(audio.get("value"), str):
            location = audio["value"]
        elif isinstance(audio, str):
            location = audio

        if location:
            # ``location`` may be an HTTP URL, a data URI, a project macro path
            # (``{outputs}/clip.mp3``), or a plain filesystem path. ffprobe
            # cannot open macro paths, so they must be resolved first.
            try:
                return resolve_media_location(location)
            except FileLoadError as e:
                msg = f"Failed to resolve audio path {location}: {e}"
                raise ValueError(msg) from e

        if isinstance(audio, AudioArtifact):
            # For AudioArtifact with bytes, save to temp file and return path
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                    temp_file.write(audio.value)
                    return str(Path(temp_file.name).absolute())
            except OSError as e:
                msg = f"Failed to create temp file for AudioArtifact: {e}"
                raise ValueError(msg) from e

        return None

    def _analyze_audio_with_ffprobe(self, audio_url: str) -> dict[str, Any]:
        """Analyze audio file using ffprobe to extract detailed information.

        Raises ValueError with ffprobe's reason when the audio can't be read, and
        RuntimeError when the bundled ffprobe can't be found.
        """
        ffprobe_path = get_ffprobe_path()

        # "-v error" rather than "quiet" so a failure below can report why ffprobe
        # failed; the JSON stays on stdout either way.
        cmd = [ffprobe_path, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", audio_url]

        try:
            result = subprocess.run(cmd, capture_output=True, encoding="utf-8", timeout=30, check=True)  # noqa: S603
        except subprocess.TimeoutExpired as e:
            msg = "ffprobe timed out after 30s reading the audio"
            raise ValueError(msg) from e
        except (subprocess.CalledProcessError, OSError) as e:
            msg = f"ffprobe failed to read the audio: {describe_ffmpeg_failure(e)}"
            raise ValueError(msg) from e

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            msg = f"Failed to parse ffprobe output: {e}"
            raise ValueError(msg) from e

        format_info = data.get("format", {})
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None)
        if audio_stream is None:
            msg = "No audio stream found in file"
            raise ValueError(msg)

        try:
            return {
                "duration": float(format_info.get("duration", 0)),
                "format": format_info.get("format_name", "UNKNOWN"),
                "file_size": int(format_info.get("size", 0)),
                "sample_rate": int(audio_stream.get("sample_rate", 0)),
                "channels": int(audio_stream.get("channels", 0)),
                "bitrate": int(format_info.get("bit_rate", 0)),
                "codec": audio_stream.get("codec_name", "UNKNOWN"),
            }
        except (ValueError, TypeError) as e:
            msg = f"Failed to parse audio properties: {e}"
            raise ValueError(msg) from e

    def _set_default_values(self) -> None:
        """Set all output values to defaults."""
        self.parameter_output_values["duration"] = 0.0
        self.parameter_output_values["format"] = "UNKNOWN"
        self.parameter_output_values["file_size"] = 0
        self.parameter_output_values["sample_rate"] = 0
        self.parameter_output_values["channels"] = 0
        self.parameter_output_values["bitrate"] = 0
        self.parameter_output_values["codec"] = "UNKNOWN"
