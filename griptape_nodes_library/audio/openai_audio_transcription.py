from __future__ import annotations

import logging
import re
from contextlib import suppress
from typing import Any

from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["OpenAiAudioTranscription"]

MODEL_CHOICES = ["gpt-4o-transcribe", "gpt-4o-mini-transcribe", "whisper-1"]
DEFAULT_MODEL = "gpt-4o-mini-transcribe"

MODEL_MAPPING = {
    "gpt-4o-transcribe": "gpt-4o-transcribe",
    "gpt-4o-mini-transcribe": "gpt-4o-mini-transcribe",
    "whisper-1": "whisper-1",
}

RESPONSE_FORMAT_CHOICES = ["json", "verbose_json"]
DEFAULT_RESPONSE_FORMAT = "json"


class OpenAiAudioTranscription(GriptapeProxyNode):
    """Transcribe audio to text using OpenAI models via Griptape Cloud proxy.

    Supports GPT-4o transcription models and Whisper for converting speech to text.

    Inputs:
        - audio: Audio file to transcribe (mp3, mp4, mpeg, mpga, m4a, wav, webm, flac)
        - model: Transcription model to use
        - language: ISO-639-1 language code to improve accuracy
        - prompt: Optional text to guide transcription style
        - response_format: Output format (json or verbose_json)
        - temperature: Sampling temperature (0 = deterministic)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the proxy
        - output (str): Transcribed text from the audio
        - words (list): Word-level timing data (only with verbose_json)
        - segments (list): Segment-level data with timing (only with verbose_json)
        - detected_language (str): Detected language of the audio
        - duration (float): Duration of the audio in seconds
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Transcribe audio to text using OpenAI models"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterAudio(
                name="audio",
                default_value=None,
                tooltip="Audio to transcribe. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac. Max 25MB.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True, "expander": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Transcription model to use",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_CHOICES)},
                ui_options={"display_name": "Model"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="language",
                default_value=None,
                tooltip="ISO-639-1 language code (e.g. en, es, fr). Providing this improves accuracy and speed.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                placeholder_text="Auto-detect",
                ui_options={"display_name": "Language"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value=None,
                tooltip="Optional text to guide transcription style or provide context. Should match the audio language.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Optional context for the transcription...",
                ui_options={"display_name": "Prompt"},
            )
        )

        # Advanced parameters
        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced_group:
            ParameterString(
                name="response_format",
                default_value=DEFAULT_RESPONSE_FORMAT,
                tooltip="Output format. Use verbose_json for word/segment timestamps.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=RESPONSE_FORMAT_CHOICES)},
                ui_options={"display_name": "Response Format"},
            )

            ParameterFloat(
                name="temperature",
                default_value=0.0,
                tooltip="Sampling temperature between 0 and 1. Higher values produce more random output.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=1.0)},
                ui_options={"display_name": "Temperature"},
            )
        self.add_node_element(advanced_group)

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="output",
                tooltip="Transcribed text from the audio",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                multiline=True,
                placeholder_text="Transcribed text will appear here...",
                ui_options={"display_name": "Output", "pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="words",
                tooltip="Word-level timing data (only with verbose_json response format)",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="segments",
                tooltip="Segment-level data with timing (only with verbose_json response format)",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="detected_language",
                tooltip="Detected language of the audio",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="duration",
                tooltip="Duration of the audio in seconds",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the transcription result or any errors",
            result_details_placeholder="Transcription status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    def _log(self, message: str) -> None:
        with suppress(Exception):
            safe_message = re.sub(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+", r"\1[REDACTED]", message)
            safe_message = re.sub(r"(?i)(bearer\s+)[^\s,;]+", r"\1[REDACTED]", safe_message)
            safe_message = re.sub(
                r"(?i)\b(api[_-]?key|password|secret|token)\b\s*[:=]\s*([^\s,;]+)",
                r"\1=[REDACTED]",
                safe_message,
            )
            logger.info(safe_message)

    async def _resolve_audio_data_uri(self, audio_value: Any) -> str | None:
        """Resolve an audio parameter value to a base64 data URI.

        Args:
            audio_value: The value from the audio parameter (AudioUrlArtifact, dict, or string)

        Returns:
            str | None: The base64 data URI, or None if resolution failed
        """
        if audio_value is None:
            return None

        # Handle AudioUrlArtifact
        if isinstance(audio_value, AudioUrlArtifact):
            audio_url = audio_value.value
        elif isinstance(audio_value, dict):
            audio_url = audio_value.get("value") or audio_value.get("url", "")
        elif isinstance(audio_value, str):
            audio_url = audio_value
        else:
            return None

        if not audio_url:
            return None

        try:
            return await File(audio_url).aread_data_uri(fallback_mime="audio/mpeg")
        except FileLoadError as e:
            self._log(f"Failed to load audio: {e}")
            return None

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for OpenAI audio transcription."""
        audio_value = self.get_parameter_value("audio")
        if audio_value is None:
            msg = "No audio provided. Please connect an audio source."
            raise ValueError(msg)

        audio_data_uri = await self._resolve_audio_data_uri(audio_value)
        if not audio_data_uri:
            msg = "Failed to load audio file. Please check the audio input."
            raise ValueError(msg)

        language = self.get_parameter_value("language")
        prompt = self.get_parameter_value("prompt")
        response_format = self.get_parameter_value("response_format") or DEFAULT_RESPONSE_FORMAT
        temperature = self.get_parameter_value("temperature")

        payload: dict[str, Any] = {
            "file": audio_data_uri,
            "response_format": response_format,
        }

        if language:
            payload["language"] = language

        if prompt:
            payload["prompt"] = prompt

        if temperature is not None and temperature != 0.0:
            payload["temperature"] = temperature

        if response_format == "verbose_json":
            payload["timestamp_granularities"] = ["word", "segment"]

        self._log("Built transcription payload")
        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the OpenAI transcription result and set output parameters."""
        text = result_json.get("text")
        if text is None:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Transcription completed but no text was found in the response.",
            )
            return

        self.parameter_output_values["output"] = text

        # Verbose JSON fields
        words = result_json.get("words")
        if words is not None:
            self.parameter_output_values["words"] = words

        segments = result_json.get("segments")
        if segments is not None:
            self.parameter_output_values["segments"] = segments

        detected_language = result_json.get("language")
        if detected_language is not None:
            self.parameter_output_values["detected_language"] = detected_language

        duration = result_json.get("duration")
        if duration is not None:
            self.parameter_output_values["duration"] = duration

        self._set_status_results(was_successful=True, result_details="Transcription completed successfully.")

    def _set_safe_defaults(self) -> None:
        """Set safe default values for outputs."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["output"] = None
        self.parameter_output_values["words"] = None
        self.parameter_output_values["segments"] = None
        self.parameter_output_values["detected_language"] = None
        self.parameter_output_values["duration"] = None
