from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts import TextArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.memory.structure import Run
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_parameter import ModelAccessParameter
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.button import Button
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.agents.griptape_nodes_agent import GriptapeNodesAgent as GtAgent
from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.agent_utils import unwrap_agent, wrap_agent

logger = logging.getLogger(__name__)

__all__ = ["TranscribeAudio"]

MODEL_CHOICES = ["whisper-1"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Deprecated models and their replacements (kept for backward-compat with saved graphs)
DEPRECATED_MODELS = {
    "gpt-4o-mini-transcribe": "whisper-1",
    "gpt-4o-transcribe": "whisper-1",
}

MODEL_MAPPING = {
    "whisper-1": "whisper-1",
}

RESPONSE_FORMAT_CHOICES = ["json", "verbose_json"]
DEFAULT_RESPONSE_FORMAT = "json"


class TranscribeAudio(GriptapeProxyNode):
    """Transcribe audio to text using OpenAI models via the Griptape Cloud model proxy.

    Routing transcription through the proxy means users no longer need to set their own
    ``OPENAI_API_KEY`` — the request is authenticated with the Griptape Cloud API key and
    billed against Griptape Cloud credits. Supports GPT-4o transcription models and Whisper.

    Inputs:
        - audio: Audio file to transcribe (mp3, mp4, mpeg, mpga, m4a, wav, webm, flac)
        - model: Transcription model to use
        - language: ISO-639-1 language code to improve accuracy
        - prompt: Optional text to guide transcription style
        - response_format: Output format (json or verbose_json)
        - temperature: Sampling temperature (0 = deterministic)

    Outputs:
        - output (str): Transcribed text from the audio
        - words (list): Word-level timing data (only with verbose_json)
        - segments (list): Segment-level data with timing (only with verbose_json)
        - detected_language (str): Detected language of the audio
        - duration (float): Duration of the audio in seconds
        - generation_id (str): Generation ID from the proxy (in the Status group)
        - provider_response (dict): Verbatim response from the proxy
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "audio"
        self.description = "Transcribe audio to text using OpenAI models via Griptape Cloud proxy"

        # License-policy helper for the "model" dropdown. Owns the model list,
        # installs the Options + refresh Button traits, applies per-row
        # decoration + badge, exposes pick_permitted_default / query_for_denial.
        self._model_access = ModelAccessParameter(
            node=self,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
        )

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            Parameter(
                name="agent",
                type="Agent",
                output_type="Agent",
                tooltip="Optional agent to pass through the pipeline. The transcribed text is added to the agent's conversation memory.",
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterAudio(
                name="audio",
                default_value=None,
                tooltip="Audio to transcribe. Supported formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, flac. Max 25MB.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"clickable_file_browser": True, "expander": True},
            )
        )

        model_param = ParameterString(
            name="model",
            default_value=self._model_access.pick_permitted_default() or DEFAULT_MODEL,
            tooltip="Transcription model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "audio transcription model"},
        )
        self.add_parameter(model_param)
        self._model_access.install(model_param)

        self.add_node_element(
            ParameterMessage(
                name="model_deprecation_notice",
                title="Model Deprecated",
                variant="info",
                value="",
                traits={
                    Button(
                        full_width=True,
                        on_click=lambda _, __: self.hide_message_by_name("model_deprecation_notice"),
                    )
                },
                button_text="Dismiss",
                hide=True,
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
                placeholder_text="The transcribed text",
                ui_options={"display_name": "output", "pulse_on_run": True},
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

        # Status parameters MUST be last; this also injects generation_id, generation_status,
        # and the Refresh affordance into the Status group.
        self._create_status_parameters(
            result_details_tooltip="Details about the transcription result or any errors",
            result_details_placeholder="Transcription status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model":
            self._model_access.on_value_changed(value)
        if parameter.name == "response_format":
            if value == "verbose_json":
                self.show_parameter_by_name("segments")
                self.show_parameter_by_name("detected_language")
                self.show_parameter_by_name("duration")
            else:
                self.hide_parameter_by_name("segments")
                self.hide_parameter_by_name("detected_language")
                self.hide_parameter_by_name("duration")
        return super().after_value_set(parameter, value)

    def before_value_set(self, parameter: Parameter, value: Any) -> Any:
        if parameter.name == "model" and isinstance(value, str) and value in DEPRECATED_MODELS:
            replacement = DEPRECATED_MODELS[value]
            message = self.get_message_by_name_or_element_id("model_deprecation_notice")
            if message is not None:
                message.value = (
                    f"The '{value}' model has been deprecated and replaced with '{replacement}'. "
                    "Please save your workflow to apply this change."
                )
                self.show_message_by_name("model_deprecation_notice")
            value = replacement
        elif parameter.name == "model" and isinstance(value, str):
            self.hide_message_by_name("model_deprecation_notice")
        return super().before_value_set(parameter, value)

    def _get_api_model_id(self) -> str:
        """Get the API model ID for this generation."""
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

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
            # ``File`` resolves project macro paths (e.g. ``{outputs}/clip.mp3``) emitted by
            # upstream nodes, which a plain HTTP GET against the value would not.
            return await File(audio_url).aread_data_uri(fallback_mime="audio/mpeg")
        except FileLoadError as e:
            self._log(f"Failed to load audio: {e}")
            return None

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for OpenAI audio transcription."""
        # License-policy runtime gate. SuccessFailure idiom: route the denial
        # through _set_status_results and return an empty payload so the outer
        # proxy pipeline sees was_successful=False rather than an exception.
        # Runs BEFORE the proxy's own INVOKE_MODEL gate so a denied model fails
        # immediately instead of after payload construction.
        model_value = self.get_parameter_value("model")
        denial = self._model_access.query_for_denial(model_value)
        if denial is not None:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=denial.reason())
            return {}

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

        self._log("Built transcription payload")
        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
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

        # Thread the agent through using the new wire format (unwrap_agent / wrap_agent).
        # The proxy handles transcription so no agent.run() is called — we append a Run
        # directly rather than using insert_false_memory (which replaces runs[-1]).
        tool_configs: list = []
        ruleset_configs: list = []
        try:
            agent_input = self.get_parameter_value("agent")
            if isinstance(agent_input, dict):
                agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(agent_input)
                agent = GtAgent().from_dict(agent_core_dict)
            else:
                agent = GtAgent()
            if agent.conversation_memory is not None:
                agent.conversation_memory.runs.append(
                    Run(
                        input=TextArtifact("I'm passing you some audio to transcribe."),
                        output=TextArtifact(
                            f"<Thought>I temporarily used an Audio Transcription tool</Thought>{text}"
                            '\n<THOUGHT>\nmeta={"used_tool": true, "tool": "AudioTranscriptionTool"}\n</THOUGHT>'
                        ),
                    )
                )
            self.parameter_output_values["agent"] = wrap_agent(
                agent.to_dict(),
                tool_configs,
                ruleset_configs,
                provider=agent_input.get("provider") if isinstance(agent_input, dict) else None,
            )
        except Exception as e:
            logger.warning("TranscribeAudio: failed to thread agent: %s", e)

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
        self.parameter_output_values["agent"] = None
        self.parameter_output_values["output"] = None
        self.parameter_output_values["words"] = None
        self.parameter_output_values["segments"] = None
        self.parameter_output_values["detected_language"] = None
        self.parameter_output_values["duration"] = None
