from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts import TextArtifact
from griptape.artifacts.audio_url_artifact import AudioUrlArtifact
from griptape.memory.structure import Run
from griptape_nodes.exe_types.core_types import (
    NodeMessageResult,
    Parameter,
    ParameterGroup,
    ParameterMessage,
    ParameterMode,
)
from griptape_nodes.exe_types.param_types.parameter_audio import ParameterAudio
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.access_events import (
    QueryModelAccessForNodeRequest,
    QueryModelAccessForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
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

        # Cache the per-provider denial map at node-init so before_value_set
        # can render the notice without a round-trip on every selection change.
        # _build_payload() re-asks the engine so a hook decision that flipped
        # between creation and run still wins.
        self._denial_by_provider_id: dict[str, CheckpointDenial] = self._fetch_denial_map()
        default_model = self._pick_default_model()

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

        self.add_parameter(
            ParameterString(
                name="model",
                default_value=default_model,
                tooltip="Transcription model to use",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={
                    Options(choices=list(MODEL_CHOICES)),
                    Button(
                        icon="list-restart",
                        size="icon",
                        variant="secondary",
                        on_click=self._on_refresh_access_click,
                        tooltip="Refresh available models",
                    ),
                },
                ui_options=self._model_ui_options(),
            )
        )

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

        # Reflect the initial selection: a node born with a denied default gets
        # the badge immediately.
        self._update_model_access_badge(default_model)

    def _update_model_access_badge(self, value: Any) -> None:
        """Set or clear the `model` parameter's badge based on the selected value's policy verdict.

        Reads the cached ``_denial_by_provider_id`` (built at init) so this is a
        local map lookup, not a round-trip. ``_build_payload`` re-asks the engine
        at run-time so a hook decision that changed between node creation and
        run is still enforced.
        """
        param = self.get_parameter_by_name("model")
        if param is None:
            return
        if not isinstance(value, str):
            param.clear_badge()
            return
        denial = self._denial_by_provider_id.get(value)
        if denial is None:
            param.clear_badge()
            return
        param.set_badge(
            variant="error",
            title="Model Not Permitted",
            message=f"Model `{value}` is not permitted. Running this node will fail.\n\nReason(s): {denial.reason()}",
            icon="shield-off",
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "model":
            self._update_model_access_badge(value)
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

    def _fetch_denial_map(self) -> dict[str, CheckpointDenial]:
        """Return `{provider_model_id: CheckpointDenial}` for every denied catalog model.

        Asks the engine which of the node's declared catalog models the
        authorization hook denies, keyed by the upstream provider's name (which
        is what the dropdown stores). Also populates ``_catalog_id_by_provider_id``
        so the runtime check in ``_build_payload`` can translate the dropdown
        name back to the catalog id the engine policy actually matches on.
        Empty on engine failure -- internal errors must not silently strip the
        dropdown.
        """
        self._catalog_id_by_provider_id: dict[str, str] = {}
        result = GriptapeNodes.handle_request(QueryModelAccessForNodeRequest(node_type=type(self).__name__))
        if not isinstance(result, QueryModelAccessForNodeResultSuccess):
            return {}
        denials: dict[str, CheckpointDenial] = {}
        for verdict in result.verdicts:
            if verdict.provider_model_id is not None:
                self._catalog_id_by_provider_id[verdict.provider_model_id] = verdict.model_id
                if verdict.denial is not None:
                    denials[verdict.provider_model_id] = verdict.denial
        return denials

    def _pick_default_model(self) -> str:
        """Return `DEFAULT_MODEL` if it's allowed; otherwise the first allowed entry.

        Falls back to `DEFAULT_MODEL` when every entry is denied so the dropdown
        still has a value the user can see and the warning notice can render.
        The existing `INSTANTIATE_NODE` gate prevents the node from being created
        at all when its single declared model is denied, so this fallback only
        matters if the two checkpoints disagree.
        """
        if DEFAULT_MODEL not in self._denial_by_provider_id:
            return DEFAULT_MODEL
        for choice in MODEL_CHOICES:
            if choice not in self._denial_by_provider_id:
                return choice
        return DEFAULT_MODEL

    def _model_ui_options(self) -> dict[str, Any]:
        """Build the `ui_options` dict for the model parameter, including per-row decoration."""
        data: list[dict[str, str]] = []
        for choice in MODEL_CHOICES:
            if choice in self._denial_by_provider_id:
                data.append({"name": choice, "icon": "shield-off", "subtitle": "Not permitted by your license"})
            else:
                data.append({"name": choice})
        return {
            "display_name": "audio transcription model",
            "data": data,
            "dropdown_row_icons": True,
            "dropdown_row_subtitles": True,
        }

    def _fetch_runtime_denial(self, model_provider_id: str) -> CheckpointDenial | None:
        """Ask the engine whether this specific model is permitted, right now.

        Translates the dropdown name (e.g. ``"whisper-1"``) to the catalog id the
        engine policy matches on (e.g. ``"gtc_whisper_1"``) via the mapping built
        at init time. Falls through to ``None`` on any failure -- a missing
        mapping means the dropdown got out of sync with the manifest; an engine
        ``Failure`` means an internal error. In neither case do we want to gate
        user work.
        """
        catalog_id = self._catalog_id_by_provider_id.get(model_provider_id)
        if catalog_id is None:
            return None
        result = GriptapeNodes.handle_request(
            QueryModelAccessForNodeRequest(
                node_type=type(self).__name__,
                candidate_model_ids=[catalog_id],
            )
        )
        if not isinstance(result, QueryModelAccessForNodeResultSuccess) or not result.verdicts:
            return None
        return result.verdicts[0].denial

    def _on_refresh_access_click(
        self, _button: Button, _button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:
        """Re-query the engine and refresh the dropdown decoration + current-selection badge.

        Fires when the user clicks the inline refresh button next to the model
        dropdown. Useful when a license / permission state has changed under the
        running engine (e.g. the user upgraded their plan) and the artist wants
        the dropdown to reflect it without recreating the node or reloading the
        workflow.
        """
        self._denial_by_provider_id = self._fetch_denial_map()
        model_param = self.get_parameter_by_name("model")
        if model_param is not None:
            model_param.update_ui_options(self._model_ui_options())
        self._update_model_access_badge(self.get_parameter_value("model"))
        return None

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
        # Runtime entitlement gate. Re-asks the engine rather than reading the
        # cached _denial_by_provider_id so a hook decision that changed between
        # node creation and run is honored.
        model_input = self.get_parameter_value("model")
        if isinstance(model_input, str):
            denial = self._fetch_runtime_denial(model_input)
            if denial is not None:
                msg = f"Cannot run {type(self).__name__}: '{model_input}' is not permitted. {denial.reason()}"
                raise RuntimeError(msg)

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
