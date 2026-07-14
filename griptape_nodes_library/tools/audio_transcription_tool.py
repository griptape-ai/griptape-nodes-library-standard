import openai
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.utils.model_invocation import declare_model_invocation_sync

API_KEY_ENV_VAR = "OPENAI_API_KEY"
SERVICE = "OpenAI"
DEFAULT_MODEL = "whisper-1"


class AudioTranscription(BaseTool):
    """Hands an agent a config for an `AudioTranscriptionTool` built at tool-use time.

    Architecture note: unlike `StructuredDataExtractor`/`PromptSummary`, this node
    does not build the underlying `OpenAiAudioTranscriptionDriver` itself.
    `process()` only emits a serializable `{"tool_type": ..., "model": ...}` config;
    `griptape_nodes_library.utils.agent_utils.build_tool_from_config` rebuilds that
    config into a live driver + tool later, from inside whichever node (Agent,
    DescribeImage, McpTask, ...) actually consumes the tool -- with no reference
    back to this node instance. The driver's `run` is therefore never reachable
    from here the way `BaseTool._gate_prompt_driver` reaches it for the other two
    tools: there is no live driver to wrap, and stashing this node on the config
    dict to reach one later would break the dict's serialization contract (agent
    wire format / workflow save-load) and would amount to casting whatever node
    happens to rebuild the tool into this node's identity.

    The nearest correct node-owned gating boundary is therefore here, at
    config-build time: this node declares the invocation for the model it is
    about to hand off, using its own identity, and fails closed if that is
    denied -- so a denied model never leaves this node as a usable tool config.
    This is coarser than the other two tools' per-call gate (it runs once per
    `process()` rather than once per actual transcription), which is the tradeoff
    of the config-dict/deferred-build design; it is not a per-invocation gate.
    """

    def process(self) -> None:
        model = self.parameter_values.get("model", DEFAULT_MODEL) or DEFAULT_MODEL

        declaration = declare_model_invocation_sync(self, model)
        if declaration.failed():
            details = str(declaration.result_details or f"{self.name}: model invocation was not permitted.")
            raise RuntimeError(details)

        self.parameter_output_values["tool"] = {"tool_type": "AudioTranscription", "model": model}

    def validate_before_workflow_run(self) -> list[Exception] | None:
        exceptions = []
        if self.parameter_values.get("driver", None):
            return exceptions
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            msg = f"{API_KEY_ENV_VAR} is not defined"
            exceptions.append(KeyError(msg))
            return exceptions
        try:
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
        except openai.AuthenticationError as e:
            exceptions.append(e)
        return exceptions if exceptions else None
