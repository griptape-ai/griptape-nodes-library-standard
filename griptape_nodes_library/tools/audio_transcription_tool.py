import openai
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.tools.base_tool import BaseTool

API_KEY_ENV_VAR = "OPENAI_API_KEY"
SERVICE = "OpenAI"
DEFAULT_MODEL = "whisper-1"


class AudioTranscription(BaseTool):
    def process(self) -> None:
        model = self.parameter_values.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
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
