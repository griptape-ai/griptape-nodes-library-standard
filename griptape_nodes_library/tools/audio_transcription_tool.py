import openai
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.utils.node_error_utils import error_fields, error_response, missing_secret_error

API_KEY_ENV_VAR = "OPENAI_API_KEY"
API_KEY_URL = "https://platform.openai.com/api-keys"
SERVICE = "OpenAI"
DEFAULT_MODEL = "whisper-1"


class AudioTranscription(BaseTool):
    def process(self) -> None:
        model = self.parameter_values.get("model", DEFAULT_MODEL) or DEFAULT_MODEL
        self.parameter_output_values["tool"] = {"tool_type": "AudioTranscription", "model": model}

    def validate_before_workflow_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []
        if self.parameter_values.get("driver", None):
            return exceptions
        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_ENV_VAR)
        if not api_key:
            exceptions.append(missing_secret_error(API_KEY_ENV_VAR, key_url=API_KEY_URL))
            return exceptions
        try:
            client = openai.OpenAI(api_key=api_key)
            client.models.list()
        except openai.AuthenticationError as e:
            body = e.body if isinstance(e.body, dict) else {}
            exceptions.append(
                NodeError(
                    body.get("message") or e.message,
                    fields=error_fields(
                        status_code=e.status_code,
                        error_code=body.get("code"),
                        request_id=e.request_id,
                    ),
                    response=error_response(body),
                )
            )
        return exceptions if exceptions else None
