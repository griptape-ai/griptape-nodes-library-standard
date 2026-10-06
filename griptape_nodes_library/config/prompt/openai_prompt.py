from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "OpenAI"
API_KEY_URL = "https://platform.openai.com/api-keys"
API_KEY_ENV_VAR = "OPENAI_API_KEY"
MODEL_CHOICES = [
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-4o",
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
]
DEFAULT_MODEL = MODEL_CHOICES[0]


class OpenAiPrompt(BasePrompt):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Update the 'model' parameter for OpenAi specifics.
        # call openai listmodels to get available models
        import openai

        available_models = [model.id for model in openai.Client().models.list().data]
        # This node offers whatever models the account can see rather than a
        # declared list, so it owns the dropdown trait instead of routing through
        # `_install_model_access`.
        model_param = self.get_parameter_by_name("model")
        if model_param is not None:
            model_param.add_trait(Options(choices=available_models))
        self._update_option_choices(
            param="model", choices=available_models, default=available_models[0] if available_models else ""
        )

        self.remove_parameter_element_by_name("seed")

        self.remove_parameter_element_by_name("top_k")

        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        config = self._build_model_config(
            ModelProvider.OPENAI,
            self.get_parameter_value("model"),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
