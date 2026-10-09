from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Cohere"
API_KEY_URL = "https://dashboard.cohere.com/api-keys"
API_KEY_ENV_VAR = "COHERE_API_KEY"
MODEL_CHOICES = ["command-r-plus"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Command R+": "command-r-plus",
    "cohere_command_r_plus": "command-r-plus",
}


class CoherePrompt(BasePrompt):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        self._replace_param_by_name(
            param_name="min_p", new_param_name="p", tooltip=None, default_value=0.9, ui_options=None
        )
        self._replace_param_by_name(param_name="top_k", new_param_name="k")

        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("response_format")

    def process(self) -> None:
        """Map Cohere's `p` and `k` parameters to `top_p` and `top_k`."""
        self._raise_if_model_denied()

        settings = self._common_settings()
        for param_name, setting_name in (("p", "top_p"), ("k", "top_k")):
            value = self.get_parameter_value(param_name)
            if value is not None:
                settings[setting_name] = value

        config = self._build_model_config(
            ModelProvider.COHERE,
            self._get_selected_model_id(),
            settings=settings,
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
