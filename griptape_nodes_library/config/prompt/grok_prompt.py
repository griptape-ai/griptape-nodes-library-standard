from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Grok"
API_KEY_URL = "https://console.x.ai"
API_KEY_ENV_VAR = "GROK_API_KEY"
MODEL_CHOICES = [
    "grok-3-beta",
    "grok-3-fast-beta",
    "grok-3-mini-beta",
    "grok-3-mini-fast-beta",
    "grok-2-vision-1212",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Grok 2 Vision": "grok-2-vision-1212",
    "Grok 3 Beta": "grok-3-beta",
    "Grok 3 Fast Beta": "grok-3-fast-beta",
    "Grok 3 Mini Beta": "grok-3-mini-beta",
    "Grok 3 Mini Fast Beta": "grok-3-mini-fast-beta",
    "xai_grok_2_vision_1212": "grok-2-vision-1212",
    "xai_grok_3_beta": "grok-3-beta",
    "xai_grok_3_fast_beta": "grok-3-fast-beta",
    "xai_grok_3_mini_beta": "grok-3-mini-beta",
    "xai_grok_3_mini_fast_beta": "grok-3-mini-fast-beta",
}


class GrokPrompt(BasePrompt):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        # Remove `top_k` parameter as it's not used by Grok.
        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("top_k")

        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.GROK,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
