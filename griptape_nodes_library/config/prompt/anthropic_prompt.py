from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Anthropic"
API_KEY_URL = "https://console.anthropic.com/settings/keys"
API_KEY_ENV_VAR = "ANTHROPIC_API_KEY"
MODEL_CHOICES = [
    "claude-opus-4-7",
    "claude-sonnet-4-6",
    "claude-haiku-4-5",
]
DEFAULT_MODEL = MODEL_CHOICES[1]  # claude-sonnet-4-6

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Claude Haiku 4.5": "claude-haiku-4-5",
    "Claude Opus 4.7": "claude-opus-4-7",
    "Claude Sonnet 4.6": "claude-sonnet-4-6",
    "gtc_claude_haiku_4_5": "claude-haiku-4-5",
    "gtc_claude_opus_4_7": "claude-opus-4-7",
    "gtc_claude_sonnet_4_6": "claude-sonnet-4-6",
    # Dated model versions
    "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
    "claude-3-5-haiku-20241022": "claude-haiku-4-5",
    "claude-3-opus-20240229": "claude-opus-4-7",
    "claude-3-sonnet-20240229": "claude-sonnet-4-6",
    "claude-3-haiku-20240307": "claude-haiku-4-5",
    "claude-3-7-sonnet-20250219": "claude-sonnet-4-6",
    "claude-sonnet-4-20250514": "claude-sonnet-4-6",
    "claude-opus-4-20250514": "claude-opus-4-7",
    # -latest variants
    "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
    "claude-3-5-opus-latest": "claude-opus-4-7",
    "claude-3-5-haiku-latest": "claude-haiku-4-5",
    # Superseded current-generation models
    "claude-haiku-4-5-20251001": "claude-haiku-4-5",
    "claude-opus-4-6": "claude-opus-4-7",
}


class AnthropicPrompt(BasePrompt):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        self._replace_param_by_name(
            param_name="min_p", new_param_name="top_p", tooltip=None, default_value=0.9, ui_options=None
        )

        self.remove_parameter_element_by_name("seed")

    def process(self) -> None:
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.ANTHROPIC,
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
