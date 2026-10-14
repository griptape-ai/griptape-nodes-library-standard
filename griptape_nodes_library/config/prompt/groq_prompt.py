from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

# --- Constants ---

SERVICE = "Groq"
BASE_URL = "https://api.groq.com/openai/v1"
API_KEY_URL = "https://console.groq.com/keys"
API_KEY_ENV_VAR = "GROQ_API_KEY"
MODEL_CHOICES = [
    "gemma2-9b-it",
    "meta-llama/llama-guard-4-12b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
    "llama3-8b-8192",
    "allam-2-7b",
    "deepseek-r1-distill-llama-70b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "Allam 2 7B": "allam-2-7b",
    "DeepSeek R1 Distill Llama 70B": "deepseek-r1-distill-llama-70b",
    "Gemma 2 9B IT": "gemma2-9b-it",
    "Llama 3 70B 8192": "llama3-70b-8192",
    "Llama 3 8B 8192": "llama3-8b-8192",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 4 Maverick 17B 128E Instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "Llama 4 Scout 17B 16E Instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "Llama Guard 4 12B": "meta-llama/llama-guard-4-12b",
    "groq_allam_2_7b": "allam-2-7b",
    "groq_deepseek_r1_distill_llama_70b": "deepseek-r1-distill-llama-70b",
    "groq_gemma2_9b_it": "gemma2-9b-it",
    "groq_llama3_70b_8192": "llama3-70b-8192",
    "groq_llama3_8b_8192": "llama3-8b-8192",
    "groq_llama_3_1_8b_instant": "llama-3.1-8b-instant",
    "groq_llama_3_3_70b_versatile": "llama-3.3-70b-versatile",
    "groq_llama_4_maverick_17b_128e_instruct": "meta-llama/llama-4-maverick-17b-128e-instruct",
    "groq_llama_4_scout_17b_16e_instruct": "meta-llama/llama-4-scout-17b-16e-instruct",
    "groq_llama_guard_4_12b": "meta-llama/llama-guard-4-12b",
}


class GroqPrompt(BasePrompt):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._install_model_access(
            model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL, deprecated_values=LEGACY_MODEL_VALUES
        )

        self.remove_parameter_element_by_name("seed")

        self.remove_parameter_element_by_name("top_k")

        self._replace_param_by_name(param_name="min_p", new_param_name="top_p", default_value=0.9)

    def process(self) -> None:
        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.GROQ,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
            base_url=BASE_URL,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
