from typing import Any

from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.prompt.base_prompt import TOP_P_TOOLTIP, BasePrompt
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

        self._install_model_access(model_choices=MODEL_CHOICES, default_model=DEFAULT_MODEL)

        self.remove_parameter_element_by_name("seed")

        self.remove_parameter_element_by_name("top_k")

        self._replace_param_by_name(
            param_name="min_p", new_param_name="top_p", tooltip=TOP_P_TOOLTIP, default_value=0.9
        )

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        """Keep a saved model the list doesn't offer, so the run can name it instead of using the default.

        `Options` rewrites a value outside its choices to the first choice, and checks only while a
        value is set. A workflow loads its saved values with `initial_setup`, so an unlisted saved model
        is accepted for that one assignment. The choices are put back afterwards, so a re-saved
        workflow doesn't store a list that hides later changes to `MODEL_CHOICES`.
        """
        is_saved_model = initial_setup and param_name == "model" and isinstance(value, str) and value
        options = self._model_options() if is_saved_model else None
        if options is None or value in options.choices:
            return super().set_parameter_value(
                param_name,
                value,
                initial_setup=initial_setup,
                emit_change=emit_change,
                skip_before_value_set=skip_before_value_set,
            )
        choices = options.choices
        options.choices = [*choices, value]
        try:
            return super().set_parameter_value(
                param_name,
                value,
                initial_setup=initial_setup,
                emit_change=emit_change,
                skip_before_value_set=skip_before_value_set,
            )
        finally:
            options.choices = choices

    def _model_options(self) -> Options | None:
        parameter = self.get_parameter_by_name("model")
        if parameter is None:
            return None
        return next(iter(parameter.find_elements_by_type(Options)), None)

    def _unlisted_model_error(self) -> ValueError | None:
        model = self._get_selected_model_id()
        if model in MODEL_CHOICES:
            return None
        return ValueError(
            f"Model '{model}' is not one of the {SERVICE} models this node offers. Pick one from the list."
        )

    def process(self) -> None:
        unlisted_model_error = self._unlisted_model_error()
        if unlisted_model_error is not None:
            raise unlisted_model_error

        self._raise_if_model_denied()

        config = self._build_model_config(
            ModelProvider.OPENAI,
            self._get_selected_model_id(),
            api_key_secret=API_KEY_ENV_VAR,
        )
        self.parameter_output_values["prompt_model_config"] = config

    def validate_before_workflow_run(self) -> list[Exception] | None:
        unlisted_model_error = self._unlisted_model_error()
        if unlisted_model_error is not None:
            return [unlisted_model_error]
        return self._validate_api_key(
            service_name=SERVICE,
            api_key_env_var=API_KEY_ENV_VAR,
            api_key_url=API_KEY_URL,
        )
