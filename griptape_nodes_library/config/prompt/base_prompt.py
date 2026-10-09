"""Base class for prompt model configuration nodes."""

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter

from griptape_nodes_library.config.base_driver import BaseDriver
from griptape_nodes_library.llm.model_config import (
    PROMPT_MODEL_CONFIG_TYPE,
    USE_NATIVE_TOOLS_OPTION,
    ModelConfig,
    ModelProvider,
)


class BasePrompt(BaseDriver):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        driver_parameter = self.get_parameter_by_name("driver")
        if driver_parameter is not None:
            driver_parameter.name = "prompt_model_config"
            driver_parameter.output_type = PROMPT_MODEL_CONFIG_TYPE
            driver_parameter._ui_options = {"display_name": "prompt model config"}

        self.add_parameter(
            Parameter(
                name="message",
                type="str",
                default_value="⚠️ This node requires an API key to function.",
                tooltip="",
                allowed_modes={},  # type: ignore  # noqa: PGH003
                ui_options={"is_full_width": True, "multiline": True, "hide": True},
            )
        )
        # Parameter for model selection. Subclasses either call
        # `_install_model_access` to offer a license-filtered dropdown of their
        # models, or add their own trait when the choices are theirs to manage
        # (a free-text field, or a list fetched from the provider at runtime).
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="Select the model you want to use from the available options.",
            )
        )

        self.add_parameter(
            Parameter(
                name="temperature",
                input_types=["float"],
                type="float",
                output_type="float",
                default_value=0.1,
                tooltip="Temperature for creativity. Higher values will be more creative.",
                ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
            )
        )
        self.add_parameter(
            Parameter(
                name="max_attempts_on_fail",
                input_types=["int"],
                type="int",
                output_type="int",
                default_value=2,
                tooltip="Maximum attempts on failure",
                ui_options={"slider": {"min_val": 1, "max_val": 100}},
            )
        )

        self.add_parameter(
            Parameter(
                name="seed",
                input_types=["int"],
                type="int",
                output_type="int",
                default_value=10342349342,
                tooltip="Seed for random number generation",
            )
        )

        self.add_parameter(
            Parameter(
                name="min_p",
                input_types=["float"],
                type="float",
                output_type="float",
                default_value=0.1,
                tooltip="Minimum probability for sampling. Lower values will be more random.",
                ui_options={"slider": {"min_val": 0.0, "max_val": 1.0}, "step": 0.01},
            )
        )

        self.add_parameter(
            Parameter(
                name="top_k",
                input_types=["int"],
                type="int",
                output_type="int",
                default_value=50,
                tooltip="Limits the number of tokens considered for each step of the generation. Prevents the model from focusing too narrowly on the top choices.",
            )
        )

        self.add_parameter(
            Parameter(
                name="use_native_tools",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=True,
                tooltip="Use native tool calling. Prompted tool use is unsupported, so agents with tools fail when False.",
            )
        )

        self.add_parameter(
            Parameter(
                name="max_tokens",
                input_types=["int"],
                type="int",
                output_type="int",
                default_value=-1,
                tooltip="Maximum tokens to generate. If <=0, it will use the default based on the tokenizer.",
            )
        )

        # Kept so saved workflows load; pydantic-ai has no equivalent.
        self.add_parameter(
            Parameter(
                name="stream",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=True,
                tooltip="",
                ui_options={"hide": True},
            )
        )

    def _common_settings(self) -> dict[str, Any]:
        """Return generation settings supported by pydantic-ai."""
        settings: dict[str, Any] = {}
        for name in ("temperature", "seed", "top_k", "top_p"):
            value = self.get_parameter_value(name) if self.get_parameter_by_name(name) is not None else None
            if value is not None:
                settings[name] = value

        max_tokens = self.get_parameter_value("max_tokens")
        if max_tokens is not None and max_tokens > 0:
            settings["max_tokens"] = max_tokens
        return settings

    def _build_model_config(  # noqa: PLR0913
        self,
        provider: ModelProvider,
        model: str,
        *,
        settings: dict[str, Any] | None = None,
        api_key_secret: str | None = None,
        base_url: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> ModelConfig:
        # griptape counted total attempts; SDK clients count retries after the first.
        max_attempts = self.get_parameter_value("max_attempts_on_fail")
        max_retries = None if max_attempts is None else max(max_attempts - 1, 0)
        options = dict(options or {})
        if self.get_parameter_value("use_native_tools") is False:
            options[USE_NATIVE_TOOLS_OPTION] = False
        return ModelConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_secret=api_key_secret,
            settings=self._common_settings() if settings is None else settings,
            max_retries=max_retries,
            options=options,
        )

    def process(self) -> None:
        """Raise because the base node has no model provider."""
        msg = f"{type(self).__name__} does not configure a model provider. Use a provider-specific prompt node."
        raise NotImplementedError(msg)
