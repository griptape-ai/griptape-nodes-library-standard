"""Defines the BasePrompt node, an abstract base class for prompt model config nodes.

This module provides the `BasePrompt` class, which serves as a foundation
for creating specific prompt model config nodes within the Griptape
Nodes framework. It inherits from `BaseDriver` and defines common parameters
used by most LLM providers (like temperature, model, etc.). Subclasses
should inherit from `BasePrompt` and override the `process` method to emit a
`ModelConfig` for their provider.
"""

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter

from griptape_nodes_library.config.base_driver import BaseDriver
from griptape_nodes_library.llm.model_config import PROMPT_MODEL_CONFIG_TYPE, ModelConfig, ModelProvider


class BasePrompt(BaseDriver):
    """Abstract base node for configuring prompt models.

    Inherits from `BaseDriver` and provides a standard set of parameters common
    to many Large Language Model (LLM) providers, such as temperature,
    model selection, and token limits.

    It renames the inherited 'driver' output parameter to 'prompt_model_config'
    to clearly indicate its purpose in the context of prompt configuration.

    Key Features for Subclasses:
    - Defines common LLM parameters accessible via `self.parameter_values`.
    - Provides `_build_model_config` to turn the base parameters into a `ModelConfig`.
    - Provides `_validate_api_key` to standardize API key validation logic.
    - Provides `_install_model_access` to turn the 'model' parameter into a
      license-filtered dropdown of driver-specific models.
    Note: The `process` method in this base class has no provider to describe and
    raises. Direct use of `BasePrompt` is not intended.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the BasePrompt node.

        Sets up the node by calling the superclass initializer, renaming the
        inherited 'driver' output parameter to 'prompt_model_config', and
        adding standard parameters common across various prompt drivers.
        """
        super().__init__(**kwargs)

        # Rename the inherited output parameter for clarity in this context.
        # The base 'BaseDriver' likely outputs a generic 'driver', but here we
        # specifically output a 'Prompt Model Config'.
        driver_parameter = self.get_parameter_by_name("driver")
        if driver_parameter is not None:
            driver_parameter.name = "prompt_model_config"
            driver_parameter.output_type = PROMPT_MODEL_CONFIG_TYPE
            driver_parameter._ui_options = {"display_name": "prompt model config"}

        # --- Common Prompt Driver Parameters ---
        # These parameters represent settings frequently used by LLM prompt drivers.
        # Subclasses will typically use these values when instantiating their specific driver.

        # Parameter for user messages.
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

        # Parameter controlling randomness/creativity in generation.
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
        # Parameter for retry logic upon driver failure.
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

        # Parameter for reproducibility (if supported by the driver).
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

        # Parameter for nucleus sampling (alternative/complement to temperature).
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

        # Parameter for limiting the sampling pool (top-k sampling).
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

        # Parameter to enable/disable model-specific tool use capabilities.
        self.add_parameter(
            Parameter(
                name="use_native_tools",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=True,
                tooltip="Use native tools for the LLM.",
            )
        )

        # Parameter to limit the length of the generated response.
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

        # Parameter to enable/disable streaming output from the driver.
        self.add_parameter(
            Parameter(
                name="stream",
                input_types=["bool"],
                type="bool",
                output_type="bool",
                default_value=True,
                tooltip="",
            )
        )

    def _common_settings(self) -> dict[str, Any]:
        """Collects the generation settings shared by the prompt nodes.

        A parameter a subclass removed (or one whose value is `None`) is skipped,
        and `max_tokens` is only included when it is greater than 0.
        `stream` and `use_native_tools` have no pydantic-ai equivalent and are ignored.
        """
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
        """Builds the `ModelConfig` this node outputs.

        Args:
            provider: Which API the config targets.
            model: The provider's model id.
            settings: Replaces the shared settings from `_common_settings` when given.
            api_key_secret: Name of the secret holding the API key. Never the key itself.
            base_url: Endpoint override.
            options: Provider-specific extras.
        """
        max_retries = self.get_parameter_value("max_attempts_on_fail")
        return ModelConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key_secret=api_key_secret,
            settings=self._common_settings() if settings is None else settings,
            max_retries=max_retries,
            options=options or {},
        )

    def process(self) -> None:
        """Subclasses MUST override this to set 'prompt_model_config' to their `ModelConfig`.

        Typical shape: read the node's parameters, call `_build_model_config` with
        the provider, model id and API key secret name, and assign the result to
        `self.parameter_output_values["prompt_model_config"]`.

        Raises:
            NotImplementedError: Always; the base node has no provider.
        """
        msg = f"{type(self).__name__} does not configure a model provider. Use a provider-specific prompt node."
        raise NotImplementedError(msg)
