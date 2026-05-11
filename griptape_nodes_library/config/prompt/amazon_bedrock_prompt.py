"""Defines the AmazonBedrockPrompt node for configuring the Amazon Bedrock Prompt Driver.

This module provides the `AmazonBedrockPrompt` class, which allows users
to configure and utilize the Amazon Bedrock prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt`, sets
Amazon Bedrock specific model options, requires a Amazon API Keys via
node configuration, and instantiates the `AmazonBedrockPromptDriver`.
"""

from typing import Any

import boto3  # pyright: ignore[reportMissingImports]
from griptape.drivers.prompt.amazon_bedrock import AmazonBedrockPromptDriver as GtAmazonBedrockPromptDriver
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt

# --- Constants ---

SERVICE = "Amazon"
API_KEY_URL = "https://console.aws.amazon.com/iam/home?#/security_credentials"
AWS_ACCESS_KEY_ID_ENV_VAR = "AWS_ACCESS_KEY_ID"
AWS_DEFAULT_REGION_ENV_VAR = "AWS_DEFAULT_REGION"
AWS_SECRET_ACCESS_KEY_ENV_VAR = "AWS_SECRET_ACCESS_KEY"  # noqa: S105

# Claude 4.x on Bedrock is only available via cross-region inference profiles
# (us.-prefixed IDs). DeepSeek R1 likewise. DeepSeek V3.2 ships as a standard
# foundation model and needs no prefix. Titan models are a legacy fallback.
MODEL_CHOICES = [
    "us.anthropic.claude-opus-4-7",
    "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "deepseek.v3.2",
    "us.deepseek.r1-v1:0",
    "us.meta.llama3-3-70b-instruct-v1:0",
    "us.meta.llama3-1-70b-instruct-v1:0",
    "amazon.titan-text-premier-v1:0",
    "amazon.titan-text-express-v1",
    "amazon.titan-text-lite-v1",
]
DEFAULT_MODEL = MODEL_CHOICES[0]

# Retired Bedrock model IDs that may appear in saved workflows. Maps each to
# its current replacement so the node auto-rewrites on load.
DEPRECATED_MODELS = {
    # Claude 3 family (retired by Anthropic Feb 19, 2026)
    "anthropic.claude-3-5-sonnet-20241022-v2:0": "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-3-5-sonnet-20240620-v1:0": "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-3-5-haiku-20241022-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-3-opus-20240229-v1:0": "us.anthropic.claude-opus-4-7",
    "anthropic.claude-3-sonnet-20240229-v1:0": "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-3-haiku-20240307-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-sonnet-4-6",
    "us.anthropic.claude-3-7-sonnet-20250219-v1:0": "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-sonnet-4-20250514-v1:0": "us.anthropic.claude-sonnet-4-6",
    # Bare Claude 4.x IDs (require inference profile prefix)
    "anthropic.claude-opus-4-7": "us.anthropic.claude-opus-4-7",
    "anthropic.claude-sonnet-4-6": "us.anthropic.claude-sonnet-4-6",
    "anthropic.claude-haiku-4-5-20251001-v1:0": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
}


class AmazonBedrockPrompt(BasePrompt):
    """Node for configuring and providing a Amazon Bedrock Chat Prompt Driver.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    customizes the available models to those supported by Amazon Bedrock,
    removes parameters not applicable to Amazon Bedrock (like 'seed'), and
    requires a Amazon Bedrock API key to be set in the node's configuration
    under the 'Amazon Bedrock' service.

    The `process` method gathers the configured parameters and the API key,
    utilizes the `_get_common_driver_args` helper from `BasePrompt`, adds
    Amazon Bedrock specific configurations, then instantiates a
    `AmazonBedrockPromptDriver` and assigns it to the 'prompt_model_config'
    output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the AmazonBedrockPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to use Amazon Bedrock specific models and sets a default.
        It also removes the 'seed' parameter inherited from `BasePrompt` as it's
        not directly supported by the Amazon Bedrock driver implementation.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Amazon Bedrock offers a lot of different models, so instead of using
        # a dropdown, we'll provide just a text field for the user to enter the model name
        # and set the default to the first model in the list.
        # TODO: https://github.com/griptape-ai/griptape-nodes/issues/876
        self._remove_options_trait(param="model")
        param = self.get_parameter_by_name("model")
        if param is not None:
            param.default_value = MODEL_CHOICES[0]

        # Add deprecation notice message element.
        self.add_node_element(
            ParameterMessage(
                name="model_deprecation_notice",
                title="Model Deprecation Notice",
                variant="info",
                value="",
                traits={
                    Button(
                        full_width=True,
                        on_click=lambda _, __: self.hide_message_by_name("model_deprecation_notice"),
                    )
                },
                button_text="Dismiss",
                hide=True,
            )
        )

        # Remove unused parameters for Amazon Bedrock
        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("min_p")
        self.remove_parameter_element_by_name("top_k")

        # Amazon Bedrock tends to fail if max_tokens isn't set
        self.set_parameter_value("max_tokens", 100)

    def before_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> Any:
        if parameter.name == "model":
            if isinstance(value, str) and value in DEPRECATED_MODELS:
                replacement = DEPRECATED_MODELS[value]
                message = self.get_message_by_name_or_element_id("model_deprecation_notice")
                if message is None:
                    raise RuntimeError("model_deprecation_notice message element not found")  # noqa: TRY003, EM101
                message.value = f"The '{value}' model has been deprecated. The model has been updated to '{replacement}'. Please save your workflow to apply this change."
                self.show_message_by_name("model_deprecation_notice")
                value = replacement
            else:
                self.hide_message_by_name("model_deprecation_notice")

        return super().before_value_set(parameter, value)

    def start_session(self) -> boto3.Session:
        """Starts a session with Amazon Bedrock using the provided AWS credentials."""
        aws_access_key_id = GriptapeNodes.SecretsManager().get_secret(AWS_ACCESS_KEY_ID_ENV_VAR)
        aws_secret_access_key = GriptapeNodes.SecretsManager().get_secret(AWS_SECRET_ACCESS_KEY_ENV_VAR)
        aws_default_region = GriptapeNodes.SecretsManager().get_secret(AWS_DEFAULT_REGION_ENV_VAR)

        try:
            session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=aws_default_region,
            )
        except Exception as e:
            msg = f"Failed to create AWS session for node {self.name}. Please check your AWS credentials and region."
            raise RuntimeError(msg) from e
        return session

    def process(self) -> None:
        """Processes the node configuration to create a AmazonBedrockPromptDriver.

        Retrieves parameter values set on the node and the required API key from
        the node's configuration system. It constructs the arguments dictionary
        for the `AmazonBedrockPromptDriver`, instantiates the
        driver, and assigns it to the 'prompt_model_config' output parameter.

        Raises:
            KeyError: If the Amazon Bedrock API key is not found in the node configuration
                      (though `validate_before_workflow_run` should prevent this during execution).
        """
        # Retrieve all parameter values set on the node UI or via input connections.
        params = self.parameter_values

        # --- Get Common Driver Arguments ---
        # Use the helper method from BasePrompt to get args like temperature, stream, max_attempts, etc.
        common_args = self._get_common_driver_args(params)

        # Start a session with the AWS Credentials.
        session = self.start_session()

        # --- Prepare Amazon Bedrock Specific Arguments ---
        specific_args = {}

        # Get the selected model.
        specific_args["model"] = self.get_parameter_value("model")

        # --- Combine Arguments and Instantiate Driver ---
        # Combine common arguments with Amazon Bedrock specific arguments.
        # Specific args take precedence if there's an overlap (though unlikely here).
        all_kwargs = {**common_args, **specific_args}

        # Create the Amazon Bedrock prompt driver instance.
        driver = GtAmazonBedrockPromptDriver(session=session, **all_kwargs)

        # Set the output parameter 'prompt_model_config'.
        self.parameter_output_values["prompt_model_config"] = driver

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Validates that the Amazon Bedrock API keys are configured correctly.

        Calls the base class helper `_validate_api_key` with Amazon Bedrock-specific
        configuration details.
        """
        exceptions = []
        api_keys_to_check = [
            AWS_ACCESS_KEY_ID_ENV_VAR,
            AWS_SECRET_ACCESS_KEY_ENV_VAR,
            AWS_DEFAULT_REGION_ENV_VAR,
        ]
        for key in api_keys_to_check:
            valid_key = self._validate_api_key(service_name=SERVICE, api_key_env_var=key, api_key_url=API_KEY_URL)
            if valid_key is not None:
                exceptions.append(valid_key)
        return exceptions if any(exceptions) else None
