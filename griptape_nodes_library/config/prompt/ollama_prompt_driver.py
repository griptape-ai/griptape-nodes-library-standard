"""Defines the OllamaPrompt node for configuring an Ollama prompt model.

This module provides the `OllamaPrompt` class, which allows users
to configure and utilize the Ollama prompt service within the Griptape
Nodes framework. It inherits common prompt parameters from `BasePrompt` and
emits a `ModelConfig`.
"""

import logging
from typing import Any

import httpx
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterGroup, ParameterMessage
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.prompt.base_prompt import BasePrompt
from griptape_nodes_library.llm.model_config import ModelProvider

logger = logging.getLogger("griptape_nodes")


class OllamaConnectionError(Exception):
    """Exception raised when unable to connect to Ollama server or retrieve models."""


# --- Constants ---

DEFAULT_PORT = "11434"
DEFAULT_BASE_URL = "http://127.0.0.1"
# Ollama lists models with their tag, so validation matches the tagged name.
DEFAULT_MODEL = "llama3.2:latest"
MODEL_LIST_TIMEOUT_SECONDS = 5.0
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
# Status text older versions of this node stored as the model value; never a model name.
LEGACY_STATUS_PREFIXES = ("⚠️", "📝", "🔄")


class OllamaPrompt(BasePrompt):
    """Node for configuring an Ollama prompt model.

    Inherits from `BasePrompt` to leverage common LLM parameters. This node
    lists the models installed on the Ollama server, adds Ollama-specific
    parameters like base_url and port, and does not require an API key since
    Ollama runs locally.

    The `process` method turns the configured parameters into a `ModelConfig` that
    names the secrets to use, and assigns it to the 'prompt_model_config' output parameter.
    """

    def __init__(self, **kwargs) -> None:
        """Initializes the OllamaPrompt node.

        Calls the superclass initializer, then modifies the inherited 'model'
        parameter to list the models installed on the Ollama server. It also adds
        Ollama-specific parameters for base_url and port.
        """
        super().__init__(**kwargs)

        # --- Customize Inherited Parameters ---

        # Replace the model parameter with one that has Button trait
        self.remove_parameter_element_by_name("model")

        models = self._fetch_models_or_none()
        choices = models or [DEFAULT_MODEL]

        self.model_param = Parameter(
            name="model",
            input_types=["str"],
            type="str",
            output_type="str",
            default_value=choices[0],
            tooltip="Select the model you want to use from the available options.",
            traits={
                Options(choices=choices),
                Button(
                    icon="list-restart",
                    size="icon",
                    variant="secondary",
                    on_click=self._on_refresh_clicked,
                ),
            },
        )
        self.add_parameter(self.model_param)

        self.move_element_to_position(self.model_param.name, self.get_element_index("prompt_model_config") + 1)

        # Remove parameters not typically used by Ollama
        self.remove_parameter_element_by_name("seed")
        self.remove_parameter_element_by_name("min_p")
        self.remove_parameter_element_by_name("top_k")

        # Add Ollama-specific parameters
        with ParameterGroup(name="connection_settings", ui_options={"collapsed": True}) as ollama_group:
            Parameter(
                name="base_url", default_value=DEFAULT_BASE_URL, type="str", tooltip="Base URL for the Ollama server"
            )
            Parameter(name="port", default_value=DEFAULT_PORT, type="str", tooltip="Port for the Ollama server")
        self.add_node_element(ollama_group)
        self.move_element_to_position(ollama_group.name, self.get_element_index("model") + 1)

        # Message for when Ollama server is not installed/running
        self.install_ollama_message = ParameterMessage(
            name="install_ollama_message",
            title="Install Ollama",
            value=f"To use the Ollama Prompt Models, you will need to have the Ollama server running.\nYou can download the Ollama server from {OLLAMA_DOWNLOAD_URL}.",
            variant="warning",
            button_link=OLLAMA_DOWNLOAD_URL,
            button_text="Download Ollama",
            button_icon="download",
        )
        self.add_node_element(self.install_ollama_message)
        self.move_element_to_position(self.install_ollama_message.name, "first")

        # Message for when Ollama is running but no models are available
        self.install_models_message = ParameterMessage(
            name="install_models_message",
            title="Install Models",
            value="Ollama is running but no models are available.\nInstall models using 'ollama pull <model>' command or visit the Ollama documentation for more information.",
            variant="info",
            button_link="https://docs.ollama.com/",
            button_text="Ollama Documentation",
            button_icon="book",
        )
        self.add_node_element(self.install_models_message)
        self.move_element_to_position(self.install_models_message.name, "first")

        self._update_ollama_messages(models)

    def _host(self) -> str:
        base_url = self.get_parameter_value("base_url") or DEFAULT_BASE_URL
        port = self.get_parameter_value("port") or DEFAULT_PORT
        return f"{base_url.rstrip('/')}:{port}"

    def _fetch_models(self) -> list[str]:
        """Return the models installed on the Ollama server, sorted by name.

        Raises:
            OllamaConnectionError: If the server can't be reached or its model list can't be read.
        """
        try:
            response = httpx.get(f"{self._host()}/api/tags", timeout=MODEL_LIST_TIMEOUT_SECONDS)
            response.raise_for_status()
            return sorted(model.get("model") or model["name"] for model in response.json().get("models", []))
        except Exception as e:
            msg = f"Unable to get available models from Ollama at {self._host()}: {e}"
            raise OllamaConnectionError(msg) from e

    def _fetch_models_or_none(self) -> list[str] | None:
        """Return the installed models, or None if the Ollama server can't be reached."""
        try:
            return self._fetch_models()
        except OllamaConnectionError as e:
            logger.warning("%s: %s", self.name, e)
            return None

    def _update_ollama_messages(self, models: list[str] | None) -> None:
        """Show the install message matching the server state: unreachable, no models, or ready."""
        if models is None:
            self.show_message_by_name("install_ollama_message")
            self.hide_message_by_name("install_models_message")
        elif not models:
            self.hide_message_by_name("install_ollama_message")
            self.show_message_by_name("install_models_message")
        else:
            self.hide_message_by_name("install_ollama_message")
            self.hide_message_by_name("install_models_message")

    def _refresh_model_list(self) -> None:
        """Re-read the server's models into the model choices and update the install messages.

        Keeps the selected model when the server still has it, or when the server can't list models,
        so a workflow opened before Ollama starts keeps its model. Status messages live in the install
        messages, never in the model value.
        """
        models = self._fetch_models_or_none()
        current = self.get_parameter_value("model")
        if not current or str(current).startswith(LEGACY_STATUS_PREFIXES):
            current = None

        if models and current not in models:
            selected = models[0]
        else:
            selected = current or DEFAULT_MODEL
        self._update_option_choices(param="model", choices=models or [selected], default=selected)
        self._update_ollama_messages(models)

    def _on_refresh_clicked(
        self,
        button: Button,  # noqa: ARG002
        button_details: ButtonDetailsMessagePayload,  # noqa: ARG002
    ) -> NodeMessageResult | None:
        self._refresh_model_list()
        return None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Handle parameter value changes.

        Refreshes model list when:
        - Base URL or port changes (indicating different Ollama server)
        """
        if parameter.name in ["base_url", "port"]:
            self._refresh_model_list()

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Performs pre-run validation checks for the node.

        Checks if Ollama is installed and if the selected model is available.

        Returns:
            A list of Exception objects if validation fails, otherwise None.
        """
        selected_model = self.get_parameter_value("model")
        if not selected_model:
            return [ValueError(f"{self.name}: No model selected.")]

        try:
            available_models = self._fetch_models()
        except OllamaConnectionError as e:
            logger.warning("%s: %s", self.name, e)
            msg = (
                f"{self.name}: Can't reach the Ollama server at {self._host()}. Start Ollama "
                f"(download it from {OLLAMA_DOWNLOAD_URL}), or check the base_url and port settings."
            )
            return [OllamaConnectionError(msg)]

        if not available_models:
            msg = f"{self.name}: Ollama is running but has no models. Install one with 'ollama pull <model>'."
            return [ValueError(msg)]

        if selected_model not in available_models:
            msg = (
                f"{self.name}: Model '{selected_model}' is not installed in Ollama. "
                f"Available models: {', '.join(available_models)}"
            )
            return [ValueError(msg)]

        return None

    def process(self) -> None:
        """Emits the `ModelConfig` for the selected Ollama model.

        Builds the OpenAI-compatible endpoint (`/v1`) from the base_url and port
        parameters. Ollama needs no API key.
        """
        base_url = self.get_parameter_value("base_url") or DEFAULT_BASE_URL
        port = self.get_parameter_value("port") or DEFAULT_PORT

        config = self._build_model_config(
            ModelProvider.OLLAMA,
            self.get_parameter_value("model"),
            base_url=f"{base_url.rstrip('/')}:{port}/v1",
        )
        self.parameter_output_values["prompt_model_config"] = config
