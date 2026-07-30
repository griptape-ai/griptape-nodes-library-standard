from typing import cast

from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.retained_mode.events.agent_events import (
    ListAgentProvidersRequest,
    ListAgentProvidersResultSuccess,
    ListProviderModelsRequest,
    ListProviderModelsResultSuccess,
    ProviderConfig,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

_GRIPTAPE_CLOUD_PROVIDER = ProviderConfig(name="griptape_cloud", type="griptape_cloud", model="")


class ProviderSelectionComponent:
    def __init__(self, node, model_param, *, gtc_model_choices, gtc_model_data):
        # adds model_provider parameter to the node (buttons wired to self)
        self._node = node
        self._model_param = model_param
        self._gtc_model_choices = gtc_model_choices
        self._gtc_model_data = gtc_model_data

        provider_names = self._fetch_provider_names()

        self._node.add_parameter(
            Parameter(
                name="model_provider",
                type="str",
                default_value=provider_names[0] if provider_names else "griptape_cloud",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Choose a provider. Refresh to see all configured providers.",
                traits={
                    Options(choices=provider_names),
                    Button(
                        icon="list-restart",
                        size="icon",
                        variant="secondary",
                        on_click=self._refresh_providers_button,
                    ),
                },
                ui_options={"display_name": "provider"},
            )
        )
        self._node.add_parameter(model_param)

    def on_provider_changed(self, provider_name: str) -> None:
        self.update_model_choices_for_provider(provider_name)

    def hide(self) -> None:
        self._node.hide_parameter_by_name("model")
        self._node.hide_parameter_by_name("model_provider")

    def show(self) -> None:
        self._node.show_parameter_by_name("model")
        self._node.show_parameter_by_name("model_provider")

    def uses_griptape_cloud_driver(self) -> bool:
        if self._node.get_parameter_value("agent") is not None:
            return False
        if isinstance(self._node.get_parameter_value("model"), BasePromptDriver):
            return False
        provider_name = self._node.get_parameter_value("model_provider") or "griptape_cloud"
        return provider_name == "griptape_cloud"

    def _fetch_providers(self) -> list[ProviderConfig]:

        _FALLBACK = [_GRIPTAPE_CLOUD_PROVIDER]
        try:
            result = GriptapeNodes.handle_request(ListAgentProvidersRequest())
            if not isinstance(result, ListAgentProvidersResultSuccess):
                return _FALLBACK
            return cast(ListAgentProvidersResultSuccess, result).providers or _FALLBACK
        except Exception:
            return _FALLBACK

    def _fetch_provider_names(self) -> list[str]:
        providers = self._fetch_providers()
        return [p.name for p in providers] or ["griptape_cloud"]

    def resolve_provider_api_key(self, provider_config: "ProviderConfig") -> str:
        secret_name = provider_config.api_key_secret_name or ""
        if secret_name:
            return (
                GriptapeNodes.SecretsManager().get_secret(secret_name, should_error_on_not_found=False) or "not-needed"
            )
        return "not-needed"

    def _refresh_providers_button(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        """Refresh the provider dropdown from the engine."""
        provider_names = self._fetch_provider_names()
        current = self._node.get_parameter_value("model_provider") or "griptape_cloud"
        default = current if current in provider_names else (provider_names[0] if provider_names else "griptape_cloud")
        self._node._update_option_choices(param="model_provider", choices=provider_names, default=default)
        return None

    def _refresh_models_button(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        """Refresh the model dropdown for the currently selected provider."""
        provider_name = self._node.get_parameter_value("model_provider") or "griptape_cloud"
        self.update_model_choices_for_provider(provider_name)
        return None

    def update_model_choices_for_provider(self, provider_name: str) -> None:
        if provider_name == "griptape_cloud":
            # Use a curated vision-only subset rather than the full GTC model list.
            models = self._gtc_model_choices
            vision_names = set(self._gtc_model_choices)
            new_data = [entry for entry in self._gtc_model_data if entry["name"] in vision_names]
        else:
            models = self.fetch_models_for_provider(provider_name)
            new_data = [{"name": m, "icon": "", "args": {}} for m in models]
        default = models[0] if models else (self._gtc_model_choices[0] if self._gtc_model_choices else "griptape_cloud")
        self._node._update_option_choices(param="model", choices=models, default=default)
        param = self._node.get_parameter_by_name("model")
        if param:
            param.update_ui_options_key("data", new_data)

    def fetch_models_for_provider(self, provider_name: str) -> list[str]:
        try:
            providers = self._fetch_providers()
            provider_config = next((p for p in providers if p.name == provider_name), None)
            if provider_config is None:
                return self._gtc_model_choices
            result = GriptapeNodes.handle_request(
                ListProviderModelsRequest(
                    provider=provider_config.type,
                    base_url=provider_config.base_url or "",
                    api_key=self.resolve_provider_api_key(provider_config),
                )
            )
            if isinstance(result, ListProviderModelsResultSuccess):
                return cast(ListProviderModelsResultSuccess, result).models or self._gtc_model_choices
        except Exception:
            pass
        return self._gtc_model_choices
