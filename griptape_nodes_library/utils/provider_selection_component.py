from typing import cast

from griptape.drivers.prompt.base_prompt_driver import BasePromptDriver
from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.retained_mode.events.agent_events import (
    ListAgentProvidersRequest,
    ListAgentProvidersResultSuccess,
    ListProviderModelsRequest,
    ListProviderModelsResultSuccess,
    ProviderConfig,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

_GRIPTAPE_CLOUD_PROVIDER = ProviderConfig(name="griptape_cloud", type="griptape_cloud", model="")


class ProviderSelectionComponent:
    """Swaps the model dropdown's choices between Griptape Cloud and a third-party provider.

    The Griptape Cloud branch offers ``model_access``'s own choices -- the
    license-gated Griptape Cloud model ids ``ModelAccessComponent`` decorates.
    A third-party provider's models are not Griptape Cloud models: that branch
    offers whatever the provider itself reports (``fetch_models_for_provider``)
    and is exempt from license gating. The two vocabularies are never
    interchangeable -- a Griptape Cloud id must never be offered for, or reach a
    driver through, the third-party branch, and vice versa.
    """

    def __init__(
        self,
        node,
        *,
        model_provider_param: Parameter,
        model_access: ModelAccessComponent,
        default_model: str | None = None,
    ):
        """Attach provider-selection UI to an already-added ``model_provider_param``.

        ``model_provider_param`` must already be attached to ``node`` (via
        ``node.add_parameter``) **before** this component is constructed, and it
        must not already carry ``Options`` or ``Button`` traits — this component
        installs them, mirroring the contract of ``ModelAccessComponent``.
        """
        self._node = node
        self._model_access = model_access
        model_choices = model_access.model_choices
        self._default_model = default_model or (model_choices[0] if model_choices else "")

        if self._node.get_parameter_by_name(model_provider_param.name) is not model_provider_param:
            msg = (
                f"ProviderSelectionComponent: parameter '{model_provider_param.name}' is not attached to "
                f"node '{self._node.name}'. Call node.add_parameter(model_provider_param) BEFORE "
                "constructing the component."
            )
            raise ValueError(msg)

        provider_names = self._fetch_provider_names()

        default = provider_names[0] if provider_names else "griptape_cloud"
        model_provider_param.set_default_value(default)
        model_provider_param.add_trait(Options(choices=provider_names))
        model_provider_param.add_trait(
            Button(
                icon="list-restart",
                size="icon",
                variant="secondary",
                on_click=self._refresh_providers_button,
            )
        )

    def on_provider_changed(self, provider_name: str) -> None:
        failure = self.update_model_choices_for_provider(provider_name)
        if failure is not None:
            logger.error("%s: %s", self._node.name, failure)

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
        failure = self.update_model_choices_for_provider(provider_name)
        if failure is not None:
            return NodeMessageResult(success=False, details=failure, response=button_details)
        return None

    def update_model_choices_for_provider(self, provider_name: str) -> str | None:
        """Point the model dropdown at ``provider_name``'s models.

        Returns ``None`` once the dropdown offers that provider's models, or a
        message naming why the dropdown was left on the previous provider's.
        Callers surface it: the refresh button as a failed ``NodeMessageResult``,
        a provider change as a logged error.
        """
        if provider_name == "griptape_cloud":
            # The component's own choices ARE Griptape Cloud model ids; offer them as-is.
            default = self._model_access.pick_permitted_default() or self._default_model
            self._node._update_option_choices(param="model", choices=self._model_access.model_choices, default=default)
            # Restore the component's per-row license decoration and badge; the
            # _update_option_choices call above only refreshed choices and value.
            self._model_access.reinstall_options()
            return None
        # A third-party provider's models are not Griptape Cloud models -- offer whatever
        # the provider itself reports, never the Griptape Cloud choices (meaningless and
        # ungate-able for this provider's own driver).
        models = self.fetch_models_for_provider(provider_name)
        if not models:
            # Emptying the dropdown wedges it: `_update_option_choices` assigns
            # `choices=[]` before rejecting the empty default it was handed, and an empty
            # choice list survives save/reload through the serialized `simple_dropdown`,
            # so every later assignment indexes `choices[0]` on an empty list. Leave the
            # dropdown on the previous provider's models and report the failure instead.
            return (
                f"Provider '{provider_name}' reported no models, so the model dropdown still offers the "
                "previous provider's. Check that the provider is reachable and its API key is set, then "
                "refresh the model list."
            )
        self._node._update_option_choices(param="model", choices=models, default=models[0])
        param = self._node.get_parameter_by_name("model")
        if param:
            param.update_ui_options_key("data", [{"name": m, "icon": "", "args": {}} for m in models])
        return None

    def fetch_models_for_provider(self, provider_name: str) -> list[str]:
        """Fetch a third-party provider's own model list.

        Returns an empty list for an unconfigured or unreachable provider, naming
        the reason in the log -- the Griptape Cloud choices are not this provider's
        model names, so they are never a valid substitute here: offering one would
        risk it reaching this provider's driver unresolved. Callers must treat the
        empty list as a failure rather than as this provider's model list.
        """
        try:
            providers = self._fetch_providers()
            provider_config = next((p for p in providers if p.name == provider_name), None)
            if provider_config is None:
                logger.warning("Provider '%s' is not among the configured providers.", provider_name)
                return []
            result = GriptapeNodes.handle_request(
                ListProviderModelsRequest(
                    provider=provider_config.type,
                    base_url=provider_config.base_url or "",
                    api_key=self.resolve_provider_api_key(provider_config),
                )
            )
            if isinstance(result, ListProviderModelsResultSuccess):
                return cast(ListProviderModelsResultSuccess, result).models or []
            logger.warning("Provider '%s' did not return a model list (%s).", provider_name, type(result).__name__)
        except Exception as error:
            logger.warning("Listing models for provider '%s' failed: %s", provider_name, error)
        return []
