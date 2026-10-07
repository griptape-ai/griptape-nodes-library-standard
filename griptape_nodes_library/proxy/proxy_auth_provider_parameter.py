from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button

from griptape_nodes_library.proxy.proxy_api_key_providers import ProxyApiKeyProviderConfig
from griptape_nodes_library.utils.node_error_utils import missing_secret_error, missing_secret_message

__all__ = ["ProxyAuthProviderParameter"]


class ProxyAuthProviderParameter:
    def __init__(
        self,
        *,
        node: Any,
        provider_config: ProxyApiKeyProviderConfig,
        parameter_name: str = "api_key_provider",
        on_label: str = "Customer",
        off_label: str = "Griptape",
    ) -> None:
        self._node = node
        self._provider_config = provider_config
        self.parameter_name = parameter_name
        self.on_label = on_label
        self.off_label = off_label
        self.message_name = f"{parameter_name}_message"

    @property
    def secret_name(self) -> str:
        return self._provider_config.api_key_name

    @property
    def provider_id(self) -> str:
        return self._provider_config.provider_id

    def add_parameters(self) -> None:
        self._node.add_parameter(
            ParameterBool(
                name=self.parameter_name,
                default_value=False,
                tooltip=f"Use your own {self._provider_config.secret_label.lower()} instead of the default one",
                allow_input=False,
                allow_output=False,
                on_label=self.on_label,
                off_label=self.off_label,
                ui_options={"display_name": self._provider_config.parameter_display_name},
                traits={
                    Button(
                        icon="key",
                        tooltip="Open secrets settings",
                        button_link=f"#settings-secrets?filter={self.secret_name}",
                    )
                },
            )
        )
        self._node.add_node_element(
            ParameterMessage(
                name=self.message_name,
                variant="info",
                title=f"{self._provider_config.provider_name} {self._provider_config.secret_label}",
                value=(
                    f"To use your own {self._provider_config.provider_name} "
                    f"{self._provider_config.secret_label}, visit:\n{self._provider_config.api_key_url}\n"
                    f"to obtain a valid credential.\n\n"
                    f"Then set {self.secret_name} in "
                    f"[Settings → Secrets](#settings-secrets?filter={self.secret_name})."
                ),
                button_link=f"#settings-secrets?filter={self.secret_name}",
                button_text="Open Secrets",
                button_icon="key",
                markdown=True,
                hide=True,
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name != self.parameter_name:
            return

        if value:
            if not self._check_secret_set(self.secret_name):
                self._node.show_message_by_name(self.message_name)
        else:
            self._node.hide_message_by_name(self.message_name)

    def is_user_auth_enabled(self) -> bool:
        return bool(self._node.get_parameter_value(self.parameter_name) or False)

    def is_user_api_enabled(self) -> bool:
        return self.is_user_auth_enabled()

    def get_user_auth_info(self) -> str | None:
        if not self.is_user_auth_enabled():
            return None
        return self._get_secret(self.secret_name)

    def _check_secret_set(self, secret_name: str) -> bool:
        secret_value = GriptapeNodes.SecretsManager().get_secret(secret_name)
        if secret_value is None:
            return False
        if isinstance(secret_value, str):
            return bool(secret_value.strip())
        return bool(secret_value)

    def _get_secret(self, secret_name: str) -> str:
        secret_value = GriptapeNodes.SecretsManager().get_secret(secret_name)
        if not secret_value:
            config = self._provider_config
            msg = (
                f"{missing_secret_message(secret_name)} Or switch "
                f"{config.parameter_display_name} to {self.off_label} to use Griptape's key."
            )
            raise missing_secret_error(
                secret_name,
                message=msg,
                key_url=config.api_key_url,
                key_url_label=f"Get a {config.provider_name} {config.secret_label}",
            )
        return secret_value
