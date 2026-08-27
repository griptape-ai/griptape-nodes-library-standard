from __future__ import annotations

import httpx
import pytest

import griptape_nodes_library.proxy.provider_asset_access as access_module
from griptape_nodes_library.proxy.provider_asset_access import (
    API_KEY_NAME,
    LICENSE_SECRET_NAME,
    PROXY_API_KEY_ENV_VAR,
    ProviderAssetAccessOutcome,
    ProxyCredential,
    check_provider_asset_access,
    missing_proxy_credential_message,
    resolve_proxy_credential,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def _stub_credential(monkeypatch: pytest.MonkeyPatch, credential: ProxyCredential | None = None) -> None:
    """Pretend a credential is configured so the probe path is exercised, not the missing-key one."""
    resolved = credential if credential is not None else ProxyCredential(value="test-key", source=API_KEY_NAME)
    monkeypatch.setattr(access_module, "resolve_proxy_credential", lambda *_args, **_kwargs: resolved)


def _stub_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
    _stub_credential(monkeypatch)
    monkeypatch.setattr(access_module.httpx, "get", lambda *_args, **_kwargs: response)


def test_access_granted_on_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_get(monkeypatch, _FakeResponse(200))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.GRANTED
    assert result.has_access is True


def test_access_granted_on_404_with_not_found_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    # The entitlement gate runs before the asset handler, so a 404 "provider asset not found"
    # proves the caller passed the gate.
    _stub_get(monkeypatch, _FakeResponse(404, '{"error":"provider asset not found."}'))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.GRANTED


def test_404_without_marker_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_get(monkeypatch, _FakeResponse(404, '{"error":"not found"}'))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE


@pytest.mark.parametrize(
    "body",
    [
        '{"error":"Your organization is not entitled to use this feature."}',
        '{"error":"This license is not permitted to perform this action."}',
    ],
)
def test_403_is_denied(monkeypatch: pytest.MonkeyPatch, body: str) -> None:
    _stub_get(monkeypatch, _FakeResponse(403, body))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.DENIED
    assert result.is_denied is True
    assert "request access" in result.detail.lower()


def test_401_is_indeterminate_not_denied(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_get(monkeypatch, _FakeResponse(401, "token_not_valid"))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False


@pytest.mark.parametrize("status", [500, 502, 503])
def test_server_error_is_indeterminate_not_denied(monkeypatch: pytest.MonkeyPatch, status: int) -> None:
    # A server error must NOT be reported as "no access" — it should surface the real failure.
    _stub_get(monkeypatch, _FakeResponse(status, "internal server error"))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False
    assert str(status) in result.detail


def test_network_error_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*_args, **_kwargs):
        raise httpx.ConnectError("connection refused")

    _stub_credential(monkeypatch)
    monkeypatch.setattr(access_module.httpx, "get", _raise)
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False


def test_missing_api_key_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_credential(monkeypatch, ProxyCredential(value=None))
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False
    # Names both credentials the proxy accepts, not just the API key.
    assert "license" in result.detail.lower()
    assert API_KEY_NAME in result.detail


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str | None]) -> None:
    """Make GriptapeNodes.SecretsManager().get_secret() read from an in-memory dict.

    Mirrors the real manager's contract: an absent secret is None, a secret registered with a
    blank value is that blank string.
    """
    monkeypatch.setattr(
        access_module.GriptapeNodes,
        "SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, name, **_kwargs: secrets.get(name)})(),
    )


def test_resolve_prefers_proxy_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy env override wins over both the License and the API key secret.
    monkeypatch.setenv(PROXY_API_KEY_ENV_VAR, "env-override")
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_credential().value == "env-override"


def test_resolve_prefers_license_over_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no env override, a configured License wins over the API key.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_credential().value == "the-license"


def test_resolve_falls_back_to_api_key_without_license(monkeypatch: pytest.MonkeyPatch) -> None:
    # License-only is the new path; the API-key-only path must still work unchanged.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_credential().value == "the-api-key"


def test_resolve_uses_license_when_api_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reported case: License configured, no GT_CLOUD_API_KEY set.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: None})
    assert resolve_proxy_credential().value == "the-license"


def test_resolve_returns_none_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: None})
    assert resolve_proxy_credential().value is None


def test_resolve_skips_blank_api_key_and_records_it(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reported case: GT_CLOUD_API_KEY registered as "" reads as configured everywhere a user
    # would look, and resolves to nothing.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: ""})
    credential = resolve_proxy_credential()
    assert credential.value is None
    assert credential.blank_sources == (API_KEY_NAME,)


def test_resolve_treats_whitespace_only_secret_as_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "   ", API_KEY_NAME: "\n"})
    credential = resolve_proxy_credential()
    assert credential.value is None
    assert credential.blank_sources == (LICENSE_SECRET_NAME, API_KEY_NAME)


def test_resolve_falls_through_blank_license_to_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # A blank License must not shadow a usable API key, and must still be reported.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "", API_KEY_NAME: "the-api-key"})
    credential = resolve_proxy_credential()
    assert credential.value == "the-api-key"
    assert credential.source == API_KEY_NAME
    assert credential.blank_sources == (LICENSE_SECRET_NAME,)


def test_resolve_does_not_report_a_blank_proxy_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # The override is a debug knob, so a user reading the message must not be sent after it.
    monkeypatch.setenv(PROXY_API_KEY_ENV_VAR, "  ")
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: ""})
    credential = resolve_proxy_credential()
    assert credential.value is None
    assert credential.blank_sources == (API_KEY_NAME,)


def test_resolve_falls_through_blank_proxy_override_to_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROXY_API_KEY_ENV_VAR, "")
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: None})
    credential = resolve_proxy_credential()
    assert credential.value == "the-license"
    assert credential.blank_sources == ()


def test_resolve_strips_surrounding_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: "  the-api-key\n"})
    assert resolve_proxy_credential().value == "the-api-key"


def test_resolve_survives_unavailable_secrets_manager(monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolution runs in contexts where the SecretsManager may not be reachable; the caller needs
    # a credential message, not an unrelated exception.
    monkeypatch.delenv(PROXY_API_KEY_ENV_VAR, raising=False)

    def _raise() -> None:
        raise RuntimeError("no engine here")

    monkeypatch.setattr(access_module.GriptapeNodes, "SecretsManager", _raise)
    credential = resolve_proxy_credential()
    assert credential.value is None
    assert credential.blank_sources == ()


def test_missing_message_names_both_credentials() -> None:
    message = missing_proxy_credential_message(ProxyCredential(value=None), attempted="run My Node")
    assert "Attempted to run My Node." in message
    # A License-only user must not be sent after an API key alone.
    assert "license" in message.lower()
    assert API_KEY_NAME in message
    assert "Griptape Cloud" in message
    assert "blank" not in message


def test_missing_message_calls_out_a_blank_credential() -> None:
    message = missing_proxy_credential_message(
        ProxyCredential(value=None, blank_sources=(API_KEY_NAME,)), attempted="run My Node"
    )
    assert f"{API_KEY_NAME} is set to a blank value" in message


def test_missing_message_calls_out_several_blank_credentials() -> None:
    message = missing_proxy_credential_message(
        ProxyCredential(value=None, blank_sources=(LICENSE_SECRET_NAME, API_KEY_NAME)), attempted="run My Node"
    )
    assert f"{LICENSE_SECRET_NAME}, {API_KEY_NAME} are set to blank values" in message
