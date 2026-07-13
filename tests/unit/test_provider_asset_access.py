from __future__ import annotations

import httpx
import pytest

import griptape_nodes_library.proxy.provider_asset_access as access_module
from griptape_nodes_library.proxy.provider_asset_access import (
    API_KEY_NAME,
    LICENSE_SECRET_NAME,
    ProviderAssetAccessOutcome,
    check_provider_asset_access,
    resolve_proxy_api_key,
)


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


@pytest.fixture(autouse=True)
def stub_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a key is configured so the probe path is exercised (not the missing-key path)."""
    monkeypatch.setattr(access_module, "resolve_proxy_api_key", lambda *_args, **_kwargs: "test-key")


def _stub_get(monkeypatch: pytest.MonkeyPatch, response: _FakeResponse) -> None:
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

    monkeypatch.setattr(access_module.httpx, "get", _raise)
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False


def test_missing_api_key_is_indeterminate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(access_module, "resolve_proxy_api_key", lambda *_args, **_kwargs: None)
    result = check_provider_asset_access()
    assert result.outcome is ProviderAssetAccessOutcome.INDETERMINATE
    assert result.is_denied is False


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str | None]) -> None:
    """Make GriptapeNodes.SecretsManager().get_secret() read from an in-memory dict."""
    monkeypatch.setattr(
        access_module.GriptapeNodes,
        "SecretsManager",
        lambda: type("S", (), {"get_secret": lambda self, name: secrets.get(name)})(),
    )


def test_resolve_prefers_proxy_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    # GT_CLOUD_PROXY_API_KEY wins over both the License and the API key secret.
    monkeypatch.setenv("GT_CLOUD_PROXY_API_KEY", "env-override")
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_api_key() == "env-override"


def test_resolve_prefers_license_over_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # With no env override, a configured License wins over the API key.
    monkeypatch.delenv("GT_CLOUD_PROXY_API_KEY", raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_api_key() == "the-license"


def test_resolve_falls_back_to_api_key_without_license(monkeypatch: pytest.MonkeyPatch) -> None:
    # License-only is the new path; the API-key-only path must still work unchanged.
    monkeypatch.delenv("GT_CLOUD_PROXY_API_KEY", raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: "the-api-key"})
    assert resolve_proxy_api_key() == "the-api-key"


def test_resolve_uses_license_when_api_key_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    # The reported case: License configured, no GT_CLOUD_API_KEY set.
    monkeypatch.delenv("GT_CLOUD_PROXY_API_KEY", raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: "the-license", API_KEY_NAME: None})
    assert resolve_proxy_api_key() == "the-license"


def test_resolve_returns_none_when_nothing_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GT_CLOUD_PROXY_API_KEY", raising=False)
    _stub_secrets(monkeypatch, {LICENSE_SECRET_NAME: None, API_KEY_NAME: None})
    assert resolve_proxy_api_key() is None
