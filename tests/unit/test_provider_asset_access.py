from __future__ import annotations

import httpx
import pytest

import griptape_nodes_library.proxy.provider_asset_access as access_module
from griptape_nodes_library.proxy.provider_asset_access import (
    ProviderAssetAccessOutcome,
    check_provider_asset_access,
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
