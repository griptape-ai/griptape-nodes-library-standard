"""Griptape Cloud credential resolution: License or API key, License first."""

from __future__ import annotations

import pytest

import griptape_nodes_library.utils.cloud_credential_utils as cloud_credential_utils
from griptape_nodes_library.utils.cloud_credential_utils import (
    missing_credential_message,
    resolve_cloud_api_key,
)

_LICENSE = "header.payload.signature"
"""A Griptape Nodes License is a JWT: three dot-separated segments."""


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]) -> None:
    class _FakeSecrets:
        def get_secret(self, name: str, **_kwargs: object) -> str | None:
            return secrets.get(name)

    monkeypatch.setattr(cloud_credential_utils.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


def test_resolves_api_key_when_only_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_API_KEY": "gt-the-api-key"})

    assert resolve_cloud_api_key() == "gt-the-api-key"


def test_resolves_license_when_no_api_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    # The license-only user this whole module exists for.
    _stub_secrets(monkeypatch, {"GRIPTAPE_NODES_LICENSE": _LICENSE})

    assert resolve_cloud_api_key() == _LICENSE


def test_license_wins_when_both_are_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {"GRIPTAPE_NODES_LICENSE": _LICENSE, "GT_CLOUD_API_KEY": "gt-the-api-key"})

    assert resolve_cloud_api_key() == _LICENSE


def test_returns_empty_string_when_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    # "" not None: the value feeds driver `api_key` fields typed `str`.
    _stub_secrets(monkeypatch, {})

    assert resolve_cloud_api_key() == ""


def test_missing_credential_message_names_both_credentials() -> None:
    message = missing_credential_message("run the Agent")

    assert message.startswith("Attempted to run the Agent. Failed because ")
    # A license-only user must not be sent after an API key alone.
    assert "license" in message.lower()
    assert "GT_CLOUD_API_KEY" in message
