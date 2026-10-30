"""FileManager tool: bucket fetch is lazy and honors the configured Cloud deployment."""

from __future__ import annotations

from typing import Any

import httpx
import pytest

import griptape_nodes_library.tools.file_manager_tool as file_manager_module
from griptape_nodes_library.tools.file_manager_tool import LOCATIONS, FileManager, buckets_url


def _stub_secrets(monkeypatch: pytest.MonkeyPatch, secrets: dict[str, str]) -> None:
    class _FakeSecrets:
        def get_secret(self, name: str, **_kwargs: object) -> str | None:
            return secrets.get(name)

    monkeypatch.setattr(file_manager_module.GriptapeNodes, "SecretsManager", lambda: _FakeSecrets())


def _record_requests(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    urls: list[str] = []

    def _get(url: str, **_kwargs: Any) -> httpx.Response:
        urls.append(url)
        return httpx.Response(
            200,
            json={"buckets": [{"name": "assets", "bucket_id": "bucket-123"}]},
            request=httpx.Request("GET", url),
        )

    monkeypatch.setattr(file_manager_module.httpx, "get", _get)
    return urls


def test_create_sends_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _record_requests(monkeypatch)

    FileManager(name="file_manager")

    assert urls == []


def test_create_succeeds_when_bucket_fetch_would_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: Any, **_kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("unreachable")

    monkeypatch.setattr(file_manager_module.httpx, "get", _fail)

    FileManager(name="file_manager")


def test_workspace_location_sends_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    urls = _record_requests(monkeypatch)
    node = FileManager(name="file_manager")
    node.parameter_values["file_location"] = LOCATIONS[0]

    node.process()

    assert urls == []
    assert node.parameter_output_values["tool"]["file_location"] == LOCATIONS[0]


def test_cloud_location_fetches_buckets_from_configured_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_BASE_URL": "https://cloud.example.test"})
    urls = _record_requests(monkeypatch)
    node = FileManager(name="file_manager")
    node.parameter_values["file_location"] = LOCATIONS[1]
    node.parameter_values["bucket_id"] = "assets"

    node.process()

    assert urls == ["https://cloud.example.test/api/buckets"]
    assert node.parameter_output_values["tool"]["bucket_id"] == "bucket-123"


def test_buckets_url_defaults_to_griptape_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {})

    assert buckets_url() == "https://cloud.griptape.ai/api/buckets"


def test_buckets_url_keeps_base_path(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_BASE_URL": "https://example.test/cloud"})

    assert buckets_url() == "https://example.test/cloud/api/buckets"


@pytest.mark.parametrize("base_url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"])
def test_buckets_url_allows_http_on_loopback(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_BASE_URL": base_url})

    assert buckets_url() == f"{base_url}/api/buckets"


@pytest.mark.parametrize("base_url", ["http://cloud.example.test", "ftp://cloud.example.test", "cloud.example.test"])
def test_buckets_url_rejects_non_https(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    _stub_secrets(monkeypatch, {"GT_CLOUD_BASE_URL": base_url})

    with pytest.raises(ValueError, match="must be an https URL"):
        buckets_url()
