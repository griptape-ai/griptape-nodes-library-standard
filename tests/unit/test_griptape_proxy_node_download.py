from __future__ import annotations

from typing import Any

import httpx
import pytest

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration


class _FakeResponse:
    def __init__(self, *, status_code: int = 200, content: bytes = b"data") -> None:
        self.status_code = status_code
        self.content = content

    def raise_for_status(self) -> None:
        if httpx.codes.is_error(self.status_code):
            request = httpx.Request("GET", "https://provider.example/asset")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, get_impl: Any) -> None:
    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, timeout: int, headers: dict[str, str] | None = None) -> _FakeResponse:
            return await get_impl(url, timeout, headers)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)


@pytest.mark.asyncio
async def test_download_retries_once_on_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def get_impl(_url: str, _timeout: int, _headers: dict[str, str] | None) -> _FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectError("boom")
        return _FakeResponse(content=b"video-bytes")

    _install_fake_client(monkeypatch, get_impl)

    result = await Flux2ImageGeneration._download_bytes_from_url("https://provider.example/asset")

    assert result == b"video-bytes"
    assert calls == 2


@pytest.mark.asyncio
async def test_download_does_not_retry_on_client_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def get_impl(_url: str, _timeout: int, _headers: dict[str, str] | None) -> _FakeResponse:
        nonlocal calls
        calls += 1
        return _FakeResponse(status_code=403)

    _install_fake_client(monkeypatch, get_impl)

    with pytest.raises(httpx.HTTPStatusError):
        await Flux2ImageGeneration._download_bytes_from_url("https://provider.example/asset")

    # 4xx is permanent; must fail fast without a retry.
    assert calls == 1


@pytest.mark.asyncio
async def test_download_retries_once_on_server_error_then_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def get_impl(_url: str, _timeout: int, _headers: dict[str, str] | None) -> _FakeResponse:
        nonlocal calls
        calls += 1
        return _FakeResponse(status_code=503)

    _install_fake_client(monkeypatch, get_impl)

    with pytest.raises(httpx.HTTPStatusError):
        await Flux2ImageGeneration._download_bytes_from_url("https://provider.example/asset")

    # 5xx is transient; one retry (2 attempts total) before giving up.
    assert calls == 2
