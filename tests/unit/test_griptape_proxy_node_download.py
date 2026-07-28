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

        async def get(self, url: str, timeout: int) -> _FakeResponse:
            return await get_impl(url, timeout)

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    async def noop_sleep(_: float) -> None:
        pass

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)


@pytest.mark.asyncio
async def test_download_retries_once_on_transient_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def get_impl(_url: str, _timeout: int) -> _FakeResponse:
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

    async def get_impl(_url: str, _timeout: int) -> _FakeResponse:
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

    async def get_impl(_url: str, _timeout: int) -> _FakeResponse:
        nonlocal calls
        calls += 1
        return _FakeResponse(status_code=503)

    _install_fake_client(monkeypatch, get_impl)

    with pytest.raises(httpx.HTTPStatusError):
        await Flux2ImageGeneration._download_bytes_from_url("https://provider.example/asset")

    # 5xx is transient; one retry (2 attempts total) before giving up.
    assert calls == 2


@pytest.mark.asyncio
async def test_download_and_save_failure_reports_unsuccessful_and_surfaces_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def get_impl(_url: str, _timeout: int) -> _FakeResponse:
        return _FakeResponse(status_code=403)

    _install_fake_client(monkeypatch, get_impl)

    node = Flux2ImageGeneration(name="Flux2")

    status_calls: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: status_calls.append(kwargs)  # type: ignore[method-assign]

    url = "https://provider.example/asset"
    await node._download_and_save(url, "image_url", lambda v, n: {"value": v, "name": n}, media_kind="image")

    assert node.parameter_output_values["image_url"] is None
    assert len(status_calls) == 1
    assert status_calls[0]["was_successful"] is False
    assert url in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_download_and_save_success_saves_and_reports_successful(monkeypatch: pytest.MonkeyPatch) -> None:
    async def get_impl(_url: str, _timeout: int) -> _FakeResponse:
        return _FakeResponse(content=b"image-bytes")

    _install_fake_client(monkeypatch, get_impl)

    node = Flux2ImageGeneration(name="Flux2")

    class _SavedFile:
        location = "project/files/output.png"
        name = "output.png"

    class _Dest:
        async def awrite_bytes(self, _data: bytes) -> _SavedFile:
            return _SavedFile()

    class _OutputFile:
        def build_file(self, **_extra: Any) -> _Dest:
            return _Dest()

    node._output_file = _OutputFile()  # type: ignore[assignment]

    status_calls: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: status_calls.append(kwargs)  # type: ignore[method-assign]

    await node._download_and_save(
        "https://provider.example/asset",
        "image_url",
        lambda v, n: {"value": v, "name": n},
        media_kind="image",
    )

    assert node.parameter_output_values["image_url"] == {"value": "project/files/output.png", "name": "output.png"}
    assert len(status_calls) == 1
    assert status_calls[0]["was_successful"] is True
    assert "output.png" in status_calls[0]["result_details"]
