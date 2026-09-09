from __future__ import annotations

from typing import Any

import httpx
import pytest

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy.hosted_artifacts import (
    ArtifactKind,
    HostedArtifactError,
    artifact_download_headers,
    fetch_hosted_artifacts,
)
from griptape_nodes_library.proxy.provider_asset_access import ProxyCredential

PROXY_BASE = "https://cloud.griptape.ai/api/proxy/v2/"
GENERATION_ID = "gen-abc"

PRESIGNED_URL = "https://account.blob.core.windows.net/generations/gen-abc/artifacts/0.mp4?sig=abc"
STREAMING_URL = "https://cloud.griptape.ai/api/proxy/v2/generations/gen-abc/artifacts/0"


class _FakeJsonResponse:
    def __init__(self, payload: Any, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if httpx.codes.is_error(self.status_code):
            request = httpx.Request("GET", "https://cloud.griptape.ai")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> Any:
        return self._payload


class _FakeBytesResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def _install_list_client(monkeypatch: pytest.MonkeyPatch, payload: Any, calls: list[str]) -> None:
    class FakeAsyncClient:
        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(
            self, url: str, headers: dict[str, str] | None = None, timeout: int | None = None
        ) -> _FakeJsonResponse | _FakeBytesResponse:
            calls.append(url)
            if url.endswith("/artifacts"):
                return _FakeJsonResponse(payload)
            return _FakeBytesResponse(b"media-bytes")

    monkeypatch.setattr("griptape_nodes_library.proxy.hosted_artifacts.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)


def _build_node(monkeypatch: pytest.MonkeyPatch) -> tuple[Flux2ImageGeneration, list[dict[str, Any]]]:
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.resolve_proxy_credential",
        lambda _name: ProxyCredential(value="test-key", source="GT_CLOUD_API_KEY"),
    )

    node = Flux2ImageGeneration(name="Flux2")
    node._proxy_base = PROXY_BASE

    status_calls: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: status_calls.append(kwargs)  # type: ignore[method-assign]
    return node, status_calls


def _install_output_file(node: Flux2ImageGeneration) -> None:
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


def test_presigned_url_is_fetched_without_a_bearer_token() -> None:
    # A presigned URL carries its own credentials and refuses a request that also
    # sends an Authorization header.
    assert artifact_download_headers(PRESIGNED_URL, "test-key") == {}


def test_streaming_route_is_fetched_with_a_bearer_token() -> None:
    assert artifact_download_headers(STREAMING_URL, "test-key") == {"Authorization": "Bearer test-key"}


@pytest.mark.asyncio
async def test_fetch_orders_by_index_and_skips_unfetchable_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "artifacts": [
            {"index": 1, "kind": "image", "url": "https://example/1.webp", "content_type": "image/webp"},
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb", "size_bytes": 12},
            {"index": 2, "kind": "image"},
        ]
    }
    _install_list_client(monkeypatch, payload, [])

    artifacts = await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")

    assert [(artifact.index, artifact.kind) for artifact in artifacts] == [(0, "model_3d"), (1, "image")]
    assert artifacts[0].size_bytes == 12
    assert artifacts[1].content_type == "image/webp"


@pytest.mark.asyncio
async def test_unexpected_payload_shape_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_list_client(monkeypatch, {"unexpected": True}, [])

    with pytest.raises(HostedArtifactError):
        await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")


@pytest.mark.asyncio
async def test_list_is_read_once_per_generation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    payload = {
        "artifacts": [
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb"},
            {"index": 1, "kind": "image", "url": "https://example/1.webp"},
        ]
    }
    _install_list_client(monkeypatch, payload, calls)
    node, _status_calls = _build_node(monkeypatch)

    mesh = await node._hosted_artifact(GENERATION_ID, kind=ArtifactKind.MODEL_3D)
    preview = await node._hosted_artifact(GENERATION_ID, kind=ArtifactKind.IMAGE)

    assert mesh.index == 0
    assert preview.index == 1
    assert calls == [f"{PROXY_BASE}generations/{GENERATION_ID}/artifacts"]


@pytest.mark.asyncio
async def test_position_selects_within_a_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "artifacts": [
            {"index": 0, "kind": "image", "url": "https://example/0.png"},
            {"index": 1, "kind": "image", "url": "https://example/1.png"},
        ]
    }
    _install_list_client(monkeypatch, payload, [])
    node, _status_calls = _build_node(monkeypatch)

    second = await node._hosted_artifact(GENERATION_ID, kind=ArtifactKind.IMAGE, position=1)

    assert second.index == 1


@pytest.mark.asyncio
async def test_missing_kind_reports_what_is_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"artifacts": [{"index": 0, "kind": "image", "url": "https://example/0.png"}]}
    _install_list_client(monkeypatch, payload, [])
    node, _status_calls = _build_node(monkeypatch)

    with pytest.raises(HostedArtifactError, match="0:image"):
        await node._hosted_artifact(GENERATION_ID, kind=ArtifactKind.VIDEO)


@pytest.mark.asyncio
async def test_save_writes_the_artifact_and_reports_success(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"artifacts": [{"index": 0, "kind": "image", "url": PRESIGNED_URL}]}
    _install_list_client(monkeypatch, payload, [])
    node, status_calls = _build_node(monkeypatch)
    _install_output_file(node)

    saved = await node._save_generated_media(
        GENERATION_ID,
        "image_url",
        lambda v, n: {"value": v, "name": n},
        kind=ArtifactKind.IMAGE,
        media_kind="image",
    )

    assert saved is True
    assert node.parameter_output_values["image_url"] == {"value": "project/files/output.png", "name": "output.png"}
    assert status_calls[0]["was_successful"] is True
    assert "output.png" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_save_reports_failure_when_nothing_is_hosted(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_list_client(monkeypatch, {"artifacts": []}, [])
    node, status_calls = _build_node(monkeypatch)
    _install_output_file(node)

    saved = await node._save_generated_media(
        GENERATION_ID,
        "image_url",
        lambda v, n: {"value": v, "name": n},
        kind=ArtifactKind.IMAGE,
        media_kind="image",
    )

    # A generation billed upstream whose media cannot be retrieved is a failure.
    assert saved is False
    assert node.parameter_output_values["image_url"] is None
    assert status_calls[0]["was_successful"] is False
    assert "could not be retrieved" in status_calls[0]["result_details"]
