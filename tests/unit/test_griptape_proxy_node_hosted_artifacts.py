from __future__ import annotations

import logging
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
    assert artifact_download_headers(PRESIGNED_URL, "test-key", PROXY_BASE) == {}


def test_streaming_route_is_fetched_with_a_bearer_token() -> None:
    assert artifact_download_headers(STREAMING_URL, "test-key", PROXY_BASE) == {"Authorization": "Bearer test-key"}


def test_streaming_route_shape_on_a_foreign_host_is_fetched_without_a_bearer_token() -> None:
    # The bearer token authenticates to the proxy; it must never be sent to another
    # host, even one whose path happens to match the proxy's streaming-route shape.
    foreign_url = "https://evil.example.com/api/proxy/v2/generations/gen-abc/artifacts/0"
    assert artifact_download_headers(foreign_url, "test-key", PROXY_BASE) == {}


@pytest.mark.asyncio
async def test_fetch_orders_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "artifacts": [
            {"index": 1, "kind": "image", "url": "https://example/1.webp", "content_type": "image/webp"},
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb", "size_bytes": 12},
        ]
    }
    _install_list_client(monkeypatch, payload, [])

    artifacts = await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")

    assert [(artifact.index, artifact.kind) for artifact in artifacts] == [(0, "model_3d"), (1, "image")]
    assert artifacts[0].size_bytes == 12
    assert artifacts[1].content_type == "image/webp"


@pytest.mark.asyncio
async def test_fetch_succeeds_on_a_genuine_tail_truncation(monkeypatch: pytest.MonkeyPatch) -> None:
    # The proxy hosting a prefix of what a provider produced (indices 0, 1 of an
    # expected 3) is a trustworthy short list: nothing was dropped, and the surviving
    # indices are consecutive.
    payload = {
        "artifacts": [
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb"},
            {"index": 1, "kind": "image", "url": "https://example/1.webp"},
        ]
    }
    _install_list_client(monkeypatch, payload, [])

    artifacts = await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")

    assert [artifact.index for artifact in artifacts] == [0, 1]


@pytest.mark.asyncio
async def test_fetch_raises_when_an_entry_is_dropped(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A dropped interior entry can look exactly like a legitimate tail truncation
    # once the malformed entry held the highest index, so any drop is untrustworthy.
    payload = {
        "artifacts": [
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb"},
            {"index": 2, "kind": "image"},
        ]
    }
    _install_list_client(monkeypatch, payload, [])

    with caplog.at_level(logging.WARNING, logger="griptape_nodes"), pytest.raises(HostedArtifactError, match="dropped"):
        await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")

    assert any("dropping it" in record.message for record in caplog.records)


@pytest.mark.asyncio
async def test_fetch_raises_on_a_gap_with_no_dropped_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both entries are individually well-formed, but index 1 is simply absent: a
    # gap the proxy itself reported, not a client-side drop.
    payload = {
        "artifacts": [
            {"index": 0, "kind": "model_3d", "url": "https://example/0.glb"},
            {"index": 2, "kind": "image", "url": "https://example/2.webp"},
        ]
    }
    _install_list_client(monkeypatch, payload, [])

    with pytest.raises(HostedArtifactError, match=r"\[0, 2\]"):
        await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")


@pytest.mark.asyncio
async def test_fetch_bounds_a_dropped_entrys_size_in_the_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # A malformed entry from a misbehaving proxy could be arbitrarily large; the log
    # line must stay bounded rather than dumping it whole.
    oversized_note = "x" * 10_000
    payload = {"artifacts": [{"index": 0, "kind": "image", "note": oversized_note}]}
    _install_list_client(monkeypatch, payload, [])

    with caplog.at_level(logging.WARNING, logger="griptape_nodes"), pytest.raises(HostedArtifactError):
        await fetch_hosted_artifacts(PROXY_BASE, GENERATION_ID, "test-key")

    dropped_lines = [record.message for record in caplog.records if "dropping it" in record.message]
    assert dropped_lines
    assert all(len(line) < len(oversized_note) for line in dropped_lines)


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
async def test_negative_position_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"artifacts": [{"index": 0, "kind": "image", "url": "https://example/0.png"}]}
    _install_list_client(monkeypatch, payload, [])
    node, _status_calls = _build_node(monkeypatch)

    with pytest.raises(HostedArtifactError, match="nonnegative"):
        await node._hosted_artifact(GENERATION_ID, kind=ArtifactKind.IMAGE, position=-1)


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


@pytest.mark.asyncio
async def test_process_generation_clears_a_stale_artifact_list(monkeypatch: pytest.MonkeyPatch) -> None:
    # A presigned artifact URL is short-lived, so a run must not reuse the list an
    # earlier run or Refresh saw.
    node, _status_calls = _build_node(monkeypatch)
    node._hosted_artifact_lists[GENERATION_ID] = ["stale"]  # type: ignore[list-item]

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("no key")

    monkeypatch.setattr(node, "_validate_api_key", _raise, raising=False)

    with pytest.raises(ValueError, match="no key"):
        await node._process_generation()

    assert node._hosted_artifact_lists == {}


@pytest.mark.asyncio
async def test_refresh_async_clears_a_stale_artifact_list(monkeypatch: pytest.MonkeyPatch) -> None:
    node, status_calls = _build_node(monkeypatch)
    node.parameter_output_values["generation_id"] = GENERATION_ID
    node._hosted_artifact_lists[GENERATION_ID] = ["stale"]  # type: ignore[list-item]

    def _raise(*_args: Any, **_kwargs: Any) -> str:
        raise ValueError("no key")

    monkeypatch.setattr(node, "_validate_api_key", _raise, raising=False)

    await node._refresh_async()

    assert node._hosted_artifact_lists == {}
    assert status_calls[0]["was_successful"] is False


def _record_status_and_execution(node: Flux2ImageGeneration) -> list[dict[str, Any]]:
    # Unlike _build_node's recorder, this also updates _execution_succeeded, matching
    # the real _set_status_results, so _refresh_completed's tri-state check can be
    # exercised against how the fake reports status.
    status_calls: list[dict[str, Any]] = []

    def _set_status_results(*, was_successful: bool, result_details: str) -> None:
        status_calls.append({"was_successful": was_successful, "result_details": result_details})
        node._execution_succeeded = was_successful

    node._set_status_results = _set_status_results  # type: ignore[method-assign]
    return status_calls


@pytest.mark.asyncio
async def test_refresh_completed_keeps_a_reported_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    # A Refresh must not paper over a _parse_result failure (e.g. media it could not
    # retrieve) with the unconditional success _refresh_completed used to report.
    node, _ = _build_node(monkeypatch)
    status_calls = _record_status_and_execution(node)

    async def _fetch_generation_result(_generation_id: str) -> dict[str, Any]:
        return {"status": "COMPLETED"}

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        node._set_status_results(was_successful=False, result_details="media could not be retrieved")

    monkeypatch.setattr(node, "_fetch_generation_result", _fetch_generation_result, raising=False)
    monkeypatch.setattr(node, "_parse_result", _parse_result, raising=False)

    await node._refresh_completed(GENERATION_ID)

    assert len(status_calls) == 1
    assert status_calls[0]["was_successful"] is False


@pytest.mark.asyncio
async def test_refresh_completed_reports_a_reported_success(monkeypatch: pytest.MonkeyPatch) -> None:
    node, _ = _build_node(monkeypatch)
    status_calls = _record_status_and_execution(node)

    async def _fetch_generation_result(_generation_id: str) -> dict[str, Any]:
        return {"status": "COMPLETED"}

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        node._set_status_results(was_successful=True, result_details="saved")

    monkeypatch.setattr(node, "_fetch_generation_result", _fetch_generation_result, raising=False)
    monkeypatch.setattr(node, "_parse_result", _parse_result, raising=False)

    await node._refresh_completed(GENERATION_ID)

    assert status_calls[-1]["was_successful"] is True


@pytest.mark.asyncio
async def test_refresh_completed_reports_success_when_parse_result_reports_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression guard: _execution_succeeded is None both before this run and after
    # _clear_execution_status, so a _parse_result that never calls _set_status_results
    # itself (most of them delegate to _save_generated_media, whose own success path
    # is the terminal statement) must still result in a reported success, not a
    # swallowed one.
    node, _ = _build_node(monkeypatch)
    status_calls = _record_status_and_execution(node)

    async def _fetch_generation_result(_generation_id: str) -> dict[str, Any]:
        return {"status": "COMPLETED"}

    async def _parse_result(_result_json: dict[str, Any], _generation_id: str) -> None:
        return None

    monkeypatch.setattr(node, "_fetch_generation_result", _fetch_generation_result, raising=False)
    monkeypatch.setattr(node, "_parse_result", _parse_result, raising=False)

    await node._refresh_completed(GENERATION_ID)

    assert len(status_calls) == 1
    assert status_calls[0]["was_successful"] is True
