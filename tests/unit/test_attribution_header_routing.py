"""The attributed dict has to reach the billable request, and only that one.

`test_griptape_cloud_headers.py` asserts each build site's `attribution=` flag, and
`test_attribution.py` asserts what the factory returns for a given flag. Neither watches the
gap between them: a function that builds two dicts and hands them to two requests can route
them backwards and pass both. That is a silent mis-bill in the direction that matters -- the
POST that spends goes out unattributed, and the platform emits no metric for a missing
header, so nothing on either end reports it.

Both sites here build the pair in one function and hand the halves to different requests:

- `_process_generation` -> `_submit_and_poll`, where the submit POST spends and the status
  GETs do not, plus the `/result` GET that `_fetch_generation_result` builds for itself.
- `_append_private_asset` -> `_create_provider_asset`, where the registering POST spends and
  the asset-status GETs do not.

The tests run the real call chain against a recording client and read the headers off the
wire, so a swapped argument fails rather than a re-assertion of the same source line.
"""

from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.assets.byteplus_provider_asset_reference import (
    ASSET_KIND_IMAGE,
    create_provider_asset_reference,
)
from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.video.seedance_2_0_video_generation import Seedance20VideoGeneration

# The marker a stubbed attributed build leaves on the dict. A literal rather than the real
# header name: these tests are about which dict arrives, and pinning the name is
# `test_attribution.py`'s job -- it reads the name off the engine's result so a rename never
# touches a call site.
ATTRIBUTED = "X-Test-Attribution"


async def _fake_build_headers(bearer_token: str, *, attribution: bool) -> dict[str, str]:
    """Stand in for `build_griptape_cloud_headers_async`, tagging the attributed dict."""
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    if attribution:
        headers[ATTRIBUTED] = "v1"
    return headers


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.headers = {"content-type": "application/json"}

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


class _RecordingClient:
    """An `httpx.AsyncClient` stand-in that records how each request was authorized.

    `calls` collects `(method, url, attributed)`. Instances share one list because the code
    under test opens a fresh `async with httpx.AsyncClient()` per leg, so per-instance
    recording would throw away everything but the last one.
    """

    calls: list[tuple[str, str, bool]] = []  # noqa: RUF012
    replies: dict[str, dict[str, Any]] = {}  # noqa: RUF012

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        return None

    def _record(self, method: str, url: str, headers: dict[str, str]) -> _Response:
        type(self).calls.append((method, url, ATTRIBUTED in headers))
        reply = next((body for suffix, body in self.replies.items() if url.endswith(suffix)), {})
        return _Response(reply)

    async def post(self, url: str, *, headers: dict[str, str], **_: Any) -> _Response:
        return self._record("POST", url, headers)

    async def get(self, url: str, *, headers: dict[str, str], **_: Any) -> _Response:
        return self._record("GET", url, headers)


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> type[_RecordingClient]:
    monkeypatch.setattr(_RecordingClient, "calls", [])
    monkeypatch.setattr(_RecordingClient, "replies", {})
    return _RecordingClient


@pytest.mark.asyncio
async def test_only_the_submit_post_carries_attribution(
    monkeypatch: pytest.MonkeyPatch, recorder: type[_RecordingClient]
) -> None:
    """A whole proxy run: the submit POST spends, the status GET and `/result` GET do not."""
    proxy = "griptape_nodes_library.proxy.griptape_proxy_node"
    monkeypatch.setattr(f"{proxy}.httpx.AsyncClient", recorder)
    monkeypatch.setattr(f"{proxy}.build_griptape_cloud_headers_async", _fake_build_headers)
    recorder.replies = {
        "models/flux-2": {"generation_id": "gen-1"},
        "generations/gen-1": {"status": "COMPLETED"},
        "generations/gen-1/result": {"artifact_url": "https://example.invalid/out.png"},
    }

    class _Permitted:
        result_details = None

        def failed(self) -> bool:
            return False

    async def _permit(*_: Any, **__: Any) -> _Permitted:
        return _Permitted()

    monkeypatch.setattr(f"{proxy}.declare_model_invocation", _permit)

    node = Flux2ImageGeneration(name="Flux2")
    # Not part of the routing question: the license gate, the credential lookup, the BYOK
    # probe, the payload, and the model-specific parse all sit outside the header path.
    node._model_access = None
    monkeypatch.setattr(type(node), "_validate_api_key", lambda self: "tok")
    monkeypatch.setattr(type(node), "_prepare_user_auth_info", lambda self: None)
    monkeypatch.setattr(type(node), "_get_api_model_id", lambda self: "flux-2")
    monkeypatch.setattr(type(node), "_build_payload", _empty_payload)
    monkeypatch.setattr(type(node), "_parse_result", _noop_parse)

    await node._process_generation()

    assert recorder.calls == [
        ("POST", "https://cloud.griptape.ai/api/proxy/v2/models/flux-2", True),
        ("GET", "https://cloud.griptape.ai/api/proxy/v2/generations/gen-1", False),
        ("GET", "https://cloud.griptape.ai/api/proxy/v2/generations/gen-1/result", False),
    ]


@pytest.mark.asyncio
async def test_refresh_re_reads_without_attribution(
    monkeypatch: pytest.MonkeyPatch, recorder: type[_RecordingClient]
) -> None:
    """Refresh re-reads a generation already paid for, so neither of its GETs may attribute."""
    proxy = "griptape_nodes_library.proxy.griptape_proxy_node"
    monkeypatch.setattr(f"{proxy}.httpx.AsyncClient", recorder)
    monkeypatch.setattr(f"{proxy}.build_griptape_cloud_headers_async", _fake_build_headers)
    recorder.replies = {
        "generations/gen-1": {"status": "COMPLETED"},
        "generations/gen-1/result": {"artifact_url": "https://example.invalid/out.png"},
    }

    node = Flux2ImageGeneration(name="Flux2")
    node.parameter_output_values["generation_id"] = "gen-1"
    monkeypatch.setattr(type(node), "_validate_api_key", lambda self: "tok")
    monkeypatch.setattr(type(node), "_parse_result", _noop_parse)

    await node._refresh_async()

    assert recorder.calls
    assert not any(attributed for _, _, attributed in recorder.calls)


@pytest.mark.asyncio
async def test_only_the_asset_registration_post_carries_attribution(
    monkeypatch: pytest.MonkeyPatch, recorder: type[_RecordingClient]
) -> None:
    """Registering a private asset spends; polling it to ACTIVE does not."""
    seedance = "griptape_nodes_library.video.seedance_common"
    monkeypatch.setattr(f"{seedance}.httpx.AsyncClient", recorder)
    monkeypatch.setattr(f"{seedance}.build_griptape_cloud_headers_async", _fake_build_headers)
    recorder.replies = {
        "assets": {"provider_asset_id": "prov-1"},
        "assets/prov-1": {"status": "ACTIVE", "asset_id": "asset-1"},
    }

    node = Seedance20VideoGeneration(name="Seedance20")
    monkeypatch.setattr(type(node), "_validate_api_key", lambda self: "tok")
    monkeypatch.setattr(
        type(node),
        "_resolve_public_url_for_asset",
        lambda self, ref, *, asset_kind: "https://public.example/portrait.png",
    )

    ref = create_provider_asset_reference(value="https://public.example/portrait.png", asset_kind=ASSET_KIND_IMAGE)
    asset_url = await node._append_private_asset(ref, expected_kind=ASSET_KIND_IMAGE, label="reference_images")

    assert asset_url == "asset://asset-1"
    assert recorder.calls == [
        ("POST", "https://cloud.griptape.ai/api/proxy/v2/assets", True),
        ("GET", "https://cloud.griptape.ai/api/proxy/v2/assets/prov-1", False),
    ]


async def _empty_payload(self: Any) -> dict[str, Any]:
    return {}


async def _noop_parse(self: Any, *_: Any, **__: Any) -> None:
    return None
