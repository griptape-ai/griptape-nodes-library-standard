"""A generation the provider fails after accepting it has to fail the node.

The poll sees FAILED or ERRORED once the provider has rejected the work -- a moderation refusal
of an input image, say. The node has to route that like any other failure: down a wired Failed
output, or out to stop the run. It must not end RESOLVED and let downstream nodes run on empty
outputs.
"""

from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import GenerationFailedError

HEADERS = {"Authorization": "Bearer key"}
GENERATION_ID = "gen-failed"
PROVIDER_ERROR = "InputImageSensitiveContentDetected.PrivacyInformation"


class StatusResponse:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._body


class TerminalStatusClient:
    """httpx.AsyncClient stand-in whose poll answers with one canned terminal status."""

    body: dict[str, Any]
    polls: int = 0

    async def __aenter__(self) -> TerminalStatusClient:
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str], timeout: int) -> StatusResponse:  # noqa: ARG002
        type(self).polls += 1
        return StatusResponse(type(self).body)

    async def post(self, url: str, json: Any = None, headers: dict[str, str] | None = None, timeout: int = 0) -> Any:  # noqa: A002, ARG002
        return StatusResponse({"generation_id": GENERATION_ID})


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Flux2ImageGeneration:
    TerminalStatusClient.body = {"status": "FAILED", "status_detail": {"details": PROVIDER_ERROR}}
    TerminalStatusClient.polls = 0
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", TerminalStatusClient)

    async def noop_sleep(_: float) -> None:
        return None

    async def no_headers(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        return HEADERS

    async def permitted(*_args: Any, **_kwargs: Any) -> Any:
        return type("Permitted", (), {"failed": lambda self: False})()

    async def empty_payload() -> dict[str, Any]:
        return {}

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)
    monkeypatch.setattr(
        "griptape_nodes_library.proxy.griptape_proxy_node.build_griptape_cloud_headers_async", no_headers
    )
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.declare_model_invocation", permitted)

    built = Flux2ImageGeneration(name="Flux2")
    built._validate_api_key = lambda: "key"  # type: ignore[method-assign]
    built._set_safe_defaults = lambda: None  # type: ignore[method-assign]
    built._prepare_user_auth_info = lambda: None  # type: ignore[method-assign]
    built._build_payload = empty_payload  # type: ignore[method-assign]
    built._get_api_model_id = lambda: "flux"  # type: ignore[method-assign]
    built._model_access = None
    return built


def _statuses(target: Flux2ImageGeneration) -> list[dict[str, Any]]:
    captured: list[dict[str, Any]] = []
    target._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]
    return captured


@pytest.mark.parametrize("status", ["FAILED", "ERRORED"])
@pytest.mark.asyncio
async def test_the_poll_raises_without_retrying(node: Flux2ImageGeneration, status: str) -> None:
    TerminalStatusClient.body = {"status": status, "status_detail": {"details": PROVIDER_ERROR}}
    statuses = _statuses(node)

    with pytest.raises(GenerationFailedError) as raised:
        await node._poll_generation_status(GENERATION_ID, HEADERS)

    assert TerminalStatusClient.polls == 1
    assert PROVIDER_ERROR in str(raised.value)
    assert node.parameter_output_values["generation_status"] == status
    assert statuses[-1]["was_successful"] is False
    assert statuses[-1]["result_details"] == str(raised.value)


@pytest.mark.asyncio
async def test_with_nothing_wired_the_failure_stops_the_run(node: Flux2ImageGeneration) -> None:
    node._has_outgoing_connections = lambda _parameter: False  # type: ignore[method-assign]
    _statuses(node)

    with pytest.raises(GenerationFailedError) as raised:
        await node.aprocess()

    assert PROVIDER_ERROR in str(raised.value)
    assert node.parameter_output_values["generation_id"] == GENERATION_ID


@pytest.mark.asyncio
async def test_a_wired_failure_output_is_taken(node: Flux2ImageGeneration) -> None:
    node._has_outgoing_connections = lambda _parameter: True  # type: ignore[method-assign]
    statuses = _statuses(node)

    await node.aprocess()

    assert statuses[-1]["was_successful"] is False
    assert PROVIDER_ERROR in statuses[-1]["result_details"]
    assert node.parameter_output_values["generation_status"] == "FAILED"
