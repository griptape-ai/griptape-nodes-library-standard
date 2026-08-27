from __future__ import annotations

import asyncio
from typing import Any, NamedTuple

import pytest

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration
from griptape_nodes_library.proxy.griptape_proxy_node import CancelOutcome

HEADERS = {"Authorization": "Bearer key"}
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_SERVER_ERROR = 500


class StatusResponse:
    def __init__(self, status: str) -> None:
        self._status = status

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return {"status": self._status}


class CancelResponse:
    def __init__(self, status_code: int, status: str) -> None:
        self.status_code = status_code
        self._status = status

    @property
    def is_success(self) -> bool:
        return HTTP_OK <= self.status_code < HTTP_BAD_REQUEST

    def json(self) -> dict[str, Any]:
        return {"status": self._status}


class FakeAsyncClient:
    """httpx.AsyncClient stand-in that records cancel POSTs and serves canned statuses."""

    cancel_urls: list[str] = []  # noqa: RUF012
    cancel_status_code: int = HTTP_OK
    # Status the proxy reports back on the cancel response.
    cancel_reported_status: str = "CANCELLED"
    cancel_error: Exception | None = None
    # Status the polled generation reports; QUEUED keeps the poll loop in flight.
    poll_status: str = "QUEUED"

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str], timeout: int) -> StatusResponse:  # noqa: ARG002
        return StatusResponse(type(self).poll_status)

    async def post(self, url: str, headers: dict[str, str], timeout: int) -> CancelResponse:  # noqa: ARG002
        cls = type(self)
        cls.cancel_urls.append(url)
        if cls.cancel_error is not None:
            raise cls.cancel_error
        return CancelResponse(cls.cancel_status_code, cls.cancel_reported_status)


class Harness(NamedTuple):
    node: Flux2ImageGeneration
    status_calls: list[dict[str, Any]]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    """A Flux node wired to the fake HTTP client, with status writes captured."""
    FakeAsyncClient.cancel_urls = []
    FakeAsyncClient.cancel_status_code = HTTP_OK
    FakeAsyncClient.cancel_reported_status = "CANCELLED"
    FakeAsyncClient.cancel_error = None
    FakeAsyncClient.poll_status = "QUEUED"
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", FakeAsyncClient)

    node = Flux2ImageGeneration(name="Flux2")
    status_calls: list[dict[str, Any]] = []
    node._set_status_results = lambda **kwargs: status_calls.append(kwargs)  # type: ignore[method-assign]
    return Harness(node, status_calls)


def _cancelled_generation_ids() -> list[str]:
    """The generation ids the fake client saw a cancel POST for."""
    return [url.rsplit("/", 2)[-2] for url in FakeAsyncClient.cancel_urls]


@pytest.mark.asyncio
async def test_task_cancellation_cancels_generation_server_side(harness: Harness) -> None:
    """Cancelling the node's task mid-sleep must POST the proxy's cancel endpoint."""
    node = harness.node
    task = asyncio.create_task(node._poll_generation_status("gen-1", HEADERS))
    # One poll happens, then the loop parks in asyncio.sleep(poll_interval) — the state a
    # user cancel almost always lands in, since the default poll interval is 5s.
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert _cancelled_generation_ids() == ["gen-1"]
    assert harness.status_calls[-1]["was_successful"] is False
    assert "will not be billed" in harness.status_calls[-1]["result_details"]
    assert node.parameter_output_values["generation_id"] == "gen-1"
    assert node.parameter_output_values["generation_status"] == "CANCELLED"


@pytest.mark.asyncio
async def test_cooperative_flag_cancels_generation_server_side(harness: Harness) -> None:
    """A cancel that only sets the cooperative flag is honoured at the top of the loop."""
    harness.node.request_cancellation()

    result = await harness.node._poll_generation_status("gen-2", HEADERS)

    assert result is None
    assert _cancelled_generation_ids() == ["gen-2"]
    assert "will not be billed" in harness.status_calls[-1]["result_details"]


@pytest.mark.asyncio
async def test_already_dispatched_generation_reports_billing(harness: Harness) -> None:
    """A 400 means the work already started: expected, not an error, but the user is told it is billed."""
    FakeAsyncClient.cancel_status_code = HTTP_BAD_REQUEST
    FakeAsyncClient.cancel_reported_status = "RUNNING"
    harness.node.request_cancellation()

    result = await harness.node._poll_generation_status("gen-3", HEADERS)

    assert result is None
    assert harness.status_calls[-1]["was_successful"] is False
    assert "and be billed" in harness.status_calls[-1]["result_details"]
    # The status must report where the generation actually is, not claim a cancellation
    # that the proxy refused.
    assert harness.node.parameter_output_values["generation_status"] == "RUNNING"
    # generation_id survives so the Refresh affordance can still retrieve the result.
    assert harness.node.parameter_output_values["generation_id"] == "gen-3"


@pytest.mark.asyncio
async def test_failed_cancel_request_does_not_mask_cancellation(harness: Harness) -> None:
    """A cancel POST that blows up must not replace the CancelledError that is unwinding."""
    FakeAsyncClient.cancel_error = RuntimeError("connection reset")

    task = asyncio.create_task(harness.node._poll_generation_status("gen-4", HEADERS))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert "could not be confirmed" in harness.status_calls[-1]["result_details"]


@pytest.mark.asyncio
async def test_request_generation_cancel_outcomes(harness: Harness) -> None:
    """Each proxy answer maps to the outcome that drives the user-facing message."""
    node = harness.node
    assert await node._request_generation_cancel("gen-5", HEADERS) is CancelOutcome.CANCELLED

    FakeAsyncClient.cancel_status_code = HTTP_BAD_REQUEST
    assert await node._request_generation_cancel("gen-5", HEADERS) is CancelOutcome.ALREADY_STARTED

    FakeAsyncClient.cancel_status_code = HTTP_SERVER_ERROR
    assert await node._request_generation_cancel("gen-5", HEADERS) is CancelOutcome.UNKNOWN

    FakeAsyncClient.cancel_error = RuntimeError("boom")
    assert await node._request_generation_cancel("gen-5", HEADERS) is CancelOutcome.UNKNOWN


@pytest.mark.asyncio
async def test_stale_cancellation_flag_does_not_cancel_a_new_run(harness: Harness) -> None:
    """A flag left set by a cancelled run must not cancel the generation the next run submits.

    The engine only clears the flag in BaseNode.clear_node(), which the flow's cancel path
    does not reach for a node cancelled mid-resolution, so a re-run starts with it still set.
    """
    node = harness.node
    node.request_cancellation()

    def raise_missing_key() -> str:
        msg = "missing key"
        raise ValueError(msg)

    # Bail out immediately after the flag is cleared, so this pins the ordering: the clear
    # has to happen before the run can reach anything that reads the flag.
    node._validate_api_key = raise_missing_key  # type: ignore[method-assign]
    node._handle_api_key_validation_error = lambda _e: None  # type: ignore[method-assign]

    await node._process_generation()

    assert node.is_cancellation_requested is False
    assert FakeAsyncClient.cancel_urls == []


@pytest.mark.asyncio
async def test_completed_generation_is_not_cancelled(harness: Harness) -> None:
    """A generation that reaches a terminal state must never be sent a cancel."""
    FakeAsyncClient.poll_status = "COMPLETED"

    result = await harness.node._poll_generation_status("gen-6", HEADERS)

    assert result is not None
    assert result["status"] == "COMPLETED"
    assert FakeAsyncClient.cancel_urls == []
