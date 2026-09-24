"""A budget refusal from Griptape Cloud has to stop the node, not be retried or ignored.

Cloud answers an over-budget call with 403 and a body naming every budget that refused. The
proxy node makes three calls that can be answered that way -- submit, poll, fetch result -- and
each one used to mishandle it differently: submit reported the generic HTTP message, poll treated
the refusal as a transient error and re-asked until the timeout ran out, and the result fetch
returned None so the run carried on with empty outputs.

The wording itself is the engine's (`griptape_nodes.utils.budget_refusal`) and is tested there.
What these pin is that the node reaches for it, raises rather than continues, and scopes the
recognition to Griptape's own host -- a 403 from a third-party API a node also talks to is not
Griptape's to blame on a budget.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from griptape_nodes.utils.budget_refusal import BUDGET_HALT_PREFIX, BudgetExceededError

from griptape_nodes_library.image.flux_2_image_generation import Flux2ImageGeneration

HEADERS = {"Authorization": "Bearer key"}
GENERATION_ID = "gen-refused"
OTHER_HOST = "api.example-vendor.com"

REFUSAL_BODY: dict[str, Any] = {
    "error": "budget_exceeded",
    "message": "Budget limit reached (tight).",
    "blocked_by": [
        {
            "budget_id": "3f1c6b4e-0000-4000-8000-000000000001",
            "budget_name": "tight",
            "scope_type": "ORG",
            "reset_period": "MONTHLY",
            "enforcement": "HARD",
            "limit_credits": 100,
            "spent_credits": 90,
            "remaining_credits": 10,
            "requested_credits": 50,
            "frozen": False,
        }
    ],
    "effective_remaining_credits": 10,
    "spend_id": "9a2d5e70-0000-4000-8000-00000000000f",
}
"""A 403 body recorded from Cloud's own `SpendHold.as_refusal()` tests."""


def _http_error(*, host: str, status: int = 403, body: object = REFUSAL_BODY) -> httpx.HTTPStatusError:
    """The exception a proxy call raises when the far end answers with an error status."""
    request = httpx.Request("POST", f"https://{host}/api/proxy/v2/models/flux")
    response = httpx.Response(status, json=body, request=request)
    return httpx.HTTPStatusError("refused", request=request, response=response)


class RefusingClient:
    """httpx.AsyncClient stand-in whose every call is answered with one canned error."""

    error: httpx.HTTPStatusError

    async def __aenter__(self) -> RefusingClient:
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        return None

    async def get(self, url: str, headers: dict[str, str], timeout: int) -> Any:  # noqa: ARG002
        raise type(self).error

    async def post(self, url: str, json: Any = None, headers: dict[str, str] | None = None, timeout: int = 0) -> Any:  # noqa: A002, ARG002
        raise type(self).error


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> Flux2ImageGeneration:
    """A Flux node whose HTTP calls all fail with a Griptape Cloud budget refusal.

    Sleeping is stubbed out so a test that asserts the poll loop does *not* retry still
    finishes promptly if the behaviour regresses, rather than sitting out the real timeout.
    """
    RefusingClient.error = _http_error(host="cloud.griptape.ai")
    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.httpx.AsyncClient", RefusingClient)

    async def noop_sleep(_: float) -> None:
        return None

    monkeypatch.setattr("griptape_nodes_library.proxy.griptape_proxy_node.asyncio.sleep", noop_sleep)

    built = Flux2ImageGeneration(name="Flux2")
    built._validate_api_key = lambda: "key"  # type: ignore[method-assign]
    built._set_safe_defaults = lambda: None  # type: ignore[method-assign]
    return built


def _statuses(target: Flux2ImageGeneration) -> list[dict[str, Any]]:
    """Capture what the node writes to its own status parameter."""
    captured: list[dict[str, Any]] = []
    target._set_status_results = lambda **kwargs: captured.append(kwargs)  # type: ignore[method-assign]
    return captured


class TestTheRunStopsWhereverTheRefusalLands:
    """Each of the three proxy calls has to end the node, and say the same thing when it does."""

    @pytest.mark.asyncio
    async def test_a_refused_submission_halts(self, node: Flux2ImageGeneration) -> None:
        statuses = _statuses(node)

        with pytest.raises(BudgetExceededError) as raised:
            await node._submit_generation({}, HEADERS, "flux")

        assert str(raised.value).startswith(BUDGET_HALT_PREFIX)
        assert "tight" in str(raised.value)
        assert statuses[-1]["was_successful"] is False
        assert statuses[-1]["result_details"] == str(raised.value)

    @pytest.mark.asyncio
    async def test_a_refused_poll_halts_instead_of_retrying(self, node: Flux2ImageGeneration) -> None:
        """A refusal is settled, not transient.

        The poll loop retries an HTTP error until the timeout, which for a refusal means the
        artist waits out the whole timeout to be told what the first poll already knew. The
        generation_id survives so Refresh can still reach the generation afterwards.
        """
        polls = 0
        original_get = RefusingClient.get

        async def counting_get(self: RefusingClient, url: str, headers: dict[str, str], timeout: int) -> Any:
            nonlocal polls
            polls += 1
            return await original_get(self, url, headers=headers, timeout=timeout)

        RefusingClient.get = counting_get  # type: ignore[method-assign]
        try:
            statuses = _statuses(node)

            with pytest.raises(BudgetExceededError) as raised:
                await node._poll_generation_status(GENERATION_ID, HEADERS)
        finally:
            RefusingClient.get = original_get  # type: ignore[method-assign]

        assert polls == 1, f"The refusal was re-asked {polls} times; Cloud's answer will not change."
        assert str(raised.value).startswith(BUDGET_HALT_PREFIX)
        assert node.parameter_output_values["generation_id"] == GENERATION_ID
        assert statuses[-1]["result_details"] == str(raised.value)

    @pytest.mark.asyncio
    async def test_a_refused_result_fetch_halts_rather_than_returning_nothing(self, node: Flux2ImageGeneration) -> None:
        """Returning None here let the run continue with empty outputs and no stated reason."""
        statuses = _statuses(node)

        with pytest.raises(BudgetExceededError) as raised:
            await node._fetch_generation_result(GENERATION_ID)

        assert str(raised.value).startswith(BUDGET_HALT_PREFIX)
        assert statuses[-1]["result_details"] == str(raised.value)


class TestOnlyGriptapesOwnRefusalsAreReadThatWay:
    """A 403 is a common answer; only Griptape Cloud's carries a budget to name."""

    @pytest.mark.asyncio
    async def test_a_403_from_another_host_is_left_to_the_generic_path(
        self, node: Flux2ImageGeneration, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(RefusingClient, "error", _http_error(host=OTHER_HOST))
        _statuses(node)

        with pytest.raises(RuntimeError) as raised:
            await node._submit_generation({}, HEADERS, "flux")

        assert not isinstance(raised.value, BudgetExceededError)

    @pytest.mark.asyncio
    async def test_an_ordinary_griptape_failure_is_left_to_the_generic_path(
        self, node: Flux2ImageGeneration, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            RefusingClient, "error", _http_error(host="cloud.griptape.ai", status=500, body={"error": "boom"})
        )
        _statuses(node)

        with pytest.raises(RuntimeError) as raised:
            await node._submit_generation({}, HEADERS, "flux")

        assert not isinstance(raised.value, BudgetExceededError)


class TestRefreshReportsTheRefusalRatherThanRaisingIt:
    """Refresh is a button press on a finished node, not a run, so there is nothing to halt.

    It runs on a throwaway thread with its own event loop, where an escaping exception reaches
    nobody. The refusal has to arrive as node status instead -- the same sentence, delivered
    the only way this path can deliver it.
    """

    @pytest.mark.asyncio
    async def test_a_refused_status_check_becomes_node_status(self, node: Flux2ImageGeneration) -> None:
        statuses = _statuses(node)

        assert await node._fetch_status_for_refresh(GENERATION_ID, HEADERS) is None

        assert statuses[-1]["was_successful"] is False
        assert statuses[-1]["result_details"].startswith(BUDGET_HALT_PREFIX)
        assert "tight" in statuses[-1]["result_details"]

    @pytest.mark.asyncio
    async def test_a_refused_result_fetch_becomes_node_status(self, node: Flux2ImageGeneration) -> None:
        statuses = _statuses(node)

        await node._refresh_completed(GENERATION_ID)

        assert statuses[-1]["was_successful"] is False
        assert statuses[-1]["result_details"].startswith(BUDGET_HALT_PREFIX)
