"""The Cloud SDK drivers have to stop on a budget refusal, with the budgets still named.

Upstream's drivers mishandle a 403 twice over, and both failures are invisible from a unit test
that stubs `requests`. So these run against a real socket:

- `try_stream` raises for status inside `with requests.post(..., stream=True)`, and the exception
  outlives the connection. Only a genuine HTTP round trip releases the body the way production
  does; a stubbed response object happily returns JSON forever and the test passes on a driver
  that would lose it. `stream` defaults to True, so this is the common path, not the corner.
- `ExponentialBackoffMixin` re-runs anything outside `ignored_exception_types`. Counting the
  requests that actually arrive is what distinguishes "stopped" from "stopped eventually", and a
  settled refusal re-asked is a second call Cloud refuses plus a backoff the artist waits through.

The refusal wording belongs to the engine (`griptape_nodes.utils.budget_refusal`) and is tested
there. What these pin is that the drivers reach it, raise rather than retry, and leave a 403 that
is not a budget refusal alone.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import TYPE_CHECKING, Any

import pytest
import requests
from griptape.artifacts import ImageArtifact
from griptape.common import PromptStack
from griptape.structures import Agent
from griptape_nodes.utils.budget_refusal import BUDGET_HALT_PREFIX, BudgetExceededError

from griptape_nodes_library.utils.cloud_budget_drivers import (
    GriptapeCloudImageGenerationDriver,
    GriptapeCloudPromptDriver,
)
from griptape_nodes_library.utils.error_utils import try_throw_error

if TYPE_CHECKING:
    from collections.abc import Iterator

REFUSAL_BODY: dict[str, Any] = {
    "error": "budget_exceeded",
    "message": "Budget limit reached (tight).",
    "blocked_by": [
        {
            "budget_id": "3f1c6b4e-0000-4000-8000-000000000001",
            "budget_name": "tight",
            "scope_type": "ORGANIZATION",
            "reset_period": "MONTHLY",
            "enforcement": "HARD",
            "limit_credits": 100,
            "spent_credits": 90,
            "spent_by_cost_basis": {"billed": 90, "estimated": 0, "declared": 0},
            "includes_byok": False,
            "includes_reported": False,
            "remaining_credits": 10,
            "requested_credits": 50,
            "frozen": False,
        }
    ],
    "effective_remaining_credits": 10,
    "spend_id": "5b2d9c10-0000-4000-8000-000000000002",
}

_OTHER_403_BODY = {"detail": "You do not have permission to perform this action."}


class _Cloud:
    """A stand-in Cloud that answers every POST the same way, and counts the calls."""

    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self.body = body
        self.paths: list[str] = []
        cloud = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                cloud.paths.append(self.path)
                payload = json.dumps(cloud.body).encode()
                self.send_response(cloud.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                """Silence the default stderr access log."""

        self._server = HTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"

    def __enter__(self) -> _Cloud:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def refusing_cloud() -> Iterator[_Cloud]:
    with _Cloud(403, REFUSAL_BODY) as cloud:
        yield cloud


def _prompt_stack() -> PromptStack:
    stack = PromptStack()
    stack.add_user_message("hello")
    return stack


def _image() -> ImageArtifact:
    return ImageArtifact(value=b"\x89PNG\r\n\x1a\n", format="png", width=1, height=1)


def _assert_names_the_budget(halt: BudgetExceededError) -> None:
    assert str(halt).startswith(BUDGET_HALT_PREFIX)
    assert "tight" in str(halt)
    # The driver is several frames below whichever node is spending through it and cannot know
    # which; `NodeManager` names it on the way out.
    assert halt.node_name is None


class TestThePromptDriverStopsOnARefusal:
    @pytest.mark.parametrize("stream", [False, True])
    def test_the_refusal_halts_the_call_with_the_budget_named(self, refusing_cloud: _Cloud, *, stream: bool) -> None:
        """Streaming included: the body has to be read before the response context closes it."""
        driver = GriptapeCloudPromptDriver(
            base_url=refusing_cloud.base_url, api_key="key", model="gpt-4.1", stream=stream
        )

        with pytest.raises(BudgetExceededError) as caught:
            driver.run(_prompt_stack())

        _assert_names_the_budget(caught.value)

    @pytest.mark.parametrize("stream", [False, True])
    def test_a_settled_refusal_is_asked_once(self, refusing_cloud: _Cloud, *, stream: bool) -> None:
        """No room is not a transient failure, so re-asking only spends the artist's time."""
        driver = GriptapeCloudPromptDriver(
            base_url=refusing_cloud.base_url, api_key="key", model="gpt-4.1", stream=stream
        )

        with pytest.raises(BudgetExceededError):
            driver.run(_prompt_stack())

        assert len(refusing_cloud.paths) == 1

    def test_a_403_that_is_not_a_budget_refusal_is_left_alone(self) -> None:
        """Only Cloud's budget envelope becomes a halt; every other 403 stays what it was."""
        with _Cloud(403, _OTHER_403_BODY) as cloud:
            driver = GriptapeCloudPromptDriver(base_url=cloud.base_url, api_key="key", model="gpt-4.1", stream=False)

            with pytest.raises(requests.exceptions.HTTPError):
                driver.run(_prompt_stack())


class TestTheHaltSurvivesTheAgent:
    """Griptape's task layer catches what a driver raises and keeps it as the task's output.

    A node reads that output after the run, so unless the halt is raised again from there it
    reaches `NodeManager` as generic agent failure, or not at all.
    """

    @pytest.mark.parametrize("stream", [False, True])
    def test_the_agent_node_raises_the_halt_itself(self, refusing_cloud: _Cloud, *, stream: bool) -> None:
        driver = GriptapeCloudPromptDriver(
            base_url=refusing_cloud.base_url, api_key="key", model="gpt-4.1", stream=stream
        )
        agent = Agent(prompt_driver=driver)

        agent.run("hello")

        with pytest.raises(BudgetExceededError) as caught:
            try_throw_error(agent.output)
        _assert_names_the_budget(caught.value)


class TestTheImageDriverStopsOnARefusal:
    def test_text_to_image_halts_with_the_budget_named(self, refusing_cloud: _Cloud) -> None:
        driver = GriptapeCloudImageGenerationDriver(base_url=refusing_cloud.base_url, api_key="key")

        with pytest.raises(BudgetExceededError) as caught:
            driver.run_text_to_image(["a cat"])

        _assert_names_the_budget(caught.value)
        assert len(refusing_cloud.paths) == 1

    def test_image_variation_halts_with_the_budget_named(self, refusing_cloud: _Cloud) -> None:
        driver = GriptapeCloudImageGenerationDriver(base_url=refusing_cloud.base_url, api_key="key")

        with pytest.raises(BudgetExceededError) as caught:
            driver.run_image_variation(["a cat"], _image())

        _assert_names_the_budget(caught.value)
        assert len(refusing_cloud.paths) == 1

    def test_a_call_site_that_sets_its_own_fail_fast_list_keeps_it(self, refusing_cloud: _Cloud) -> None:
        """`create_image` passes `ignored_exception_types`; the budget halt is added, not swapped in."""
        driver = GriptapeCloudImageGenerationDriver(
            base_url=refusing_cloud.base_url,
            api_key="key",
            ignored_exception_types=(requests.exceptions.HTTPError,),
        )

        assert set(driver.ignored_exception_types) == {requests.exceptions.HTTPError, BudgetExceededError}
