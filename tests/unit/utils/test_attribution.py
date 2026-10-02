"""Best-effort contract of the attribution lookup: every unclear answer is `{}`, never a raise.

The caller is one line from spending real credits, so the interesting cases here are the four
ways the engine can fail to answer -- not the one way it succeeds. None of them may reach the
caller as an exception, and none may invent a header value.

Both spellings are held to that contract by the same tests. `attribution_header` and
`attribution_header_async` differ only in how they wait, so a claim that holds for one and not
the other is a bug in whichever was edited alone -- and the async one is the copy that gets
forgotten, while being the one every billable coroutine call site goes through. The
async-only tests at the end cover what cannot be shared: that the wait yields the loop, and
that giving up cancels the request.

The engine is stubbed rather than driven live. A live dispatch in this suite returns "no project
open", so every assertion would pass for the wrong reason: an empty workspace is not evidence
about what this code does with an answer.
"""

from __future__ import annotations

import asyncio
import inspect
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.retained_mode.events.budget_events import (
    GetAttributionContextRequest,
    GetAttributionContextResultFailure,
    GetAttributionContextResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import griptape_nodes_library.utils.attribution as attribution_module
from griptape_nodes_library.utils.attribution import attribution_header, attribution_header_async

# `base64url(b'{"v": 1}')` -- the tagless envelope the engine sends when no project is open.
_TAGLESS_ENVELOPE = "eyJ2IjoxfQ=="

_VARIANTS = pytest.mark.parametrize("variant", [attribution_header, attribution_header_async], ids=["sync", "async"])


def _ask(variant: Callable[[], Any]) -> dict[str, str]:
    """Call whichever spelling `variant` is, and return the header dict."""
    answer = variant()
    return asyncio.run(answer) if inspect.iscoroutine(answer) else answer


def _succeeding(**overrides: Any) -> GetAttributionContextResultSuccess:
    return GetAttributionContextResultSuccess(result_details="ok", **{"header_value": _TAGLESS_ENVELOPE, **overrides})


def _answering(monkeypatch: pytest.MonkeyPatch, responder: Callable[[Any], Any]) -> list[Any]:
    """Put `responder` where the engine was, and return the list of requests that reach it.

    It answers both spellings: the sync one through `handle_request`, the async one through
    `ahandle_request`.
    """
    seen: list[Any] = []

    def _handle_request(request: Any) -> Any:
        seen.append(request)
        return responder(request)

    async def _ahandle_request(request: Any) -> Any:
        return _handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle_request)
    monkeypatch.setattr(GriptapeNodes, "ahandle_request", _ahandle_request)
    return seen


def _wedging(monkeypatch: pytest.MonkeyPatch, released: threading.Event) -> None:
    """Make the engine hold every request until `released` is set.

    The async stub waits by sleeping, as a forwarded request waits on the orchestrator, so
    the loop stays free and a timeout can cancel it.
    """

    def _handle_request(_request: Any) -> Any:
        released.wait(timeout=5)
        return _succeeding()

    async def _ahandle_request(_request: Any) -> Any:
        while not released.is_set():
            await asyncio.sleep(0.005)
        return _succeeding()

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle_request)
    monkeypatch.setattr(GriptapeNodes, "ahandle_request", _ahandle_request)


@_VARIANTS
def test_a_successful_answer_becomes_the_header_the_engine_named(
    variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both halves come off the result -- the name as much as the value.

    `header_name` ships on the payload precisely so the platform can rename the header without
    an edit in every library that vendored this file. A literal here would quietly defeat that,
    and would still pass a test that used the current name, so the stub deliberately does not.
    """
    _answering(monkeypatch, lambda _: _succeeding(header_name="X-Renamed-Later", header_value="a-payload"))

    assert _ask(variant) == {"X-Renamed-Later": "a-payload"}


@_VARIANTS
def test_a_tagless_envelope_is_sent_rather_than_dropped(
    variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`{"v": 1}` is a statement, not an absence, and it is the engine's to make.

    It says "this client attributes, and no project is open". Only the engine knows whether that
    is true, so the value is forwarded unread -- no decode, no emptiness check, no repair. This
    test exists to fail a future sanity check added in good faith, which would turn a true
    statement into no statement and lose the forward-compatibility the envelope is sent for.
    """
    _answering(monkeypatch, lambda _: _succeeding(header_value=_TAGLESS_ENVELOPE))

    assert _ask(variant) == {"X-Griptape-Attribution": _TAGLESS_ENVELOPE}


@_VARIANTS
def test_a_declined_answer_sends_no_header(variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A Failure means the engine cannot describe the spend truthfully; copying that is the point.

    Not an error on this side, and not a reason to substitute an envelope of our own: the engine
    withheld a claim it could not support, and manufacturing one here would put the claim back.
    """
    _answering(monkeypatch, lambda _: GetAttributionContextResultFailure(result_details="no chain"))

    assert _ask(variant) == {}


@_VARIANTS
def test_a_raising_dispatch_sends_no_header(variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The billable call must survive anything this module can hit, including a broken engine."""

    def _explode(_request: Any) -> Any:
        msg = "the event bus is down"
        raise RuntimeError(msg)

    _answering(monkeypatch, _explode)

    assert _ask(variant) == {}


@_VARIANTS
def test_a_wedged_engine_costs_the_bound_and_not_the_call(
    variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The timeout has to be a real ceiling on the caller, not just on the answer.

    Omitting the bound inherits the engine's 30s forwarded-request timeout, and this catches
    that. It does not catch every way the bound can be given back -- the one it cannot see is
    `test_a_wedged_engine_does_not_outlive_the_process` below, which needs a real interpreter
    exit.

    The bound is patched down so the suite does not pay it; the assertion below is that the
    caller returns while the engine is still busy, which is the property, not the number.
    """
    # Meaningful only against the transport timeout it exists to escape.
    assert 0 < attribution_module._TIMEOUT_SECONDS < 30  # noqa: SLF001

    released = threading.Event()
    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.05)
    _wedging(monkeypatch, released)

    try:
        started = time.monotonic()
        headers = _ask(variant)
        elapsed = time.monotonic() - started

        assert headers == {}
        assert not released.is_set(), "the worker finished on its own; the test proved nothing"
        assert elapsed < 2
    finally:
        released.set()


@pytest.mark.parametrize(
    "call",
    ["attribution.attribution_header()", "asyncio.run(attribution.attribution_header_async())"],
    ids=["sync", "async"],
)
def test_a_wedged_engine_does_not_outlive_the_process(call: str) -> None:
    """The bound has to hold at interpreter exit too, where a thread pool would quietly void it.

    The interpreter joins every `ThreadPoolExecutor` worker at exit, however the pool was
    closed, and `asyncio.run` joins the loop's default executor, which `asyncio.to_thread`
    uses. With a lookup still blocked on a wedged engine, quitting would wait out the 30s
    transport timeout the caller-side bound just escaped: a hung node becomes a hung quit.

    The sync spelling survives this by using a daemon thread, which is never joined. The async
    spelling survives it by owning no thread at all: `wait_for` cancels the request. Either way,
    only a real interpreter exit can show it, hence the subprocess. The child wedges the engine
    for 30s, so a regression takes ~30s to exit, against well under a second.
    """
    child = textwrap.dedent(f"""
        import asyncio
        import time
        import griptape_nodes_library.utils.attribution as attribution

        class _Wedged:
            @staticmethod
            def handle_request(_request):
                time.sleep(30)

            @staticmethod
            async def ahandle_request(_request):
                await asyncio.sleep(30)

        attribution.GriptapeNodes = _Wedged
        attribution._TIMEOUT_SECONDS = 0.05
        assert {call} == {{}}
    """)

    started = time.monotonic()
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-c", child],
        cwd=Path(__file__).parents[3],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    elapsed = time.monotonic() - started

    assert completed.returncode == 0, completed.stderr
    assert elapsed < 10, f"the process took {elapsed:.1f}s to exit; the orphaned worker was joined"


@_VARIANTS
def test_the_request_is_dispatched_bare_and_silently(
    variant: Callable[[], Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing to resolve before asking, and nothing worth telling the rest of the app about.

    The request takes no arguments -- the engine dropped every dimension but `project` -- so a
    default-constructed instance is the whole payload. `broadcast_result` stays `False` or the
    success payload goes out on the WebSocket feed once per metered call.
    """
    seen = _answering(monkeypatch, lambda _: _succeeding())

    _ask(variant)

    assert seen == [GetAttributionContextRequest()]
    assert seen[0].broadcast_result is False


@pytest.mark.asyncio
async def test_the_async_wait_lets_the_rest_of_the_loop_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reason the async spelling exists, asserted directly rather than assumed from its shape.

    Everything above passes just as well if `attribution_header_async` is a coroutine wrapped
    around the blocking call -- the answers are identical, only the loop suffers, and nothing
    fails. What that costs is invisible from the outside: on a node awaiting a Cloud generation,
    the poll timer, the cancel handler, and every other node sharing the loop all stop for the
    length of an engine round trip, and for the whole bound when the engine is wedged.

    So this counts a competing task's turns while the engine is deliberately unresponsive. A
    parked loop never schedules it, and the count stays at zero.
    """
    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.2)
    released = threading.Event()
    _wedging(monkeypatch, released)

    ticks = 0

    async def _competing_work() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.005)

    task = asyncio.create_task(_competing_work())
    try:
        assert await attribution_header_async() == {}
        assert ticks > 1, "the loop was parked for the whole lookup; the async spelling is blocking"
    finally:
        released.set()
        task.cancel()


@pytest.mark.asyncio
async def test_giving_up_cancels_the_request(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timed-out async lookup leaves nothing running behind it.

    The request is awaited on the caller's loop, so `wait_for` cancels it at the bound. If it
    kept running, its answer would arrive after the Cloud call it belonged to had gone out, with
    no caller left to hand it to.
    """
    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.05)
    outcome: list[str] = []

    async def _ahandle_request(_request: Any) -> Any:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            outcome.append("cancelled")
            raise
        outcome.append("finished")
        return _succeeding()

    monkeypatch.setattr(GriptapeNodes, "ahandle_request", _ahandle_request)

    assert await attribution_header_async() == {}
    assert outcome == ["cancelled"]
