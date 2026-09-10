"""Best-effort contract of the attribution lookup: every unclear answer is `{}`, never a raise.

The caller is one line from spending real credits, so the interesting cases here are the four
ways the engine can fail to answer -- not the one way it succeeds. None of them may reach the
caller as an exception, and none may invent a header value.

Both spellings are held to that contract by the same tests. `attribution_header` and
`attribution_header_async` differ only in how they wait, so a claim that holds for one and not
the other is a bug in whichever was edited alone -- and the async one is the copy that gets
forgotten, while being the one every billable coroutine call site goes through. The two
async-only tests at the end cover the part that cannot be shared: that the wait actually yields.

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
    """Put `responder` where the engine was, and return the list of requests that reach it."""
    seen: list[Any] = []

    def _handle_request(request: Any) -> Any:
        seen.append(request)
        return responder(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", _handle_request)
    return seen


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

    def _never_answers(_request: Any) -> Any:
        released.wait(timeout=5)
        return _succeeding()

    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.05)
    _answering(monkeypatch, _never_answers)

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
    """The bound has to hold at interpreter exit too, which is where a thread pool quietly voids it.

    `ThreadPoolExecutor` registers every worker it starts with
    `concurrent.futures.thread._python_exit`, and that handler `join()`s all of them at shutdown
    no matter how the pool was closed -- `shutdown(wait=False)` returns immediately but detaches
    nothing. With the worker still blocked on a wedged engine, quitting then waits out the full
    30s transport timeout that the caller-side bound just finished escaping. The cost is not
    removed, only moved: a hung node becomes a hung quit, somewhere the timing assertions above
    cannot see it. A daemon thread is never joined, which is why this module does not use a pool.

    The async parameter is not a formality: `asyncio.to_thread` is the obvious way to write that
    wait and is a pool in disguise -- it dispatches to the running loop's default
    `ThreadPoolExecutor`, which `asyncio.run` joins in `shutdown_default_executor` on its way
    out. Measured while writing this, that spelling took 30.1s to exit and the loop's own
    `asyncio.run` accounted for 30.0s of it. Only this parameter would have caught that.

    Only a real interpreter exit can show any of it, hence the subprocess. The child wedges the
    engine for far longer than it bounds the call, so the two outcomes are unmistakable: measured
    here, a pool exits in ~30s and a daemon thread in ~0.3s.
    """
    child = textwrap.dedent(f"""
        import asyncio
        import time
        import griptape_nodes_library.utils.attribution as attribution

        class _Wedged:
            @staticmethod
            def handle_request(_request):
                time.sleep(30)

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
    _answering(monkeypatch, lambda _: released.wait(timeout=5))

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
async def test_a_late_answer_after_a_timeout_is_dropped_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The worker outlives the caller that gave up on it, and must not disturb the loop on its way back.

    Nothing joins the dispatch thread, so a timed-out lookup leaves it running with a future
    nobody is waiting on any more. `wait_for` cancelled that future on its way out, and settling
    a cancelled future raises `InvalidStateError` -- inside a `call_soon_threadsafe` callback,
    where no caller exists to catch it.

    The loop's exception handler is what makes that visible. Without it the failure is invisible
    from a test: a raise inside a loop callback never reaches the awaiting code, it is handed to
    the handler, which by default logs it and moves on. In a real node it surfaces as
    "Exception in callback" on whatever unrelated task happened to be running, some seconds after
    the Cloud call it belongs to already returned -- an error report with no path back to its
    cause, which is the worst kind to leave lying around for a reporting field.
    """
    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.05)
    reported: list[dict[str, Any]] = []
    asyncio.get_running_loop().set_exception_handler(lambda _loop, context: reported.append(context))
    answered = threading.Event()

    def _answers_late(_request: Any) -> Any:
        time.sleep(0.2)
        answered.set()
        return _succeeding()

    _answering(monkeypatch, _answers_late)

    assert await attribution_header_async() == {}

    # Hold the loop open past the late answer, so the callback runs where the handler sees it.
    await asyncio.sleep(0.4)
    assert answered.is_set(), "the engine never answered; the late-delivery path was not exercised"
    assert reported == []


def test_a_late_answer_after_the_loop_closed_is_dropped_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other way the caller can be gone: not just done waiting, but out of loop entirely.

    A node that bridges into async with `asyncio.run` closes its loop the moment the coroutine
    returns -- and on the timeout path it returns with the dispatch thread still blocked on the
    engine. `call_soon_threadsafe` on a closed loop raises `RuntimeError`, and it raises on the
    dispatch thread, where there is no caller at all: it goes to `threading.excepthook` and
    prints a traceback the user cannot connect to anything, seconds after the call it belongs to
    already succeeded.

    Recorded through that hook rather than left to pytest, so the assertion is about the thread
    rather than about which warnings the runner happens to promote.
    """
    monkeypatch.setattr(attribution_module, "_TIMEOUT_SECONDS", 0.05)
    escaped: list[Any] = []
    monkeypatch.setattr(threading, "excepthook", escaped.append)
    answered = threading.Event()

    def _answers_late(_request: Any) -> Any:
        time.sleep(0.2)
        answered.set()
        return _succeeding()

    _answering(monkeypatch, _answers_late)

    # The loop closes on the way out of `run`, while the dispatch thread is still sleeping.
    assert asyncio.run(attribution_header_async()) == {}

    assert answered.wait(timeout=5), "the engine never answered; the late-delivery path was not exercised"
    # `excepthook` fires after the target returns, so give the thread a moment to finish unwinding.
    time.sleep(0.1)
    assert escaped == []
