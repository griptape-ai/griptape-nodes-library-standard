"""Best-effort contract of `attribution_header`: every unclear answer is `{}`, never a raise.

The caller is one line from spending real credits, so the interesting cases here are the four
ways the engine can fail to answer -- not the one way it succeeds. None of them may reach the
caller as an exception, and none may invent a header value.

The engine is stubbed rather than driven live. A live dispatch in this suite returns "no project
open", so every assertion would pass for the wrong reason: an empty workspace is not evidence
about what this code does with an answer.
"""

from __future__ import annotations

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
from griptape_nodes_library.utils.attribution import attribution_header

# `base64url(b'{"v": 1}')` -- the tagless envelope the engine sends when no project is open.
_TAGLESS_ENVELOPE = "eyJ2IjoxfQ=="


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


def test_a_successful_answer_becomes_the_header_the_engine_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both halves come off the result -- the name as much as the value.

    `header_name` ships on the payload precisely so the platform can rename the header without
    an edit in every library that vendored this file. A literal here would quietly defeat that,
    and would still pass a test that used the current name, so the stub deliberately does not.
    """
    _answering(monkeypatch, lambda _: _succeeding(header_name="X-Renamed-Later", header_value="a-payload"))

    assert attribution_header() == {"X-Renamed-Later": "a-payload"}


def test_a_tagless_envelope_is_sent_rather_than_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """`{"v": 1}` is a statement, not an absence, and it is the engine's to make.

    It says "this client attributes, and no project is open". Only the engine knows whether that
    is true, so the value is forwarded unread -- no decode, no emptiness check, no repair. This
    test exists to fail a future sanity check added in good faith, which would turn a true
    statement into no statement and lose the forward-compatibility the envelope is sent for.
    """
    _answering(monkeypatch, lambda _: _succeeding(header_value=_TAGLESS_ENVELOPE))

    assert attribution_header() == {"X-Griptape-Attribution": _TAGLESS_ENVELOPE}


def test_a_declined_answer_sends_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Failure means the engine cannot describe the spend truthfully; copying that is the point.

    Not an error on this side, and not a reason to substitute an envelope of our own: the engine
    withheld a claim it could not support, and manufacturing one here would put the claim back.
    """
    _answering(monkeypatch, lambda _: GetAttributionContextResultFailure(result_details="no chain"))

    assert attribution_header() == {}


def test_a_raising_dispatch_sends_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """The billable call must survive anything this module can hit, including a broken engine."""

    def _explode(_request: Any) -> Any:
        msg = "the event bus is down"
        raise RuntimeError(msg)

    _answering(monkeypatch, _explode)

    assert attribution_header() == {}


def test_a_wedged_engine_costs_the_bound_and_not_the_call(monkeypatch: pytest.MonkeyPatch) -> None:
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
        headers = attribution_header()
        elapsed = time.monotonic() - started

        assert headers == {}
        assert not released.is_set(), "the worker finished on its own; the test proved nothing"
        assert elapsed < 2
    finally:
        released.set()


def test_a_wedged_engine_does_not_outlive_the_process() -> None:
    """The bound has to hold at interpreter exit too, which is where a thread pool quietly voids it.

    `ThreadPoolExecutor` registers every worker it starts with
    `concurrent.futures.thread._python_exit`, and that handler `join()`s all of them at shutdown
    no matter how the pool was closed -- `shutdown(wait=False)` returns immediately but detaches
    nothing. With the worker still blocked on a wedged engine, quitting then waits out the full
    30s transport timeout that the caller-side bound just finished escaping. The cost is not
    removed, only moved: a hung node becomes a hung quit, somewhere the timing assertions above
    cannot see it. A daemon thread is never joined, which is why this module does not use a pool.

    Only a real interpreter exit can show this, hence the subprocess. The child wedges the engine
    for far longer than it bounds the call, so the two outcomes are unmistakable: measured here,
    a pool exits in ~30s and a daemon thread in ~0.3s.
    """
    child = textwrap.dedent("""
        import time
        import griptape_nodes_library.utils.attribution as attribution

        class _Wedged:
            @staticmethod
            def handle_request(_request):
                time.sleep(30)

        attribution.GriptapeNodes = _Wedged
        attribution._TIMEOUT_SECONDS = 0.05
        assert attribution.attribution_header() == {}
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


def test_the_request_is_dispatched_bare_and_silently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing to resolve before asking, and nothing worth telling the rest of the app about.

    The request takes no arguments -- the engine dropped every dimension but `project` -- so a
    default-constructed instance is the whole payload. `broadcast_result` stays `False` or the
    success payload goes out on the WebSocket feed once per metered call.
    """
    seen = _answering(monkeypatch, lambda _: _succeeding())

    attribution_header()

    assert seen == [GetAttributionContextRequest()]
    assert seen[0].broadcast_result is False
