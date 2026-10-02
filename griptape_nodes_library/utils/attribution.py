"""Ask the engine which project the impending Griptape Cloud spend belongs to.

Cloud meters credits against budgets keyed by project, and only the engine knows the
project chain, so each billable request asks it once and carries the answer in a header.

Two spellings: :func:`attribution_header` for synchronous callers and
:func:`attribution_header_async` for coroutines. They share a bound and interpret an answer
in one place, :func:`_header_from`, so they cannot drift apart.

Two invariants:

- **Attribution never fails a call.** Anything short of a clear answer returns ``{}``.
- **The value is passed through, never built.** The engine's answer goes on the wire
  verbatim, including a tagless ``{"v": 1}``, which means "no project is open". A timeout
  sends no header, not a made-up envelope.

Other libraries vendor this file verbatim, so it depends only on ``griptape_nodes`` and the
standard library.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading

from griptape_nodes.retained_mode.events.budget_events import (
    GetAttributionContextRequest,
    GetAttributionContextResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes")

__all__ = ["attribution_header", "attribution_header_async"]

_UNATTRIBUTED = "Could not resolve the Griptape Cloud attribution header; billing this call unattributed."

# On a worker the request is forwarded to the orchestrator under the 30s timeout every
# forwarded request shares. A wedged orchestrator should cost a reporting field, not a
# node that looks hung.
_TIMEOUT_SECONDS = 2.0


def attribution_header() -> dict[str, str]:
    """Return ``{header_name: header_value}`` for the current project, or ``{}``.

    For synchronous callers. Inside a coroutine use :func:`attribution_header_async`,
    which does not park the event loop while the engine answers.

    Returns:
        dict[str, str]: The attribution header to merge into a Cloud request, or ``{}``
            when the engine declined, did not answer within :data:`_TIMEOUT_SECONDS`, or
            the lookup raised.
    """
    answers: queue.Queue[object] = queue.Queue(maxsize=1)
    try:
        _ask_on_a_daemon_thread(answers)
        return _header_from(answers.get(timeout=_TIMEOUT_SECONDS))
    except queue.Empty:
        return _timed_out()
    except Exception:
        # Reached only when the thread could not start. Never propagate into a call that
        # is about to spend.
        logger.warning(_UNATTRIBUTED, exc_info=True)
        return {}


async def attribution_header_async() -> dict[str, str]:
    """Await ``{header_name: header_value}`` for the current project, or ``{}``.

    Same contract and bound as :func:`attribution_header`. The request is awaited on the
    caller's loop, so a timeout cancels it rather than leaving it running.

    Returns:
        dict[str, str]: As :func:`attribution_header`.
    """
    try:
        answer = await asyncio.wait_for(GriptapeNodes.ahandle_request(GetAttributionContextRequest()), _TIMEOUT_SECONDS)
    except TimeoutError:
        return _timed_out()
    except Exception:
        # As above: never propagate into a call that is about to spend.
        logger.warning(_UNATTRIBUTED, exc_info=True)
        return {}
    return _header_from(answer)


def _ask_on_a_daemon_thread(answers: queue.Queue[object]) -> None:
    """Run the sync lookup on a thread, so the caller can stop waiting at the bound.

    A daemon thread rather than a pool, because the interpreter joins pool workers at exit:
    a wedged engine would turn a hung node into a hung quit.
    `test_a_wedged_engine_does_not_outlive_the_process` holds this.
    """

    def _ask() -> None:
        try:
            answers.put(GriptapeNodes.handle_request(GetAttributionContextRequest()))
        except Exception as error:  # noqa: BLE001
            # Handed back as the answer so the caller logs it with its cause.
            answers.put(error)

    threading.Thread(target=_ask, name="griptape-attribution", daemon=True).start()


def _timed_out() -> dict[str, str]:
    logger.warning(
        "Griptape Cloud attribution did not answer within %ss; billing this call unattributed.", _TIMEOUT_SECONDS
    )
    return {}


def _header_from(answer: object) -> dict[str, str]:
    """Turn whatever the engine handed back into the header dict, or ``{}``."""
    if isinstance(answer, BaseException):
        # Cloud has no metric for a missing header, so this log line is the only way to tell
        # a broken client from a slow one.
        logger.warning(_UNATTRIBUTED, exc_info=answer)
        return {}

    if not isinstance(answer, GetAttributionContextResultSuccess):
        # The engine declining, which it does routinely with no project open.
        logger.debug("Griptape Cloud attribution unavailable: %s", type(answer).__name__)
        return {}

    return {answer.header_name: answer.header_value}
