"""Ask the engine which project the impending Griptape Cloud spend belongs to.

Griptape Cloud meters credits against a budget, and the budget is keyed by project.
Cloud holds no project model of its own, so the request has to carry the answer in a
header naming the project chain the spend belongs to. Only the engine knows that chain,
and only at the moment of the call -- which is what this module asks it, once per Cloud
request, on behalf of
:mod:`~griptape_nodes_library.utils.griptape_cloud_headers`.

The ask is a round trip to the engine, so it comes in two spellings:
:func:`attribution_header` for synchronous callers and :func:`attribution_header_async`
for the coroutines, which are most of the billable ones. They answer identically and
share a bound; all that differs is whether the wait parks the calling thread. Keep them
that way -- the interpretation of an answer lives in one place below precisely so a fix
cannot land on one spelling and miss the other.

Two properties matter more than the header itself.

**Attribution never fails a call.** Every path out of here that is not a clear answer
returns ``{}``, so a Cloud request that would have succeeded still does. Sending spend
unattributed is a reporting problem; failing the user's generation to protect a
reporting field would be a worse trade, and one they did not ask for.

**The value is passed through, never built.** Whatever the engine hands back goes on the
wire byte-for-byte -- this module does no JSON, no base64, and no inspection. A payload
of ``{"v": 1}`` with no tags is well-formed and legal, and means "this client attributes,
and no project is open"; the engine is entitled to say that, and dropping it would
silently downgrade a true statement to no statement. What this module must never do is
*manufacture* that envelope when it has no answer at all. A timeout is not the engine
saying "no project" -- it is nobody saying anything, and the honest wire representation
of nobody saying anything is no header.

This file is the canonical implementation. Other node libraries cannot import across
each other's Python packages, so any library that needs this behavior vendors this file
verbatim rather than depending on it. Keep this module free of dependencies beyond the
engine package (`griptape_nodes.*`) and the standard library so it can be copied as-is
into another library's `utils/` directory.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import TYPE_CHECKING

from griptape_nodes.retained_mode.events.budget_events import (
    GetAttributionContextRequest,
    GetAttributionContextResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("griptape_nodes")

__all__ = ["attribution_header", "attribution_header_async"]

_UNATTRIBUTED = "Could not resolve the Griptape Cloud attribution header; billing this call unattributed."

# Bounds this module's contribution to the latency of every billable Cloud call. The
# engine's own bound is far looser: on a worker the request is forwarded to the
# orchestrator under the 30s transport timeout shared by every forwarded request, which
# is not a number this call can afford to inherit. Two seconds is long enough for a
# healthy round trip and short enough that a wedged orchestrator costs the user a
# reporting field rather than the appearance of a hung node.
_TIMEOUT_SECONDS = 2.0


def _dispatch(deliver: Callable[[object], None]) -> None:
    """Ask the engine, and hand back whatever came of it -- an answer or the exception."""
    try:
        deliver(GriptapeNodes.handle_request(GetAttributionContextRequest()))
    except Exception as error:  # noqa: BLE001
        # Carried back as a value rather than logged here, so the caller reports the cause
        # on the one path that has given up on it.
        deliver(error)


def _start_asking(deliver: Callable[[object], None]) -> None:
    """Run :func:`_dispatch` on a thread the interpreter will never wait for.

    Off-thread purely for the timeout, and `handle_request` documents itself as safe on
    arbitrary threads. The one cost of a library-owned thread is losing the
    broadcast-suppression ContextVar, and this request suppresses its own broadcast anyway,
    so there is nothing left to leak.

    `daemon=True`, and a bare thread rather than a pool, because of what happens at
    interpreter exit. `ThreadPoolExecutor` registers its workers with
    `concurrent.futures.thread._python_exit`, which `join()`s every one of them no matter how
    the pool was shut down -- `shutdown(wait=False)` does not detach anything. On the timeout
    path the worker is still blocked on the engine, so quitting would block until the engine's
    30s transport gave up: the same 30s this timeout exists to escape, moved from a hung node
    to a hung quit. This rules out `asyncio.to_thread` for the async variant too, since that
    runs on the event loop's default pool and `asyncio.run` joins it on the way out.
    """
    threading.Thread(target=_dispatch, args=(deliver,), name="griptape-attribution", daemon=True).start()


def _timed_out() -> dict[str, str]:
    # No traceback worth printing -- the timeout carries nothing the message does not.
    logger.warning(
        "Griptape Cloud attribution did not answer within %ss; billing this call unattributed.", _TIMEOUT_SECONDS
    )
    return {}


def _header_from(answer: object) -> dict[str, str]:
    """Turn whatever the engine handed back into the header dict, or ``{}``."""
    if isinstance(answer, BaseException):
        # Cloud emits no metric for a *missing* header, so this log line is the only place a
        # permanently broken client is distinguishable from a momentarily slow one. Without
        # the cause attached, every billable call reports the same unactionable sentence.
        logger.warning(_UNATTRIBUTED, exc_info=answer)
        return {}

    if not isinstance(answer, GetAttributionContextResultSuccess):
        # The engine declining to describe the spend, rather than anything going wrong.
        # Debug, not warning: an engine with no project open answers this way routinely.
        logger.debug("Griptape Cloud attribution unavailable: %s", type(answer).__name__)
        return {}

    # Verbatim, including a tagless envelope. See the module docstring.
    return {answer.header_name: answer.header_value}


def attribution_header() -> dict[str, str]:
    """Return ``{header_name: header_value}`` for the current project, or ``{}``.

    For synchronous callers. Inside a coroutine use :func:`attribution_header_async`
    instead -- this one parks the calling thread, which on an event loop means nothing
    else scheduled there advances until the engine answers.

    Returns:
        dict[str, str]: The single attribution header to merge into a Cloud request, or
            an empty dict when the engine gave no usable answer -- it reported failure,
            it did not answer within :data:`_TIMEOUT_SECONDS`, or the dispatch raised.
            Merging ``{}`` is what makes an unattributed call the fallback rather than a
            failed one.
    """
    answers: queue.Queue[object] = queue.Queue(maxsize=1)
    try:
        _start_asking(answers.put)
        return _header_from(answers.get(timeout=_TIMEOUT_SECONDS))
    except queue.Empty:
        return _timed_out()
    except Exception:
        # Deliberately broad, and reached when the thread itself could not be started. The
        # caller is one line away from spending real credits, and there is no exception from
        # here worth propagating into that -- see the module docstring.
        logger.warning(_UNATTRIBUTED, exc_info=True)
        return {}


async def attribution_header_async() -> dict[str, str]:
    """Await ``{header_name: header_value}`` for the current project, or ``{}``.

    Same contract as :func:`attribution_header`, and the same bound, but the wait yields
    the event loop instead of parking it. Every billable Cloud call made from a coroutine
    should come through here: the sync variant would freeze that loop for the whole
    worker-to-orchestrator round trip, and for the full bound when the engine is wedged.

    Returns:
        dict[str, str]: As :func:`attribution_header`.
    """
    loop = asyncio.get_running_loop()
    answer: asyncio.Future[object] = loop.create_future()

    def _settle(value: object) -> None:
        # `wait_for` may already have given up and cancelled the future.
        if not answer.done():
            answer.set_result(value)

    def _deliver(value: object) -> None:
        try:
            loop.call_soon_threadsafe(_settle, value)
        except RuntimeError:
            # The loop closed while the engine was still thinking; nobody is left to tell.
            logger.debug("Griptape Cloud attribution answered after its event loop closed.")

    try:
        _start_asking(_deliver)
        return _header_from(await asyncio.wait_for(answer, _TIMEOUT_SECONDS))
    except TimeoutError:
        return _timed_out()
    except Exception:
        # As above: never propagate into a call that is about to spend.
        logger.warning(_UNATTRIBUTED, exc_info=True)
        return {}
