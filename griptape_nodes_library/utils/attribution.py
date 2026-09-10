"""Ask the engine which project the impending Griptape Cloud spend belongs to.

Griptape Cloud meters credits against a budget, and the budget is keyed by project.
Cloud holds no project model of its own, so the request has to carry the answer in a
header naming the project chain the spend belongs to. Only the engine knows that chain,
and only at the moment of the call -- which is what this module asks it, once per Cloud
request, on behalf of
:func:`~griptape_nodes_library.utils.griptape_cloud_headers.build_griptape_cloud_headers`.

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

import logging
import queue
import threading

from griptape_nodes.retained_mode.events.budget_events import (
    GetAttributionContextRequest,
    GetAttributionContextResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes")

__all__ = ["attribution_header"]

_UNATTRIBUTED = "Could not resolve the Griptape Cloud attribution header; billing this call unattributed."

# Bounds this module's contribution to the latency of every billable Cloud call. The
# engine's own bound is far looser: on a worker the request is forwarded to the
# orchestrator under the 30s transport timeout shared by every forwarded request, which
# is not a number this call can afford to inherit. Two seconds is long enough for a
# healthy round trip and short enough that a wedged orchestrator costs the user a
# reporting field rather than the appearance of a hung node.
_TIMEOUT_SECONDS = 2.0


def attribution_header() -> dict[str, str]:
    """Return ``{header_name: header_value}`` for the current project, or ``{}``.

    Returns:
        dict[str, str]: The single attribution header to merge into a Cloud request, or
            an empty dict when the engine gave no usable answer -- it reported failure,
            it did not answer within :data:`_TIMEOUT_SECONDS`, or the dispatch raised.
            Merging ``{}`` is what makes an unattributed call the fallback rather than a
            failed one.
    """
    answers: queue.Queue[object] = queue.Queue(maxsize=1)

    def _ask() -> None:
        try:
            answers.put(GriptapeNodes.handle_request(GetAttributionContextRequest()))
        except Exception as error:  # noqa: BLE001
            # Carried back rather than logged here, so the caller reports the cause on the
            # one path that has given up on it. Nothing else reads this queue.
            answers.put(error)

    # Dispatched off-thread purely for the timeout: the call sites are a mix of sync and
    # async, so `asyncio.wait_for` is unavailable to a helper that has to serve both.
    # `handle_request` documents itself as safe on arbitrary threads; the one cost of a
    # library-owned thread is losing the broadcast-suppression ContextVar, and this request
    # suppresses its own broadcast anyway, so there is nothing left to leak.
    #
    # `daemon=True`, and a bare thread rather than a `ThreadPoolExecutor`, because of what
    # happens at interpreter exit. A pool registers its workers with
    # `concurrent.futures.thread._python_exit`, which `join()`s every one of them no matter
    # how the pool was shut down -- `shutdown(wait=False)` does not detach anything. So on
    # the timeout path, where the worker is still blocked on the engine, quitting would
    # block until the engine's 30s transport gave up: the same 30s this timeout exists to
    # escape, moved from a hung node to a hung quit. A daemon thread is never joined.
    try:
        threading.Thread(target=_ask, name="griptape-attribution", daemon=True).start()
        answer = answers.get(timeout=_TIMEOUT_SECONDS)
    except queue.Empty:
        # No traceback worth printing -- `Empty` carries nothing the message does not.
        logger.warning(
            "Griptape Cloud attribution did not answer within %ss; billing this call unattributed.", _TIMEOUT_SECONDS
        )
        return {}
    except Exception:
        # Deliberately broad, and reached when the thread itself could not be started. The
        # caller is one line away from spending real credits, and there is no exception from
        # here worth propagating into that -- see the module docstring.
        logger.warning(_UNATTRIBUTED, exc_info=True)
        return {}

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
