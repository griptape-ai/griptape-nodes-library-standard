"""Stop a run when Griptape Cloud refuses a call over budget.

A HARD budget with no room makes Cloud answer 403 with a body naming the budgets.
:func:`raise_budget_halt` turns that error into the engine's `BudgetExceededError`,
worded for the artist, so the run stops instead of reporting a generic HTTP failure.
The halt names no node; `NodeManager` re-words it to name one.
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlsplit

from griptape_nodes.utils.budget_refusal import BudgetExceededError, refusal_from_exception
from griptape_nodes.utils.budget_refusal import describe as describe_budget_refusal
from griptape_nodes.utils.budget_refusal import log_line as budget_log_line

logger = logging.getLogger("griptape_nodes")

GRIPTAPE_CLOUD_BASE_URL = "https://cloud.griptape.ai"


def cloud_root(base_url: str | None = None) -> str:
    return (base_url or os.environ.get("GT_CLOUD_BASE_URL") or GRIPTAPE_CLOUD_BASE_URL).rstrip("/")


def budget_halt(exc: BaseException, *, base_url: str | None = None) -> BudgetExceededError | None:
    """Return the halt `exc` carries, or None when it is not a Cloud budget refusal."""
    if isinstance(exc, BudgetExceededError):
        return exc
    refusal = refusal_from_exception(exc, cloud_host=lambda: urlsplit(cloud_root(base_url)).hostname or "")
    if refusal is None:
        return None
    logger.error(budget_log_line(refusal))
    return BudgetExceededError(describe_budget_refusal(refusal), refusal)


def raise_budget_halt(exc: BaseException, *, base_url: str | None = None) -> None:
    """Raise the halt `exc` carries, if it is a Cloud budget refusal."""
    halt = budget_halt(exc, base_url=base_url)
    if halt is None:
        return
    if halt is exc:
        raise halt
    raise halt from exc
