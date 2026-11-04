"""Gate an impending model invocation on the permission layer and, for direct calls, on budgets.

Callers dispatch `declare_model_invocation` before making any network call to
the model provider and treat a failed result as do-not-invoke: the engine
clears the call by default, but a registered policy can deny it, in which
case the result reports failure and the caller must not proceed. This is a
fail-closed contract -- if the declaration fails for any reason, the model
must not be invoked.

A node that calls a provider directly (BYOK, not through the Griptape proxy)
goes through `require_model_access_sync` instead, which adds Griptape Cloud's
budget check after the permission declaration, and reports the call's token
counts with `report_model_usage_sync` afterwards; Cloud prices them. The budget
check is the opposite of the permission gate: it fails open when Cloud cannot be
asked and fails closed only on an explicit deny. Calls through the Griptape
proxy are budgeted server-side and must use neither.

This file is the canonical implementation. Other node libraries cannot import
across each other's Python packages, so any library that needs this behavior
vendors this file verbatim rather than depending on it. Keep this module free
of dependencies beyond the engine package (`griptape_nodes.*`) and the
standard library so it can be copied as-is into another library's `utils/`
directory.
"""

from __future__ import annotations

import logging
from typing import cast

from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import get_declared_models
from griptape_nodes.retained_mode.events.base_events import ForwardedException, ResultPayload
from griptape_nodes.retained_mode.events.budget_events import (
    BudgetAccessRequest,
    BudgetAccessResultSuccess,
    ReportUsageRequest,
)
from griptape_nodes.retained_mode.events.model_events import DeclareModelInvocationRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.utils.budget_refusal import BudgetExceededError, BudgetRefusal, refusal_from_check

logger = logging.getLogger("griptape_nodes")

_BUDGET_EXCEEDED_TYPE = f"{BudgetExceededError.__module__}.{BudgetExceededError.__qualname__}"

__all__ = [
    "declare_model_invocation",
    "declare_model_invocation_sync",
    "report_model_usage_sync",
    "require_budget_access_sync",
    "require_model_access_sync",
    "require_model_invocation_sync",
    "resolve_catalog_model_id",
]


def resolve_catalog_model_id(node: BaseNode, api_model_id: str) -> str | None:
    """Resolve a provider's model id to the stable catalog key the permission layer gates on.

    `api_model_id` is the provider's own name for the model: either the value a
    model dropdown stores, or a driver's `model` attribute read off whatever
    concrete driver ended up installed. The lookup is scoped to the node's own
    declared models.

    The catalog permits two entries to share one `provider_model_id`, so an
    ambiguous match resolves to nothing rather than gating against an arbitrary
    one of them. Returns None in that case, and when `api_model_id` is not a
    provider id this node declares.
    """
    declared = get_declared_models(node)
    matches = [resolved.model_id for resolved in declared if resolved.model.provider_model_id == api_model_id]
    return matches[0] if len(matches) == 1 else None


async def declare_model_invocation(node: BaseNode, api_model_id: str) -> ResultPayload:
    """Declare the impending model invocation so the permission layer can gate it.

    Resolves the concrete provider model id to the stable catalog key the
    permission system gates on, and declares that. The engine clears the
    call by default; a registered policy can deny it, in which case the
    result reports failure. The proxy enforces server-side as well; this
    runs first, so a denied call fails fast and never leaves the engine.
    """
    return await GriptapeNodes.ahandle_request(_build_declaration(node, api_model_id))


def declare_model_invocation_sync(node: BaseNode, api_model_id: str) -> ResultPayload:
    """Synchronous twin of `declare_model_invocation` for non-async call sites.

    Nodes whose model call happens outside an async context (e.g. a generator
    `process()` that runs framework drivers synchronously) declare through
    this variant. Identical contract: dispatch before any network call and
    treat a failed result as do-not-invoke.
    """
    return GriptapeNodes.handle_request(_build_declaration(node, api_model_id))


def require_model_invocation_sync(node: BaseNode, api_model_id: str | None, *, purpose: str | None = None) -> None:
    """Declare the invocation and raise if the permission layer denies it.

    The fail-closed half of `declare_model_invocation_sync`, for the common case
    where a denial should abort the node. Callers that need to recover instead of
    raising (e.g. reporting the denial through a status parameter) should call
    `declare_model_invocation_sync` and inspect the result themselves.

    `api_model_id` is optional because some drivers leave `model` unset and let
    the provider choose (`GriptapeCloudPromptDriver.model` defaults to None). An
    unidentified model cannot be gated, so that is refused rather than declared:
    declaring a null model id would ask the permission layer to rule on a model
    nobody has named, which is the one outcome a fail-closed gate must not allow.

    `purpose` names which invocation is being gated, for nodes that gate more than
    one. It is named in the raised message so a denial points at the specific call.

    Raises:
        RuntimeError: if the model is unidentified, or if the declaration was
            denied. Carries the engine's `result_details` when it says something,
            since that explains *why* the policy denied the call; otherwise a
            generic message naming the model.
    """
    subject = f"the model ({purpose})" if purpose else "the model"
    if not api_model_id or not api_model_id.strip():
        msg = (
            f"Cannot run {subject}: no model was identified, so the invocation cannot be "
            "checked against the model policy. Set a model on this node or on the "
            "driver connected to it."
        )
        raise RuntimeError(msg)
    declaration = declare_model_invocation_sync(node, api_model_id)
    if not declaration.failed():
        return
    # `result_details` carries the policy's own explanation and is a required field,
    # so `or ""` is only reached by a payload that crossed a boundary without
    # validation. Coerce either way: a blank explanation would otherwise raise a
    # RuntimeError that names no model and gives the user nothing to act on.
    details = str(declaration.result_details or "").strip()
    if not details:
        details = f"invocation of model '{api_model_id}' was not permitted."
    msg = f"Cannot run {subject}: {details}"
    raise RuntimeError(msg)


def require_model_access_sync(
    node: BaseNode,
    api_model_id: str | None,
    *,
    estimated_cost_micro_usd: int | None = None,
    purpose: str | None = None,
) -> str | None:
    """Gate a direct provider call on the permission layer, then on Griptape Cloud budgets.

    Permission runs first so a model the license forbids never reaches the budget
    check. A node can run the two halves apart instead: `require_model_invocation_sync`
    is local and fast, while `require_budget_access_sync` is a round trip to Cloud that
    belongs off the engine's event loop.

    Returns the check's correlation id, to pass to `report_model_usage_sync`, or None
    when the check never ran.

    Raises:
        RuntimeError: as `require_model_invocation_sync` and `require_budget_access_sync` do.
        BudgetExceededError: as `require_budget_access_sync` does.
    """
    require_model_invocation_sync(node, api_model_id, purpose=purpose)
    # The declaration above refused a missing model.
    return require_budget_access_sync(
        node, cast("str", api_model_id), estimated_cost_micro_usd=estimated_cost_micro_usd
    )


def require_budget_access_sync(
    node: BaseNode, api_model_id: str, *, estimated_cost_micro_usd: int | None = None
) -> str | None:
    """Ask Griptape Cloud whether a direct provider call fits its budgets; raise if it doesn't.

    A round trip to Cloud: call it from a worker thread, not the engine's event loop. It
    names the model by `api_model_id`, the provider's own name, which is also what the
    usage report sends. `estimated_cost_micro_usd` is optional; without it only a budget
    with no headroom left refuses the call. Fails open when Cloud can't be asked.

    Returns the check's correlation id, to pass to `report_model_usage_sync`, or None
    when the check never ran.

    Raises:
        BudgetExceededError: the engine's own exception, raised unwrapped so the run
            halts with its "Budget stopped this run." message. Any exception a failed
            check carries is raised the same way.
        RuntimeError: when a failed check carries no exception.
    """
    result = GriptapeNodes.handle_request(
        BudgetAccessRequest(
            model=api_model_id,
            estimated_cost_micro_usd=estimated_cost_micro_usd,
            node_type=type(node).__name__,
        )
    )
    # Any failure stops the call, not just `BudgetAccessResultFailure`: from a worker process
    # a refusal arrives flattened into a generic failure.
    if result.failed():
        exception = getattr(result, "exception", None)
        # From a worker process the refusal arrives as a ForwardedException carrying only the
        # original type's name. Rebuild it so callers that catch BudgetExceededError, and the
        # engine's halt path, still see a budget refusal. The budgets come back from the plain
        # `blocked_by` data, which survives the boundary, so the halt message still names them.
        if isinstance(exception, ForwardedException) and exception.original_type == _BUDGET_EXCEEDED_TYPE:
            blocked_by = getattr(result, "blocked_by", None)
            if isinstance(blocked_by, list):
                refusal = refusal_from_check({"blocked_by": blocked_by})
                raise BudgetExceededError(str(exception), refusal) from exception
            # Nothing to rebuild the budgets from: keep the message already worded, naming the node.
            raise BudgetExceededError(str(exception), BudgetRefusal(), node_name=node.name) from exception
        if isinstance(exception, Exception):
            raise exception
        details = str(result.result_details or "").strip() or "Budget stopped this run."
        msg = f"Cannot run {type(node).__name__}: {details}"
        raise RuntimeError(msg)
    if isinstance(result, BudgetAccessResultSuccess):
        return result.correlation_id
    # Any other payload means the check never ran; fail open like the engine does.
    logger.warning("%s: budget check returned %s; proceeding unchecked.", type(node).__name__, type(result).__name__)
    return None


def report_model_usage_sync(  # noqa: PLR0913
    node: BaseNode,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    input_audio_tokens: int = 0,
    cache_audio_read_tokens: int = 0,
    correlation_id: str | None = None,
    activity_type: str = "chat_completion",
) -> None:
    """Report a direct provider call's token counts; Griptape Cloud prices them.

    `input_tokens` is the provider's total prompt count, cached and audio tokens included.
    `input_audio_tokens` is the audio share of it, and `cache_audio_read_tokens` the cached share
    of that audio, also counted in `cache_read_tokens`. Best-effort: never raises, and the result
    is ignored.
    """
    try:
        GriptapeNodes.handle_request(
            ReportUsageRequest(
                provider=provider,
                model=model,
                activity_type=activity_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_read_tokens=cache_read_tokens,
                cache_write_tokens=cache_write_tokens,
                input_audio_tokens=input_audio_tokens,
                cache_audio_read_tokens=cache_audio_read_tokens,
                node_type=type(node).__name__,
                correlation_id=correlation_id,
            )
        )
    except Exception:
        logger.warning("%s: could not report model usage.", type(node).__name__, exc_info=True)


def _build_declaration(node: BaseNode, api_model_id: str) -> DeclareModelInvocationRequest:
    model_id = resolve_catalog_model_id(node, api_model_id)
    if model_id is None:
        logger.warning(
            "%s: '%s' is not a declared catalog model for this node; "
            "declaring the invocation with the provider model id for now.",
            node.name,
            api_model_id,
        )
        model_id = api_model_id
    return DeclareModelInvocationRequest(model_id=model_id, node_name=node.name)
