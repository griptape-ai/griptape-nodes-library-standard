"""Declare an impending model invocation so the permission layer can gate it.

Callers dispatch `declare_model_invocation` before making any network call to
the model provider and treat a failed result as do-not-invoke: the engine
clears the call by default, but a registered policy can deny it, in which
case the result reports failure and the caller must not proceed. This is a
fail-closed contract -- if the declaration fails for any reason, the model
must not be invoked.

This file is the canonical implementation. Other node libraries cannot import
across each other's Python packages, so any library that needs this behavior
vendors this file verbatim rather than depending on it. Keep this module free
of dependencies beyond the engine package (`griptape_nodes.*`) and the
standard library so it can be copied as-is into another library's `utils/`
directory.
"""

from __future__ import annotations

import logging

from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import get_declared_models
from griptape_nodes.retained_mode.events.base_events import ResultPayload
from griptape_nodes.retained_mode.events.model_events import DeclareModelInvocationRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "declare_model_invocation",
    "declare_model_invocation_sync",
    "require_model_invocation_sync",
    "resolve_catalog_model_id",
]


def resolve_catalog_model_id(node: BaseNode, api_model_id: str) -> str | None:
    """Resolve the selected provider model id to its stable catalog key.

    The lookup is scoped to the node's own declared models, so the
    provider_model_id -> stable key mapping is unambiguous: a node does not
    declare the same upstream model under two catalog keys. Returns None
    when the selection is not one of the node's declared catalog models.
    """
    matches = [
        resolved.model_id for resolved in get_declared_models(node) if resolved.model.provider_model_id == api_model_id
    ]
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


def require_model_invocation_sync(node: BaseNode, api_model_id: str | None) -> None:
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

    Raises:
        RuntimeError: if the model is unidentified, or if the declaration was
            denied. Carries the engine's `result_details` when it says something,
            since that explains *why* the policy denied the call; otherwise a
            generic message naming the model.
    """
    if not api_model_id:
        msg = (
            f"Cannot run {type(node).__name__} '{node.name}': no model is set on the driver, "
            "so the invocation cannot be checked against the model policy. "
            "Select a model on the upstream driver node."
        )
        raise RuntimeError(msg)
    declaration = declare_model_invocation_sync(node, api_model_id)
    if not declaration.failed():
        return
    # `result_details` is a required field, but coerce defensively: it carries the
    # policy's own explanation, and an empty one would otherwise produce a
    # RuntimeError that names no model and gives the user nothing to act on.
    details = str(declaration.result_details or "").strip()
    if not details:
        details = f"invocation of model '{api_model_id}' was not permitted."
    msg = f"Cannot run {type(node).__name__} '{node.name}': {details}"
    raise RuntimeError(msg)


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
