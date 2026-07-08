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

__all__ = ["declare_model_invocation", "resolve_catalog_model_id"]


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
    model_id = resolve_catalog_model_id(node, api_model_id)
    if model_id is None:
        logger.warning(
            "%s: '%s' is not a declared catalog model for this node; "
            "declaring the invocation with the provider model id for now.",
            node.name,
            api_model_id,
        )
        model_id = api_model_id
    return await GriptapeNodes.ahandle_request(DeclareModelInvocationRequest(model_id=model_id, node_name=node.name))
