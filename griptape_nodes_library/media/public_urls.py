from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from griptape.artifacts.error_artifact import ErrorArtifact
from griptape.artifacts.url_artifact import UrlArtifact
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable

    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")

# Enough to overlap round trips without one node saturating the user's upload bandwidth.
MAX_CONCURRENT_UPLOADS = 4


async def adelete_uploaded_artifacts(helpers: Iterable[PublicArtifactUrlParameter], *, node_name: str) -> None:
    """Delete every helper's upload concurrently, logging failures instead of raising them.

    For cleanup paths: one failed delete must not skip the others or mask the run's own error.
    """
    results = await asyncio.gather(*(helper.adelete_uploaded_artifact() for helper in helpers), return_exceptions=True)
    for result in results:
        if isinstance(result, Exception):
            logger.warning("%s: failed to delete a temporary upload, so it stays in the bucket: %s", node_name, result)


async def gather_limited[T](coros: Iterable[Coroutine[Any, Any, T]], limit: int = MAX_CONCURRENT_UPLOADS) -> list[T]:
    """Run coroutines at most `limit` at a time and return their results in order.

    The first failure cancels the rest and waits for them to stop before it is raised, so a
    caller's cleanup sees every upload that was started.
    """
    pending = list(coros)
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run(coro)) for coro in pending]
    except ExceptionGroup as group_error:
        # Callers expect the failure itself. The group remains only as its context in tracebacks.
        raise group_error.exceptions[0]  # noqa: B904 - `from` would replace the failure's own cause
    finally:
        # Coroutines still queued when a failure cancelled the rest never started; close them
        # so they are not reported as never awaited.
        for coro in pending:
            coro.close()
    return [task.result() for task in tasks]


# TODO: remove PublicListItemUrl and public_list_item_helpers once PublicArtifactUrlParameter can wrap a
# ParameterList, and use the engine helper for list inputs, as Kling Omni's additional_reference_videos does.
# https://github.com/griptape-ai/griptape-nodes-engine/issues/6345
class PublicListItemUrl(PublicArtifactUrlParameter):
    """Public URL and cleanup for one value from a list input.

    The engine helper reads a single parameter by name, so it can't see an item of a ParameterList.
    This one is handed the item's value instead. The parameter it wraps is never added to the node.
    """

    def __init__(self, node: BaseNode, value: Any, *, name: str, disclaimer_message: str | None = None) -> None:
        super().__init__(
            node=node,
            artifact_url_parameter=ParameterVideo(name=name),
            disclaimer_message=disclaimer_message,
        )
        self._value = value

    def _get_url_to_publish(self) -> str:
        # Mirrors the engine helper, which clears the previous run's upload path first.
        self.gtc_file_path = None
        value = self._value
        if isinstance(value, ErrorArtifact):
            msg = (
                f"Attempted to generate a public URL for an item of a list input on node '{self._node.name}'. "
                f"Failed because the upstream value is an error: {value.value}"
            )
            raise RuntimeError(msg)  # noqa: TRY004 the upstream failure is a runtime error, not a type error.
        if isinstance(value, UrlArtifact):
            return value.value
        if isinstance(value, dict):
            url: Any = value.get("value")
            return url
        return value


def public_list_item_helpers(
    node: BaseNode, values: Iterable[Any], *, name: str, disclaimer_message: str | None = None
) -> list[PublicListItemUrl]:
    """One helper per list item, in order. `name` only labels errors."""
    return [
        PublicListItemUrl(node, value, name=f"{name}_{index}", disclaimer_message=disclaimer_message)
        for index, value in enumerate(values, start=1)
    ]
