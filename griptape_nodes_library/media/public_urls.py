from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable

    from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
        PublicArtifactUrlParameter,
    )

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
