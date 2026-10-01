"""Upload media through PublicArtifactUrlParameter without blocking the event loop."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable

    from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
        PublicArtifactUrlParameter,
    )

# Enough to overlap round trips without one node saturating the user's upload bandwidth.
MAX_CONCURRENT_UPLOADS = 4


async def aget_public_url(helper: PublicArtifactUrlParameter) -> str:
    """Return the helper's public URL, uploading off the event loop when the engine supports it."""
    # Engines without the async variant only offer the blocking call.
    aget = getattr(helper, "aget_public_url_for_parameter", None)
    if aget is None:
        return helper.get_public_url_for_parameter()
    return await aget()


async def adelete_uploaded_artifact(helper: PublicArtifactUrlParameter) -> None:
    """Delete the helper's upload, off the event loop when the engine supports it."""
    adelete = getattr(helper, "adelete_uploaded_artifact", None)
    if adelete is None:
        helper.delete_uploaded_artifact()
        return
    await adelete()


async def gather_limited[T](coros: Iterable[Coroutine[Any, Any, T]], limit: int = MAX_CONCURRENT_UPLOADS) -> list[T]:
    """Run coroutines at most `limit` at a time and return their results in order.

    The first failure cancels the rest and waits for them to stop before it is raised, so a
    caller's cleanup sees every upload that was started.
    """
    semaphore = asyncio.Semaphore(limit)

    async def run(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    try:
        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(run(coro)) for coro in coros]
    except ExceptionGroup as group_error:
        # Callers expect the failure itself, not a group of one.
        raise group_error.exceptions[0]  # noqa: B904 - chaining from the group would hide the original cause
    return [task.result() for task in tasks]
