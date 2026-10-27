from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit, urlunsplit

from griptape_nodes.files.path_utils import parse_static_server_url
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from collections.abc import Coroutine, Iterable

    from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
        PublicArtifactUrlParameter,
    )

logger = logging.getLogger("griptape_nodes")

# Enough to overlap round trips without one node saturating the user's upload bandwidth.
MAX_CONCURRENT_UPLOADS = 4


def local_path_for_static_server_url(url: str) -> str | None:
    """Map one of our own static-server URLs back to the file on disk, or None if it isn't one.

    The engine's parse_static_server_url does this for the `localhost` spelling only, while the
    static server is just as often reached by a loopback or LAN IP or a container name. So the
    host is swapped for `localhost` and the engine does the rest, `/workspace/` and `/external/`
    paths alike.

    Returns None unless the result is a file on this machine, so a static server running on a
    different host degrades to base64 rather than uploading whatever the path happens to hit.
    """
    # Not percent-encoded, so `#` is part of the filename, as the engine treats it.
    parsed = urlsplit(url, allow_fragments=False)
    if parsed.scheme not in ("http", "https"):
        return None
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError:
        return None
    netloc = f"localhost:{port}"
    candidate = parse_static_server_url(
        urlunsplit(parsed._replace(netloc=netloc)), GriptapeNodes.ConfigManager().workspace_path
    )
    return str(candidate) if candidate is not None and candidate.is_file() else None


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
