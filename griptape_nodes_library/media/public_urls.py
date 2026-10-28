from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse
from uuid import uuid4

import httpx
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.media.urls import is_publicly_reachable_url

if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Iterable

    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")

# Generous enough for a provider-sized video over a slow link.
DOWNLOAD_TIMEOUT_SECONDS = 120

_MEDIA_MIME_PREFIXES = ("image/", "video/", "audio/")

# Used only when a downloaded URL reports no media type, so the upload still gets an extension.
_FALLBACK_MIME_TYPES = {
    "ImageUrlArtifact": "image/png",
    "VideoUrlArtifact": "video/mp4",
    "AudioUrlArtifact": "audio/wav",
}

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


def local_path_for_static_server_url(url: str) -> str | None:
    """Map one of our own static-server URLs back to the file on disk, or None if it isn't one.

    Mirrors StaticServerFileDriver._resolve_to_local_path, which does the same thing for the
    `localhost` spelling only: take the path after `/workspace/` and anchor it to the workspace
    directory. Query parameters are stripped because the static server appends a cachebuster.

    Returns None unless the result is a file on this machine, so a static server running on a
    different host is fetched like any other URL rather than uploading whatever the path happens to hit.
    """
    path = unquote(urlparse(url.split("?", maxsplit=1)[0]).path)
    if "/workspace/" not in path:
        return None
    workspace_relative = path.split("/workspace/", 1)[1]
    if not workspace_relative:
        return None
    workspace = GriptapeNodes.ConfigManager().workspace_path.resolve()
    candidate = (workspace / workspace_relative).resolve()
    # A `..` in the URL must not reach a file outside the workspace, which would then be uploaded.
    if not candidate.is_relative_to(workspace) or not candidate.is_file():
        return None
    return str(candidate)


async def adownload_as_data_uri(url: str, *, fallback_mime: str) -> str:
    """Fetch a URL and return its bytes as a data URI, typed by the response, the path, or the fallback."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, follow_redirects=True, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()
    for mime_type in (
        response.headers.get("content-type", "").split(";", 1)[0].strip(),
        mimetypes.guess_type(urlparse(url).path)[0] or "",
    ):
        if mime_type.startswith(_MEDIA_MIME_PREFIXES):
            break
    else:
        mime_type = fallback_mime
    return f"data:{mime_type};base64,{base64.b64encode(response.content).decode()}"


async def aupload_media_to_public_url(
    node: BaseNode,
    media_value: Any,
    *,
    artifact_type: str,
    pending_uploads: list[tuple[PublicArtifactUrlParameter, str]],
    passes_through: Callable[[str], bool] = is_publicly_reachable_url,
) -> str:
    """Return a URL a provider can fetch for the media, uploading it to Griptape Cloud when needed.

    A URL for which `passes_through` holds is returned untouched. Anything else is uploaded through a
    hidden scratch parameter, recorded in `pending_uploads` so the caller can delete the upload and
    remove the parameter once the run ends.

    The engine's upload opens with its own "is this already public?" test, which treats every http(s)
    URL not containing `localhost` as public and returns it without uploading. So a URL that reaches
    the upload is first turned into something the engine does upload: our own static server's URL
    into the file it serves, and any other URL into its downloaded bytes.

    Raises if the upload did not happen, rather than returning a URL the provider cannot fetch.
    """
    # Media arrives as a URL string, an artifact, or a serialized artifact dict.
    if isinstance(media_value, dict):
        url = media_value.get("value")
    else:
        url = getattr(media_value, "value", media_value)
    if not isinstance(url, str) or not url:
        msg = f"Could not get a public URL for a {type(media_value).__name__} value: it has no media path or URL."
        raise ValueError(msg)
    if passes_through(url):
        return url

    upload_value = url
    if url.startswith(("http://", "https://")):
        upload_value = local_path_for_static_server_url(url) or await adownload_as_data_uri(
            url, fallback_mime=_FALLBACK_MIME_TYPES.get(artifact_type, "application/octet-stream")
        )

    # Adding this scratch parameter during aprocess trips the strict-mode
    # "parameter-mutation-during-aprocess" warning. That is expected and harmless here: the
    # parameter is a transient, worker-local helper that only exists to feed the upload
    # (PublicArtifactUrlParameter reads its value locally) and is removed before the run ends,
    # so there is nothing for the orchestrator to stay in sync with.
    scratch_name = f"_media_upload_{uuid4().hex}"
    helper = PublicArtifactUrlParameter(
        node=node,
        artifact_url_parameter=Parameter(
            name=scratch_name,
            input_types=[artifact_type],
            type=artifact_type,
            default_value="",
            tooltip="",
            allowed_modes={ParameterMode.PROPERTY},
            hide=True,
            hide_property=True,
        ),
    )
    helper.add_input_parameters()
    pending_uploads.append((helper, scratch_name))
    node.set_parameter_value(scratch_name, upload_value)

    public_url = await helper.aget_public_url_for_parameter()

    # gtc_file_path is the engine's own record of having uploaded (delete_uploaded_artifact keys off
    # it). It stays None when the engine passed the value through unchanged, which a truthy return
    # would otherwise hide.
    if helper.gtc_file_path is None:
        msg = f"{url} was not uploaded. The engine treated it as already public, but the provider cannot fetch it."
        raise RuntimeError(msg)

    return public_url
