"""Build the HTTP headers for a Griptape Cloud request.

One factory owns the dict, so a header the platform wants on every billable call is
added once rather than once per call site. ``attribution`` says whether this particular
call incurs spend: control-plane requests that consume no credits pass ``False``, and
get everything except the attribution header. It is keyword-only and has no default --
a call that omits it fails at the call, which is the only failure mode available here
that is louder than under-reporting the spend.

Callers resolve the credential and this builds: folding resolution in would make
``utils`` import ``proxy``, which already imports ``utils``.

Nodes that hand an ``api_key`` to a framework driver (``GriptapeCloudPromptDriver``,
``GriptapeCloudImageGenerationDriver``, ``GriptapeCloudFileManagerDriver``) get their
``Authorization`` built inside ``griptape``. :mod:`griptape_nodes_library.utils.cloud_driver_auth`
is the bridge: it passes this dict in as the driver's ``headers``, replacing the framework's own.
``Agent.from_dict`` would rebuild a driver from ``os.environ`` with no headers at all;
:func:`~griptape_nodes_library.utils.agent_utils._restored_cloud_credentials` injects both
into the serialized dict before ``from_dict`` reads it.

Two spellings of the same factory, one sync and one async, because the attribution
lookup is a round trip to the engine and five call sites make it from a coroutine.
:func:`build_griptape_cloud_headers_async` is the one to reach for inside an ``async
def``; the sync spelling parks its thread for the length of the lookup, which on an event
loop stalls every other task scheduled there. They are otherwise the same function and
must stay so -- everything that differs between them lives in :mod:`.attribution`.
"""

from __future__ import annotations

from griptape_nodes_library.utils.attribution import attribution_header, attribution_header_async

__all__ = ["build_griptape_cloud_headers", "build_griptape_cloud_headers_async"]


# Spelled once, and once in the library: `test_only_the_factory_builds_an_authorization_header`
# reads every module for an inline `Authorization` build and requires this file to be the only
# hit. Both public factories go through here so neither can drift from the other.
def _base_headers(bearer_token: str) -> dict[str, str]:
    """The half of the dict that needs no engine round trip."""
    return {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}


def build_griptape_cloud_headers(bearer_token: str, *, attribution: bool) -> dict[str, str]:
    """Return the headers for one Griptape Cloud request.

    For synchronous callers. From inside a coroutine use
    :func:`build_griptape_cloud_headers_async`, which does not block the loop while the
    engine answers.

    Args:
        bearer_token: The already-resolved credential. Not validated here, because the
            useful error names which credential sources were checked and only the caller
            knows that.
        attribution: Whether this call incurs spend. ``True`` asks the engine which project
            the spend belongs to and adds the header naming it -- see
            :func:`~griptape_nodes_library.utils.attribution.attribution_header`, which
            answers ``{}`` rather than raising when it cannot, so a call is never failed to
            protect a reporting field. Pass ``False`` only for a request that consumes no
            credits: there is no spend to attribute, and the header would assert otherwise.

    Returns:
        dict[str, str]: A fresh dict; callers may mutate it freely -- and one per request,
            not one per node run. A caller whose requests differ in what they attribute
            builds twice rather than threading one dict through both; see
            :meth:`GriptapeProxyNode._process_generation`, which pairs an attributed submit
            with an unattributed poll.
    """
    headers = _base_headers(bearer_token)
    if attribution:
        # Merged here rather than passed in by callers: a parameter would put the header
        # back in the hands of the thirteen call sites, which is the edit this module exists
        # to prevent. `{}` when the engine has no answer, so the merge is a no-op.
        headers |= attribution_header()
    return headers


async def build_griptape_cloud_headers_async(bearer_token: str, *, attribution: bool) -> dict[str, str]:
    """Await the headers for one Griptape Cloud request.

    Identical to :func:`build_griptape_cloud_headers` in what it returns; the difference
    is that the attribution lookup yields the event loop instead of parking it. Every
    coroutine that builds Cloud headers should call this one, including the ones passing
    ``attribution=False``: that flag is a fact about today's endpoint and flipping it is a
    one-word edit, so a site on the sync spelling is a loop stall waiting to be introduced
    by an unrelated change.

    Args:
        bearer_token: As :func:`build_griptape_cloud_headers`.
        attribution: As :func:`build_griptape_cloud_headers`.

    Returns:
        dict[str, str]: As :func:`build_griptape_cloud_headers`.
    """
    headers = _base_headers(bearer_token)
    if attribution:
        headers |= await attribution_header_async()
    return headers
