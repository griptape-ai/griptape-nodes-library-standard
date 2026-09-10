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
"""

from __future__ import annotations

from griptape_nodes_library.utils.attribution import attribution_header

__all__ = ["build_griptape_cloud_headers"]


def build_griptape_cloud_headers(bearer_token: str, *, attribution: bool) -> dict[str, str]:
    """Return the headers for one Griptape Cloud request.

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
        dict[str, str]: A fresh dict; callers may mutate it freely.
            :meth:`GriptapeProxyNode._process_generation` threads the dict it gets back
            through poll and cancel, so a value that must differ between those three
            requests cannot be added here.
    """
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    if attribution:
        # Merged here rather than passed in by callers: a parameter would put the header
        # back in the hands of the eleven call sites, which is the edit this module exists
        # to prevent. `{}` when the engine has no answer, so the merge is a no-op.
        headers |= attribution_header()
    return headers
