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
``GriptapeCloudImageGenerationDriver``) get their ``Authorization`` built inside
``griptape``, so a header added here never reaches them.
"""

from __future__ import annotations

__all__ = ["build_griptape_cloud_headers"]


def build_griptape_cloud_headers(bearer_token: str, *, attribution: bool) -> dict[str, str]:
    """Return the headers for one Griptape Cloud request.

    Args:
        bearer_token: The already-resolved credential. Not validated here, because the
            useful error names which credential sources were checked and only the caller
            knows that.
        attribution: Whether this call incurs spend. The two branches return the same dict
            today; the attribution header itself lands under #601, and this is the seam it
            lands on. Pass ``False`` only for a request that consumes no credits.

    Returns:
        dict[str, str]: A fresh dict; callers may mutate it freely.
            :meth:`GriptapeProxyNode._process_generation` threads the dict it gets back
            through poll and cancel, so a value that must differ between those three
            requests cannot be added here.
    """
    return {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
