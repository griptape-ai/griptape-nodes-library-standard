"""Build the HTTP headers for Griptape Cloud requests that incur spend.

One factory owns the dict, so a header the platform wants on every billable call is
added once rather than once per call site. Control-plane requests that consume no
credits keep their own inline headers; ``tests/unit/utils/test_attribution_headers.py``
pins that split in both directions.

Callers resolve the credential and this builds: folding resolution in would make
``utils`` import ``proxy``, which already imports ``utils``.

Nodes that hand an ``api_key`` to a framework driver (``GriptapeCloudPromptDriver``,
``GriptapeCloudImageGenerationDriver``) get their ``Authorization`` built inside
``griptape``, so a header added here never reaches them.
"""

from __future__ import annotations

__all__ = ["build_attribution_headers"]


def build_attribution_headers(bearer_token: str) -> dict[str, str]:
    """Return the headers for one Griptape Cloud request.

    Args:
        bearer_token: The already-resolved credential. Not validated here, because the
            useful error names which credential sources were checked and only the caller
            knows that.

    Returns:
        dict[str, str]: A fresh dict; callers may mutate it freely.
            :meth:`GriptapeProxyNode._process_generation` threads the dict it gets back
            through poll and cancel, so a value that must differ between those three
            requests cannot be added here.
    """
    return {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
