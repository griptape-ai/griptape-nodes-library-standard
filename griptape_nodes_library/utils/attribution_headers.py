"""Build the HTTP headers for Griptape Cloud requests that incur spend.

Griptape Cloud is adding org-level budgets and usage auditing, so every
credit-consuming call will have to carry an attribution header naming the project
chain the spend belongs to. Spelled at each call site that is an N-site edit --
the kind that lands on N-2 of them and silently under-reports the rest. This
module owns the dict instead.

Despite the name, the header it emits today is ``Authorization`` (plus
``Content-Type``); ``X-Griptape-Attribution`` lands here with L2, once the engine
exposes the context to build it from. The module is named for the reason it exists
rather than its current contents, so this is also the one place the bearer token is
spelled.

Scope is the billable calls only. Control-plane requests that consume no credits
keep their own inline headers: there is no usage to attribute, so routing them here
would buy nothing and would imply the attribution header belongs on them.
``tests/unit/utils/test_attribution_headers.py`` pins that split in both directions.

Callers resolve the credential; this builds. Folding resolution in would make
``utils`` import ``proxy``, which already imports ``utils``.

``X-GTC-PROXY-AUTH-INFO`` deliberately stays at the submit site rather than being
read here: it carries the user's own provider key, and submit is the only call that
dispatches to the provider.

The "one line here" claim covers calls this library makes over HTTP itself. A node
that hands an ``api_key`` to a framework driver (``GriptapeCloudPromptDriver``,
``GriptapeCloudImageGenerationDriver``) gets its ``Authorization`` header built
inside ``griptape``, and a header added here never reaches it.
"""

from __future__ import annotations

__all__ = ["build_attribution_headers"]


def build_attribution_headers(bearer_token: str, *, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Return the headers for one Griptape Cloud request.

    The single place ``Authorization`` is spelled, so a header the platform wants on
    every billable call is added once here. Attribution is the next one to land.

    Args:
        bearer_token: The already-resolved credential. Not validated here, because the
            useful error names which credential sources were checked and only the caller
            knows that. Every caller today arrives through
            :meth:`GriptapeProxyNode._validate_api_key`, which raises first.
        extra: Per-request headers, merged last so a caller can also override a default.
            A dict rather than ``**kwargs`` because the one real case,
            ``X-GTC-PROXY-AUTH-INFO``, is not a valid Python identifier.

    Returns:
        dict[str, str]: A fresh dict; callers may mutate it freely. Note that
            :meth:`GriptapeProxyNode._process_generation` threads the dict it gets back
            through poll and cancel rather than rebuilding it, so a value that must
            differ between those three requests cannot be added here.
    """
    headers = {"Authorization": f"Bearer {bearer_token}", "Content-Type": "application/json"}
    if extra:
        headers.update(extra)
    return headers
