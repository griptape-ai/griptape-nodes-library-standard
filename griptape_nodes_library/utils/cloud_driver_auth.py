"""Attach Griptape Cloud attribution to a ``griptape`` framework driver.

Most billable calls in this library are HTTP requests the library issues itself, and
those take their headers from :func:`build_attribution_headers`. The exception is a node
that constructs a framework driver -- ``GriptapeCloudPromptDriver``,
``GriptapeCloudImageGenerationDriver``, ``GriptapeCloudFileManagerDriver`` -- and hands it
an ``api_key``. That driver builds its own ``Authorization`` header inside ``griptape``,
so a header added to the factory never reaches the wire. This module is the bridge.

Passing ``headers=`` is the extension point, but it comes with a trap, and that trap is
why this returns both kwargs rather than letting each site spell them. The drivers
declare::

    api_key: str = field(default=Factory(lambda: os.environ["GT_CLOUD_API_KEY"]))
    headers: dict = field(default=Factory(lambda self: {...}, takes_self=True), kw_only=True)

attrs evaluates the default of every field the caller omits, so a site passing only
``headers=`` still runs ``os.environ["GT_CLOUD_API_KEY"]``. On a booted engine that lookup
does not even raise: ``register_all_secrets`` plants ``GT_CLOUD_API_KEY=""`` from the
engine's default ``secrets_to_register``, so the driver authenticates with an empty bearer
and Cloud answers 401 -- without ever consulting the License the user does have. The two
kwargs have to travel together.

The extra ``Content-Type: application/json`` that rides along from the factory is inert
here. ``requests.PreparedRequest.prepare_body`` sets that header only ``if content_type
and ("content-type" not in self.headers)``, so on every ``json=``-carrying driver path the
value we send is byte-identical to the one ``requests`` would have computed.

Two gaps this cannot close, both tracked in
https://github.com/griptape-ai/griptape-nodes-library-standard/issues/595:

- ``GriptapeCloudFileManagerDriver`` declares ``headers`` as ``init=False``, so it rejects
  the kwarg outright. That site assigns after construction instead; see
  :func:`griptape_nodes_library.utils.agent_utils.build_tool`.
- Neither ``api_key`` nor ``headers`` carries ``serializable`` metadata, so
  ``Agent.from_dict`` rebuilds a Cloud driver from ``os.environ`` with no attribution
  header at all. Every site that deserializes an agent is therefore still unattributed,
  and no amount of care at the construction sites reaches them.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes_library.utils.attribution_headers import build_attribution_headers
from griptape_nodes_library.utils.cloud_credential_utils import resolve_cloud_api_key

__all__ = ["cloud_driver_auth"]


def cloud_driver_auth(bearer_token: str | None = None) -> dict[str, Any]:
    """Return the ``api_key`` and ``headers`` kwargs for a Griptape Cloud driver.

    Spread it into the constructor::

        GriptapeCloudPromptDriver(model=model, stream=True, **cloud_driver_auth())

    Args:
        bearer_token: An already-resolved credential, for the sites that resolve one
            anyway so they can report a missing credential themselves. Omit it and the
            License-aware :func:`resolve_cloud_api_key` runs here. Passing ``""``
            explicitly is honored rather than re-resolved, so a site that has already
            decided the credential is absent keeps that answer.

    Returns:
        dict[str, Any]: ``api_key`` and ``headers``, both of which must be passed --
            see the module docstring for what happens when only one is. The dict and the
            ``headers`` inside it are fresh per call.
    """
    token = resolve_cloud_api_key() if bearer_token is None else bearer_token
    return {"api_key": token, "headers": build_attribution_headers(token)}
