"""Attach Griptape Cloud attribution to a ``griptape`` framework driver.

Calls this library issues itself take their headers from
:func:`build_griptape_cloud_headers`. A node that instead hands an ``api_key`` to a framework
driver cannot: the driver builds its own ``Authorization`` header inside ``griptape``, so a
factory header never reaches the wire. ``headers=`` is the way in.

**Both kwargs have to travel together.** attrs evaluates the default of every field the caller
omits, and ``api_key`` defaults to ``os.environ["GT_CLOUD_API_KEY"]`` -- which a booted engine
plants as ``""`` via ``register_all_secrets``. A site passing only ``headers=`` therefore
authenticates with an empty bearer and takes a 401, never consulting the user's License.

The factory's ``Content-Type`` is inert here: ``requests`` would set it anyway for a ``json=``
body. The exception is ``GriptapeCloudFileManagerDriver``'s bodyless requests, which now carry
one -- meaningless rather than wrong, since neither declares a body.

Two sites this does not reach, both covered elsewhere:

- ``GriptapeCloudFileManagerDriver`` declares ``headers`` ``init=False`` and rejects the kwarg;
  :func:`griptape_nodes_library.utils.agent_utils.build_tool_from_config` assigns it after
  construction.
- An agent rebuilt from a saved dict calls no constructor;
  :func:`griptape_nodes_library.utils.agent_utils._restored_cloud_credentials` writes both
  values into the serialized dict before ``from_dict``.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes_library.utils.cloud_credential_utils import resolve_cloud_api_key
from griptape_nodes_library.utils.griptape_cloud_headers import build_griptape_cloud_headers

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
    return {"api_key": token, "headers": build_griptape_cloud_headers(token, attribution=True)}
