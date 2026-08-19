"""Resolve the bearer credential for Griptape Cloud, License or API key.

Griptape Cloud accepts two kinds of credential: a Griptape Cloud API key
(``GT_CLOUD_API_KEY``, a ``gt-`` prefixed key) and a Griptape Nodes License
(``GRIPTAPE_NODES_LICENSE``, a JWT the desktop app writes into the engine's global
``.env``). A license-only user has no API key at all, so any node that reads
``GT_CLOUD_API_KEY`` directly is unusable for them -- it fails validation with
"GT_CLOUD_API_KEY is not defined" and points them at a knob they are not meant to set.

The endpoints these nodes call all authenticate a License via the control plane's
``LicenseAuthMixin``: ``api/chat/messages`` and ``api/chat/messages/stream`` (prompt
drivers), ``api/images/generations`` and ``api/images/variations`` (image drivers), and
``api/buckets`` (the FileManager tool).

This wraps the engine's ``resolve_cloud_credential`` rather than reimplementing the
precedence, and adds the ``""`` coercion the driver call sites need. Two things it is
deliberately NOT for:

- **BYOK provider keys.** ``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, and friends are
  the user's own provider credentials; a Griptape License says nothing about them.
- **The model proxy.** Proxy nodes resolve through ``resolve_proxy_credential`` in
  ``griptape_nodes_library.proxy.provider_asset_access``, which additionally honors the
  proxy-scoped ``GT_CLOUD_PROXY_API_KEY`` override.
"""

from __future__ import annotations

from griptape_nodes.drivers.cloud_credentials import (
    MISSING_CREDENTIAL_MESSAGE,
    resolve_cloud_credential,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

__all__ = [
    "MISSING_CREDENTIAL_MESSAGE",
    "missing_credential_message",
    "resolve_cloud_api_key",
]

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
"""Secret holding the Griptape Cloud API key. The License is checked first."""


def resolve_cloud_api_key() -> str:
    """Return the Griptape Cloud bearer credential, preferring a License.

    Returns ``""`` rather than ``None`` when neither credential is set, because the
    driver fields this feeds are typed ``str`` and interpolated straight into an
    ``Authorization`` header. Callers must not treat ``""`` as their failure signal:
    report a missing credential from ``validate_before_workflow_run`` (see
    :func:`missing_credential_message`), which runs before any request is sent.
    """
    return resolve_cloud_credential(GriptapeNodes.SecretsManager(), secret_name=API_KEY_ENV_VAR) or ""


def missing_credential_message(attempted: str) -> str:
    """Build the user-facing "no Cloud credential" message for a failed action.

    Names both credentials, so a license-only user is not sent after an API key they
    are not meant to have.

    Args:
        attempted: What the node was trying to do, as a sentence fragment starting
            with a verb -- e.g. ``"run the Agent"`` or ``"describe an image"``.
    """
    return f"Attempted to {attempted}. Failed because {MISSING_CREDENTIAL_MESSAGE}"
