"""Resolve Griptape Cloud proxy config and check provider-asset (BytePlus private asset) access.

Provider-asset registration (used by Seedance 2.0 human-reference inputs) is an org-gated
feature. The only way to tell whether an org/API key may use it is to call the provider-asset
API and inspect the response, so this module exposes a small access check that probes
``GET proxy/v2/assets/<id>`` and classifies the result.

These helpers also centralize the proxy base-URL and API-key resolution so that nodes which are
not ``GriptapeProxyNode`` subclasses (e.g. the human-reference-asset DataNode) can reach the
proxy without duplicating that logic. Resolution reports what it found at each source, so a
failure message can name both credentials a user configures (see
:func:`missing_proxy_credential_message`).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
from urllib.parse import urljoin

import httpx
from griptape_nodes.drivers.cloud_credentials import MISSING_CREDENTIAL_MESSAGE
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "LICENSE_SECRET_NAME",
    "PROXY_API_KEY_ENV_VAR",
    "ProviderAssetAccess",
    "ProviderAssetAccessOutcome",
    "ProxyCredential",
    "check_provider_asset_access",
    "missing_proxy_credential_message",
    "resolve_proxy_base",
    "resolve_proxy_credential",
]

# Secret name for the Griptape Cloud API key (mirrors GriptapeProxyNode.API_KEY_NAME).
API_KEY_NAME = "GT_CLOUD_API_KEY"

# Secret name for the Griptape Nodes License. The Griptape Cloud proxy accepts a License as a
# valid credential in addition to the API key, so a License-only user (no GT_CLOUD_API_KEY set)
# can still reach the proxy. When both are configured we prefer the License.
LICENSE_SECRET_NAME = "GRIPTAPE_NODES_LICENSE"

# Env var that overrides the credential used for proxy requests only, leaving other engine
# systems that read GT_CLOUD_API_KEY untouched.
PROXY_API_KEY_ENV_VAR = "GT_CLOUD_PROXY_API_KEY"

# Probe asset id used purely to reach the provider-asset handler. It is not expected to exist;
# an access-granted org returns a 404 "provider asset not found" for it.
_PROBE_ASSET_ID = "griptape-access-probe-0000"
_ASSET_NOT_FOUND_MARKER = "provider asset not found"
_ACCESS_CHECK_TIMEOUT = 10  # seconds; keep short so it never blocks graph load for long


class ProviderAssetAccessOutcome(Enum):
    """How the access probe resolved.

    GRANTED — the request reached the provider-asset handler (the entitlement gate is applied
    before the handler runs, so reaching it proves access): HTTP 200, or 404 with the
    "provider asset not found" marker.

    DENIED — the org is not entitled to the feature. The backend gates this specifically with
    HTTP 403 (feature flag off, model-proxy entitlement missing, or a license policy denial).
    This is the only outcome that should tell the user to request access from Foundry.

    INDETERMINATE — the probe could not determine entitlement: a missing key, an auth error
    (401), a server error (5xx), a network/timeout failure, or any unexpected status. The cause
    is real but is NOT a no-access signal, so callers should surface the underlying error rather
    than claim the org lacks access, and should not block on it alone.
    """

    GRANTED = "granted"
    DENIED = "denied"
    INDETERMINATE = "indeterminate"


def resolve_proxy_base() -> str:
    """Return the proxy v2 base URL (``.../api/proxy/v2/``).

    GT_CLOUD_PROXY_BASE_URL overrides just the proxy without affecting other engine systems
    that use GT_CLOUD_BASE_URL.
    """
    base = os.getenv("GT_CLOUD_PROXY_BASE_URL") or os.getenv("GT_CLOUD_BASE_URL", "https://cloud.griptape.ai")
    base_slash = base if base.endswith("/") else base + "/"
    api_base = urljoin(base_slash, "api/")
    return urljoin(api_base, "proxy/v2/")


@dataclass(frozen=True)
class ProxyCredential:
    """The credential to send to the proxy, plus what the lookup saw along the way.

    ``value`` is None when no source held a usable credential, and ``source`` names the source
    that supplied it otherwise.

    ``blank_sources`` names sources that held a value that cannot be used as a credential (empty
    or whitespace only). It exists because the two states are indistinguishable from ``value``
    alone: ``SecretsManager.get_secret`` returns on presence rather than truthiness, so a secret
    registered as ``""`` is a hit that resolves to nothing, and both it and a never-set secret
    arrive at the call site as a falsy ``value``. A blank secret is reported as present by the
    config file, the Settings UI, and the DEBUG log, so the failure message is the only place a
    user can learn it is the problem.
    """

    value: str | None
    source: str | None = None
    blank_sources: tuple[str, ...] = ()


def resolve_proxy_credential(secret_name: str = API_KEY_NAME) -> ProxyCredential:
    """Resolve the credential for proxy requests and report what each source held.

    Resolution order, first usable value wins:

    1. ``GT_CLOUD_PROXY_API_KEY`` env var — an explicit override for the proxy credential that
       does not affect other engine systems using the ``secret_name`` secret.
    2. The Griptape Nodes License (``GRIPTAPE_NODES_LICENSE`` secret) — the proxy accepts a
       License as a valid credential, so a License-only user (no ``GT_CLOUD_API_KEY``) can still
       reach the proxy. When both a License and an API key are configured, the License wins.
    3. The Griptape Cloud API key (``secret_name``, ``GT_CLOUD_API_KEY`` by default).

    A source holding an empty or whitespace-only value is skipped rather than returned: sending
    it would buy a 401 from Griptape Cloud instead of a message naming the credentials that
    would actually work. Each such source is recorded in ``ProxyCredential.blank_sources``.

    This does not touch BYOK (bring-your-own-key) provider credentials; those are resolved
    separately and take precedence when present.
    """
    blank_sources: list[str] = []
    # Fetchers stay lazy so a winning source short-circuits the ones below it, mirroring the
    # search order in SecretsManager.get_secret.
    search_order: tuple[tuple[str, Callable[[], str | None]], ...] = (
        (PROXY_API_KEY_ENV_VAR, lambda: os.getenv(PROXY_API_KEY_ENV_VAR)),
        (LICENSE_SECRET_NAME, lambda: _read_secret(LICENSE_SECRET_NAME)),
        (secret_name, lambda: _read_secret(secret_name)),
    )
    for source, fetch in search_order:
        raw_value = fetch()
        if raw_value is None:
            continue
        value = raw_value.strip()
        if value:
            return ProxyCredential(value=value, source=source, blank_sources=tuple(blank_sources))
        blank_sources.append(source)
    return ProxyCredential(value=None, blank_sources=tuple(blank_sources))


def missing_proxy_credential_message(credential: ProxyCredential, *, attempted: str) -> str:
    """Build the user-facing message for a proxy call with no usable credential.

    Names both credentials a user configures -- the License and the API key -- so a License-only
    user is not sent after an API key they are not meant to have, and names any source that is
    configured but blank. ``GT_CLOUD_PROXY_API_KEY`` is deliberately not offered as a remedy:
    it is an override for pointing the proxy at other infrastructure, not a credential users are
    meant to set. It still appears when it is one of the blank sources, since a blank override
    silently costs the user their configured credential.

    Args:
        credential: The failed resolution, from :func:`resolve_proxy_credential`.
        attempted: What the caller was trying to do, as a sentence fragment starting with a verb
            -- e.g. ``"run Nano Banana Image Generation"``.
    """
    reason = MISSING_CREDENTIAL_MESSAGE
    if credential.blank_sources:
        names = ", ".join(credential.blank_sources)
        if len(credential.blank_sources) == 1:
            reason += f" {names} is set to a blank value, which does not count as configured."
        else:
            reason += f" {names} are set to blank values, which do not count as configured."
    return f"Attempted to {attempted}. Failed because {reason}"


def _read_secret(secret_name: str) -> str | None:
    """Return a secret's raw value, None when it is absent or cannot be read.

    Absence is routine here — an API-key user has no License and a License-only user has no API
    key — so ``should_error_on_not_found=False`` keeps an ordinary lookup out of the error log.
    A SecretsManager failure is also reported as absence: the caller's job is to describe what
    it found, and letting an unrelated exception escape would replace the credential message
    with a worse one.
    """
    try:
        return GriptapeNodes.SecretsManager().get_secret(secret_name, should_error_on_not_found=False)
    except Exception as e:
        logger.debug("Could not read secret '%s': %s", secret_name, e)
        return None


@dataclass
class ProviderAssetAccess:
    """Result of a provider-asset access probe.

    `outcome` classifies the probe (see `ProviderAssetAccessOutcome`); `detail` is a
    human-readable explanation. `has_access` is a convenience for "the org may use the feature"
    (GRANTED only). Callers distinguish DENIED (block + tell the user to request access) from
    INDETERMINATE (surface the underlying error; do not assert no-access).
    """

    outcome: ProviderAssetAccessOutcome
    detail: str

    @property
    def has_access(self) -> bool:
        return self.outcome is ProviderAssetAccessOutcome.GRANTED

    @property
    def is_denied(self) -> bool:
        return self.outcome is ProviderAssetAccessOutcome.DENIED


def check_provider_asset_access() -> ProviderAssetAccess:
    """Probe ``GET proxy/v2/assets/<probe>`` and classify whether the org may use the feature.

    The backend applies the entitlement gate before the asset handler, so the probe distinguishes
    three outcomes (see `ProviderAssetAccessOutcome`): GRANTED (reached the handler: 200, or 404
    with the not-found marker), DENIED (HTTP 403 entitlement gate), or INDETERMINATE (missing key,
    401 auth, 5xx server error, network/timeout, or any unexpected status). Only DENIED means the
    org lacks access; INDETERMINATE surfaces the real cause without claiming no-access.
    """
    credential = resolve_proxy_credential()
    if not credential.value:
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.INDETERMINATE,
            detail=missing_proxy_credential_message(credential, attempted="use provider-asset references"),
        )

    url = urljoin(resolve_proxy_base(), f"assets/{_PROBE_ASSET_ID}")
    headers = {"Authorization": f"Bearer {credential.value}"}
    try:
        response = httpx.get(url, headers=headers, timeout=_ACCESS_CHECK_TIMEOUT)
    except Exception as e:  # network/timeout — cause is real but not a no-access signal.
        logger.info("Provider-asset access probe failed to reach %s: %s", url, e)
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.INDETERMINATE,
            detail=f"Could not reach the Griptape Cloud provider-asset API ({e}). Check connectivity and try again.",
        )

    status = response.status_code
    body = response.text or ""

    # Reached the provider-asset handler -> access granted. A 404 must carry the asset-not-found
    # marker: an entitled org's probe of a non-existent asset returns 404 "provider asset not
    # found", whereas no entitlement is gated upstream with 403. Matching the marker (rather than
    # any 404) avoids mistaking some other 404 for access.
    if status == httpx.codes.OK:
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.GRANTED, detail="Provider-asset access confirmed."
        )
    if status == httpx.codes.NOT_FOUND and _ASSET_NOT_FOUND_MARKER in body.lower():
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.GRANTED, detail="Provider-asset access confirmed."
        )

    # The entitlement gate denies with 403 specifically (feature flag off, model-proxy
    # entitlement missing, or a license policy denial). This is the only no-access outcome.
    if status == httpx.codes.FORBIDDEN:
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.DENIED,
            detail=(
                "This organization does not have access to provider-asset references. "
                "An admin needs to request access to this feature from Foundry."
            ),
        )

    # Auth error — a real, distinct cause, but not an entitlement signal.
    if status == httpx.codes.UNAUTHORIZED:
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.INDETERMINATE,
            detail=(
                f"The Griptape Cloud credential from {credential.source} was rejected (HTTP 401). "
                "Verify that it is valid."
            ),
        )

    # Server error or any other unexpected status — surface it as-is, do not claim no-access.
    return ProviderAssetAccess(
        outcome=ProviderAssetAccessOutcome.INDETERMINATE,
        detail=f"Could not verify provider-asset access (HTTP {status} from Griptape Cloud).",
    )
