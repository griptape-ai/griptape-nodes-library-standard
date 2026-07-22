"""Resolve Griptape Cloud proxy config and check provider-asset (BytePlus private asset) access.

Provider-asset registration (used by Seedance 2.0 human-reference inputs) is an org-gated
feature. The only way to tell whether an org/API key may use it is to call the provider-asset
API and inspect the response, so this module exposes a small access check that probes
``GET proxy/v2/assets/<id>`` and classifies the result.

These helpers also centralize the proxy base-URL and API-key resolution so that nodes which are
not ``GriptapeProxyNode`` subclasses (e.g. the human-reference-asset DataNode) can reach the
proxy without duplicating that logic.
"""

from __future__ import annotations

import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urljoin

import httpx
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

logger = logging.getLogger("griptape_nodes")

__all__ = [
    "LICENSE_SECRET_NAME",
    "ProviderAssetAccess",
    "ProviderAssetAccessOutcome",
    "check_provider_asset_access",
    "resolve_proxy_api_key",
    "resolve_proxy_base",
]

# Secret name for the Griptape Cloud API key (mirrors GriptapeProxyNode.API_KEY_NAME).
API_KEY_NAME = "GT_CLOUD_API_KEY"

# Secret name for the Griptape Nodes License. The Griptape Cloud proxy accepts a License as a
# valid credential in addition to the API key, so a License-only user (no GT_CLOUD_API_KEY set)
# can still reach the proxy. When both are configured we prefer the License.
LICENSE_SECRET_NAME = "GRIPTAPE_NODES_LICENSE"

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


def resolve_proxy_api_key(secret_name: str = API_KEY_NAME) -> str | None:
    """Return the credential for proxy requests, or None if unavailable.

    Resolution order:

    1. ``GT_CLOUD_PROXY_API_KEY`` env var — an explicit override for the proxy credential that
       does not affect other engine systems using the ``secret_name`` secret.
    2. The Griptape Nodes License (``GRIPTAPE_NODES_LICENSE`` secret) — the proxy accepts a
       License as a valid credential, so a License-only user (no ``GT_CLOUD_API_KEY``) can still
       reach the proxy. When both a License and an API key are configured, the License wins.
    3. The Griptape Cloud API key (``secret_name``, ``GT_CLOUD_API_KEY`` by default).

    This does not touch BYOK (bring-your-own-key) provider credentials; those are resolved
    separately and take precedence when present.
    """
    proxy_key = os.getenv("GT_CLOUD_PROXY_API_KEY")
    if proxy_key:
        return proxy_key
    with suppress(Exception):
        license_key = GriptapeNodes.SecretsManager().get_secret(LICENSE_SECRET_NAME)
        if license_key:
            return license_key
    with suppress(Exception):
        return GriptapeNodes.SecretsManager().get_secret(secret_name)
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
    api_key = resolve_proxy_api_key()
    if not api_key:
        return ProviderAssetAccess(
            outcome=ProviderAssetAccessOutcome.INDETERMINATE,
            detail=f"Missing {API_KEY_NAME}. Set it in the environment/config to use provider-asset references.",
        )

    url = urljoin(resolve_proxy_base(), f"assets/{_PROBE_ASSET_ID}")
    headers = {"Authorization": f"Bearer {api_key}"}
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
            detail=f"{API_KEY_NAME} was rejected by Griptape Cloud (HTTP 401). Verify the key is valid.",
        )

    # Server error or any other unexpected status — surface it as-is, do not claim no-access.
    return ProviderAssetAccess(
        outcome=ProviderAssetAccessOutcome.INDETERMINATE,
        detail=f"Could not verify provider-asset access (HTTP {status} from Griptape Cloud).",
    )
