"""Shared URL classification helpers for media handed to third-party providers.

A provider fetches a media URL from its own infrastructure, so the only URLs worth
passing through untouched are the ones reachable from the public internet. Everything
else -- the static server's own address, a LAN host, a container name -- has to be
uploaded somewhere the provider can actually reach.

Two predicates, because two different questions get asked:

* ``is_publicly_reachable_url`` -- "can a third party fetch this at all?" Scheme-agnostic
  across http and https. This is the question at a pass-through decision.
* ``is_public_https_domain_url`` -- "will a provider that mandates HTTPS accept this?"
  LTX does mandate it, so its inputs are held to the stricter bar.

Conflating the two is a bug in both directions: the strict predicate rejects a plain
http URL a provider could have fetched, and the loose one would hand an HTTPS-only
provider a URL it refuses.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

__all__ = ["is_public_https_domain_url", "is_publicly_reachable_url"]


def _has_public_host(url: str) -> bool:
    """Return True if the URL's host is a domain name resolvable from the public internet.

    Rejects bare IPs (0.0.0.0, 192.168.x.x, 127.0.0.1, ::1) and single-label hostnames
    (localhost, container names). A substring test for ``"localhost"`` is not enough:
    ``STATIC_SERVER_HOST`` / ``static_server_base_url`` are explicitly overridable for
    tunnels and reverse proxies, so the static server's address is a loopback or LAN IP
    or a bare container name in ordinary deployments. Handing one of those to a provider
    both fails the fetch and discloses an internal host.

    Every bare IP is rejected, including a routable one. Nothing in this library hosts
    media on a naked IP, and telling a public IP from a private one correctly is more
    surface than the case is worth.
    """
    hostname = urlparse(url).hostname or ""
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return "." in hostname
    return False


def is_publicly_reachable_url(url: str) -> bool:
    """Return True for an http(s) URL a third-party provider can fetch from its own network."""
    return url.startswith(("http://", "https://")) and _has_public_host(url)


def is_public_https_domain_url(url: str) -> bool:
    """Return True only for https on a public domain, as HTTPS-only providers require.

    Stricter than ``is_publicly_reachable_url`` by rejecting plain http. Use it only where
    the provider itself mandates HTTPS (LTX does); using it as a general "can this be
    fetched" test needlessly uploads or rejects a perfectly fetchable http URL.
    """
    return url.startswith("https://") and _has_public_host(url)
