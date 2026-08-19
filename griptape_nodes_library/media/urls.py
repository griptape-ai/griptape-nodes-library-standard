"""Shared URL classification helpers for media handed to third-party providers.

A provider fetches a media URL from its own infrastructure, so the only URLs worth
passing through untouched are the ones reachable from the public internet. Everything
else -- the static server's own address, a LAN host, a container name -- has to be
uploaded somewhere the provider can actually reach.
"""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

__all__ = ["is_public_https_domain_url"]


def is_public_https_domain_url(url: str) -> bool:
    """Return True only for URLs a third-party provider can fetch: https:// with a domain name.

    Rejects http://, bare IPs (0.0.0.0, 192.168.x.x, 127.0.0.1, ::1), and single-label
    hostnames (localhost, container names) -- all of which must upload to Griptape Cloud
    instead.

    A substring test for ``"localhost"`` is not enough: ``STATIC_SERVER_HOST`` /
    ``static_server_base_url`` are explicitly overridable for tunnels and reverse proxies,
    so the static server's address can be a bare IP or a container name just as easily.
    Passing one of those to a provider both fails the fetch and discloses an internal host.
    """
    if not url.startswith("https://"):
        return False
    hostname = urlparse(url).hostname or ""
    try:
        ipaddress.ip_address(hostname)
        return False
    except ValueError:
        pass
    return "." in hostname
