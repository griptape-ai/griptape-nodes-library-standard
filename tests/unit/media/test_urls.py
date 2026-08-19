"""Tests for the shared public-URL predicate.

``is_public_https_domain_url`` decides whether a media URL can be handed to a
third-party provider as-is or has to be uploaded to Griptape Cloud first. The
provider fetches from its own infrastructure, so the only safe pass-through is
https on a resolvable domain name. Anything pointing at our own static server
must upload -- and that address is not reliably spelled ``localhost``, since
``STATIC_SERVER_HOST`` / ``static_server_base_url`` are overridable for tunnels
and reverse proxies.
"""

from __future__ import annotations

import pytest

from griptape_nodes_library.media import is_public_https_domain_url, is_publicly_reachable_url


@pytest.mark.parametrize(
    "url",
    [
        "https://public.example/style.png",
        "https://cloud.griptape.ai/api/assets/style.png?token=abc",
        "https://sub.domain.example.com:8443/a/b.mp4",
    ],
)
def test_public_https_domain_urls_are_accepted(url: str) -> None:
    assert is_public_https_domain_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        # http is never a pass-through, even on a real domain.
        "http://insecure.example.com/style.png",
        # Loopback, unspecified and LAN addresses: a bare IP is not fetchable from outside.
        "https://127.0.0.1:8124/static/style.png",
        "https://0.0.0.0:8124/static/style.png",
        "https://192.168.1.20:8124/static/style.png",
        "https://[::1]:8124/static/style.png",
        # Single-label hostnames: localhost and container/service names.
        "https://localhost:8124/static/style.png",
        "https://my-container:8124/static/style.png",
        # The static server's default spelling, which is http on localhost.
        "http://localhost:8124/workspace/static_files/style.png",
        # Not a URL at all: a filesystem path or a data URI has to upload or inline.
        "/Users/me/style.png",
        "{inputs}/style.png",
        "data:image/png;base64,AAAA",
        "",
    ],
)
def test_unreachable_urls_are_rejected(url: str) -> None:
    assert is_public_https_domain_url(url) is False


# ---------------------------------------------------------------------------
# is_publicly_reachable_url -- the scheme-agnostic "can a provider fetch it" test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://public.example/style.png",
        # The distinction that matters: plain http on a real domain is fetchable, so a provider
        # that does not mandate HTTPS can be handed it directly.
        "http://insecure.example.com/style.png",
        "http://sub.domain.example.com:8080/a/b.mp4",
    ],
)
def test_reachable_urls_are_accepted(url: str) -> None:
    assert is_publicly_reachable_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8124/workspace/static_files/style.png",
        "http://127.0.0.1:8124/static/style.png",
        "http://0.0.0.0:8124/static/style.png",
        "http://192.168.1.20:8124/static/style.png",
        "https://192.168.1.20:8124/static/style.png",
        "http://[::1]:8124/static/style.png",
        "http://my-container:8124/static/style.png",
        "/Users/me/style.png",
        "{inputs}/style.png",
        "data:image/png;base64,AAAA",
        "",
    ],
)
def test_unreachable_urls_are_rejected_regardless_of_scheme(url: str) -> None:
    assert is_publicly_reachable_url(url) is False


def test_https_bar_is_stricter_than_the_reachability_bar() -> None:
    # The two predicates differ on exactly one axis, and conflating them is what caused a plain
    # http URL to be routed to an upload that never happened.
    public_http_url = "http://insecure.example.com/style.png"

    assert is_publicly_reachable_url(public_http_url) is True
    assert is_public_https_domain_url(public_http_url) is False
