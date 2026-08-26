"""Tests that ``SplatUrlArtifact`` exposes its URL as bare text.

``to_text()`` used to prefix the value with ``"Splat file URL: "``, so anything
consuming a splat output as text had to strip the prefix first — unlike the
sibling video/audio URL artifacts, which inherit ``UrlArtifact`` unchanged.
"""

from __future__ import annotations

import pytest

from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact

URLS = [
    "https://example.com/splat_full_res_v002.spz",
    "{outputs}/splat/splat_full_res_v002.spz",
]


@pytest.mark.parametrize("url", URLS)
def test_to_text_returns_bare_url(url: str) -> None:
    assert SplatUrlArtifact(value=url).to_text() == url


@pytest.mark.parametrize("url", URLS)
def test_str_returns_bare_url(url: str) -> None:
    assert str(SplatUrlArtifact(value=url)) == url


def test_to_dict_still_mirrors_meta_into_metadata() -> None:
    meta = {"resolution": "full_res"}
    data = SplatUrlArtifact(value=URLS[0], meta=meta).to_dict()

    assert data["value"] == URLS[0]
    assert data["metadata"] == meta
