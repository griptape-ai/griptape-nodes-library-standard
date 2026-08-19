"""Tests for image_utils helpers."""

from __future__ import annotations

from griptape.artifacts import ImageUrlArtifact

from griptape_nodes_library.utils.image_utils import extract_image_url


class TestExtractImageUrl:
    """The falsy-on-missing contract callers rely on.

    Callers guard with ``if not extract_image_url(...)``. Returning the string
    ``"None"`` for a missing image would slip past that guard and only fail later
    with an opaque AttributeError, so absence must come back falsy.
    """

    def test_none_returns_empty_string(self) -> None:
        assert extract_image_url(None) == ""

    def test_none_is_falsy_not_the_literal_none_string(self) -> None:
        result = extract_image_url(None)
        assert not result
        assert result != "None"

    def test_dict_with_none_value_returns_empty_string(self) -> None:
        assert extract_image_url({"value": None}) == ""

    def test_artifact_returns_its_url(self) -> None:
        assert extract_image_url(ImageUrlArtifact("https://example.com/a.png")) == "https://example.com/a.png"

    def test_dict_returns_its_value(self) -> None:
        assert extract_image_url({"value": "https://example.com/b.png"}) == "https://example.com/b.png"

    def test_string_passes_through(self) -> None:
        assert extract_image_url("https://example.com/c.png") == "https://example.com/c.png"

    def test_empty_string_stays_empty(self) -> None:
        assert extract_image_url("") == ""
