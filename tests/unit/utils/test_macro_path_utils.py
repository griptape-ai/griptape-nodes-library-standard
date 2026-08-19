"""Tests for resolve_to_macro_path URL / filesystem handling."""

from griptape_nodes_library.utils.macro_path_utils import resolve_to_macro_path


class TestResolveToMacroPathUrls:
    """Remote URLs must be treated as external without touching the filesystem.

    Regression coverage for the bug where an S3 URL fed into a Load node was
    resolved as a local path (joined onto the cwd, "//" collapsed to "/"), and
    stat()-ing the resulting long single filename raised OSError (ENAMETOOLONG)
    instead of degrading to "external".
    """

    def test_https_url_is_external_and_unchanged(self) -> None:
        url = "https://example.com/image.png"
        result = resolve_to_macro_path(url)
        assert result.is_external is True
        assert result.resolved_path == url

    def test_http_url_is_external_and_unchanged(self) -> None:
        url = "http://example.com/image.png"
        result = resolve_to_macro_path(url)
        assert result.is_external is True
        assert result.resolved_path == url

    def test_long_signed_s3_url_does_not_raise(self) -> None:
        """A long signed URL previously crashed with 'File name too long'."""
        url = (
            "https://sg-media-usor-01.s3-accelerate.amazonaws.com/"
            "7d0ea73011daa9d36f8c3a0bdff31f7fbad21fe3/"
            "039b502dda0f485a220d380347e4fee15cba8a38/"
            "e39081a6093f10bd_image_v026_t.jpg?response-content-disposition="
            "filename%3D%22e39081a6093f10bd_image_v026_t.jpg%22"
            "&x-amz-meta-user-id=225&X-Amz-Algorithm=AWS4-HMAC-SHA256"
            "&X-Amz-Expires=900&X-Amz-Signature=" + ("a" * 400)
        )
        result = resolve_to_macro_path(url)
        assert result.is_external is True
        # The URL must be preserved verbatim — not mangled into a filesystem path.
        assert result.resolved_path == url

    def test_query_string_is_preserved(self) -> None:
        url = "https://example.com/image.png?token=abc&expires=123"
        result = resolve_to_macro_path(url)
        assert result.resolved_path == url


class TestResolveToMacroPathLocalPaths:
    """Non-URL values still go through filesystem resolution."""

    def test_nonexistent_local_path_is_external(self) -> None:
        path = "/tmp/definitely/does/not/exist/image.png"
        result = resolve_to_macro_path(path)
        assert result.is_external is True
        assert result.resolved_path == path

    def test_windows_drive_letter_not_treated_as_url(self) -> None:
        """A Windows drive letter (C:) must not be mistaken for a URL scheme."""
        path = r"C:\Users\someone\image.png"
        result = resolve_to_macro_path(path)
        # It does not exist on this (posix) test host, so it's external — but the
        # point is it went through the filesystem branch, not the URL short-circuit,
        # and returned without raising.
        assert result.is_external is True
        assert result.resolved_path == path
