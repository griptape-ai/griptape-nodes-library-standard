"""Tests for resolve_to_macro_path URL and filesystem-path handling."""

from griptape_nodes_library.utils.macro_path_utils import resolve_to_macro_path


class TestResolveToMacroPathUrls:
    """A remote URL must be treated as external, never anchored to the cwd as a path.

    Regression: a presigned S3 URL fed to a Load* node was joined onto the working
    directory and the resulting overlong path raised "[Errno 63] File name too long".
    """

    def test_presigned_s3_url_is_external(self) -> None:
        url = (
            "https://sg-media-usor-01.s3-accelerate.amazonaws.com/"
            "7d0ea73011daa9d36f8c3a0bdff31f7fbad21fe3/"
            "e39081a6093f10bd_image_v026_t.jpg"
            "?response-content-disposition=filename%3D%22e39081a6093f10bd_image_v026_t.jpg%22"
            "&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Expires=900&X-Amz-Signature=" + ("a" * 400)
        )

        result = resolve_to_macro_path(url)

        assert result.is_external is True
        assert result.resolved_path == url

    def test_plain_https_url_is_external(self) -> None:
        url = "https://example.com/images/photo.png"

        result = resolve_to_macro_path(url)

        assert result.is_external is True
        assert result.resolved_path == url

    def test_http_url_is_external(self) -> None:
        url = "http://example.com/images/photo.png"

        result = resolve_to_macro_path(url)

        assert result.is_external is True
        assert result.resolved_path == url


class TestResolveToMacroPathFilesystem:
    """Filesystem paths that don't exist (or can't be stat'd) degrade to external."""

    def test_nonexistent_local_path_is_external(self) -> None:
        path = "/tmp/does-not-exist-xyz/image.png"

        result = resolve_to_macro_path(path)

        assert result.is_external is True
        assert result.resolved_path == path

    def test_overlong_path_degrades_to_external_without_raising(self) -> None:
        """A pathological non-URL value must not raise OSError from the stat call."""
        path = "/tmp/" + ("x" * 5000)

        result = resolve_to_macro_path(path)

        assert result.is_external is True
        assert result.resolved_path == path
