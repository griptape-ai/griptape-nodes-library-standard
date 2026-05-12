"""Tests that ``ExtractKeyColors`` reads bytes via ``File`` for URL artifacts.

``ImageUrlArtifact.to_bytes()`` does ``requests.get(self.value)``, which fails
when ``.value`` is a project macro path like ``{outputs}/foo.png`` emitted by
upstream nodes that wrote through ``ProjectFileParameter``. The fix routes
URL artifacts through ``File(value).read_bytes()`` so macro paths and plain
filesystem paths resolve correctly, while raw-bytes ``ImageArtifact`` inputs
continue to use ``to_bytes()``.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.files import file as file_module

import griptape_nodes_library.image.extract_key_colors as extract_key_colors_module
from griptape_nodes_library.image.extract_key_colors import ExtractKeyColors

CANNED_BYTES = b"\x89PNG\r\n\x1a\n-CANNED-IMAGE"


@pytest.fixture
def captured_paths(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture every path passed to ``File(...)`` and return canned bytes."""
    paths: list[str] = []
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        paths.append(path)
        real_init(self, path, *args, **kwargs)

    def fake_read_bytes(self: file_module.File) -> bytes:
        return CANNED_BYTES

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", fake_read_bytes)
    return paths


def test_image_url_artifact_with_macro_path_reads_via_file(captured_paths: list[str]) -> None:
    node = ExtractKeyColors.__new__(ExtractKeyColors)
    artifact = ImageUrlArtifact("{outputs}/color_bars.png")

    result = node._image_to_bytes(artifact)

    assert result == CANNED_BYTES
    assert "{outputs}/color_bars.png" in captured_paths


def test_image_url_artifact_with_filesystem_path_reads_via_file(captured_paths: list[str]) -> None:
    node = ExtractKeyColors.__new__(ExtractKeyColors)
    artifact = ImageUrlArtifact("/abs/path/foo.png")

    result = node._image_to_bytes(artifact)

    assert result == CANNED_BYTES
    assert "/abs/path/foo.png" in captured_paths


def test_image_url_artifact_with_http_url_reads_via_file(captured_paths: list[str]) -> None:
    """File handles HTTP URLs via the engine's read pipeline, not requests.get
    bypassing macro resolution.
    """
    node = ExtractKeyColors.__new__(ExtractKeyColors)
    artifact = ImageUrlArtifact("https://example.com/foo.png")

    result = node._image_to_bytes(artifact)

    assert result == CANNED_BYTES
    assert "https://example.com/foo.png" in captured_paths


def test_dict_serialized_image_url_artifact_with_macro_path_reads_via_file(
    captured_paths: list[str],
) -> None:
    node = ExtractKeyColors.__new__(ExtractKeyColors)
    serialized = {"type": "ImageUrlArtifact", "value": "{outputs}/foo.png"}

    result = node._image_to_bytes(serialized)

    assert result == CANNED_BYTES
    assert "{outputs}/foo.png" in captured_paths


def test_image_artifact_uses_to_bytes_directly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Raw-bytes ``ImageArtifact`` inputs must NOT route through ``File`` because
    their ``.value`` is bytes, not a path/URL.
    """

    class FakeImageArtifact:
        # Mimic griptape ImageArtifact's payload location well enough for isinstance.
        pass

    # Patch the type check so isinstance(fake, ImageArtifact) is True.
    monkeypatch.setattr(extract_key_colors_module, "ImageArtifact", FakeImageArtifact)

    file_calls: list[Any] = []

    def fail_file_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        file_calls.append(path)

    monkeypatch.setattr(file_module.File, "__init__", fail_file_init)

    instance = FakeImageArtifact()
    instance.to_bytes = lambda: b"RAW-BYTES"  # type: ignore[attr-defined]

    node = ExtractKeyColors.__new__(ExtractKeyColors)
    result = node._image_to_bytes(instance)  # pyright: ignore[reportArgumentType]

    assert result == b"RAW-BYTES"
    assert file_calls == []  # File never constructed for raw-bytes artifacts


def test_file_load_error_is_wrapped_as_value_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason

    def fail_read(self: file_module.File) -> bytes:
        raise file_module.FileLoadError(FileIOFailureReason.FILE_NOT_FOUND, "missing")

    monkeypatch.setattr(file_module.File, "read_bytes", fail_read)

    node = ExtractKeyColors.__new__(ExtractKeyColors)
    artifact = ImageUrlArtifact("{outputs}/missing.png")

    with pytest.raises(ValueError, match="Failed to extract image data"):
        node._image_to_bytes(artifact)
