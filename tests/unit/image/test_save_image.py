"""Tests that ``SaveImage`` reads bytes via ``File`` for URL artifacts.

When an upstream node emits an ``ImageUrlArtifact`` whose ``.value`` is a
project macro path (``{outputs}/foo.png``), calling
``ImageUrlArtifact.to_bytes()`` does ``requests.get(self.value)`` and fails
with "No scheme supplied". The fix routes ``ImageUrlArtifact`` inputs through
``File(value).read_bytes()`` so macro paths and plain filesystem paths
resolve correctly. Raw-bytes ``ImageArtifact`` inputs continue to use
``to_bytes()``.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from griptape.artifacts import ImageArtifact
from griptape_nodes.files import file as file_module
from PIL import Image

from griptape_nodes_library.image.save_image import SaveImage


def _make_png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (1, 1), color="red").save(buf, "PNG")
    return buf.getvalue()


CANNED_BYTES = _make_png_bytes()
SAVED_LOCATION = "{outputs}/saved.png"


@pytest.fixture
def file_capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"paths": []}
    real_init = file_module.File.__init__

    def capture_init(self: file_module.File, path: Any, *args: Any, **kwargs: Any) -> None:
        captured["paths"].append(path)
        real_init(self, path, *args, **kwargs)

    def fake_read_bytes(self: file_module.File) -> bytes:
        return CANNED_BYTES

    monkeypatch.setattr(file_module.File, "__init__", capture_init)
    monkeypatch.setattr(file_module.File, "read_bytes", fake_read_bytes)
    return captured


@pytest.fixture
def stub_output_file(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    captured: dict[str, Any] = {"written": None}

    class FakeSaved:
        location = SAVED_LOCATION

    class FakeDest:
        def write_bytes(self, content: bytes) -> FakeSaved:
            captured["written"] = content
            return FakeSaved()

    class FakeOutputFile:
        _default_filename = "griptape_nodes.png"

        def add_parameter(self) -> None:
            pass

        def build_file(self) -> FakeDest:
            return FakeDest()

    monkeypatch.setattr(
        "griptape_nodes_library.image.save_image.ProjectFileParameter",
        lambda **_kwargs: FakeOutputFile(),
    )
    return captured


def test_dict_serialized_image_url_artifact_with_macro_path_reads_via_file(
    file_capture: dict[str, Any], stub_output_file: dict[str, Any]
) -> None:
    """A dict-shaped ``ImageUrlArtifact`` with a macro-path value flows through
    ``to_image_artifact`` (which returns an ``ImageUrlArtifact``) and previously
    crashed at ``image_artifact.to_bytes()``. The fix detects the URL-artifact
    case and uses ``File`` instead.
    """
    node = SaveImage(name="save")
    serialized = {"type": "ImageUrlArtifact", "value": "{outputs}/color_bars.png"}
    node.set_parameter_value("image", serialized)
    node.process()

    assert "{outputs}/color_bars.png" in file_capture["paths"]
    # SaveImage re-encodes bytes to the target output format before writing,
    # so we don't compare to the canned source bytes; just assert the write
    # happened with non-empty bytes.
    assert isinstance(stub_output_file["written"], bytes)
    assert len(stub_output_file["written"]) > 0


def test_image_artifact_uses_to_bytes_directly(file_capture: dict[str, Any], stub_output_file: dict[str, Any]) -> None:
    """Raw-bytes ``ImageArtifact`` inputs must NOT route through ``File``."""
    artifact = ImageArtifact(value=CANNED_BYTES, format="png", width=1, height=1)

    node = SaveImage(name="save")
    node.set_parameter_value("image", artifact)
    node.process()

    # ``ImageArtifact.to_bytes()`` returns the raw value; ``File`` should never
    # have been constructed for this input.
    assert file_capture["paths"] == []
    assert isinstance(stub_output_file["written"], bytes)
    assert len(stub_output_file["written"]) > 0


def test_jpeg_composites_transparent_pixels_over_black(
    file_capture: dict[str, Any], stub_output_file: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """JPEG has no alpha, so straight-alpha RGB under alpha 0 must not show through."""
    # Left half transparent, right half opaque, both with the same RGB.
    source = Image.new("RGBA", (16, 8), (200, 120, 40, 255))
    source.paste((200, 120, 40, 0), (0, 0, 8, 8))
    buf = BytesIO()
    source.save(buf, "PNG")
    artifact = ImageArtifact(value=buf.getvalue(), format="png", width=16, height=8)

    node = SaveImage(name="save")
    monkeypatch.setattr(node, "_get_target_pil_format", lambda: "JPEG")
    node.set_parameter_value("image", artifact)
    node.process()

    saved = Image.open(BytesIO(stub_output_file["written"]))
    assert saved.format == "JPEG"
    # JPEG is lossy, so sample away from the edge and allow a small tolerance.
    transparent = saved.getpixel((2, 4))
    opaque = saved.getpixel((13, 4))
    assert isinstance(transparent, tuple)
    assert isinstance(opaque, tuple)
    assert all(c < 20 for c in transparent), transparent
    assert all(abs(a - b) < 20 for a, b in zip(opaque, (200, 120, 40), strict=True)), opaque
