"""Tests for ExtractPdfImages."""

from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image
from pypdf import PdfReader, PdfWriter

from griptape_nodes_library.files import extract_pdf_images
from griptape_nodes_library.files.extract_pdf_images import ExtractPdfImages, OutputFormat, Rotation

RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> ExtractPdfImages:  # noqa: ARG001
    return ExtractPdfImages(name="test_extract_pdf_images")


def _scan(size: tuple[int, int]) -> Image.Image:
    """A solid red scan with a blue block in the top-left corner."""
    image = Image.new("RGB", size, RED)
    image.paste(BLUE, (0, 0, size[0] // 4, size[1] // 4))
    return image


def _pdf(images: list[Image.Image]) -> bytes:
    buffer = BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:])
    return buffer.getvalue()


def _decode(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data))


def test_extracts_one_image_per_page_at_stored_resolution(node: ExtractPdfImages) -> None:
    pdf = _pdf([_scan((120, 160)), _scan((80, 100))])

    result = node._extract_images(pdf, "", Rotation.NONE)

    assert [_decode(image.data).size for image in result] == [(120, 160), (80, 100)]


def test_unrotated_jpeg_keeps_its_stored_bytes(node: ExtractPdfImages) -> None:
    pdf = _pdf([_scan((120, 160))])
    stored = next(iter(PdfReader(BytesIO(pdf)).pages[0].images)).data

    result = node._extract_images(pdf, "", Rotation.NONE)

    assert result[0].data == stored
    assert result[0].extension == ".jpg"


@pytest.mark.parametrize(
    ("rotation", "size", "blue_at"),
    [
        (Rotation.CLOCKWISE_90, (160, 120), (150, 10)),
        (Rotation.ROTATE_180, (120, 160), (110, 150)),
        (Rotation.COUNTER_CLOCKWISE_90, (160, 120), (10, 110)),
    ],
)
def test_rotation_turns_the_image(
    node: ExtractPdfImages, rotation: Rotation, size: tuple[int, int], blue_at: tuple[int, int]
) -> None:
    pdf = _pdf([_scan((120, 160))])

    result = node._extract_images(pdf, "", rotation)

    rotated = _decode(result[0].data).convert("RGB")
    assert rotated.size == size
    red, _green, blue = rotated.getpixel(blue_at)  # type: ignore[misc]
    assert blue > 200  # noqa: PLR2004
    assert red < 60  # noqa: PLR2004


def test_page_rotation_is_added_to_the_chosen_rotation(node: ExtractPdfImages) -> None:
    writer = PdfWriter(clone_from=BytesIO(_pdf([_scan((120, 160))])))
    writer.pages[0].rotate(90)
    buffer = BytesIO()
    writer.write(buffer)

    quarter_turn = node._extract_images(buffer.getvalue(), "", Rotation.NONE)
    half_turn = node._extract_images(buffer.getvalue(), "", Rotation.CLOCKWISE_90)

    assert _decode(quarter_turn[0].data).size == (160, 120)
    assert _decode(half_turn[0].data).size == (120, 160)


def test_perform_extraction_saves_numbered_images_in_the_output_folder(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(_pdf([_scan((120, 160)), _scan((80, 100))]))
    out_dir = tmp_path / "extracted_v001"
    monkeypatch.setattr(node, "_resolve_output_dir", lambda: out_dir)
    node.set_parameter_value("path", str(pdf_path))

    node._perform_extraction()

    artifacts = node.parameter_output_values["images"]
    assert [Path(a.value) for a in artifacts] == [out_dir / "page_001.jpg", out_dir / "page_002.jpg"]
    assert [_decode(Path(a.value).read_bytes()).size for a in artifacts] == [(120, 160), (80, 100)]
    assert node.parameter_output_values["image_count"] == 2  # noqa: PLR2004
    assert node.parameter_output_values["output_directory"] == str(out_dir)


def test_prefix_padding_and_format_name_the_files(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(_pdf([_scan((40, 40))]))
    monkeypatch.setattr(node, "_resolve_output_dir", lambda: tmp_path)
    node.set_parameter_value("path", str(pdf_path))
    node.set_parameter_value("file_prefix", "scan")
    node.set_parameter_value("number_padding", "2")
    node.set_parameter_value("file_format", "png")

    node._perform_extraction()

    saved = Path(node.parameter_output_values["images"][0].value)
    assert saved == tmp_path / "scan_01.png"
    assert _decode(saved.read_bytes()).format == "PNG"


@pytest.mark.parametrize(
    ("output_format", "pil_format", "extension"),
    [
        (OutputFormat.PNG, "PNG", ".png"),
        (OutputFormat.JPG, "JPEG", ".jpg"),
        (OutputFormat.WEBP, "WEBP", ".webp"),
    ],
)
def test_file_format_converts_the_stored_image(
    node: ExtractPdfImages, output_format: OutputFormat, pil_format: str, extension: str
) -> None:
    pdf = _pdf([_scan((120, 160))])

    result = node._extract_images(pdf, "", Rotation.NONE, output_format)

    assert result[0].extension == extension
    assert _decode(result[0].data).format == pil_format
    assert _decode(result[0].data).size == (120, 160)


def test_original_format_keeps_png_scans_as_png(node: ExtractPdfImages) -> None:
    png = BytesIO()
    _scan((60, 80)).save(png, format="PNG")

    result = node._encode(png.getvalue(), 0, OutputFormat.ORIGINAL)

    assert result.extension == ".png"
    assert result.data == png.getvalue()


def test_pdf_without_images_returns_nothing(node: ExtractPdfImages) -> None:
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buffer = BytesIO()
    writer.write(buffer)

    assert node._extract_images(buffer.getvalue(), "", Rotation.NONE) == []


def test_encrypted_pdf_needs_the_password(node: ExtractPdfImages) -> None:
    writer = PdfWriter(clone_from=BytesIO(_pdf([_scan((40, 40))])))
    writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)

    with pytest.raises(ValueError, match="password"):
        node._extract_images(buffer.getvalue(), "wrong", Rotation.NONE)

    assert len(node._extract_images(buffer.getvalue(), "secret", Rotation.NONE)) == 1


def test_setting_path_publishes_a_preview_of_the_first_image(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(_pdf([_scan((120, 160))]))
    monkeypatch.setattr(node, "_build_preview_url", lambda path: f"http://static/{Path(path).name}.jpg")

    node.set_parameter_value("path", str(pdf_path))
    node.after_value_set(node.path, str(pdf_path))

    assert node.get_parameter_value("rotation_preview") == {
        "image_url": "http://static/scan.pdf.jpg",
        "rotation": Rotation.NONE,
    }


def test_changing_rotation_updates_the_preview_without_rebuilding_the_image(
    node: ExtractPdfImages, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: "http://static/first.jpg")
    node.after_value_set(node.path, "/some/scan.pdf")
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: pytest.fail("rebuilt the preview image"))

    node.after_value_set(node.get_parameter_by_name("rotation"), Rotation.CLOCKWISE_90)  # type: ignore[arg-type]

    assert node.get_parameter_value("rotation_preview") == {
        "image_url": "http://static/first.jpg",
        "rotation": Rotation.CLOCKWISE_90,
    }


def test_build_preview_url_saves_a_small_copy_of_the_first_image(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(_pdf([_scan((2000, 3000))]))
    requested: list[Path] = []

    class FakeStaticUrl:
        url = "http://static/preview.jpg"

    def fake_handle_request(request: Any) -> FakeStaticUrl:
        requested.append(Path(request.file_path))
        return FakeStaticUrl()

    # Serving the file is the engine's job. Stub it so this test does not start static file machinery.
    monkeypatch.setattr(extract_pdf_images, "CreateStaticFileDownloadUrlFromPathResultSuccess", FakeStaticUrl)
    monkeypatch.setattr(extract_pdf_images, "GriptapeNodes", SimpleNamespace(handle_request=fake_handle_request))

    url = node._build_preview_url(str(pdf_path))

    assert url == "http://static/preview.jpg"
    preview = _decode(requested[0].read_bytes())
    assert max(preview.size) == 1000  # noqa: PLR2004
