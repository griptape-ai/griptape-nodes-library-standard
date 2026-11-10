"""Tests for ExtractPdfImages."""

import re
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from PIL import Image
from pypdf import PdfReader, PdfWriter

from griptape_nodes_library.files import extract_pdf_images
from griptape_nodes_library.files.extract_pdf_images import ExtractPdfImages, OutputFormat, Rotation
from griptape_nodes_library.utils.directory_settings import reject_connected_file_destination

RED = (255, 0, 0)
BLUE = (0, 0, 255)


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> ExtractPdfImages:  # noqa: ARG001
    return ExtractPdfImages(name="test_extract_pdf_images")


@pytest.fixture
def real_graph(open_workflow: None, tmp_path: Path) -> ExtractPdfImages:
    """A real ExtractPdfImages node in a real flow, with outputs routed under tmp_path.

    `DirectoryDestination.create()` does real engine work (next-version lookup, mkdir, mapping the
    created path back to project-macro form) through the node's bound `WorkflowRuntime`, which only
    a node actually built by the engine -- not a stand-in object -- has.
    """
    GriptapeNodes.ConfigManager().workspace_path = tmp_path
    result = GriptapeNodes.handle_request(CreateFlowRequest(parent_flow_name=None, flow_name="TestFlow"))
    assert result.succeeded(), result.result_details
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="ExtractPdfImages", node_name="epi"))
    assert result.succeeded(), result.result_details
    found = GriptapeNodes.ObjectManager().attempt_get_object_by_name("epi")
    assert found is not None
    assert type(found).__name__ == "ExtractPdfImages"
    return found


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
    monkeypatch.setattr(
        node._directory_param,
        "build_directory",
        lambda **_kwargs: SimpleNamespace(create=lambda: SimpleNamespace(resolve=lambda: out_dir)),
    )
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: "")
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
    monkeypatch.setattr(
        node._directory_param,
        "build_directory",
        lambda **_kwargs: SimpleNamespace(create=lambda: SimpleNamespace(resolve=lambda: tmp_path)),
    )
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: "")
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

    with pytest.raises(NodeError, match="password"):
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
        "operation": "rotate",
        "media": "image",
        "image_url": "http://static/scan.pdf.jpg",
    }


def test_changing_rotation_does_not_rebuild_the_preview_image(
    node: ExtractPdfImages, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: "http://static/first.jpg")
    node.set_parameter_value("path", "/some/scan.pdf")
    node.after_value_set(node.path, "/some/scan.pdf")
    before = node.get_parameter_value("rotation_preview")
    monkeypatch.setattr(node, "_build_preview_url", lambda _path: pytest.fail("rebuilt the preview image"))

    node.set_parameter_value("rotation", Rotation.CLOCKWISE_90)
    node.after_value_set(node.get_parameter_by_name("rotation"), Rotation.CLOCKWISE_90)  # type: ignore[arg-type]

    assert node.get_parameter_value("rotation_preview") == before


def test_preview_is_the_shared_rotate_preview_driven_by_the_rotation_setting(node: ExtractPdfImages) -> None:
    preview = node.get_parameter_by_name("rotation_preview")
    assert preview is not None
    assert preview.ui_options["widget"] == "ImageProcessorPreview"
    assert preview.ui_options["synced_parameters"] == ["rotation"]
    assert node.get_parameter_by_name("rotation") is not None
    assert node.get_parameter_value("rotation_preview")["operation"] == "rotate"
    # The preview shows the PDF's first image, so the path stays editable.
    assert not node.path.hide_property


def test_loading_a_saved_path_builds_the_preview(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(_pdf([_scan((120, 160))]))
    monkeypatch.setattr(node, "_build_preview_url", lambda path: f"http://static/{Path(path).name}.jpg")

    node.set_parameter_value("path", str(pdf_path), initial_setup=True)

    assert node.get_parameter_value("rotation_preview")["image_url"] == "http://static/scan.pdf.jpg"


def test_the_preview_knows_every_rotation_the_node_offers() -> None:
    operations = (Path(__file__).parents[3] / "widgets" / "ImageProcessorPreview" / "_operations.js").read_text()
    table = re.search(r"const QUARTER_TURNS = \{(.*?)\};", operations, re.DOTALL)
    assert table is not None
    for rotation in Rotation:
        assert f'"{rotation.value}"' in table.group(1) or f"{rotation.value}:" in table.group(1), rotation


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


@pytest.mark.parametrize("output_format", [OutputFormat.ORIGINAL, OutputFormat.PNG])
def test_fax_compressed_scan_is_saved_as_a_real_png(node: ExtractPdfImages, output_format: OutputFormat) -> None:
    buffer = BytesIO()
    Image.new("1", (64, 64), 1).save(buffer, format="PDF")
    stored = next(iter(PdfReader(BytesIO(buffer.getvalue())).pages[0].images)).data
    assert _decode(stored).format == "TIFF"

    result = node._extract_images(buffer.getvalue(), "", Rotation.NONE, output_format)

    assert result[0].extension == ".png"
    assert _decode(result[0].data).format == "PNG"


def test_stored_jpeg_is_not_passed_through_as_png(node: ExtractPdfImages) -> None:
    result = node._extract_images(_pdf([_scan((60, 80))]), "", Rotation.NONE, OutputFormat.PNG)

    assert _decode(result[0].data).format == "PNG"


def test_entering_the_password_after_the_path_builds_the_preview(
    node: ExtractPdfImages, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pdf_path = tmp_path / "protected.pdf"
    pdf_path.write_bytes(_pdf([_scan((60, 80))]))
    password = {"value": ""}

    def build(_path: str) -> str:
        return "http://static/preview.jpg" if password["value"] == "secret" else ""

    monkeypatch.setattr(node, "_build_preview_url", build)
    node.set_parameter_value("path", str(pdf_path))
    node.after_value_set(node.path, str(pdf_path))
    assert node.get_parameter_value("rotation_preview")["image_url"] == ""

    password["value"] = "secret"
    node.set_parameter_value("password", "secret")
    node.after_value_set(node.get_parameter_by_name("password"), "secret")  # type: ignore[arg-type]

    assert node.get_parameter_value("rotation_preview")["image_url"] == "http://static/preview.jpg"


def test_failure_raises_when_failed_is_unwired(node: ExtractPdfImages) -> None:
    node.set_parameter_value("path", "unused.pdf")
    node._perform_extraction = lambda: (_ for _ in ()).throw(NodeError("No embedded images found"))  # type: ignore[method-assign]
    node._has_outgoing_connections = lambda _param: False  # type: ignore[method-assign]
    with pytest.raises(NodeError, match="No embedded images"):
        node._run_extraction()
    assert node.parameter_output_values["was_successful"] is False


def test_failure_follows_failed_when_wired(node: ExtractPdfImages) -> None:
    node._perform_extraction = lambda: (_ for _ in ()).throw(NodeError("No embedded images found"))  # type: ignore[method-assign]
    node._has_outgoing_connections = lambda _param: True  # type: ignore[method-assign]
    node._run_extraction()
    assert node.parameter_output_values["was_successful"] is False
    assert "No embedded images" in node.parameter_output_values["result_details"]


def test_a_typed_folder_name_creates_a_versioned_folder(real_graph: ExtractPdfImages, tmp_path: Path) -> None:
    real_graph.set_parameter_value("directory", "scans")

    # Each resolve actually creates the folder, so the next one takes the next version.
    first = real_graph._directory_param.build_directory().create().resolve()
    second = real_graph._directory_param.build_directory().create().resolve()

    assert first == tmp_path / "outputs" / "scans_v001"
    assert second == tmp_path / "outputs" / "scans_v002"


def test_old_default_directory_value_is_migrated_on_load(node: ExtractPdfImages) -> None:
    """A saved workflow holding the pre-redesign default is rewritten to the new default."""
    node.set_parameter_value("directory", "{outputs}/extracted_v{###}", initial_setup=True)

    assert node.get_parameter_value("directory") == "extracted"


def test_a_file_settings_node_connected_to_directory_raises(real_graph: ExtractPdfImages) -> None:
    result = GriptapeNodes.handle_request(CreateNodeRequest(node_type="FileOutputSettings", node_name="fos"))
    assert result.succeeded(), result.result_details
    result = GriptapeNodes.handle_request(
        CreateConnectionRequest(
            source_node_name="fos",
            source_parameter_name="file_destination",
            target_node_name=real_graph.name,
            target_parameter_name="directory",
        )
    )
    assert result.succeeded(), result.result_details

    with pytest.raises(NodeError, match="names a file, not a folder"):
        reject_connected_file_destination(real_graph)
