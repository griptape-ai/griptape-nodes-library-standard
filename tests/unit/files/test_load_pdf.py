"""Tests for LoadPdf's PDF viewer."""

from pathlib import Path

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files.load_pdf import LoadPdf


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> LoadPdf:  # noqa: ARG001
    return LoadPdf(name="test_load_pdf")


def test_viewer_is_the_page_viewer_widget_synced_to_the_password(node: LoadPdf) -> None:
    viewer = node.get_parameter_by_name("pdf_viewer")
    assert viewer is not None
    assert viewer.ui_options["widget"] == "PdfPageViewer"
    assert viewer.ui_options["synced_parameters"] == ["password"]
    assert node.get_parameter_by_name("password") is not None


def test_setting_path_gives_the_viewer_the_pdfs_url(node: LoadPdf, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_static_url", lambda path: f"http://static/{Path(path).name}")

    node.set_parameter_value("path", "/some/scan.pdf")

    assert node.get_parameter_value("pdf_viewer") == {"url": "http://static/scan.pdf"}


def test_the_outputs_are_unchanged(node: LoadPdf) -> None:
    for name in ("text", "pages", "page_count"):
        assert node.get_parameter_by_name(name) is not None
