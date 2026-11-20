"""Tests for PdfViewerMixin."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.files import pdf_viewer
from griptape_nodes_library.files.pdf_viewer import PdfViewerMixin


class ViewerNode(PdfViewerMixin, ControlNode):
    PDF_VIEWER_SYNCED: ClassVar[list[str]] = ["password"]

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.path = Parameter(
            name="path", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, type="str", default_value=""
        )
        self.add_parameter(self.path)
        self.add_parameter(
            Parameter(name="password", allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY}, type="str")
        )
        self._add_pdf_viewer_parameter("The PDF's pages.")


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> ViewerNode:  # noqa: ARG001
    return ViewerNode(name="test_pdf_viewer")


def test_viewer_is_the_page_viewer_widget_synced_to_the_nodes_parameters(node: ViewerNode) -> None:
    viewer = node.get_parameter_by_name("pdf_viewer")
    assert viewer is not None
    assert viewer.ui_options["widget"] == "PdfPageViewer"
    assert viewer.ui_options["synced_parameters"] == ["password"]
    assert node.get_parameter_value("pdf_viewer") == {"url": ""}


def test_setting_path_gives_the_viewer_the_pdfs_url(node: ViewerNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_static_url", lambda path: f"http://static/{Path(path).name}")

    node.set_parameter_value("path", "/some/scan.pdf")

    assert node.get_parameter_value("pdf_viewer") == {"url": "http://static/scan.pdf"}


def test_clearing_the_path_clears_the_viewer(node: ViewerNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_static_url", lambda _path: "http://static/scan.pdf")
    node.set_parameter_value("path", "/some/scan.pdf")

    node.set_parameter_value("path", "")

    assert node.get_parameter_value("pdf_viewer") == {"url": ""}


def test_changing_another_parameter_leaves_the_viewer_alone(node: ViewerNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_static_url", lambda _path: "http://static/scan.pdf")
    node.set_parameter_value("path", "/some/scan.pdf")
    before = node.get_parameter_value("pdf_viewer")
    monkeypatch.setattr(node, "_static_url", lambda _path: pytest.fail("looked the PDF up again"))

    node.set_parameter_value("password", "secret")

    assert node.get_parameter_value("pdf_viewer") == before


def test_loading_a_saved_path_points_the_viewer_at_it(node: ViewerNode, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(node, "_static_url", lambda path: f"http://static/{Path(path).name}")

    node.set_parameter_value("path", "/some/scan.pdf", initial_setup=True)

    assert node.get_parameter_value("pdf_viewer") == {"url": "http://static/scan.pdf"}


@pytest.mark.parametrize("saved", [{"url": "http://static/old.pdf"}, {"operation": "rotate", "image_url": "x.jpg"}])
def test_a_saved_viewer_value_loading_after_the_path_does_not_replace_the_fresh_url(
    node: ViewerNode, monkeypatch: pytest.MonkeyPatch, saved: dict[str, Any]
) -> None:
    monkeypatch.setattr(node, "_static_url", lambda path: f"http://static/{Path(path).name}")

    node.set_parameter_value("path", "/some/scan.pdf", initial_setup=True)
    node.set_parameter_value("pdf_viewer", saved, initial_setup=True)

    assert node.get_parameter_value("pdf_viewer") == {"url": "http://static/scan.pdf"}


def test_a_saved_viewer_value_loading_before_the_path_still_ends_on_the_fresh_url(
    node: ViewerNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(node, "_static_url", lambda path: f"http://static/{Path(path).name}")

    node.set_parameter_value("pdf_viewer", {"url": "http://static/old.pdf"}, initial_setup=True)
    node.set_parameter_value("path", "/some/scan.pdf", initial_setup=True)

    assert node.get_parameter_value("pdf_viewer") == {"url": "http://static/scan.pdf"}


def test_static_url_is_the_engines_url_for_the_file(node: ViewerNode, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeStaticUrl:
        url = "http://static/scan.pdf"

    # Serving the file is the engine's job. Stub it so this test does not start static file machinery.
    monkeypatch.setattr(pdf_viewer, "CreateStaticFileDownloadUrlFromPathResultSuccess", FakeStaticUrl)
    monkeypatch.setattr(pdf_viewer, "GriptapeNodes", SimpleNamespace(handle_request=lambda _request: FakeStaticUrl()))

    assert node._static_url("/some/scan.pdf") == "http://static/scan.pdf"


def test_static_url_is_empty_when_the_engine_cannot_serve_the_file(
    node: ViewerNode, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(_request: Any) -> None:
        msg = "no static server"
        raise RuntimeError(msg)

    monkeypatch.setattr(pdf_viewer, "GriptapeNodes", SimpleNamespace(handle_request=refuse))

    assert node._static_url("/some/scan.pdf") == ""
