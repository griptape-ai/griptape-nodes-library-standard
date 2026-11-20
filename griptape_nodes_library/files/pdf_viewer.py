from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.widget import Widget


class PdfViewerMixin(BaseNode):
    """Gives a node that reads a PDF a viewer for its pages, shown above its settings.

    A node calls _add_pdf_viewer_parameter() after adding its `path` parameter. The PdfPageViewer widget
    draws the pages in the browser with pdf.js and keeps the current page itself, so stepping through
    them never reaches the engine. The node only gives it the PDF's URL. The parameters named in
    PDF_VIEWER_SYNCED reach the widget as synced_parameters: `password` for a protected PDF, and
    `rotation` to turn each page as it will be extracted.
    """

    # A node that already has a parameter called "pdf_viewer", or saved workflows that hold an older
    # name for it, name the widget's parameter something else.
    PDF_VIEWER_PARAMETER_NAME: ClassVar[str] = "pdf_viewer"
    PDF_VIEWER_SYNCED: ClassVar[list[str]] = ["password"]

    def _add_pdf_viewer_parameter(self, tooltip: str) -> None:
        self.add_parameter(
            ParameterDict(
                name=self.PDF_VIEWER_PARAMETER_NAME,
                default_value={"url": ""},
                tooltip=tooltip,
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Widget(
                        name="PdfPageViewer",
                        library="Griptape Nodes Library",
                        synced_parameters=list(self.PDF_VIEWER_SYNCED),
                    )
                },
            )
        )

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        """Point the viewer at the PDF when a workflow loads, since a saved URL may be out of date.

        Loading replays the saved values in parameter order, so the viewer's own saved value can arrive after
        the path's and replace the fresh URL. It refreshes the viewer too.
        """
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        if initial_setup and param_name in {"path", self.PDF_VIEWER_PARAMETER_NAME}:
            self._update_pdf_viewer()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "path":
            self._update_pdf_viewer()
        return super().after_value_set(parameter, value)

    def _update_pdf_viewer(self) -> None:
        """Give the viewer widget the PDF's URL. It draws the pages itself, so only the URL crosses over."""
        path = self.get_parameter_value("path") or ""
        viewer = {"url": self._static_url(path) if path else ""}
        if viewer == self.get_parameter_value(self.PDF_VIEWER_PARAMETER_NAME):
            return
        self.set_parameter_value(self.PDF_VIEWER_PARAMETER_NAME, viewer)
        self.publish_update_to_parameter(self.PDF_VIEWER_PARAMETER_NAME, viewer)

    def _static_url(self, path: str) -> str:
        """Return the static server's URL for a file, or an empty string when there is none."""
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=File(path).resolve())
            )
        except Exception as e:
            logger.warning(f"{self.name}: could not build a preview URL for {path!r}: {e}")
            return ""
        if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            return result.url
        logger.warning(f"{self.name}: the engine would not serve {path!r} to the viewer: {result.result_details}")
        return ""
