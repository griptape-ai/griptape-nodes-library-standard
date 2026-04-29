from typing import Any

from griptape.loaders import PdfLoader
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode, NodeDependencies
from griptape_nodes.traits.file_system_picker import FileSystemPicker


class LoadPdf(ControlNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.path = Parameter(
            name="path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            input_types=["str"],
            type="str",
            default_value="",
            tooltip="Path to the local PDF file to load.",
        )
        self.path.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=False,
                multiple=False,
                file_types=[".pdf"],
            )
        )
        self.add_parameter(self.path)

        self.add_parameter(
            Parameter(
                name="password",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=["str"],
                type="str",
                default_value="",
                tooltip="Optional password for password-protected PDFs. Leave blank if not needed.",
            )
        )

        self.add_parameter(
            Parameter(
                name="text",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value="",
                tooltip="The full text content of the PDF, with all pages joined by two newlines.",
                ui_options={"multiline": True, "placeholder_text": "PDF text will appear here."},
            )
        )

        self.add_parameter(
            Parameter(
                name="pages",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="list",
                default_value=[],
                tooltip="A list of strings, one entry per page of the PDF, in order.",
            )
        )

        self.add_parameter(
            Parameter(
                name="page_count",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="int",
                default_value=0,
                tooltip="The total number of pages in the PDF.",
            )
        )

    def get_node_dependencies(self) -> NodeDependencies | None:
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()
        value = self.get_parameter_value("path")
        if value and isinstance(value, str):
            deps.static_files.add(value)
        return deps

    def process(self) -> None:
        path = self.get_parameter_value("path")
        password = self.get_parameter_value("password") or ""

        loader = PdfLoader()
        data = loader.fetch(path)
        pages_artifact = loader.try_parse(data, password=password or None)
        page_texts = [page.value for page in pages_artifact]

        full_text = "\n\n".join(page_texts)
        page_count = len(page_texts)

        self.parameter_output_values["text"] = full_text
        self.parameter_output_values["pages"] = page_texts
        self.parameter_output_values["page_count"] = page_count

        self.parameter_values["text"] = full_text
        self.parameter_values["pages"] = page_texts
        self.parameter_values["page_count"] = page_count
