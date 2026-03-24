import os
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.files.file import File
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

from griptape_nodes_library.utils.file_utils import SUPPORTED_TEXT_EXTENSIONS


class LoadText(ControlNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # Add output parameters
        self.path = Parameter(
            name="path",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            input_types=["str"],
            type="str",
            default_value="",
            tooltip="The full path to the loaded file.",
        )
        self.path.add_trait(
            FileSystemPicker(
                allow_files=True,
                allow_directories=False,
                multiple=False,
                file_types=list(SUPPORTED_TEXT_EXTENSIONS),
            )
        )

        self.add_parameter(self.path)

        self.add_parameter(
            Parameter(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="str",
                default_value="",
                tooltip="The text content of the loaded file.",
                ui_options={"multiline": True, "placeholder_text": "Text will load here."},
            )
        )

    def process(self) -> None:
        # Get the selected file
        text_path = self.get_parameter_value("path")

        # Load file content based on extension
        ext = os.path.splitext(text_path)[1]  # noqa: PTH122
        if ext.lower() == ".pdf":
            page_texts = [
                "".join(element.get_text() for element in page_layout if isinstance(element, LTTextContainer))
                for page_layout in extract_pages(text_path)
            ]
            output_text = "\n\n".join(page_texts)
        else:
            output_text = File(text_path).read_text()

        # Set output values
        self.parameter_output_values["path"] = text_path
        self.parameter_output_values["output"] = output_text

        # Also set in parameter_values for get_value compatibility
        self.parameter_values["path"] = text_path
        self.parameter_values["output"] = output_text
