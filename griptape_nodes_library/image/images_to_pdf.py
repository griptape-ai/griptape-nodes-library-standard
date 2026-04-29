from io import BytesIO
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact, UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list
from PIL import Image

from griptape_nodes_library.utils.image_utils import load_pil_from_url


class ImagesToPdf(ControlNode):
    """Convert multiple images to a single multi-page PDF file.

    Takes a list of ImageUrlArtifact objects and combines them into a single PDF,
    with each image as a separate page. Images are automatically converted to RGB
    mode as required by PDF format.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Input parameter for list of images
        self.images = ParameterList(
            name="images",
            input_types=[
                "ImageArtifact",
                "ImageUrlArtifact",
                "str",
                "list",
                "list[ImageArtifact]",
                "list[ImageUrlArtifact]",
            ],
            default_value=[],
            tooltip="List of images to convert to PDF",
            allowed_modes={ParameterMode.INPUT},
            ui_options={"expander": True, "display_name": "Images"},
        )
        self.add_parameter(self.images)

        # Output parameter showing final path
        self.output = Parameter(
            name="output",
            type="UrlArtifact",
            default_value=None,
            tooltip="Final path where PDF was saved",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.output)

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="output.pdf",
        )
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Normalize image inputs when the list is set."""
        super().after_value_set(parameter, value)

        # Convert string paths to ImageUrlArtifact by uploading to static storage
        if parameter.name == "images" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("images", updated_list)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that images list is not empty."""
        exceptions: list[Exception] = []

        images = self.get_parameter_list_value("images") or []
        if not images:
            msg = f"{self.name}: Images parameter is required and cannot be empty"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def process(self) -> None:
        """Convert list of images to a multi-page PDF file."""
        images = self.get_parameter_list_value("images") or []

        # Normalize string paths to ImageUrlArtifact during processing
        # (handles cases where values come from connections and bypass after_value_set)
        images = normalize_artifact_list(images, ImageUrlArtifact, accepted_types=(ImageArtifact,))

        logger.info(f"{self.name}: Converting {len(images)} images to PDF")

        # Load all images and convert to RGB
        pil_images = []
        for idx, image_artifact in enumerate(images):
            if not isinstance(image_artifact, ImageUrlArtifact):
                msg = f"{self.name}: Item {idx} is not an ImageUrlArtifact"
                raise TypeError(msg)

            pil_image = load_pil_from_url(image_artifact.value)

            # Convert to RGB (PDF doesn't support RGBA)
            if pil_image.mode == "RGBA":
                # Create white background for transparent images
                rgb_image = Image.new("RGB", pil_image.size, (255, 255, 255))
                rgb_image.paste(pil_image, mask=pil_image.split()[3])
                pil_images.append(rgb_image)
            elif pil_image.mode != "RGB":
                pil_images.append(pil_image.convert("RGB"))
            else:
                pil_images.append(pil_image)

        if not pil_images:
            msg = f"{self.name}: No valid images to convert"
            raise ValueError(msg)

        # Create PDF in memory
        pdf_buffer = BytesIO()
        if len(pil_images) == 1:
            # Single image PDF
            pil_images[0].save(pdf_buffer, format="PDF")
        else:
            # Multi-page PDF
            first_image = pil_images[0]
            rest_images = pil_images[1:]
            first_image.save(
                pdf_buffer,
                format="PDF",
                save_all=True,
                append_images=rest_images,
            )

        pdf_bytes = pdf_buffer.getvalue()
        pdf_buffer.close()

        # Save PDF
        dest = self._output_file.build_file()
        saved = dest.write_bytes(pdf_bytes)
        logger.debug(f"{self.name}: PDF saved as {saved.location}")

        # Set output
        self.parameter_output_values["output"] = UrlArtifact(saved.location)
