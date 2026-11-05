from dataclasses import dataclass
from functools import partial
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact, UrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline, ImageFormat, ImageMetadata
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input, normalize_artifact_list

from griptape_nodes_library.utils.image_utils import load_image_from_url

# Transparent pixels are composited over this, since PDF pages here hold JPEG, which has no alpha
PAGE_BACKGROUND = (255, 255, 255)

# Page size is the image's pixel size at this resolution, in PDF points (1/72 inch)
_POINTS_PER_INCH = 72.0
_PAGE_DPI = 72.0

# Header, plus a comment line of high bytes marking the file as binary
_PDF_HEADER = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"

# The catalog and page tree come first; each page then takes three objects: image, page, contents
_CATALOG_ID = 1
_PAGES_ID = 2
_FIRST_PAGE_OBJECT_ID = 3
_OBJECTS_PER_PAGE = 3


@dataclass(frozen=True)
class _PdfPage:
    """One page of a PDF: a JPEG-encoded RGB image filling the page."""

    jpeg: bytes
    width: int
    height: int


def _pdf_page_from_image(image: GriptapeImage) -> _PdfPage:
    """Flatten an image onto white as 8-bit RGB and encode it as a page's JPEG, without metadata."""
    rgb = GriptapeImagePipeline(image).to_uint8().flatten(PAGE_BACKGROUND).end()
    return _PdfPage(jpeg=rgb.encode(ImageFormat.JPEG, metadata=ImageMetadata()), width=rgb.width, height=rgb.height)


def _pdf_from_pages(pages: list[_PdfPage]) -> bytes:
    """Write a PDF with one page per image, each page the image's size at ``_PAGE_DPI``.

    Each image is a DCTDecode (JPEG) image XObject drawn over the whole page.
    """
    page_ids = [_FIRST_PAGE_OBJECT_ID + index * _OBJECTS_PER_PAGE + 1 for index in range(len(pages))]
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects: list[bytes] = [
        f"<< /Type /Catalog /Pages {_PAGES_ID} 0 R >>".encode("ascii"),
        f"<< /Type /Pages /Kids [ {kids} ] /Count {len(pages)} >>".encode("ascii"),
    ]
    scale = _POINTS_PER_INCH / _PAGE_DPI
    for page, page_id in zip(pages, page_ids, strict=True):
        image_id = page_id - 1
        contents_id = page_id + 1
        page_width = page.width * scale
        page_height = page.height * scale
        image_header = (
            f"<< /Type /XObject /Subtype /Image /Width {page.width} /Height {page.height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(page.jpeg)} >>"
        )
        objects.append(image_header.encode("ascii") + b"\nstream\n" + page.jpeg + b"\nendstream")
        objects.append(
            (
                f"<< /Type /Page /Parent {_PAGES_ID} 0 R /MediaBox [ 0 0 {page_width:g} {page_height:g} ] "
                f"/Resources << /ProcSet [ /PDF /ImageC ] /XObject << /image {image_id} 0 R >> >> "
                f"/Contents {contents_id} 0 R >>"
            ).encode("ascii")
        )
        contents = f"q {page_width:g} 0 0 {page_height:g} 0 0 cm /image Do Q".encode("ascii")
        objects.append(f"<< /Length {len(contents)} >>".encode("ascii") + b"\nstream\n" + contents + b"\nendstream")

    output = bytearray(_PDF_HEADER)
    offsets = []
    for object_id, body in enumerate(objects, start=_CATALOG_ID):
        offsets.append(len(output))
        output += f"{object_id} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_offset = len(output)
    # Object 0 heads the free list; each entry is exactly 20 bytes
    output += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")
    for offset in offsets:
        output += f"{offset:010d} 00000 n \n".encode("ascii")
    output += (
        f"trailer\n<< /Size {len(objects) + 1} /Root {_CATALOG_ID} 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    ).encode("ascii")
    return bytes(output)


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
            item_converters=[
                partial(normalize_artifact_input, artifact_type=ImageUrlArtifact, accepted_types=(ImageArtifact,))
            ],
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

    def validate_before_node_run(self) -> list[Exception] | None:
        """Validate that images list is not empty."""
        exceptions: list[Exception] = []

        images = self.get_parameter_list_value("images") or []
        if not images:
            msg = "Connect at least one image to 'Images'."
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def process(self) -> None:
        """Convert list of images to a multi-page PDF file."""
        images = self.get_parameter_list_value("images") or []

        # Normalize string paths to ImageUrlArtifact during processing
        # (handles cases where values come from connections and bypass after_value_set)
        images = normalize_artifact_list(images, ImageUrlArtifact, accepted_types=(ImageArtifact,))

        logger.info(f"{self.name}: Converting {len(images)} images to PDF")

        # Load all images and convert to RGB pages
        pages = []
        for idx, image_artifact in enumerate(images):
            if not isinstance(image_artifact, ImageUrlArtifact):
                msg = f"Item {idx + 1} in 'Images' is a {type(image_artifact).__name__}, not an image."
                raise TypeError(msg)

            image = load_image_from_url(image_artifact.value)

            # Flatten onto white (the JPEG each page holds has no alpha)
            pages.append(_pdf_page_from_image(image))

        if not pages:
            msg = "'Images' has no images to convert. Connect at least one image."
            raise ValueError(msg)

        # Create PDF in memory
        pdf_bytes = _pdf_from_pages(pages)

        # Save PDF
        dest = self._output_file.build_file()
        saved = dest.write_bytes(pdf_bytes)
        logger.debug(f"{self.name}: PDF saved as {saved.location}")

        # Set output
        self.parameter_output_values["output"] = UrlArtifact(saved.location)
