import pathlib
from enum import StrEnum
from io import BytesIO
from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, NodeDependencies, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.files.project_file import ProjectFileDestination
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.file_system_picker import FileSystemPicker
from griptape_nodes.traits.options import Options
from PIL import Image
from pypdf import PdfReader

from griptape_nodes_library.image.image_preview import ImagePreviewMixin
from griptape_nodes_library.utils.versioned_output_directory import (
    build_directory_parameter,
    clear_directory_value,
    configure_directory_with_settings_node,
    normalize_directory_value,
    resolve_output_dir,
)

DEFAULT_DIRECTORY = "{outputs}/extracted_v{###}"
DEFAULT_PREFIX = "page"
DEFAULT_PADDING = 3
PADDING_OPTIONS = ["1", "2", "3", "4", "5", "6", "7", "8"]

JPEG_QUALITY = 95
WEBP_QUALITY = 95
FULL_TURN = 360
PREVIEW_MAX_SIZE = 1000
PREVIEW_QUALITY = 80


class Rotation(StrEnum):
    NONE = "none"
    CLOCKWISE_90 = "90° clockwise"
    ROTATE_180 = "180°"
    COUNTER_CLOCKWISE_90 = "90° counter-clockwise"


CLOCKWISE_DEGREES = {
    Rotation.NONE: 0,
    Rotation.CLOCKWISE_90: 90,
    Rotation.ROTATE_180: 180,
    Rotation.COUNTER_CLOCKWISE_90: 270,
}

# Image.transpose rotates counter-clockwise, so a clockwise turn is the complementary constant.
TRANSPOSE_FOR_CLOCKWISE_DEGREES = {
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}


class OutputFormat(StrEnum):
    ORIGINAL = "original"
    PNG = "png"
    JPG = "jpg"
    WEBP = "webp"


# Image formats whose stored bytes can be written to disk unchanged.
STORED_FORMATS = {"JPEG": OutputFormat.JPG, "PNG": OutputFormat.PNG}


class ExtractedImage:
    """An image pulled out of a PDF page, encoded and ready to write."""

    def __init__(self, data: bytes, extension: str) -> None:
        self.data = data
        self.extension = extension


class ExtractPdfImages(ImagePreviewMixin, SuccessFailureNode):
    """Extract the images embedded in a PDF, such as the pages of a scanned document.

    Images come out at the resolution they are stored at in the PDF. Nothing is resampled.
    With the 'original' format and no rotation, JPEG images are written byte-for-byte.
    """

    PREVIEW_OPERATION = "rotate"
    PREVIEW_PARAMETERS: ClassVar[list[str]] = ["rotation"]
    # The preview shows the first image in the PDF, which is not a parameter, so _preview_extras supplies it.
    PREVIEW_IMAGES: ClassVar[dict[str, str]] = {}
    PREVIEW_PARAMETER_NAME = "rotation_preview"

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
            tooltip="Path to the local PDF file to extract images from.",
            ui_options={"placeholder_text": "Path to a PDF file"},
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
            ParameterString(
                name="password",
                default_value="",
                placeholder_text="Only needed for password-protected PDFs",
                tooltip="Optional password for password-protected PDFs. Leave blank if not needed.",
            )
        )

        self._preview_cache: tuple[tuple[str, str], str] | None = None
        self._add_preview_parameter()

        rotation = ParameterString(
            name="rotation",
            default_value=Rotation.NONE,
            tooltip="Rotate every extracted image. Use this when the scans are stored sideways. "
            "Any rotation the PDF already applies to a page is added on top of this.",
        )
        rotation.add_trait(Options(choices=[r.value for r in Rotation]))
        self.add_parameter(rotation)

        file_format = ParameterString(
            name="file_format",
            default_value=OutputFormat.ORIGINAL,
            tooltip="Image format to save. 'original' keeps the format stored in the PDF (JPEG or PNG), "
            "so scans are not re-encoded unless rotated.",
        )
        file_format.add_trait(Options(choices=[f.value for f in OutputFormat]))
        self.add_parameter(file_format)

        self.add_parameter(
            build_directory_parameter(
                default_directory=DEFAULT_DIRECTORY,
                tooltip=(
                    "Folder the images are saved in. Use {###} for auto-incrementing versions "
                    "(e.g. {outputs}/extracted_v{###} → extracted_v001, extracted_v002, …). "
                    "Click the gear to connect a Directory Output Settings node."
                ),
                on_configure_clicked=self._on_configure_output_dir_clicked,
                on_connection_removed=self._on_output_dir_connection_removed,
            )
        )

        self.add_parameter(
            ParameterString(
                name="file_prefix",
                default_value=DEFAULT_PREFIX,
                placeholder_text=DEFAULT_PREFIX,
                tooltip="Filename prefix for each saved image (e.g. 'page' → 'page_001.png').",
            )
        )

        padding = ParameterString(
            name="number_padding",
            default_value=str(DEFAULT_PADDING),
            tooltip="Number of digits used to zero-pad image numbers in filenames.",
        )
        padding.add_trait(Options(choices=PADDING_OPTIONS))
        self.add_parameter(padding)

        self.add_parameter(
            Parameter(
                name="images",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="list[ImageUrlArtifact]",
                type="list[ImageUrlArtifact]",
                default_value=[],
                tooltip="The extracted images, in page order. A page with several images contributes each of them.",
            )
        )

        self.add_parameter(
            Parameter(
                name="image_count",
                allowed_modes={ParameterMode.OUTPUT},
                output_type="int",
                default_value=0,
                tooltip="The number of images extracted.",
            )
        )

        self.add_parameter(
            ParameterString(
                name="output_directory",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Folder the images were saved in.",
                placeholder_text="This will be automatically set to the output directory",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the extraction, or why it failed.",
            result_details_placeholder="Results will appear here after the node runs.",
        )

    # ── Lifecycle hooks ─────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "directory":
            normalize_directory_value(self, value)
        super().after_value_set(parameter, value)

    def get_node_dependencies(self) -> NodeDependencies | None:
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()
        value = self.get_parameter_value("path")
        if value and isinstance(value, str):
            deps.static_files.add(value)
        return deps

    def validate_before_node_run(self) -> list[Exception] | None:
        if not self.get_parameter_value("path"):
            return [ValueError("Set 'path' to the PDF file to extract images from.")]
        return None

    # ── Output directory ────────────────────────────────────────────────────────

    def _on_configure_output_dir_clicked(
        self, _button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult:
        return configure_directory_with_settings_node(self, button_details)

    def _on_output_dir_connection_removed(self, *_: object) -> None:
        clear_directory_value(self)

    def _resolve_output_dir(self) -> pathlib.Path:
        return resolve_output_dir(self, DEFAULT_DIRECTORY)

    # ── Preview ─────────────────────────────────────────────────────────────────

    def _is_preview_source(self, parameter_name: str) -> bool:
        # A protected PDF has no preview until the password is right, so the password counts too.
        return parameter_name in {"path", "password"}

    def _preview_extras(self) -> dict[str, Any]:
        """The first image in the PDF, for the preview widget. The rotation reaches it from its own parameter."""
        path = self.get_parameter_value("path") or ""
        password = self.get_parameter_value("password") or ""
        if self._preview_cache is not None and self._preview_cache[0] == (path, password):
            return {"image_url": self._preview_cache[1]}
        try:
            image_url = self._build_preview_url(path) if path else ""
        except Exception as e:
            logger.warning(f"{self.name}: could not build a rotation preview for {path!r}: {e}")
            image_url = ""
        self._preview_cache = ((path, password), image_url)
        return {"image_url": image_url}

    def _build_preview_url(self, path: str) -> str:
        """Save a small copy of the PDF's first image to temp storage and return its URL."""
        reader = PdfReader(BytesIO(File(path).read_bytes()))
        if reader.is_encrypted and not reader.decrypt(self.get_parameter_value("password") or ""):
            return ""
        for page in reader.pages:
            for pdf_image in page.images:
                with Image.open(BytesIO(pdf_image.data)) as image:
                    thumbnail = _flatten_to_rgb(image)
                thumbnail.thumbnail((PREVIEW_MAX_SIZE, PREVIEW_MAX_SIZE))
                # The preview shows the rotation the PDF applies on top of the node's setting.
                page_turn = page.rotation % FULL_TURN
                if page_turn:
                    thumbnail = thumbnail.transpose(TRANSPOSE_FOR_CLOCKWISE_DEGREES[page_turn])
                buffer = BytesIO()
                thumbnail.save(buffer, format="JPEG", quality=PREVIEW_QUALITY)
                destination = ProjectFileDestination.from_situation(
                    "pdf_preview.jpg", "save_temp_file", node_name=self.name
                )
                saved = destination.write_bytes(buffer.getvalue())
                result = GriptapeNodes.handle_request(
                    CreateStaticFileDownloadUrlFromPathRequest(file_path=File(saved.location).resolve())
                )
                if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                    return result.url
                return ""
        return ""

    # ── Process ─────────────────────────────────────────────────────────────────

    def process(self) -> AsyncResult[None]:
        self.parameter_output_values["images"] = []
        self.parameter_output_values["image_count"] = 0
        self.parameter_output_values["output_directory"] = ""
        yield lambda: self._run_extraction()

    def _run_extraction(self) -> None:
        self._clear_execution_status()
        try:
            self._perform_extraction()
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self._handle_failure_exception(exc)
            return
        count = self.parameter_output_values["image_count"]
        self._set_status_results(
            was_successful=True,
            result_details=f"Extracted {count} image(s) to '{self.parameter_output_values['output_directory']}'.",
        )

    def _perform_extraction(self) -> None:
        path = self.get_parameter_value("path")
        password = self.get_parameter_value("password") or ""
        rotation = Rotation(self.get_parameter_value("rotation") or Rotation.NONE)
        output_format = OutputFormat(self.get_parameter_value("file_format") or OutputFormat.ORIGINAL)
        prefix = self.get_parameter_value("file_prefix") or DEFAULT_PREFIX
        padding = int(self.get_parameter_value("number_padding") or DEFAULT_PADDING)

        extracted = self._extract_images(File(path).read_bytes(), password, rotation, output_format)
        if not extracted:
            msg = f"No embedded images found in '{path}'. The PDF may hold text and vector graphics only."
            raise NodeError(msg)

        output_dir = self._resolve_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[ImageUrlArtifact] = []
        for index, image in enumerate(extracted, start=1):
            image_path = output_dir / f"{prefix}_{index:0{padding}d}{image.extension}"
            image_path.write_bytes(image.data)
            artifacts.append(ImageUrlArtifact(str(image_path.resolve())))

        self.parameter_output_values["images"] = artifacts
        self.parameter_output_values["image_count"] = len(artifacts)
        self.parameter_output_values["output_directory"] = str(output_dir)

    def _extract_images(
        self,
        pdf_bytes: bytes,
        password: str,
        rotation: Rotation,
        output_format: OutputFormat = OutputFormat.ORIGINAL,
    ) -> list[ExtractedImage]:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted and not reader.decrypt(password):
            msg = "The PDF is password protected and the password is missing or wrong. Set 'password'."
            raise NodeError(msg)

        extracted: list[ExtractedImage] = []
        for page in reader.pages:
            clockwise_degrees = (CLOCKWISE_DEGREES[rotation] + page.rotation) % FULL_TURN
            extracted.extend(self._encode(image.data, clockwise_degrees, output_format) for image in page.images)
        return extracted

    def _encode(self, data: bytes, clockwise_degrees: int, output_format: OutputFormat) -> ExtractedImage:
        with Image.open(BytesIO(data)) as image:
            stored_format = STORED_FORMATS.get(image.format or "")
            match output_format:
                case OutputFormat.ORIGINAL:
                    # Formats other than JPEG and PNG (fax-compressed TIFF, JPEG 2000) are re-encoded as PNG.
                    target = stored_format or OutputFormat.PNG
                case OutputFormat.PNG | OutputFormat.JPG | OutputFormat.WEBP:
                    target = output_format
                case _:
                    msg = f"Unknown file format: {output_format!r}"
                    raise ValueError(msg)
            extension = f".{target.value}"

            if clockwise_degrees == 0 and target == stored_format:
                # Untouched images keep their stored bytes, so JPEGs are not re-encoded.
                return ExtractedImage(data, extension)

            if clockwise_degrees:
                image = image.transpose(TRANSPOSE_FOR_CLOCKWISE_DEGREES[clockwise_degrees])  # noqa: PLW2901
            return ExtractedImage(_save_as(image, target), extension)


def _flatten_to_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB copy, flattening any transparency onto white."""
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    return image.convert("RGB")


def _save_as(image: Image.Image, target: OutputFormat) -> bytes:
    buffer = BytesIO()
    match target:
        case OutputFormat.JPG:
            if image.mode not in ("RGB", "L", "CMYK"):
                image = _flatten_to_rgb(image)  # noqa: PLW2901
            image.save(buffer, format="JPEG", quality=JPEG_QUALITY, subsampling=0)
        case OutputFormat.PNG:
            if image.mode == "CMYK":
                image = image.convert("RGB")  # noqa: PLW2901
            image.save(buffer, format="PNG")
        case OutputFormat.WEBP:
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGBA" if "A" in image.mode else "RGB")  # noqa: PLW2901
            image.save(buffer, format="WEBP", quality=WEBP_QUALITY)
        case _:
            msg = f"Cannot save as {target!r}."
            raise ValueError(msg)
    return buffer.getvalue()
