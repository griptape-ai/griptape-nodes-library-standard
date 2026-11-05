import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline, ImageFormat
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.file_utils import generate_filename
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_image_from_url,
    validate_image_format,
)

# Formats with no alpha channel, where a transparent image is flattened onto white
_OPAQUE_FORMATS = frozenset({ImageFormat.JPEG, ImageFormat.BMP})
_WHITE = (255, 255, 255)


class BaseImageProcessor(SuccessFailureNode, ABC):
    """Base class for image processing nodes with common functionality."""

    # Default image properties constants
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080
    DEFAULT_QUALITY = 95

    # Common resample filter options for different quality needs
    RESAMPLE_FILTER_OPTIONS: ClassVar[dict[str, str]] = {
        "nearest": "Nearest neighbor (fastest, lowest quality)",
        "box": "Box (good for downscaling)",
        "bilinear": "Bilinear (good balance)",
        "hamming": "Hamming (good for downscaling)",
        "bicubic": "Bicubic (high quality, slower)",
        "lanczos": "Lanczos (highest quality, slowest)",
    }

    # Common image format options
    IMAGE_FORMAT_OPTIONS: ClassVar[dict[str, str]] = {
        "auto": "Auto (preserve input format)",
        "PNG": "PNG (lossless, good for graphics)",
        "JPEG": "JPEG (lossy, good for photos)",
        "WEBP": "WebP (modern, good compression)",
    }

    # Extended format options for specialized use cases
    EXTENDED_FORMAT_OPTIONS: ClassVar[dict[str, str]] = {
        "BMP": "BMP (uncompressed, large files)",
        "TIFF": "TIFF (high quality, large files)",
    }

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add image input parameter
        self.add_parameter(
            ParameterImage(
                name="input_image",
                tooltip="The image to process",
                ui_options={
                    "clickable_file_browser": True,
                    "expander": True,
                },
            )
        )

        self._setup_custom_parameters()

        # Add output format parameter
        format_param = ParameterString(
            name="output_format",
            default_value="auto",
            tooltip="Output image format. Choose 'auto' to preserve input format, or select a specific format.",
        )
        format_param.add_trait(Options(choices=list(self.IMAGE_FORMAT_OPTIONS.keys())))
        self.add_parameter(format_param)

        # Add quality parameter for lossy formats
        quality_param = ParameterString(
            name="quality",
            default_value=str(self.DEFAULT_QUALITY),
            tooltip="Image quality for lossy formats (1-100, higher is better)",
        )
        quality_param.add_trait(Options(choices=["50", "60", "70", "80", "85", "90", "95", "100"]))
        self.add_parameter(quality_param)

        # Add output parameter
        self.add_parameter(
            ParameterImage(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The processed image",
                ui_options={"pulse_on_run": True, "expander": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="output.png",
        )
        self._output_file.add_parameter()

        # Add status parameters using the SuccessFailureNode helper method
        self._create_status_parameters(
            result_details_tooltip="Details about the image processing result",
            result_details_placeholder="Details on the image processing will be presented here.",
            parameter_group_initially_collapsed=True,
        )

        self._setup_logging_group()

    @abstractmethod
    def _setup_custom_parameters(self) -> None:
        """Setup custom parameters specific to this image processor. Override in subclasses."""

    @abstractmethod
    def _get_processing_description(self) -> str:
        """Get a description of what this processor does. Override in subclasses."""

    @abstractmethod
    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the image. Override in subclasses."""

    def _setup_logging_group(self) -> None:
        """Setup the common logging parameter group."""
        with ParameterGroup(name="Logs") as logs_group:
            Parameter(
                name="logs",
                type="str",
                tooltip="Displays processing logs and detailed events if enabled.",
                ui_options={"multiline": True, "placeholder_text": "Logs"},
                allowed_modes={ParameterMode.OUTPUT},
            )
        logs_group.ui_options = {"hide": True}  # Hide the logs group by default
        self.add_node_element(logs_group)

    def _validate_image_input(self) -> list[Exception] | None:
        """Common image input validation."""
        exceptions = []

        # Only validate if there's actually a value to validate
        image = self.parameter_values.get("input_image")
        if image is not None and (not hasattr(image, "value") or not image.value):
            msg = "'Input Image' has no image data. Connect a valid image to 'Input Image'."
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_image_input_data(self) -> tuple[GriptapeImage, str]:
        """Get the image and its detected format."""
        image = self.parameter_values.get("input_image")

        if not image:
            msg = "Connect an image to 'Input Image'."
            raise ValueError(msg)

        # Convert to ImageUrlArtifact if needed
        if hasattr(image, "to_dict"):
            image = dict_to_image_url_artifact(image.to_dict())

        # Ensure we have a valid image artifact with a value
        if not hasattr(image, "value") or not image.value:
            msg = "'Input Image' has no image data. Connect a valid image to 'Input Image'."
            raise ValueError(msg)

        loaded = load_image_from_url(image.value)

        # Detect format
        detected_format = self._detect_image_format(loaded)

        return loaded, detected_format

    def _detect_image_format(self, image: GriptapeImage) -> str:
        """Detect the format of the image."""
        match image.format:
            case ImageFormat.MPO:
                # MPO is read, not written; its first frame is a JPEG
                return ImageFormat.JPEG
            case ImageFormat() as image_format:
                return image_format
            case None:
                # Default to PNG if format can't be detected
                return ImageFormat.PNG
            case _:
                msg = f"Unknown image format: {image.format!r}"
                raise ValueError(msg)

    def _get_output_format(self, input_format: str) -> str:
        """Get the output format based on user preference."""
        output_format = self.get_parameter_value("output_format") or "auto"

        if output_format == "auto":
            return input_format

        return output_format

    def _get_quality_setting(self) -> int:
        """Get the quality setting for lossy formats."""
        quality_value = self.get_parameter_value("quality")
        if quality_value is None:
            return self.DEFAULT_QUALITY
        return int(quality_value)

    def _create_temp_output_file(self, format_extension: str) -> tuple[str, Path]:
        """Create a temporary output file and return path."""
        with tempfile.NamedTemporaryFile(suffix=f".{format_extension.lower()}", delete=False) as output_file:
            output_path = Path(output_file.name)
        return str(output_path), output_path

    def _prepared_for_format(self, image: GriptapeImage, image_format: ImageFormat) -> GriptapeImage:
        """Return the image ready to encode as ``image_format``.

        A transparent image is flattened onto white for formats without alpha. Other channel
        layouts a format can't hold are converted when the image is encoded.
        """
        if image_format in _OPAQUE_FORMATS and image.channels in (2, 4):
            logger.warning(f"{self.name}: Flattening transparency onto white for {image_format} format")
            return GriptapeImagePipeline(image).flatten(_WHITE).end()
        return image

    def _encode(self, image: GriptapeImage, format_extension: str) -> bytes:
        """Encode the image in the named format, at the quality setting for lossy formats.

        Metadata the image carries (text, EXIF, color profile, XMP) is kept where the format can
        hold it.

        Raises:
            ValueError: If the format can't be written
        """
        try:
            image_format = validate_image_format(format_extension, "output_format")
        except ValueError as e:
            msg = f"Unsupported image format '{format_extension}'. Pick another 'Output Format'."
            raise ValueError(msg) from e
        prepared = self._prepared_for_format(image, image_format)
        match image_format:
            case ImageFormat.JPEG:
                return prepared.encode(image_format, quality=self._get_quality_setting(), optimize=True)
            case ImageFormat.WEBP:
                return prepared.encode(image_format, quality=self._get_quality_setting())
            case ImageFormat.PNG:
                return prepared.encode(image_format, optimize=True)
            case _:
                return prepared.encode(image_format)

    def _save_image_artifact(self, image: GriptapeImage, format_extension: str) -> ImageUrlArtifact:
        """Save the image and return an ImageUrlArtifact."""
        image_bytes = self._encode(image, format_extension)
        dest = self._output_file.build_file()
        saved = dest.write_bytes(image_bytes)
        return ImageUrlArtifact(saved.location)

    def _image_to_bytes(self, image: GriptapeImage, format_extension: str) -> bytes:
        """Encode the image in the specified format, at the quality setting for lossy formats."""
        return self._encode(image, format_extension)

    def _cleanup_temp_file(self, file_path: Path) -> None:
        """Clean up temporary file with error handling."""
        try:
            file_path.unlink(missing_ok=True)
        except Exception as e:
            self.append_value_to_parameter("logs", f"Warning: Failed to clean up temporary file: {e}\n")

    def _log_image_properties(self, image: GriptapeImage) -> None:
        """Log detected image properties."""
        self.append_value_to_parameter(
            "logs",
            f"Processing image: {image.width}x{image.height}, channels: {image.channels}, "
            f"pixel type: {image.pixel_type}, format: {image.format}\n",
        )

    def _log_format_detection(self, detected_format: str) -> None:
        """Log detected image format."""
        self.append_value_to_parameter("logs", f"Detected image format: {detected_format}\n")

    def validate_before_node_run(self) -> list[Exception] | None:
        """Common image input validation."""
        exceptions = []

        # Use base class validation for image input
        base_exceptions = self._validate_image_input()
        if base_exceptions:
            exceptions.extend(base_exceptions)

        # Add custom validation from subclasses
        custom_exceptions = self._validate_custom_parameters()
        if custom_exceptions:
            exceptions.extend(custom_exceptions)

        # If there are validation errors, set the status to failure
        if exceptions:
            error_messages = [str(e) for e in exceptions]
            error_details = f"Validation failed: {'; '.join(error_messages)}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")

        return exceptions if exceptions else None

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate custom parameters. Override in subclasses if needed."""
        return None

    def _process(self, image: GriptapeImage, detected_format: str, **kwargs) -> None:
        """Common processing wrapper."""
        self.append_value_to_parameter("logs", f"{self._get_processing_description()}\n")

        # Process image using the custom implementation
        processed_image = self._process_image(image, **kwargs)

        # Get output format
        output_format = self._get_output_format(detected_format)

        # Get output suffix from subclass
        suffix = self._get_output_suffix(**kwargs)

        # Save processed image
        output_artifact = self._save_image_artifact(processed_image, output_format)

        self.append_value_to_parameter(
            "logs", f"Successfully processed image with suffix: {suffix}.{output_format.lower()}\n"
        )

        # Save to parameter
        self.parameter_output_values["output"] = output_artifact

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get the output filename suffix. Override in subclasses if needed."""
        return ""

    def process(self) -> AsyncResult[None]:
        """Main workflow execution method following LoadImage pattern."""
        # Reset execution state and result details at the start of each run
        self._clear_execution_status()

        # Clear output values to prevent downstream nodes from getting stale data on errors
        self.parameter_output_values["output"] = None

        # Get input image
        input_image = self.get_parameter_value("input_image")

        if input_image is None:
            error_details = "No input image provided"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            self._handle_failure_exception(RuntimeError(error_details))
            return

        try:
            # Get image input data
            image, detected_format = self._get_image_input_data()
            self._log_format_detection(detected_format)
            self._log_image_properties(image)

            # Get custom parameters from subclasses
            custom_params = self._get_custom_parameters()

            # Initialize logs
            self.append_value_to_parameter("logs", f"[Processing {self._get_processing_description()}..]\n")

            # Run the image processing
            self.append_value_to_parameter("logs", "[Started image processing..]\n")
            yield lambda: self._process(image, detected_format, **custom_params)
            self.append_value_to_parameter("logs", "[Finished image processing.]\n")

            # Success case
            success_details = (
                f"Successfully processed image: {self._get_processing_description()} ({image.width}x{image.height})"
            )
            self._set_status_results(was_successful=True, result_details=f"SUCCESS: {success_details}")
            logger.info(f"{self.__class__.__name__} '{self.name}': {success_details}")

        except Exception as e:
            error_details = f"Failed to process image: {e}"
            self._set_status_results(was_successful=False, result_details=f"FAILURE: {error_details}")
            self._handle_failure_exception(e)

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get custom parameters for processing. Override in subclasses if needed."""
        return {}

    def _generate_filename(self, suffix: str = "", extension: str = "png") -> str:
        """Generate a meaningful filename based on workflow and node information."""
        return generate_filename(
            node_name=self.name,
            suffix=suffix,
            extension=extension,
        )

    def _generate_processed_image_filename(self, extension: str = "png") -> str:
        """Generate a meaningful filename for processed images with processing parameters."""
        # Get the processing suffix from the node's _get_output_suffix method if it exists
        processing_suffix = ""
        try:
            # Get current processing parameters
            custom_params = self._get_custom_parameters()
            if custom_params:
                processing_suffix = self._get_output_suffix(**custom_params)
        except Exception:
            # If _get_output_suffix doesn't exist or fails, use empty suffix
            processing_suffix = ""

        return self._generate_filename(processing_suffix, extension)
