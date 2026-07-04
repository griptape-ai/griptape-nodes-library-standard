from enum import StrEnum, auto
from io import BytesIO
from pathlib import Path
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger
from PIL import Image

from griptape_nodes_library.utils.situation_utils import add_situation_parameter

from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    image_to_bytes,
    load_image_from_url_artifact,
    validate_pil_format,
)

PREVIEW_LENGTH = 50


class SaveImageStatus(StrEnum):
    """Status enum for save image operations."""

    SUCCESS = auto()
    WARNING = auto()
    FAILURE = auto()


def to_image_artifact(image: ImageArtifact | dict) -> ImageArtifact | ImageUrlArtifact:
    """Normalise a raw dict wire-value into an artifact; pass ImageArtifact through unchanged."""
    if isinstance(image, dict):
        return dict_to_image_url_artifact(image)
    return image


class SaveImage(SuccessFailureNode):
    """Save an image to a file."""

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        # Add image input parameter
        self.add_parameter(
            ParameterImage(
                name="image",
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                tooltip="The image to save to file",
            )
        )

        # Add status parameters using the helper method
        self._create_status_parameters(
            result_details_tooltip="Details about the image save operation result",
            result_details_placeholder="Details on the save attempt will be presented here.",
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="griptape_nodes.png",
        )
        add_situation_parameter(self, self._output_file)
        self._output_file.add_parameter()

    def after_value_set(self, parameter: Parameter, value: object, **kwargs: object) -> None:
        if parameter.name == "situation":
            self._output_file._situation_name = str(value)
        super().after_value_set(parameter, value, **kwargs)

    def _get_target_pil_format(self) -> str:
        """Determine the target PIL format string from the output_file parameter."""
        param_value = self.get_parameter_value("output_file")
        filename = param_value if isinstance(param_value, str) and param_value else self._output_file._default_filename
        ext = Path(filename).suffix.lstrip(".").upper()
        return "JPEG" if ext == "JPG" else ext

    def _extract_format_from_artifact(self, image_artifact: Any) -> str | None:
        """Detect the format of an image artifact (e.g. 'png', 'jpeg'), or None if unknown."""
        # Try to get format from PIL Image
        if hasattr(image_artifact, "value"):
            try:
                if isinstance(image_artifact.value, bytes):
                    pil_image = Image.open(BytesIO(image_artifact.value))
                    if pil_image.format:
                        return pil_image.format.lower()
            except Exception:
                logger.debug("Failed to extract format from PIL Image", exc_info=True)

        # Check artifact metadata
        if hasattr(image_artifact, "meta") and image_artifact.meta:
            meta = image_artifact.meta
            if isinstance(meta, dict):
                # Check for format in meta
                if "format" in meta and isinstance(meta["format"], str):
                    return meta["format"].lower()
                # Check for content_type (e.g., "image/png" -> "png")
                if "content_type" in meta and isinstance(meta["content_type"], str) and "/" in meta["content_type"]:
                    return meta["content_type"].split("/")[1].lower()

        return None

    def process(self) -> None:
        # Reset execution state and result details at the start of each run
        self._clear_execution_status()

        image = self.get_parameter_value("image")
        self.parameter_output_values["image"] = image

        if not image:
            # Blank image is a warning, not a failure
            warning_details = "No image provided to save"
            logger.warning(warning_details)
            self._handle_execution_result(
                status=SaveImageStatus.WARNING,
                saved_path="",
                input_info="No image input",
                details=warning_details,
            )
            return

        # Captured here so all downstream error paths can include source details
        input_info = self._get_input_info(image)

        # Convert ImageUrlArtifact to ImageArtifact if needed
        processed_image = image
        if isinstance(image, ImageUrlArtifact):
            try:
                processed_image = load_image_from_url_artifact(image)
            except Exception as e:
                error_details = f"Failed to load image from URL: {e!s}"
                self._handle_error_with_graceful_exit(error_details, e, input_info)
                return

        # Convert to appropriate artifact type
        try:
            image_artifact = to_image_artifact(processed_image)
        except Exception as e:
            error_details = f"Failed to convert image to artifact: {e!s}"
            self._handle_error_with_graceful_exit(error_details, e, input_info)
            return

        # Detect incoming format for logging
        source_format = self._extract_format_from_artifact(image_artifact)
        logger.debug("Source image format detected: %s", source_format)

        # Determine and validate target format from output filename
        target_format = self._get_target_pil_format()
        try:
            validate_pil_format(target_format)
        except ValueError as e:
            error_details = f"Unsupported output file extension: {e!s}"
            self._handle_error_with_graceful_exit(error_details, e, input_info)
            return

        # Re-encode image bytes to the target format and write to disk
        try:
            if isinstance(image_artifact, ImageUrlArtifact):
                # ``ImageUrlArtifact.to_bytes()`` issues an HTTP GET against ``self.value``,
                # which fails when the value is a project macro path emitted by an upstream
                # node that wrote through ``ProjectFileParameter`` (e.g. ``CreateColorBars``).
                image_bytes = File(image_artifact.value).read_bytes()
            else:
                image_bytes = image_artifact.to_bytes()
            pil_image = Image.open(BytesIO(image_bytes))
            if target_format == "JPEG" and pil_image.mode in ("RGBA", "LA", "P"):
                pil_image = pil_image.convert("RGB")
            converted_bytes = image_to_bytes(pil_image, target_format)
            dest = self._output_file.build_file()
            saved = dest.write_bytes(converted_bytes)
            saved_path = saved.location
        except Exception as e:
            error_details = f"Failed to save image: {e!s}"
            self._handle_error_with_graceful_exit(error_details, e, input_info)
            return

        # Success case — override the image output with the real saved path so
        # downstream nodes (and Nuke, via the End node) receive the on-disk file
        # rather than the transient localhost static-file URL that the input
        # artifact carries after ParameterImage normalization.
        # Mirrors the convention in CreateColorBars and FluxImageGeneration.
        self.parameter_output_values["image"] = ImageUrlArtifact(value=saved_path)
        success_details = "Image saved successfully"
        self._handle_execution_result(
            status=SaveImageStatus.SUCCESS,
            saved_path=saved_path,
            input_info=input_info,
            details=success_details,
        )
        logger.info("Saved image: %s", saved_path)

    def _get_input_info(self, image: Any) -> str:
        """Return a short human-readable description of the image source."""
        input_type = type(image).__name__
        if isinstance(image, dict):
            return f"Dictionary input with type: {image.get('type', 'unknown')}"
        if isinstance(image, ImageUrlArtifact):
            return f"ImageUrlArtifact with URL: {image.value}"
        return f"ImageArtifact of type: {input_type}"

    def _get_input_info_for_failure(self, image: Any) -> str:
        """Return a detailed description of the image source, including a value preview for dicts."""
        input_type = type(image).__name__
        if isinstance(image, dict):
            input_info = f"Dictionary input with type: {image.get('type', 'unknown')}"
            if "value" in image:
                value_str = str(image["value"])
                value_preview = value_str[:PREVIEW_LENGTH] + "..." if len(value_str) > PREVIEW_LENGTH else value_str
                input_info += f", value preview: {value_preview}"
            return input_info
        if isinstance(image, ImageUrlArtifact):
            return f"ImageUrlArtifact with URL: {image.value}"
        return f"ImageArtifact of type: {input_type}"

    def _handle_execution_result(
        self,
        status: SaveImageStatus,
        saved_path: str,
        input_info: str,
        details: str,
        exception: Exception | None = None,
    ) -> None:
        """Set status outputs and log based on the outcome of a save attempt."""
        match status:
            case SaveImageStatus.FAILURE:
                # Use the richer variant here — it includes a value preview for dict inputs
                detailed_input_info = self._get_input_info_for_failure(self.get_parameter_value("image"))

                failure_details = f"Image save failed\nInput: {detailed_input_info}\nError: {details}"

                if exception:
                    failure_details += f"\nException type: {type(exception).__name__}"
                    if exception.__cause__:
                        failure_details += f"\nCause: {exception.__cause__}"

                self._set_status_results(was_successful=False, result_details=f"{status}: {failure_details}")
                logger.error("Error saving image: %s", details)

            case SaveImageStatus.WARNING:
                result_details = f"No image to save (warning)\nInput: {input_info}\nResult: No file created"

                self._set_status_results(was_successful=True, result_details=f"{status}: {result_details}")

            case SaveImageStatus.SUCCESS:
                result_details = f"Image saved successfully\nInput: {input_info}\nSaved to: {saved_path}"

                self._set_status_results(was_successful=True, result_details=f"{status}: {result_details}")

    def _handle_error_with_graceful_exit(self, error_details: str, exception: Exception, input_info: str) -> None:
        """Record failure status and re-raise only when no failure output is connected."""
        self._handle_execution_result(
            status=SaveImageStatus.FAILURE,
            saved_path="",
            input_info=input_info,
            details=error_details,
            exception=exception,
        )
        # Use the helper to handle exception based on connection status
        self._handle_failure_exception(RuntimeError(error_details))
