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


class ImagePreviewMixin(BaseNode):
    """Gives an image node a live preview of its result above its settings.

    A node names its operation in PREVIEW_OPERATION and the parameters that drive it in
    PREVIEW_PARAMETERS, then calls _add_preview_parameter() after adding its input image. The
    ImageProcessorPreview widget draws that operation in the browser, reading those parameters directly
    as synced_parameters, so it redraws while the user drags them. The node only pushes the URLs of the
    images in PREVIEW_IMAGES; it still processes the real image when it runs.
    """

    PREVIEW_OPERATION: ClassVar[str | None] = None
    PREVIEW_PARAMETERS: ClassVar[list[str]] = []
    # Image parameters the preview shows, mapped to the key the widget reads each one's URL from
    PREVIEW_IMAGES: ClassVar[dict[str, str]] = {"input_image": "image_url"}

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Give the preview widget the new image when one of its images changes."""
        if self.PREVIEW_OPERATION is not None and parameter.name in self.PREVIEW_IMAGES:
            self._update_preview()
        return super().after_value_set(parameter, value)

    def _add_preview_parameter(self) -> None:
        """Add the preview widget. It shows the input image, so the plain input row would be a second copy."""
        if self.PREVIEW_OPERATION is None:
            return

        input_image = self.get_parameter_by_name("input_image")
        if input_image is not None:
            input_image.hide_property = True

        self.add_parameter(
            ParameterDict(
                name="preview",
                default_value={"operation": self.PREVIEW_OPERATION},
                tooltip="Preview of the result. The image is only processed when the node runs.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Widget(
                        name="ImageProcessorPreview",
                        library="Griptape Nodes Library",
                        synced_parameters=list(self.PREVIEW_PARAMETERS),
                    )
                },
            )
        )

    def _update_preview(self) -> None:
        """Give the preview widget browser URLs for its images. Its settings reach it from their own parameters."""
        preview: dict[str, Any] = {"operation": self.PREVIEW_OPERATION}
        for parameter_name, key in self.PREVIEW_IMAGES.items():
            preview[key] = self._resolve_image_url(self.get_parameter_value(parameter_name))
        if preview == self.get_parameter_value("preview"):
            return
        self.set_parameter_value("preview", preview)
        self.publish_update_to_parameter("preview", preview)

    def _resolve_image_url(self, image: Any) -> str:
        """Return a browser-accessible URL for an image artifact or its dict form."""
        raw = image.get("value") if isinstance(image, dict) else getattr(image, "value", image)
        if not raw or not isinstance(raw, str):
            return ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw
        try:
            resolved = File(raw).resolve()
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
        except Exception as e:
            logger.warning(f"{self.name}: could not build a preview URL for {raw!r}: {e}")
            return raw
        if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            return result.url
        return raw
