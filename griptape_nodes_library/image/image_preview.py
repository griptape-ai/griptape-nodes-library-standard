import tempfile
from pathlib import Path
from typing import Any, ClassVar

import PIL
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
from PIL import ImageFont


class ImagePreviewMixin(BaseNode):
    """Gives an image node a live preview of its result above its settings.

    A node names its operation in PREVIEW_OPERATION and the parameters that drive it in
    PREVIEW_PARAMETERS, then calls _add_preview_parameter() after adding its input image. The
    ImageProcessorPreview widget draws that operation in the browser, reading those parameters directly
    as synced_parameters, so it redraws while the user drags them. The node only pushes the URLs of the
    images in PREVIEW_IMAGES, plus anything _preview_extras() adds; it still processes the real image
    when it runs.
    """

    PREVIEW_OPERATION: ClassVar[str | None] = None
    PREVIEW_PARAMETERS: ClassVar[list[str]] = []
    # Image parameters the preview shows, mapped to the key the widget reads each one's URL from
    PREVIEW_IMAGES: ClassVar[dict[str, str]] = {"input_image": "image_url"}
    # A node that already has a parameter called "preview" names the widget's parameter something else
    PREVIEW_PARAMETER_NAME: ClassVar[str] = "preview"

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Give the preview widget the new image when one of its images changes."""
        if self.PREVIEW_OPERATION is not None and parameter.name in self.PREVIEW_IMAGES:
            self._update_preview()
        return super().after_value_set(parameter, value)

    def _add_preview_parameter(self) -> None:
        """Add the preview widget. It shows its images, so their plain input rows would be second copies."""
        if self.PREVIEW_OPERATION is None:
            return

        for parameter_name in self.PREVIEW_IMAGES:
            shown = self.get_parameter_by_name(parameter_name)
            if shown is not None:
                shown.hide_property = True

        self.add_parameter(
            ParameterDict(
                name=self.PREVIEW_PARAMETER_NAME,
                default_value={"operation": self.PREVIEW_OPERATION, **self._preview_extras()},
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

    def _preview_extras(self) -> dict[str, Any]:
        """Data the widget needs that isn't a parameter, such as an image's mode. Override to add some."""
        return {}

    def _update_preview(self) -> None:
        """Give the preview widget browser URLs for its images. Its settings reach it from their own parameters."""
        preview: dict[str, Any] = {"operation": self.PREVIEW_OPERATION}
        for parameter_name, key in self.PREVIEW_IMAGES.items():
            preview[key] = self._resolve_image_url(self.get_parameter_value(parameter_name))
        preview.update(self._preview_extras())
        if preview == self.get_parameter_value(self.PREVIEW_PARAMETER_NAME):
            return
        self.set_parameter_value(self.PREVIEW_PARAMETER_NAME, preview)
        self.publish_update_to_parameter(self.PREVIEW_PARAMETER_NAME, preview)

    def _resolve_image_url(self, image: Any) -> str:
        """Return a browser-accessible URL for an image artifact or its dict form."""
        raw = image.get("value") if isinstance(image, dict) else getattr(image, "value", image)
        if not raw or not isinstance(raw, str):
            return ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw
        return self._static_url(raw)

    def _static_url(self, path: str) -> str:
        """Return the static server's URL for a file, or the path itself when there is no server to ask."""
        try:
            resolved = File(path).resolve()
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
        except Exception as e:
            logger.warning(f"{self.name}: could not build a preview URL for {path!r}: {e}")
            return path
        if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            return result.url
        return path

    def _pillow_font_url(self) -> str:
        """URL of the font Pillow's ImageFont.load_default uses, so text previews match the node.

        The font comes from the installed Pillow rather than the library. It is written to the same path
        every time, so a URL saved with a workflow still resolves once a node has been created again.
        """
        path = Path(tempfile.gettempdir()) / f"griptape-nodes-pillow-{PIL.__version__}-default-font.ttf"
        if not path.exists():
            font = ImageFont.load_default(size=10)
            if not isinstance(font, ImageFont.FreeTypeFont) or font.font_bytes is None:
                return ""
            path.write_bytes(font.font_bytes)
        return self._static_url(str(path))
