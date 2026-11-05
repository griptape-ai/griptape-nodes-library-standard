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

from griptape_nodes_library.utils.image_utils import load_pil_from_url


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

    def set_parameter_value(
        self,
        param_name: str,
        value: Any,
        *,
        initial_setup: bool = False,
        emit_change: bool = True,
        skip_before_value_set: bool = False,
    ) -> None:
        """Keep the preview in step while a workflow loads.

        Loading sets values with initial_setup, which skips after_value_set, so a saved image would
        never reach the preview and a saved font URL would never be refreshed. The preview is worked out
        again whenever a value it is built from loads, so it ends up right whatever order they load in.
        """
        super().set_parameter_value(
            param_name,
            value,
            initial_setup=initial_setup,
            emit_change=emit_change,
            skip_before_value_set=skip_before_value_set,
        )
        if not initial_setup or self.PREVIEW_OPERATION is None:
            return
        if param_name == self.PREVIEW_PARAMETER_NAME or self._is_preview_source(param_name):
            self._update_preview()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Give the preview widget the new image when one of its images changes."""
        if self.PREVIEW_OPERATION is not None and self._is_preview_source(parameter.name):
            self._update_preview()
        elif parameter.name == self.PREVIEW_PARAMETER_NAME and isinstance(value, dict):
            self._refresh_session_extras(value)
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
                default_value={
                    "operation": self.PREVIEW_OPERATION,
                    **self._preview_extras(),
                    **self._preview_session_extras(),
                },
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

    def _is_preview_source(self, parameter_name: str) -> bool:
        """Whether the preview is built from this parameter's value. Override for sources it doesn't list."""
        return parameter_name in self.PREVIEW_IMAGES

    def _preview_extras(self) -> dict[str, Any]:
        """Data the widget needs that isn't a parameter, such as an image's mode. Override to add some."""
        return {}

    def _preview_session_extras(self) -> dict[str, Any]:
        """Data that only holds for this engine session, such as a URL the static server hands out.

        A saved workflow carries the values from the session it was saved in, so these are worked out
        again whenever the preview's value is set, including when the workflow loads.
        """
        return {}

    def _refresh_session_extras(self, value: dict[str, Any]) -> None:
        """Swap this session's values into a preview value that holds stale ones."""
        session = self._preview_session_extras()
        if all(value.get(key) == entry for key, entry in session.items()):
            return
        refreshed = {**value, **session}
        self.set_parameter_value(self.PREVIEW_PARAMETER_NAME, refreshed)
        self.publish_update_to_parameter(self.PREVIEW_PARAMETER_NAME, refreshed)

    def _update_preview(self) -> None:
        """Give the preview widget browser URLs for its images. Its settings reach it from their own parameters."""
        preview: dict[str, Any] = {"operation": self.PREVIEW_OPERATION}
        for parameter_name, key in self.PREVIEW_IMAGES.items():
            preview[key] = self._resolve_image_url(self.get_parameter_value(parameter_name))
        preview.update(self._preview_extras())
        preview.update(self._preview_session_extras())
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

    def _image_mode(self, parameter_name: str) -> str:
        """The Pillow mode of an image parameter's image, or "" when it can't be read.

        The browser decodes every image to RGBA, so nodes that treat RGB, RGBA and greyscale images
        differently send the mode with the preview.
        """
        return self._mode_of(self.get_parameter_value(parameter_name))

    def _mode_of(self, image: Any) -> str:
        """The Pillow mode of an image artifact or its dict form, or "" when it can't be read."""
        raw = image.get("value") if isinstance(image, dict) else getattr(image, "value", image)
        if not raw or not isinstance(raw, str):
            return ""
        try:
            return load_pil_from_url(raw).mode
        except Exception as e:
            logger.debug(f"{self.name}: could not read the mode of {raw!r} for the preview: {e}")
            return ""

    def _pillow_font_url(self) -> str:
        """URL of the font Pillow's ImageFont.load_default uses, so text previews match the node.

        The font comes from the installed Pillow rather than the library, written to a temporary file.
        Nodes return it from _preview_session_extras, since the path and URL differ between machines,
        Pillow versions and sessions. Where the static server can't serve the file, the preview falls
        back to the browser's sans-serif.
        """
        path = Path(tempfile.gettempdir()) / f"griptape-nodes-pillow-{PIL.__version__}-default-font.ttf"
        if not path.exists():
            font = ImageFont.load_default(size=10)
            if not isinstance(font, ImageFont.FreeTypeFont) or font.font_bytes is None:
                return ""
            path.write_bytes(font.font_bytes)
        return self._static_url(str(path))
