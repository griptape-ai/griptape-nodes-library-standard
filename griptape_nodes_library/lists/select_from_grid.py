import json
from typing import Any

from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.audio_utils import is_audio_url_artifact
from griptape_nodes_library.utils.video_utils import is_video_url_artifact

_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".avi", ".mkv"})
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"})

_THUMBNAIL_MAX_DIM = 300


class SelectFromGrid(ControlNode):
    """Displays a list of items in a selectable grid widget.

    Accepts a list of strings, image/video/audio artifacts, or dicts. The user
    selects items by clicking thumbnails in the grid. The output is a filtered
    list containing only the selected items in the same format as the input.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.list_input = Parameter(
            name="list",
            input_types=["list"],
            type="list",
            allowed_modes={ParameterMode.INPUT},
            tooltip="List of items to display in the grid. Supports strings, image/video/audio artifacts, and dicts.",
        )
        self.add_parameter(self.list_input)

        self.grid_param = ParameterDict(
            name="grid",
            default_value={"items": [], "selected_indices": [], "columns": 3, "layout": "square"},
            tooltip="Interactive grid selector — click items to select them.",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Widget(name="SelectFromGrid", library="Griptape Nodes Library")},
        )
        self.add_parameter(self.grid_param)

        self.selected_items = Parameter(
            name="selected_items",
            type="list",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The selected items from the grid, in the same format as the input list.",
        )
        self.add_parameter(self.selected_items)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == self.list_input.name:
            self._sync_grid_items(value)
        elif parameter.name == self.grid_param.name:
            self._update_output_from_grid(value)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        grid_value = self.get_parameter_value(self.grid_param.name) or {}
        selected_indices = grid_value.get("selected_indices", [])
        list_values = self.get_parameter_value(self.list_input.name) or []
        selected = [list_values[i] for i in selected_indices if isinstance(i, int) and i < len(list_values)]
        self.parameter_output_values[self.selected_items.name] = selected

    def _sync_grid_items(self, list_values: Any) -> None:
        """Serialize the incoming list to widget-friendly items and push to the grid parameter."""
        current = self.get_parameter_value(self.grid_param.name) or {}

        if not isinstance(list_values, list):
            self.set_parameter_value(
                self.grid_param.name,
                {
                    "columns": current.get("columns", 3),
                    "layout": current.get("layout", "square"),
                    "items": [],
                    "selected_indices": [],
                },
            )
            return

        widget_items = [self._serialize_item(item) for item in list_values]

        # Preserve the selection when the list length is unchanged (e.g. node re-run
        # with the same inputs). Reset only when items are added or removed, because
        # existing selected indices may no longer point to the right items.
        current_len = len(current.get("items", []))
        kept_indices = current.get("selected_indices", []) if len(widget_items) == current_len else []

        self.set_parameter_value(
            self.grid_param.name,
            {
                "columns": current.get("columns", 3),
                "layout": current.get("layout", "square"),
                "items": widget_items,
                "selected_indices": kept_indices,
            },
        )

    def _update_output_from_grid(self, grid_value: Any) -> None:
        """Update the output parameter when the user changes the selection in the widget."""
        if not isinstance(grid_value, dict):
            return
        selected_indices = grid_value.get("selected_indices", [])
        list_values = self.get_parameter_value(self.list_input.name) or []
        selected = [list_values[i] for i in selected_indices if isinstance(i, int) and i < len(list_values)]
        self.parameter_output_values[self.selected_items.name] = selected
        self.publish_update_to_parameter(self.selected_items.name, selected)

    def _serialize_item(self, item: Any) -> dict[str, Any]:
        """Convert a Python item to a JSON-serializable dict for the grid widget."""
        # Dicts with a "value" key — display the inner value, optionally with a label
        if isinstance(item, dict) and "value" in item:
            result = self._serialize_item(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result

        # Image artifacts
        if isinstance(item, (ImageUrlArtifact, ImageArtifact)):
            url = self._resolve_artifact_url(item)
            return {"type": "image", "url": url}

        # Video artifacts
        if is_video_url_artifact(item):
            url = self._resolve_url_string(item.value)
            return {"type": "video", "url": url}

        # Audio artifacts
        if is_audio_url_artifact(item) or isinstance(item, AudioArtifact):
            url = self._resolve_url_string(getattr(item, "value", ""))
            return {"type": "audio", "url": url}

        # Strings — detect media by file extension
        if isinstance(item, str):
            lower = item.lower()
            dot = lower.rfind(".")
            ext = lower[dot:] if dot != -1 else ""
            if ext in _IMAGE_EXTENSIONS:
                return {"type": "image", "url": self._resolve_image_url(item)}
            if ext in _VIDEO_EXTENSIONS:
                return {"type": "video", "url": self._resolve_url_string(item)}
            if ext in _AUDIO_EXTENSIONS:
                return {"type": "audio", "url": self._resolve_url_string(item)}
            return {"type": "text", "value": item}

        # Dicts without a "value" key — render as formatted JSON
        if isinstance(item, dict):
            return {"type": "dict", "value": json.dumps(item, indent=2, default=str)}

        # Primitives and anything else
        return {"type": "text", "value": str(item)}

    def _make_thumbnail(self, image_bytes: bytes) -> tuple[bytes, str]:
        """Resize image bytes to _THUMBNAIL_MAX_DIM on the longest side.

        Returns (thumbnail_bytes, extension). Falls back to original bytes + "png" on error.
        """
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            if max(img.width, img.height) > _THUMBNAIL_MAX_DIM:
                img.thumbnail((_THUMBNAIL_MAX_DIM, _THUMBNAIL_MAX_DIM), Image.LANCZOS)
            out = io.BytesIO()
            try:
                img.save(out, format="WEBP", quality=75)
                return out.getvalue(), "webp"
            except Exception:
                out = io.BytesIO()
                img.save(out, format="PNG")
                return out.getvalue(), "png"
        except Exception:
            return image_bytes, "png"

    def _resolve_artifact_url(self, artifact: ImageUrlArtifact | ImageArtifact) -> str:
        """Resolve an image artifact to a browser-accessible thumbnail URL."""
        if isinstance(artifact, ImageArtifact):
            try:
                thumb, ext = self._make_thumbnail(artifact.value)
                return GriptapeNodes.StaticFilesManager().save_static_file(
                    thumb, f"grid_thumb_{id(artifact)}.{ext}"
                )
            except Exception:
                return ""
        # ImageUrlArtifact — value is a URL string (possibly macro://)
        return self._resolve_image_url(artifact.value)

    def _resolve_image_url(self, path: str) -> str:
        """Resolve a path to a browser URL, thumbnailing local image files."""
        if not path:
            return ""
        if path.startswith(("http://", "https://")):
            return path
        try:
            resolved = File(path).resolve()
        except Exception:
            resolved = path
        # Load, thumbnail, and serve the local file
        try:
            import pathlib

            image_bytes = pathlib.Path(resolved).read_bytes()
            thumb, ext = self._make_thumbnail(image_bytes)
            return GriptapeNodes.StaticFilesManager().save_static_file(
                thumb, f"grid_thumb_{hash(resolved)}.{ext}"
            )
        except Exception:
            pass
        # Fallback: resolve to a static download URL without thumbnailing
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return path

    def _resolve_url_string(self, path: str) -> str:
        """Resolve a file path or macro:// URL to a browser-accessible HTTP URL."""
        if not path:
            return ""
        if path.startswith(("http://", "https://")):
            return path
        try:
            resolved = File(path).resolve()
        except Exception:
            resolved = path
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return path
