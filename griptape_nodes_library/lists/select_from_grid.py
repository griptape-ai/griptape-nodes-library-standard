import json
import pathlib
from typing import Any

from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape_nodes.common.macro_parser import ParsedMacro
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.artifact_events import (
    GetPreviewForArtifactRequest,
    GetPreviewForArtifactResultSuccess,
)
from griptape_nodes.retained_mode.events.project_events import MacroPath
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.audio_utils import is_audio_url_artifact
from griptape_nodes_library.utils.video_utils import is_video_url_artifact

# Used only when list items are plain strings — artifacts are detected by type, not extension.
_IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg"})
_VIDEO_EXTENSIONS = frozenset({".mp4", ".webm", ".mov", ".avi", ".mkv"})
_AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a"})

# Thumbnails are resolved in chunks so the browser receives incremental updates
# rather than waiting for all items to be processed in one silent blocking pass.
_RESOLVE_CHUNK_SIZE = 100


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
            default_value={
                "items": [],
                "selected_indices": [],
                "columns": 3,
                "layout": "grid",
                "settings": {"multi_select": True},
            },
            tooltip="Interactive grid selector — click items to select them.",
            allowed_modes={ParameterMode.PROPERTY},
            traits={Widget(name="SelectFromGrid", library="Griptape Nodes Library")},
        )
        self.add_parameter(self.grid_param)

        self.multi_select = Parameter(
            name="multi_select",
            type="bool",
            default_value=True,
            allowed_modes={ParameterMode.PROPERTY},
            tooltip="Allow selecting multiple items. When disabled, only one item can be selected at a time.",
        )
        self.add_parameter(self.multi_select)

        self.selected_items = Parameter(
            name="selected_items",
            output_type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The selected items from the grid, in the same format as the input list.",
        )
        self.add_parameter(self.selected_items)

        self.selected_item = Parameter(
            name="selected_item",
            output_type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            hide_property=True,
            tooltip="The selected item from the grid, in the same format as the input list.",
        )
        self.add_parameter(self.selected_item)
        self.hide_parameter_by_name("selected_item")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == self.list_input.name:
            self._sync_grid_items(value)
        elif parameter.name == self.grid_param.name:
            self._update_output_from_grid(value)
        elif parameter.name == self.multi_select.name:
            self._apply_multi_select(bool(value))
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        list_values = self.get_parameter_value(self.list_input.name) or []
        current = self.get_parameter_value(self.grid_param.name) or {}

        base: dict = {
            "columns": current.get("columns", 3),
            "layout": current.get("layout", "grid"),
            "settings": current.get("settings", {"multi_select": True}),
            "show_labels": current.get("show_labels", True),
            "label_size": current.get("label_size", 10),
        }
        kept_indices = current.get("selected_indices", [])

        # Phase 2 — resolve image previews via the engine's async preview generator,
        # pushing chunk updates so the browser receives thumbnails incrementally.
        widget_items = list(current.get("items", [self._serialize_item_placeholder(v) for v in list_values]))
        for chunk_start in range(0, len(list_values), _RESOLVE_CHUNK_SIZE):
            chunk_end = min(chunk_start + _RESOLVE_CHUNK_SIZE, len(list_values))
            changed = False
            for i in range(chunk_start, chunk_end):
                item = list_values[i]
                if self._is_image_item(item):
                    widget_items[i] = await self._serialize_image_item_async(item)
                    changed = True
                elif self._is_video_item(item):
                    widget_items[i] = await self._serialize_video_item_async(item)
                    changed = True
            if changed:
                self.set_parameter_value(
                    self.grid_param.name,
                    {**base, "items": list(widget_items), "selected_indices": kept_indices},
                )

        # Output selected items
        selected = [list_values[i] for i in kept_indices if isinstance(i, int) and i < len(list_values)]
        is_multi = self.get_parameter_value(self.multi_select.name)
        if is_multi is None:
            is_multi = True
        if is_multi:
            self.parameter_output_values[self.selected_items.name] = selected
        else:
            self.parameter_output_values[self.selected_item.name] = selected[0] if selected else None

    def _sync_grid_items(self, list_values: Any) -> None:
        """Phase 1 — push immediately-serializable items; image items show spinners until aprocess runs."""
        current = self.get_parameter_value(self.grid_param.name) or {}

        base: dict = {
            "columns": current.get("columns", 3),
            "layout": current.get("layout", "grid"),
            "settings": current.get("settings", {"multi_select": True}),
            "show_labels": current.get("show_labels", True),
            "label_size": current.get("label_size", 10),
        }

        if not isinstance(list_values, list):
            self.set_parameter_value(
                self.grid_param.name,
                {**base, "items": [], "selected_indices": []},
            )
            return

        current_len = len(current.get("items", []))
        # All items resolve synchronously in Phase 1 — images get a direct file URL
        # (no thumbnail generation), everything else resolves as normal. aprocess
        # upgrades image cells to engine-generated previews when the node runs.
        phase1_items = [self._serialize_item_sync(item) for item in list_values]
        kept_indices = current.get("selected_indices", []) if len(phase1_items) == current_len else []

        self.set_parameter_value(
            self.grid_param.name,
            {**base, "items": phase1_items, "selected_indices": kept_indices},
        )

    def _apply_multi_select(self, is_multi: bool) -> None:
        """Switch between multi-select and single-select mode."""
        current = self.get_parameter_value(self.grid_param.name) or {}
        settings = dict(current.get("settings", {}))
        settings["multi_select"] = is_multi
        self.set_parameter_value(
            self.grid_param.name,
            {**current, "settings": settings, "selected_indices": []},
        )
        if is_multi:
            self.show_parameter_by_name("selected_items")
            self.hide_parameter_by_name("selected_item")
        else:
            self.hide_parameter_by_name("selected_items")
            self.show_parameter_by_name("selected_item")

    def _update_output_from_grid(self, grid_value: Any) -> None:
        """Update the output parameter when the user changes the selection in the widget."""
        if not isinstance(grid_value, dict):
            return
        selected_indices = grid_value.get("selected_indices", [])
        list_values = self.get_parameter_value(self.list_input.name) or []
        selected = [list_values[i] for i in selected_indices if isinstance(i, int) and i < len(list_values)]
        is_multi = self.get_parameter_value(self.multi_select.name)
        if is_multi is None:
            is_multi = True
        if is_multi:
            self.parameter_output_values[self.selected_items.name] = selected
            self.publish_update_to_parameter(self.selected_items.name, selected)
        else:
            item = selected[0] if selected else None
            self.parameter_output_values[self.selected_item.name] = item
            self.publish_update_to_parameter(self.selected_item.name, item)

    @staticmethod
    def _is_video_item(item: Any) -> bool:
        """Return True if the item will be rendered as a video cell."""
        if isinstance(item, dict) and "value" in item:
            return SelectFromGrid._is_video_item(item["value"])
        if is_video_url_artifact(item):
            return True
        if isinstance(item, str):
            return pathlib.Path(item).suffix.lower() in _VIDEO_EXTENSIONS
        return False

    @staticmethod
    def _is_image_item(item: Any) -> bool:
        """Return True if the item will be rendered as an image cell."""
        if isinstance(item, dict) and "value" in item:
            return SelectFromGrid._is_image_item(item["value"])
        if isinstance(item, (ImageUrlArtifact, ImageArtifact)):
            return True
        if isinstance(item, str):
            return pathlib.Path(item).suffix.lower() in _IMAGE_EXTENSIONS
        return False

    @staticmethod
    def _extract_media_label(item: Any) -> str:
        """Return the filename component of a media item's path or URL, or empty string."""
        if isinstance(item, str):
            return pathlib.Path(item).name
        name = getattr(item, "name", "") or ""
        if name:
            return name
        value = getattr(item, "value", "") or ""
        return pathlib.Path(str(value)).name if isinstance(value, str) and value else ""

    def _serialize_item_placeholder(self, item: Any) -> dict[str, Any]:
        """Return a lightweight placeholder for an item with no resolved URL.

        The JS widget renders a loading spinner for any image/video item whose
        url is absent, allowing the grid to appear immediately while Phase 2
        resolves thumbnails.
        """
        if isinstance(item, dict) and "value" in item:
            result = self._serialize_item_placeholder(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result
        if isinstance(item, (ImageUrlArtifact, ImageArtifact)):
            label = self._extract_media_label(item)
            return {"type": "image", "url": "", **({"label": label} if label else {})}
        if is_video_url_artifact(item):
            label = self._extract_media_label(item)
            return {"type": "video", "url": "", **({"label": label} if label else {})}
        if is_audio_url_artifact(item) or isinstance(item, AudioArtifact):
            label = self._extract_media_label(item)
            return {"type": "audio", "url": "", **({"label": label} if label else {})}
        if isinstance(item, str):
            lower = item.lower()
            dot = lower.rfind(".")
            ext = lower[dot:] if dot != -1 else ""
            label = pathlib.Path(item).name if ext else ""
            if ext in _IMAGE_EXTENSIONS:
                return {"type": "image", "url": "", **({"label": label} if label else {})}
            if ext in _VIDEO_EXTENSIONS:
                return {"type": "video", "url": "", **({"label": label} if label else {})}
            if ext in _AUDIO_EXTENSIONS:
                return {"type": "audio", "url": "", **({"label": label} if label else {})}
            return {"type": "text", "value": item}
        if isinstance(item, dict):
            return {"type": "dict", "value": ""}
        return {"type": "text", "value": str(item)}

    def _serialize_item(self, item: Any) -> dict[str, Any]:
        """Convert a non-image item to a JSON-serializable dict for the grid widget.

        Image items are handled asynchronously in aprocess via _serialize_image_item_async.
        """
        if isinstance(item, dict) and "value" in item:
            result = self._serialize_item(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result

        if is_video_url_artifact(item):
            url = self._resolve_url_string(getattr(item, "value", ""))
            label = self._extract_media_label(item)
            return {"type": "video", "url": url, **({"label": label} if label else {})}

        if is_audio_url_artifact(item) or isinstance(item, AudioArtifact):
            url = self._resolve_url_string(getattr(item, "value", ""))
            label = self._extract_media_label(item)
            return {"type": "audio", "url": url, **({"label": label} if label else {})}

        if isinstance(item, str):
            lower = item.lower()
            dot = lower.rfind(".")
            ext = lower[dot:] if dot != -1 else ""
            label = pathlib.Path(item).name if ext else ""
            if ext in _VIDEO_EXTENSIONS:
                return {"type": "video", "url": self._resolve_url_string(item), **({"label": label} if label else {})}
            if ext in _AUDIO_EXTENSIONS:
                return {"type": "audio", "url": self._resolve_url_string(item), **({"label": label} if label else {})}
            if ext not in _IMAGE_EXTENSIONS:
                return {"type": "text", "value": item}

        if isinstance(item, dict):
            return {"type": "dict", "value": json.dumps(item, indent=2, default=str)}

        return {"type": "text", "value": str(item)}

    def _serialize_item_sync(self, item: Any) -> dict[str, Any]:
        """Like _serialize_item but resolves image paths to a direct URL without preview generation.

        Gives instant feedback in Phase 1; aprocess replaces these with proper engine previews.
        """
        if isinstance(item, dict) and "value" in item:
            result = self._serialize_item_sync(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result
        if isinstance(item, (ImageUrlArtifact, ImageArtifact)):
            path = getattr(item, "value", "")
            url = self._resolve_url_string(str(path))
            label = self._extract_media_label(item)
            return {"type": "image", "url": url, **({"label": label} if label else {})}
        if isinstance(item, str):
            ext = pathlib.Path(item).suffix.lower()
            label = pathlib.Path(item).name if ext else ""
            if ext in _IMAGE_EXTENSIONS:
                url = self._resolve_url_string(item)
                return {"type": "image", "url": url, **({"label": label} if label else {})}
        return self._serialize_item(item)

    async def _serialize_image_item_async(self, item: Any) -> dict[str, Any]:
        """Resolve an image item to a grid dict, generating a preview via the engine."""
        if isinstance(item, dict) and "value" in item:
            result = await self._serialize_image_item_async(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result
        label = self._extract_media_label(item)
        path = item if isinstance(item, str) else getattr(item, "value", "")
        url = await self._resolve_image_url_async(str(path))
        return {"type": "image", "url": url, **({"label": label} if label else {})}

    async def _serialize_video_item_async(self, item: Any) -> dict[str, Any]:
        """Resolve a video item, adding an engine-generated thumbnail as the poster frame."""
        if isinstance(item, dict) and "value" in item:
            result = await self._serialize_video_item_async(item["value"])
            if "label" in item:
                result["label"] = str(item["label"])
            return result
        label = self._extract_media_label(item)
        path = item if isinstance(item, str) else getattr(item, "value", "")
        url = self._resolve_url_string(str(path))
        thumbnail = await self._resolve_preview_url_async(str(path), "Video")
        return {
            "type": "video",
            "url": url,
            **({"thumbnail": thumbnail} if thumbnail else {}),
            **({"label": label} if label else {}),
        }

    async def _resolve_preview_url_async(self, path: str, provider: str) -> str:
        """Ask the engine to generate (or load cached) preview for any artifact type, return a browser URL."""
        if not path or path.startswith(("http://", "https://")):
            return ""
        try:
            resolved = File(path).resolve()
        except Exception:
            resolved = path
        try:
            result = await GriptapeNodes.ahandle_request(
                GetPreviewForArtifactRequest(
                    macro_path=MacroPath(ParsedMacro(resolved), {}),
                    artifact_provider_name=provider,
                )
            )
            if isinstance(result, GetPreviewForArtifactResultSuccess):
                preview_path = (
                    result.paths_to_preview
                    if isinstance(result.paths_to_preview, str)
                    else next(iter(result.paths_to_preview.values()))
                )
                url_result = GriptapeNodes.handle_request(
                    CreateStaticFileDownloadUrlFromPathRequest(file_path=preview_path)
                )
                if isinstance(url_result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                    return url_result.url
        except Exception:
            pass
        return ""

    async def _resolve_image_url_async(self, path: str) -> str:
        """Resolve an image path to a browser URL via the engine's preview generator."""
        if not path:
            return ""
        if path.startswith(("http://", "https://")):
            return path
        url = await self._resolve_preview_url_async(path, "Image")
        if url:
            return url
        # Fallback: serve the original file directly without preview generation
        try:
            resolved = File(path).resolve()
        except Exception:
            resolved = path
        try:
            url_result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved)
            )
            if isinstance(url_result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return url_result.url
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
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return path
