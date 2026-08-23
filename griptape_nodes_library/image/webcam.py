import logging
import re
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget

logger = logging.getLogger(__name__)

_IDLE: dict[str, Any] = {
    "state": "idle",
    "gallery_items": [],
    "selected_index": -1,
    "gallery_count": 0,
}


class Webcam(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._updating_selection = False
        self._items: list | None = None

        self.add_parameter(
            ParameterDict(
                name="snapshot",
                default_value={**_IDLE, "gallery_items": []},
                tooltip="Webcam snapshot widget. Capture frames and build a gallery.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="WebcamCapture", library="Griptape Nodes Library")},
            )
        )

        # Hidden parameter: sole Python-owned store for accumulated gallery items.
        # Never written to by JS events, so it is safe to read even when `snapshot`
        # has been overwritten by a new incoming capture.
        gallery_store = Parameter(
            name="gallery_store",
            type="list",
            default_value=[],
            allowed_modes={ParameterMode.PROPERTY},
            hide_property=True,
        )
        self.add_parameter(gallery_store)
        gallery_store.hide = True

        self.add_parameter(
            ParameterImage(
                name="image",
                tooltip="The selected image from the gallery (latest capture by default).",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="images",
                output_type=ParameterTypeBuiltin.ALL.value,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="All captured images.",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="webcam/snapshot.jpg",
        )
        self._output_file.add_parameter()

    # ── Gallery helpers ───────────────────────────────────────────────────────

    def _get_items(self) -> list:
        if self._items is None:
            self._items = list(self.get_parameter_value("gallery_store") or [])
        return list(self._items)

    def _commit_items(self, items: list) -> None:
        self._items = list(items)
        self.set_parameter_value("gallery_store", list(items))

    def _safe_node_name(self) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", self.name)

    def _temp_rel_path(self, seq: int) -> str:
        return f"temp/_wc_{self._safe_node_name()}_{seq}.jpg"

    # ── Parameter events ──────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "snapshot" and isinstance(value, dict):
            match value.get("state"):
                case "requesting_upload_url":
                    self._handle_requesting_upload_url(value)
                    return
                case "accepted":
                    self._handle_accepted(value)
                    return
                case "selected":
                    if not self._updating_selection:
                        self._handle_selected(value)
                    return
                case "clear_gallery":
                    self._handle_clear_gallery(value)
                    return
                case _:
                    pass
        return super().after_value_set(parameter, value)

    # ── State handlers ────────────────────────────────────────────────────────

    def _handle_requesting_upload_url(self, snapshot: dict) -> None:
        seq = snapshot.get("_emitSeq", 0)
        try:
            server_url = GriptapeNodes.StaticFilesManager().static_server_base_url
            rel_path = self._temp_rel_path(seq)
            upload_ready = {
                "state": "upload_ready",
                "_uploadUrl": f"{server_url}/static-uploads/{rel_path}",
                "_emitSeq": seq,
            }
            self.set_parameter_value("snapshot", upload_ready)
            self.publish_update_to_parameter("snapshot", upload_ready)
        except Exception:
            logger.warning("webcam [%s]: failed to build upload URL for seq %d; dropping capture", self.name, seq, exc_info=True)
            # Echo "processed" with no new item so the JS pending thumbnail clears
            # and the capture queue can continue draining.
            items = self._get_items()
            failed = {
                "state": "processed",
                "gallery_items": list(items),
                "selected_index": len(items) - 1 if items else -1,
                "gallery_count": len(items),
                "_emitSeq": seq,
            }
            self.set_parameter_value("snapshot", failed)
            self.publish_update_to_parameter("snapshot", failed)

    def _handle_accepted(self, snapshot: dict) -> None:
        seq = snapshot.get("_emitSeq", 0)
        workspace = GriptapeNodes.ConfigManager().workspace_path
        temp_path = workspace / self._temp_rel_path(seq)
        items = self._get_items()
        try:
            if not temp_path.exists():
                # Temp file missing means either the PUT failed (upload error) or
                # this is a duplicate after_value_set call (file already consumed).
                # Either way there's nothing to save — echo "processed" to let JS
                # clear the pending thumbnail and drain the queue.
                pass
            else:
                # Read and immediately delete so any duplicate call finds the file
                # gone and skips processing — idempotency without cross-session state.
                image_bytes = temp_path.read_bytes()
                temp_path.unlink(missing_ok=True)
                saved = self._output_file.build_file().write_bytes(image_bytes)
                artifact = ImageUrlArtifact(saved.location)
                url = self._resolve_url(artifact.value)
                items.append({"url": url, "_path": artifact.value})
                self._commit_items(items)
                self.parameter_output_values["image"] = artifact
                self.parameter_output_values["images"] = self._all_artifacts(items)
                self.publish_update_to_parameter("image", artifact)
                self.publish_update_to_parameter("images", self.parameter_output_values["images"])
        except Exception:
            logger.warning("webcam [%s]: failed to save snapshot for seq %d; dropping capture", self.name, seq, exc_info=True)

        # Always echo "processed" so the JS clears the pending thumbnail,
        # whether the save succeeded, failed, or was a duplicate call.
        processed = {
            "state": "processed",
            "gallery_items": list(items),
            "selected_index": len(items) - 1 if items else -1,
            "gallery_count": len(items),
            "_emitSeq": seq,
        }
        self.set_parameter_value("snapshot", processed)
        self.publish_update_to_parameter("snapshot", processed)

    def _handle_selected(self, snapshot: dict) -> None:
        self._updating_selection = True
        try:
            selected_index = snapshot.get("selected_index", 0)
            items = self._get_items()

            artifact = self._artifact_at(items, selected_index)
            self.parameter_output_values["image"] = artifact
            self.parameter_output_values["images"] = self._all_artifacts(items)
            self.publish_update_to_parameter("image", artifact)
            self.publish_update_to_parameter("images", self.parameter_output_values["images"])

            new_stored = {
                # Use "selection_confirmed", NOT "processed" — the JS "processed"
                # handler removes a pending thumb and resets _processing, which
                # would corrupt capture state if a capture is in-flight.
                "state": "selection_confirmed",
                "gallery_items": list(items),
                "selected_index": selected_index,
                "gallery_count": len(items),
                "_emitSeq": snapshot.get("_emitSeq", 0),
            }
            self.set_parameter_value("snapshot", new_stored)
            # publish_update_to_parameter intentionally omitted: JS handles selection
            # locally and deliberately ignores "selection_confirmed" in handleUpdate,
            # so broadcasting would be a no-op. Calling it could also race with an
            # in-flight capture that is listening for "processed".
        finally:
            self._updating_selection = False

    def _handle_clear_gallery(self, snapshot: dict) -> None:
        self._commit_items([])

        self.parameter_output_values["image"] = None
        self.parameter_output_values["images"] = []
        self.publish_update_to_parameter("image", None)
        self.publish_update_to_parameter("images", [])

        idle = {**_IDLE, "gallery_items": [], "_emitSeq": snapshot.get("_emitSeq", 0)}
        self.set_parameter_value("snapshot", idle)
        self.publish_update_to_parameter("snapshot", idle)

    # ── Artifact helpers ──────────────────────────────────────────────────────

    def _artifact_at(self, items: list, index: int) -> ImageUrlArtifact | None:
        if isinstance(index, int) and 0 <= index < len(items):
            path = items[index].get("_path", "")
            return ImageUrlArtifact(path) if path else None
        return None

    def _all_artifacts(self, items: list) -> list[ImageUrlArtifact]:
        return [ImageUrlArtifact(item["_path"]) for item in items if item.get("_path")]

    def _resolve_url(self, path: str) -> str:
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

    # ── Run ───────────────────────────────────────────────────────────────────

    def process(self) -> None:
        items = self._get_items()
        if not items:
            msg = "No images captured. Use the webcam widget to capture snapshots before running."
            raise ValueError(msg)
        stored = self.get_parameter_value("snapshot") or {}
        selected_index = stored.get("selected_index", len(items) - 1)
        artifact = self._artifact_at(items, selected_index) or self._artifact_at(items, len(items) - 1)
        self.parameter_output_values["image"] = artifact
        self.parameter_output_values["images"] = self._all_artifacts(items)
