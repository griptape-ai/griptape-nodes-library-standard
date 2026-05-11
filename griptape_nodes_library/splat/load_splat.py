"""LoadSplat — wired splat aggregator with self-contained inline viewer."""

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, DataNode
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.splat.parameter_splat import ParameterSplat
from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact

logger = logging.getLogger("griptape_nodes")

VIEWER_STATE_DEFAULTS: dict[str, Any] = {
    "splats": {"100k": None, "500k": None, "full_res": None},
    "defaults": {
        "flip_coordinates": True,
        "enable_lod": True,
        "lod_scale": 1.0,
        "max_sh": 3,
    },
}

INPUT_TO_RESOLUTION_KEY = {
    "splat_100k": "100k",
    "splat_500k": "500k",
    "splat_full_res": "full_res",
}
OUTPUT_PRIORITY = ["splat_full_res", "splat_500k", "splat_100k"]


class LoadSplat(DataNode):
    """View Gaussian splats inline.

    Wire one or more SplatUrlArtifacts (typically from WorldLabsWorldGeneration's
    three resolution outputs). The widget renders inline and exposes its own
    resolution dropdown, LoD knobs, and load button — pick what you want and
    click Load. No node re-run required to switch resolutions.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self.category = "Splat"
        self.description = "Aggregate wired splats and view inline (resolution + LoD selection live in the widget)."

        # --- Wired inputs ---------------------------------------------------
        self.add_parameter(
            ParameterSplat(
                name="splat_100k",
                tooltip="Wire a splat here, or click to pick a local file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                clickable_file_browser=True,
                ui_options={"display_name": "Splat (100k)"},
            )
        )
        self.add_parameter(
            ParameterSplat(
                name="splat_500k",
                tooltip="Wire a splat here, or click to pick a local file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                clickable_file_browser=True,
                ui_options={"display_name": "Splat (500k)"},
            )
        )
        self.add_parameter(
            ParameterSplat(
                name="splat_full_res",
                tooltip="Wire a splat here, or click to pick a local file.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                clickable_file_browser=True,
                ui_options={"display_name": "Splat (Full Res)"},
            )
        )

        # --- Widget surface (the viewer) ------------------------------------
        # JSON-serialized dict because dict-typed parameters don't propagate
        # value changes to widgets at edit time, but string-typed ones do.
        # The widget parses JSON on its end.
        self.add_parameter(
            Parameter(
                name="viewer_state",
                type="str",
                output_type="str",
                default_value=json.dumps(VIEWER_STATE_DEFAULTS),
                tooltip="Aggregated splat URLs + viewer defaults (JSON). Driven by the wired inputs.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="SplatViewer", library="Griptape Nodes Library")},
                ui_options={"display_name": "Viewer", "is_full_width": True},
            )
        )

        # --- Pass-through output --------------------------------------------
        self.add_parameter(
            ParameterSplat(
                name="splat_out",
                tooltip="The highest-fidelity wired splat (full → 500k → 100k) for downstream wiring.",
                allowed_modes={ParameterMode.OUTPUT},
                clickable_file_browser=False,
                settable=False,
                ui_options={"display_name": "Splat Out"},
            )
        )

    # ------------------------------------------------------------------ reactive
    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name in INPUT_TO_RESOLUTION_KEY:
            upstream_value = self._read_upstream_value(source_node, source_parameter)
            if upstream_value is not None:
                self.set_parameter_value(target_parameter.name, upstream_value)
        self._refresh_viewer_state()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter.name in INPUT_TO_RESOLUTION_KEY:
            self.set_parameter_value(target_parameter.name, None)
            self._refresh_viewer_state()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in INPUT_TO_RESOLUTION_KEY:
            self._refresh_viewer_state()
        return super().after_value_set(parameter, value)

    def _read_upstream_value(self, source_node: BaseNode, source_parameter: Parameter) -> Any:
        for fn in [
            lambda: source_node.get_parameter_value(source_parameter.name),
            lambda: getattr(source_node, "parameter_output_values", {}).get(source_parameter.name),
            lambda: getattr(source_node, "parameter_values", {}).get(source_parameter.name),
        ]:
            try:
                v = fn()
                if v is not None:
                    return v
            except Exception:
                continue
        return None

    def _refresh_viewer_state(self) -> None:
        new_state = self._build_viewer_state()
        new_state_json = json.dumps(new_state)

        try:
            self.set_parameter_value("viewer_state", new_state_json)
        except Exception as e:
            logger.warning("[LoadSplat] set_parameter_value(viewer_state) failed: %s", e)

        try:
            GriptapeNodes.handle_request(
                SetParameterValueRequest(
                    parameter_name="viewer_state",
                    node_name=self.name,
                    value=new_state_json,
                )
            )
        except Exception:
            pass

        try:
            self.publish_update_to_parameter("viewer_state", new_state_json)
        except Exception:
            pass

        self.parameter_output_values["viewer_state"] = new_state_json

    def _build_viewer_state(self) -> dict[str, Any]:
        splats: dict[str, Any] = {}
        for input_name, key in INPUT_TO_RESOLUTION_KEY.items():
            raw = self.get_parameter_value(input_name)
            splats[key] = self._splat_to_widget_entry(raw)
        return {
            "splats": splats,
            "defaults": dict(VIEWER_STATE_DEFAULTS["defaults"]),
        }

    @staticmethod
    def _splat_to_widget_entry(raw: Any) -> dict[str, Any] | None:
        """Convert a wired SplatUrlArtifact (or string) to a widget-friendly dict."""
        if not raw:
            return None
        if isinstance(raw, str):
            url = raw.strip()
            return {"url": url, "meta": {}} if url else None

        url = getattr(raw, "value", None)
        meta_obj = getattr(raw, "meta", None)
        if not isinstance(url, str) or not url:
            if isinstance(raw, dict):
                url = raw.get("value") or raw.get("url")
                meta_obj = raw.get("meta") or raw.get("metadata") or {}
        if not isinstance(url, str) or not url:
            return None
        try:
            meta = dict(meta_obj or {})
        except Exception:
            meta = {}
        return {"url": url, "meta": meta}

    # ------------------------------------------------------------------ process
    def process(self) -> None:
        """Emit the highest-fidelity wired splat as splat_out for downstream nodes."""
        self._refresh_viewer_state()
        for input_name in OUTPUT_PRIORITY:
            value = self.get_parameter_value(input_name)
            if value:
                self.parameter_output_values["splat_out"] = self._coerce_to_splat(value)
                return
        self.parameter_output_values["splat_out"] = None

    @staticmethod
    def _coerce_to_splat(raw: Any) -> SplatUrlArtifact | None:
        if isinstance(raw, SplatUrlArtifact):
            return raw
        if isinstance(raw, str) and raw.strip():
            return SplatUrlArtifact(value=raw.strip(), meta={})
        url = getattr(raw, "value", None)
        if isinstance(url, str) and url:
            try:
                meta = dict(getattr(raw, "meta", None) or {})
            except Exception:
                meta = {}
            return SplatUrlArtifact(value=url, meta=meta)
        if isinstance(raw, dict):
            url = raw.get("value") or raw.get("url")
            if isinstance(url, str) and url:
                meta = raw.get("meta") or raw.get("metadata") or {}
                return SplatUrlArtifact(value=url, meta=dict(meta))
        return None
