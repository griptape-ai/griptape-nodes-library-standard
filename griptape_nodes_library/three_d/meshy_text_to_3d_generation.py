from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_three_d import Parameter3D
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

logger = logging.getLogger("griptape_nodes")

__all__ = ["MeshyTextTo3DGeneration"]

# Model ID mapping
MODEL_MAPPING = {
    "Meshy 5": "meshy-5",
    "Meshy 6": "meshy-6",
    "Latest": "latest",
}
DEFAULT_MODEL = "Latest"

# Model type options
MODEL_TYPE_OPTIONS = ["standard", "lowpoly"]
DEFAULT_MODEL_TYPE = "standard"

# Topology options
TOPOLOGY_OPTIONS = ["triangle", "quad"]
DEFAULT_TOPOLOGY = "triangle"

# Symmetry mode options
SYMMETRY_MODE_OPTIONS = ["off", "auto", "on"]
DEFAULT_SYMMETRY_MODE = "auto"

# Pose mode options
POSE_MODE_OPTIONS = ["", "a-pose", "t-pose"]
DEFAULT_POSE_MODE = ""

# Origin position options
ORIGIN_AT_OPTIONS = ["bottom", "center"]
DEFAULT_ORIGIN_AT = "bottom"

# Output format options
OUTPUT_FORMAT_OPTIONS = ["glb", "obj", "fbx", "stl", "usdz", "3mf"]
DEFAULT_OUTPUT_FORMAT = "glb"


class MeshyTextTo3DGeneration(GriptapeProxyNode):
    """Generate 3D models from text using Meshy via Griptape model proxy.

    This node uses Meshy's Text to 3D API to generate 3D models in preview mode,
    producing geometry without textures. For textured models, use the refine workflow.

    Inputs:
        - prompt (str): Text description of the 3D model (max 600 characters)
        - ai_model (str): Model version to use (meshy-5, meshy-6, or latest)
        - model_type (str): Model architecture (standard or lowpoly)
        - target_polycount (int): Target polygon count (100-300,000)
        - topology (str): Mesh topology (triangle or quad)
        - symmetry_mode (str): Symmetry mode (off, auto, or on)
        - pose_mode (str): Character pose (empty, a-pose, or t-pose)
        - target_formats (list): Output file formats
        - auto_size (bool): Auto-estimate real-world dimensions
        - origin_at (str): Origin position (bottom or center, requires auto_size=true)

    Outputs:
        - generation_id (str): Generation ID (task ID) from Meshy API
        - provider_response (dict): Raw provider response with all task details
        - model_url (ThreeDUrlArtifact): Primary GLB model download URL
        - model_urls (dict): All model format URLs (glb, obj, fbx, etc.)
        - thumbnail_url (str): Preview thumbnail image URL
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate 3D models from text using Meshy via Griptape model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the 3D model to generate (max 600 characters)",
                multiline=True,
                placeholder_text="A simple wooden chair",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="ai_model",
                default_value=DEFAULT_MODEL,
                tooltip="Model version to use",
                allow_output=False,
                traits={Options(choices=list(MODEL_MAPPING.keys()))},
                ui_options={"display_name": "AI Model"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="model_type",
                default_value=DEFAULT_MODEL_TYPE,
                tooltip="Model architecture type",
                allow_output=False,
                traits={Options(choices=MODEL_TYPE_OPTIONS)},
                ui_options={"display_name": "Model Type"},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="target_polycount",
                default_value=30000,
                tooltip="Target polygon count for the mesh (100-300,000)",
                allow_output=False,
                traits={Slider(min_val=100, max_val=300000)},
                ui_options={"display_name": "Target Polycount"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="topology",
                default_value=DEFAULT_TOPOLOGY,
                tooltip="Mesh topology type",
                allow_output=False,
                traits={Options(choices=TOPOLOGY_OPTIONS)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="symmetry_mode",
                default_value=DEFAULT_SYMMETRY_MODE,
                tooltip="Apply symmetry to the model",
                allow_output=False,
                traits={Options(choices=SYMMETRY_MODE_OPTIONS)},
                ui_options={"display_name": "Symmetry Mode"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="pose_mode",
                default_value=DEFAULT_POSE_MODE,
                tooltip="Character pose for human-like models",
                allow_output=False,
                traits={Options(choices=POSE_MODE_OPTIONS)},
                ui_options={"display_name": "Pose Mode"},
            )
        )

        self.add_parameter(
            ParameterList(
                name="target_formats",
                input_types=["str", "list", "list[str]"],
                default_value=[DEFAULT_OUTPUT_FORMAT],
                tooltip="Output file formats",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"expander": True, "display_name": "Target Formats"},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="auto_size",
                default_value=False,
                tooltip="Automatically estimate real-world dimensions",
                allow_output=False,
                ui_options={"display_name": "Auto Size"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="origin_at",
                default_value=DEFAULT_ORIGIN_AT,
                tooltip="Origin position (requires auto_size=true)",
                allow_output=False,
                traits={Options(choices=ORIGIN_AT_OPTIONS)},
                ui_options={"display_name": "Origin At"},
            )
        )

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID (task ID) from Meshy API",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            Parameter3D(
                name="model_url",
                tooltip="Primary GLB model as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True, "display_name": "3D Model"},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="model_urls",
                tooltip="All model format URLs (glb, obj, fbx, etc.)",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "All Model URLs"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="thumbnail_url",
                tooltip="Preview thumbnail image URL",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "Thumbnail URL"},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="model.glb",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _get_api_model_id(self) -> str:
        """Map friendly model name to API model ID."""
        model = self.get_parameter_value("ai_model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    async def _build_payload(self) -> dict[str, Any]:
        """Build the request payload for Meshy Text to 3D API."""
        # Gather parameter values
        prompt = self.get_parameter_value("prompt") or ""
        if not prompt.strip():
            msg = "Prompt is required for 3D generation."
            raise ValueError(msg)

        model_type = self.get_parameter_value("model_type") or DEFAULT_MODEL_TYPE
        target_polycount = self.get_parameter_value("target_polycount") or 30000
        topology = self.get_parameter_value("topology") or DEFAULT_TOPOLOGY
        symmetry_mode = self.get_parameter_value("symmetry_mode") or DEFAULT_SYMMETRY_MODE
        pose_mode = self.get_parameter_value("pose_mode") or DEFAULT_POSE_MODE
        auto_size = self.get_parameter_value("auto_size") or False
        origin_at = self.get_parameter_value("origin_at") or DEFAULT_ORIGIN_AT

        # Get target formats
        target_formats_raw = self.get_parameter_list_value("target_formats") or [DEFAULT_OUTPUT_FORMAT]
        # Ensure it's a list of strings
        target_formats: list[str] = []
        for fmt in target_formats_raw:
            if isinstance(fmt, str):
                target_formats.append(fmt)

        if not target_formats:
            target_formats = [DEFAULT_OUTPUT_FORMAT]

        # Build payload according to Meshy API spec
        payload: dict[str, Any] = {
            "mode": "preview",  # This node only does preview (geometry without textures)
            "prompt": prompt,
            "model_type": model_type,
            "topology": topology,
            "target_polycount": target_polycount,
            "symmetry_mode": symmetry_mode,
            "target_formats": target_formats,
            "auto_size": auto_size,
        }

        # Add optional parameters
        if pose_mode:
            payload["pose_mode"] = pose_mode

        if auto_size and origin_at:
            payload["origin_at"] = origin_at

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        """Parse the Meshy result and set output parameters."""
        # Extract model URLs
        model_urls = result_json.get("model_urls", {})
        thumbnail_url = result_json.get("thumbnail_url", "")

        if not model_urls:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details="No model URLs in response.")
            return

        # Get the primary GLB URL
        glb_url = model_urls.get("glb")
        if not glb_url:
            # Fallback to first available format
            glb_url = next(iter(model_urls.values()), None)

        if not glb_url:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details="No valid model URL found.")
            return

        # Download and save the primary model
        try:
            from griptape_nodes.files.file import File

            model_bytes = await File(glb_url).aread_bytes()
            if model_bytes:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(model_bytes)
                self.parameter_output_values["model_url"] = ThreeDUrlArtifact(
                    value=saved.location,
                    meta={"filename": saved.name},
                )
                self.parameter_output_values["model_urls"] = model_urls
                self.parameter_output_values["thumbnail_url"] = thumbnail_url
                self._set_status_results(was_successful=True, result_details="3D model generated successfully.")
            else:
                self._set_safe_defaults()
                self._set_status_results(was_successful=False, result_details="Failed to download model file.")
        except Exception as e:
            self._log(f"Error downloading model: {e}")
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=f"Failed to download model: {e}")

    def _set_safe_defaults(self) -> None:
        """Clear all output parameters on error."""
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["model_url"] = None
        self.parameter_output_values["model_urls"] = {}
        self.parameter_output_values["thumbnail_url"] = ""
