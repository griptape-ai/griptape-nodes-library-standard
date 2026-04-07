from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_three_d import Parameter3D
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode
from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact

logger = logging.getLogger("griptape_nodes")

__all__ = ["RodinBang3DEdit"]

# Output format options
GEOMETRY_FORMAT_OPTIONS = ["glb", "usdz", "fbx", "obj", "stl"]
DEFAULT_GEOMETRY_FORMAT = "glb"

# Material options
MATERIAL_OPTIONS = ["PBR", "Shaded", "None", "All"]
DEFAULT_MATERIAL = "PBR"

# Resolution options
RESOLUTION_OPTIONS = ["Basic", "High"]
DEFAULT_RESOLUTION = "Basic"

# Strength range
DEFAULT_STRENGTH = 5
MIN_STRENGTH = 2
MAX_STRENGTH = 12


class RodinBang3DEdit(GriptapeProxyNode):
    """Split 3D models into parts using Rodin Bang! via Griptape model proxy.

    Takes a previously generated Rodin Gen-2 asset and splits it into
    individual parts. Higher strength values produce more pieces.

    Inputs:
        - asset_id (str): UUID of a previous Rodin Gen-2 generation task
        - strength (int): Splitting intensity (2-12). Higher values produce more pieces.
        - geometry_file_format (str): Output 3D file format (glb, usdz, fbx, obj, stl)
        - material (str): Material type (PBR, Shaded, None, All)
        - resolution (str): Texture resolution (Basic=2K, High=4K)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response from the model proxy
        - model_url (ThreeDUrlArtifact): Primary generated 3D model part
        - all_files (list): URLs of all generated part files
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Split 3D models into parts using Rodin Bang! via Griptape model proxy"

        # --- INPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="asset_id",
                tooltip="UUID of a previous Rodin Gen-2 generation task to split into parts",
                placeholder_text="Enter a Rodin Gen-2 asset UUID...",
                allow_output=False,
                ui_options={"display_name": "Asset ID"},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="strength",
                default_value=DEFAULT_STRENGTH,
                tooltip="Controls splitting intensity. Higher values produce more pieces.",
                allow_output=False,
                min_val=MIN_STRENGTH,
                max_val=MAX_STRENGTH,
                traits={Slider(min_val=MIN_STRENGTH, max_val=MAX_STRENGTH)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="geometry_file_format",
                default_value=DEFAULT_GEOMETRY_FORMAT,
                tooltip="Output 3D file format",
                allow_output=False,
                traits={Options(choices=GEOMETRY_FORMAT_OPTIONS)},
                ui_options={"display_name": "File Format"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="material",
                default_value=DEFAULT_MATERIAL,
                tooltip="Material type: PBR (physically based), Shaded (baked lighting), None, or All",
                allow_output=False,
                traits={Options(choices=MATERIAL_OPTIONS)},
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                default_value=DEFAULT_RESOLUTION,
                tooltip="Texture resolution: Basic (2K) or High (4K)",
                allow_output=False,
                traits={Options(choices=RESOLUTION_OPTIONS)},
            )
        )

        # --- OUTPUT PARAMETERS ---
        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
                allow_input=False,
                allow_property=False,
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
                tooltip="Primary generated 3D model part",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True, "display_name": "3D Model"},
            )
        )

        self.add_parameter(
            Parameter(
                name="all_files",
                output_type="list",
                type="list",
                tooltip="URLs of all generated part files",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"display_name": "All Files"},
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
            result_details_tooltip="Details about the 3D edit result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _get_api_model_id(self) -> str:
        return "rodin-bang"

    async def _build_payload(self) -> dict[str, Any]:
        asset_id = self.get_parameter_value("asset_id") or ""
        if not asset_id.strip():
            msg = "An asset_id is required. Provide the UUID of a previous Rodin Gen-2 generation."
            raise ValueError(msg)

        strength = self.get_parameter_value("strength") or DEFAULT_STRENGTH
        geometry_file_format = self.get_parameter_value("geometry_file_format") or DEFAULT_GEOMETRY_FORMAT
        material = self.get_parameter_value("material") or DEFAULT_MATERIAL
        resolution = self.get_parameter_value("resolution") or DEFAULT_RESOLUTION

        payload: dict[str, Any] = {
            "asset_id": asset_id.strip(),
            "strength": strength,
            "geometry_file_format": geometry_file_format,
            "material": material,
            "resolution": resolution,
        }

        return payload

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        self.parameter_output_values["provider_response"] = result_json

        # Get download URLs - the proxy returns 'downloads' list at top level
        files = result_json.get("downloads", [])
        if not files:
            # Try nested in result object
            result = result_json.get("result", {})
            if isinstance(result, dict):
                files = result.get("downloads", [])

        if not files:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no files were found in the response.",
            )
            return

        await self._save_model_files(files)

    async def _save_model_files(self, files: list[dict[str, Any]]) -> None:
        """Download and save the generated 3D model part files."""
        requested_format = self.get_parameter_value("geometry_file_format") or DEFAULT_GEOMETRY_FORMAT

        all_file_urls: list[str] = []
        primary_url: str | None = None
        primary_filename: str | None = None

        for idx, file_info in enumerate(files):
            file_url = file_info.get("url")
            file_name = file_info.get("name", f"part_{idx}.{requested_format}")

            if not file_url:
                continue

            try:
                self._log(f"Downloading file: {file_name}")
                file_bytes = await self._download_bytes_from_url(file_url)

                if file_bytes:
                    dest = self._output_file.build_file()
                    saved = await dest.awrite_bytes(file_bytes)
                    all_file_urls.append(saved.location)
                    self._log(f"Saved file: {saved.name}")

                    # Track primary model file
                    if file_name.lower().endswith(f".{requested_format}") and primary_url is None:
                        primary_url = saved.location
                        primary_filename = saved.name
                    elif primary_url is None:
                        primary_url = saved.location
                        primary_filename = saved.name

            except Exception as e:
                self._log(f"Failed to save file {file_name}: {e}")

        # Set outputs
        self.parameter_output_values["all_files"] = all_file_urls

        if primary_url:
            self.parameter_output_values["model_url"] = ThreeDUrlArtifact(
                value=primary_url,
                meta={"filename": primary_filename, "format": requested_format},
            )
            self._set_status_results(
                was_successful=True,
                result_details=f"3D model split successfully. Saved {len(all_file_urls)} file(s).",
            )
        else:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but failed to save model files.",
            )

    def _extract_error_message(self, response_json: dict[str, Any]) -> str:
        """Extract error message, handling Rodin's application-level error format."""
        if not response_json:
            return f"{self.name} generation failed with no error details provided by API."

        # Rodin returns errors with "error" code and "message" description
        error_code = response_json.get("error")
        message = response_json.get("message")
        if error_code and message:
            if isinstance(message, list):
                message = "; ".join(str(m) for m in message)
            return f"{self.name}: {error_code}: {message}"

        return super()._extract_error_message(response_json)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["model_url"] = None
        self.parameter_output_values["all_files"] = []

    def _handle_payload_build_error(self, e: Exception) -> None:
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return

        super()._handle_payload_build_error(e)
