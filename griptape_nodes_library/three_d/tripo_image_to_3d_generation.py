from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_three_d import Parameter3D
from griptape_nodes.traits.options import Options

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.three_d._tripo_utils import (
    TripoCapability,
    TripoEndpoint,
    add_model_version_parameter,
    default_version,
    parse_tripo_task_result,
    supports,
)

logger = logging.getLogger("griptape_nodes")

__all__ = ["TripoImageTo3DGeneration"]


TRIPO_MODEL_ID = "tripo-image-to-3d"
TRIPO_ENDPOINT = TripoEndpoint.IMAGE

TEXTURE_QUALITY_OPTIONS = ["standard", "detailed"]
DEFAULT_TEXTURE_QUALITY = "standard"

GEOMETRY_QUALITY_OPTIONS = ["standard", "detailed"]
DEFAULT_GEOMETRY_QUALITY = "standard"

TEXTURE_ALIGNMENT_OPTIONS = ["original_image", "geometry"]
DEFAULT_TEXTURE_ALIGNMENT = "original_image"


class TripoImageTo3DGeneration(GriptapeProxyNode):
    """Generate 3D models from a reference image using Tripo via the Griptape model proxy.

    Inputs:
        - image (ImageArtifact / ImageUrlArtifact / str): Reference image
        - model_version (str): Tripo model version
        - texture (bool): Generate textures
        - pbr (bool): Produce a PBR material
        - texture_quality (str): "standard" or "detailed" (HD)
        - geometry_quality (str): "standard" or "detailed" (v3.0/v3.1 only)
        - texture_alignment (str): "original_image" or "geometry"
        - enable_image_autofix (bool): Let Tripo auto-correct the input image

    Outputs:
        - generation_id (str): Generation ID from the proxy
        - provider_response (dict): Verbatim proxy response
        - model_url (ThreeDUrlArtifact): Generated GLB model
        - preview_image (ImageUrlArtifact): Rendered preview, if available
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate 3D models from a reference image using Tripo via the Griptape model proxy"

        self.add_parameter(
            ParameterImage(
                name="image",
                tooltip="Reference image for 3D generation (PNG/JPEG/WEBP, <=10 MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Image"},
            )
        )

        add_model_version_parameter(self, TRIPO_ENDPOINT)

        self.add_parameter(
            ParameterBool(
                name="texture",
                default_value=True,
                tooltip="Generate surface textures (costs more when enabled)",
                allow_output=False,
                ui_options={"display_name": "Texture"},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="pbr",
                default_value=True,
                tooltip="Produce a PBR material. When true, texture is also forced on.",
                allow_output=False,
                ui_options={"display_name": "PBR"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="texture_quality",
                default_value=DEFAULT_TEXTURE_QUALITY,
                tooltip="Texture quality. 'detailed' = HD textures (costs more).",
                allow_output=False,
                traits={Options(choices=TEXTURE_QUALITY_OPTIONS)},
                ui_options={"display_name": "Texture Quality"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="geometry_quality",
                default_value=DEFAULT_GEOMETRY_QUALITY,
                tooltip="Geometry quality (v3.0 and v3.1 only). 'detailed' costs more.",
                allow_output=False,
                traits={Options(choices=GEOMETRY_QUALITY_OPTIONS)},
                ui_options={"display_name": "Geometry Quality"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="texture_alignment",
                default_value=DEFAULT_TEXTURE_ALIGNMENT,
                tooltip=(
                    "Whether to align textures to the original image or the generated geometry. "
                    "Only meaningful when texture is enabled."
                ),
                allow_output=False,
                traits={Options(choices=TEXTURE_ALIGNMENT_OPTIONS)},
                ui_options={"display_name": "Texture Alignment"},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="enable_image_autofix",
                default_value=False,
                tooltip="Let Tripo auto-correct the input image before generation",
                allow_output=False,
                hide=True,
                ui_options={"display_name": "Image Autofix"},
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
                tooltip="Generated 3D model (GLB)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True, "display_name": "3D Model"},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="preview_image",
                tooltip="Rendered preview of the generated 3D model",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"pulse_on_run": True, "display_name": "Preview"},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="tripo_model.glb",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the 3D generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

        self._update_parameter_visibility_for_model(default_version(TRIPO_ENDPOINT))

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _update_parameter_visibility_for_model(self, model_version: str) -> None:
        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.TEXTURE):
            self.show_parameter_by_name("texture")
            self.show_parameter_by_name("pbr")
            self.show_parameter_by_name("texture_quality")
        else:
            self.hide_parameter_by_name("texture")
            self.hide_parameter_by_name("pbr")
            self.hide_parameter_by_name("texture_quality")

        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.TEXTURE_ALIGNMENT):
            self.show_parameter_by_name("texture_alignment")
        else:
            self.hide_parameter_by_name("texture_alignment")

        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.GEOMETRY_QUALITY):
            self.show_parameter_by_name("geometry_quality")
        else:
            self.hide_parameter_by_name("geometry_quality")

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name == "model_version":
            self._update_parameter_visibility_for_model(value)

    def _get_api_model_id(self) -> str:
        return TRIPO_MODEL_ID

    async def _build_payload(self) -> dict[str, Any]:
        image_value = self.get_parameter_value("image")
        data_uri = await prepare_media_data_uri(image_value, kind="image", node_name=self.name)
        if not data_uri:
            msg = "A reference image is required for Tripo image-to-3D generation."
            raise ValueError(msg)

        model_version = self.get_parameter_value("model_version") or default_version(TRIPO_ENDPOINT)
        payload: dict[str, Any] = {
            "image": data_uri,
            "model_version": model_version,
            "enable_image_autofix": bool(self.get_parameter_value("enable_image_autofix")),
        }

        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.TEXTURE):
            payload["texture"] = bool(self.get_parameter_value("texture"))
            payload["pbr"] = bool(self.get_parameter_value("pbr"))
            payload["texture_quality"] = self.get_parameter_value("texture_quality") or DEFAULT_TEXTURE_QUALITY

        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.TEXTURE_ALIGNMENT):
            payload["texture_alignment"] = self.get_parameter_value("texture_alignment") or DEFAULT_TEXTURE_ALIGNMENT

        if supports(TRIPO_ENDPOINT, model_version, TripoCapability.GEOMETRY_QUALITY):
            payload["geometry_quality"] = self.get_parameter_value("geometry_quality") or DEFAULT_GEOMETRY_QUALITY

        return payload

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        await parse_tripo_task_result(self, result_json)

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["model_url"] = None
        self.parameter_output_values["preview_image"] = None

    def _handle_payload_build_error(self, e: Exception) -> None:
        if isinstance(e, ValueError):
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details=str(e))
            self._handle_failure_exception(e)
            return
        super()._handle_payload_build_error(e)
