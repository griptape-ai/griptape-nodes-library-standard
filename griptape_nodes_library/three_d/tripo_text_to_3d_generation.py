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

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.three_d._tripo_utils import parse_tripo_task_result

logger = logging.getLogger("griptape_nodes")

__all__ = ["TripoTextTo3DGeneration"]


TRIPO_MODEL_ID = "tripo-text-to-3d"

MODEL_VERSION_OPTIONS = [
    "P1-20260311",
    "Turbo-v1.0-20250506",
    "v3.1-20260211",
    "v3.0-20250812",
    "v2.5-20250123",
    "v2.0-20240919",
    "v1.4-20240625",
]
DEFAULT_MODEL_VERSION = "v2.5-20250123"

TEXTURE_QUALITY_OPTIONS = ["standard", "detailed"]
DEFAULT_TEXTURE_QUALITY = "standard"

GEOMETRY_QUALITY_OPTIONS = ["standard", "detailed"]
DEFAULT_GEOMETRY_QUALITY = "standard"

# Parameters only available on certain model versions
GEOMETRY_QUALITY_SUPPORTED_VERSIONS = {"v3.0-20250812", "v3.1-20260211"}
TEXTURE_PARAMS_SUPPORTED_VERSIONS = {"P1-20260311", "Turbo-v1.0-20250506", "v3.1-20260211", "v3.0-20250812", "v2.5-20250123", "v2.0-20240919"}

_MODEL_VERSION_BADGE = (
    "**P1-20260311** — Premium flagship model\n"
    "**Turbo-v1.0-20250506** — Fastest generation\n"
    "**v3.1-20260211** / **v3.0-20250812** — High quality with geometry control\n"
    "**v2.5-20250123** / **v2.0-20240919** — Standard quality (default)\n"
    "**v1.4-20240625** — Legacy; prompt only\n\n"
    "[Model docs](https://docs.tripo3d.ai/model-generation/text-to-model-p1-20260311.html)"
)


class TripoTextTo3DGeneration(GriptapeProxyNode):
    """Generate 3D models from text prompts using Tripo via the Griptape model proxy.

    Inputs:
        - prompt (str): Text description of the 3D model to generate
        - negative_prompt (str): Optional negative prompt
        - model_version (str): Tripo model version (e.g. v2.5-20250123, P1-20260311)
        - texture (bool): Whether to generate textures (default True)
        - pbr (bool): Whether to produce a PBR material (default True)
        - texture_quality (str): "standard" or "detailed" (HD)
        - geometry_quality (str): "standard" or "detailed" (v3.0/v3.1 only)

    Outputs:
        - generation_id (str): Generation ID from the proxy
        - provider_response (dict): Verbatim proxy response
        - model_url (ThreeDUrlArtifact): Generated GLB model
        - preview_image (ImageUrlArtifact): Rendered preview, if available
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate 3D models from a text prompt using Tripo via the Griptape model proxy"

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the 3D model to generate (required, <=1024 chars)",
                multiline=True,
                placeholder_text="A small red cube...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                default_value="",
                tooltip="Optional negative prompt (<=255 chars)",
                allow_output=False,
                ui_options={"display_name": "Negative Prompt"},
            )
        )

        model_version_param = ParameterString(
            name="model_version",
            default_value=DEFAULT_MODEL_VERSION,
            tooltip="Tripo model version. See badge for details on what each version supports.",
            allow_output=False,
            traits={Options(choices=MODEL_VERSION_OPTIONS)},
            ui_options={"display_name": "Model Version"},
        )
        model_version_param.set_badge(
            variant="info",
            title="Model Versions",
            message=_MODEL_VERSION_BADGE,
        )
        self.add_parameter(model_version_param)

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

        self._update_parameter_visibility_for_model(DEFAULT_MODEL_VERSION)

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _update_parameter_visibility_for_model(self, model_version: str) -> None:
        if model_version in TEXTURE_PARAMS_SUPPORTED_VERSIONS:
            self.show_parameter_by_name("texture")
            self.show_parameter_by_name("pbr")
            self.show_parameter_by_name("texture_quality")
            self.show_parameter_by_name("negative_prompt")
        else:
            self.hide_parameter_by_name("texture")
            self.hide_parameter_by_name("pbr")
            self.hide_parameter_by_name("texture_quality")
            self.hide_parameter_by_name("negative_prompt")

        if model_version in GEOMETRY_QUALITY_SUPPORTED_VERSIONS:
            self.show_parameter_by_name("geometry_quality")
        else:
            self.hide_parameter_by_name("geometry_quality")

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name == "model_version":
            self._update_parameter_visibility_for_model(value)

    def _get_api_model_id(self) -> str:
        return TRIPO_MODEL_ID

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            msg = "Prompt is required for Tripo text-to-3D generation."
            raise ValueError(msg)

        model_version = self.get_parameter_value("model_version") or DEFAULT_MODEL_VERSION
        payload: dict[str, Any] = {
            "prompt": prompt,
            "model_version": model_version,
        }

        if model_version in TEXTURE_PARAMS_SUPPORTED_VERSIONS:
            payload["texture"] = bool(self.get_parameter_value("texture"))
            payload["pbr"] = bool(self.get_parameter_value("pbr"))
            payload["texture_quality"] = self.get_parameter_value("texture_quality") or DEFAULT_TEXTURE_QUALITY
            negative_prompt = (self.get_parameter_value("negative_prompt") or "").strip()
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt

        if model_version in GEOMETRY_QUALITY_SUPPORTED_VERSIONS:
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
