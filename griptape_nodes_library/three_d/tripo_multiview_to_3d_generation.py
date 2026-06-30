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
from griptape_nodes_library.three_d._tripo_utils import parse_tripo_task_result

logger = logging.getLogger("griptape_nodes")

__all__ = ["TripoMultiviewTo3DGeneration"]


TRIPO_MODEL_ID = "tripo-multiview-to-3d"

# Slot order Tripo expects for multiview_to_model. Front is mandatory; at least
# two views must be provided ("Do not use less than two images to generate").
SLOT_PARAM_NAMES = ["front_image", "left_image", "back_image", "right_image"]
MIN_IMAGES = 2

MODEL_VERSION_OPTIONS = [
    "P1-20260311",
    "v2.5-20250123",
    "v2.0-20240919",
    "v1.4-20240625",
]
DEFAULT_MODEL_VERSION = "P1-20260311"

TEXTURE_QUALITY_OPTIONS = ["standard", "detailed"]
DEFAULT_TEXTURE_QUALITY = "standard"

TEXTURE_ALIGNMENT_OPTIONS = ["original_image", "geometry"]
DEFAULT_TEXTURE_ALIGNMENT = "original_image"

# Parameters only available on certain model versions
TEXTURE_PARAMS_SUPPORTED_VERSIONS = {
    "P1-20260311",
    "v2.5-20250123",
    "v2.0-20240919",
}
TEXTURE_ALIGNMENT_SUPPORTED_VERSIONS = {
    "P1-20260311",
    "v2.5-20250123",
    "v2.0-20240919",
}

_MODEL_VERSION_BADGE = (
    "**P1-20260311** — Premium flagship model (default)\n"
    "**v2.5-20250123** / **v2.0-20240919** — Standard quality\n"
    "**v1.4-20240625** — Legacy; basic parameters only\n\n"
    "[Model docs](https://docs.tripo3d.ai/model-generation/multiview-to-model-p1-20260311.html)"
)


class TripoMultiviewTo3DGeneration(GriptapeProxyNode):
    """Generate 3D models from up to four view images using Tripo via the Griptape model proxy.

    Tripo's multiview-to-3D task accepts front/left/back/right reference views,
    producing more accurate geometry than a single image. The front view is
    required and at least two views must be supplied.

    Inputs:
        - front_image (ImageArtifact / ImageUrlArtifact / str): Front view (required)
        - left_image (ImageArtifact / ImageUrlArtifact / str): Left view
        - back_image (ImageArtifact / ImageUrlArtifact / str): Back view
        - right_image (ImageArtifact / ImageUrlArtifact / str): Right view
        - model_version (str): Tripo model version
        - texture (bool): Generate textures
        - pbr (bool): Produce a PBR material
        - texture_quality (str): "standard" or "detailed" (HD)
        - texture_alignment (str): "original_image" or "geometry"

    Outputs:
        - provider_response (dict): Verbatim proxy response
        - model_url (ThreeDUrlArtifact): Generated GLB model
        - preview_image (ImageUrlArtifact): Rendered preview, if available
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate 3D models from up to four view images using Tripo via the Griptape model proxy"

        self.add_parameter(
            ParameterImage(
                name="front_image",
                tooltip="Front view image (required; PNG/JPEG/WEBP, <=20 MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Front Image"},
            )
        )
        self.add_parameter(
            ParameterImage(
                name="left_image",
                tooltip="Left view image (optional; PNG/JPEG/WEBP, <=20 MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Left Image"},
            )
        )
        self.add_parameter(
            ParameterImage(
                name="back_image",
                tooltip="Back view image (optional; PNG/JPEG/WEBP, <=20 MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Back Image"},
            )
        )
        self.add_parameter(
            ParameterImage(
                name="right_image",
                tooltip="Right view image (optional; PNG/JPEG/WEBP, <=20 MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"display_name": "Right Image"},
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
        else:
            self.hide_parameter_by_name("texture")
            self.hide_parameter_by_name("pbr")
            self.hide_parameter_by_name("texture_quality")

        if model_version in TEXTURE_ALIGNMENT_SUPPORTED_VERSIONS:
            self.show_parameter_by_name("texture_alignment")
        else:
            self.hide_parameter_by_name("texture_alignment")

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name == "model_version":
            self._update_parameter_visibility_for_model(value)

    def _get_api_model_id(self) -> str:
        return TRIPO_MODEL_ID

    async def _build_payload(self) -> dict[str, Any]:
        # Assemble the ordered [front, left, back, right] images list, using None
        # for omitted slots. The proxy maps these positionally onto Tripo's
        # fixed-length-4 files array.
        images: list[str | None] = []
        for param_name in SLOT_PARAM_NAMES:
            image_value = self.get_parameter_value(param_name)
            data_uri = await prepare_media_data_uri(image_value, kind="image", node_name=self.name)
            images.append(data_uri or None)

        if not images[0]:
            msg = "A front image is required for Tripo multiview-to-3D generation."
            raise ValueError(msg)

        present_count = sum(1 for image in images if image)
        if present_count < MIN_IMAGES:
            msg = (
                f"Tripo multiview-to-3D requires at least {MIN_IMAGES} images "
                "(do not use less than two images to generate)."
            )
            raise ValueError(msg)

        model_version = self.get_parameter_value("model_version") or DEFAULT_MODEL_VERSION
        payload: dict[str, Any] = {
            "images": images,
            "model_version": model_version,
        }

        if model_version in TEXTURE_PARAMS_SUPPORTED_VERSIONS:
            payload["texture"] = bool(self.get_parameter_value("texture"))
            payload["pbr"] = bool(self.get_parameter_value("pbr"))
            payload["texture_quality"] = self.get_parameter_value("texture_quality") or DEFAULT_TEXTURE_QUALITY

        if model_version in TEXTURE_ALIGNMENT_SUPPORTED_VERSIONS:
            payload["texture_alignment"] = self.get_parameter_value("texture_alignment") or DEFAULT_TEXTURE_ALIGNMENT

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
