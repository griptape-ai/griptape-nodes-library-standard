from __future__ import annotations

import contextlib
import json as _json
import logging
from typing import Any, ClassVar

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.artifact_url.public_artifact_url_parameter import (
    PublicArtifactUrlParameter,
)
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.proxy.provider_asset_access import resolve_proxy_api_key
from griptape_nodes_library.utils.image_utils import extract_image_url

logger = logging.getLogger("griptape_nodes")

__all__ = ["OmnihumanSubjectDetection"]


class OmnihumanSubjectDetection(GriptapeProxyNode):
    """Detect and locate subjects in an image, returning masks and bounding boxes.

    This is Step 2 of the OmniHuman workflow (optional). It detects subjects in the image
    and provides profile images, mask images, and bounding box coordinates. This step can
    be skipped if there's no need to specify a subject to speak during video generation.

    Inputs:
        - image_url (str): URL of the image to analyze for subject detection

    Outputs:
        - mask_image_urls (list[ImageUrlArtifact]): URLs of the subject mask images
        - contains_subject (bool): Whether the image contains a human subject
        - was_successful (bool): Whether the detection succeeded
        - result_details (str): Details about the detection result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"
    MODEL_IDS: ClassVar[list[str]] = [
        "omnihuman-1-5-subject-detection",
    ]
    # Migrates values saved before the dropdown stored the provider's own model id.
    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = {
        "OmniHuman 1.5 Subject Detection": "omnihuman-1-5-subject-detection",
        "gtc_omnihuman_1_5_subject_detection": "omnihuman-1-5-subject-detection",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Detect subjects and generate masks using OmniHuman Subject Detection via Griptape Cloud"

        # INPUTS
        model_id_param = ParameterString(
            name="model_id",
            default_value=self.MODEL_IDS[0],
            tooltip="Model identifier to use for detection",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_id_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_id_param,
            model_choices=self.MODEL_IDS,
            default_model=self.MODEL_IDS[0],
            deprecated_values=self.LEGACY_MODEL_VALUES,
        )

        self._public_image_url_parameter = PublicArtifactUrlParameter(
            node=self,
            artifact_url_parameter=ParameterImage(
                name="image_url",
                tooltip="URL of the image to analyze for subject detection.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={
                    "placeholder_text": "https://example.com/image.jpg",
                    "display_name": "Image URL",
                },
            ),
            disclaimer_message="The OmniHuman service utilizes this URL to access the image for subject detection.",
        )
        self._public_image_url_parameter.add_input_parameters()

        # OUTPUTS
        self.add_parameter(
            Parameter(
                name="mask_image_urls",
                type="list",
                output_type="list",
                tooltip="List of mask image URLs for detected subjects",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="contains_subject",
                tooltip="Whether the image contains a human subject",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        # Create status parameters for success/failure tracking
        self._create_status_parameters(
            result_details_tooltip="Details about the subject detection result or any errors",
            result_details_placeholder="Detection status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        """Log a message."""
        with contextlib.suppress(Exception):
            logger.info("%s: %s", self.name, message)

    async def aprocess(self) -> None:
        await self._process_generation()

    async def _process_generation(self) -> None:
        try:
            await super()._process_generation()
        finally:
            self._public_image_url_parameter.delete_uploaded_artifact()

    def _validate_api_key(self) -> str:
        """Validate that the API key is available."""
        api_key = resolve_proxy_api_key(self.API_KEY_NAME)
        if not api_key:
            msg = f"{self.name} is missing {self.API_KEY_NAME}. Ensure it's set in the environment/config."
            raise ValueError(msg)
        return api_key

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []
        if not extract_image_url(self.get_parameter_value("image_url")):
            exceptions.append(ValueError(self._missing_image_message()))
        return exceptions if exceptions else None

    def _missing_image_message(self) -> str:
        return f"{self.name} requires an input image. Set the Image URL parameter or connect an image to it."

    async def _build_payload(self) -> dict[str, Any]:
        provider_model_id = self._get_selected_model_id()
        image_value = extract_image_url(self.get_parameter_value("image_url"))
        if not image_value:
            msg = self._missing_image_message()
            raise ValueError(msg)

        # OmniHuman downloads the image server-side, so it needs a publicly
        # reachable URL rather than an inline data URI.
        image_url = self._public_image_url_parameter.get_public_url_for_parameter()

        return {
            "req_key": self._get_req_key(provider_model_id),
            "image_url": image_url,
        }

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        self._process_response(result_json)

    def _get_req_key(self, model_id: str) -> str:
        """Get the request key based on model_id."""
        if model_id == "omnihuman-1-5-subject-detection":
            return "realman_avatar_object_detection_cv"

        msg = f"Unsupported model_id: {model_id}"
        raise ValueError(msg)

    def _process_response(self, response_json: dict[str, Any]) -> None:
        """Process the API response from Griptape Cloud proxy."""
        # Extract provider response from Griptape Cloud format
        resp_data = _json.loads(response_json.get("data", {}).get("resp_data", {}))

        contains_human = resp_data.get("status") == 1
        mask_urls = resp_data.get("object_detection_result", {}).get("mask", {}).get("url", [])

        self.parameter_output_values["contains_subject"] = contains_human
        self.parameter_output_values["mask_image_urls"] = mask_urls

        result_msg = f"Subject detection completed successfully. response: {resp_data}. "
        self._set_status_results(
            was_successful=True,
            result_details=result_msg,
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["mask_image_urls"] = []
        self.parameter_output_values["contains_subject"] = False
