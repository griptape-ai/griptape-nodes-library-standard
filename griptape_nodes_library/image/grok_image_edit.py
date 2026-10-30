from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, ClassVar

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["GrokImageEdit"]


class GrokImageEdit(GriptapeProxyNode):
    """Edit images using Grok image models via Griptape model proxy.

    Inputs:
        - model (str): Grok image model to use
        - image (ImageUrlArtifact): Input image to edit (required)
        - additional_images (list): Up to 4 more source images sent alongside 'image'
        - prompt (str): Editing prompt
        - n (int): Number of images to generate (1-10)
        - quality (str): Output quality (low, medium, high)
        - resolution (str): Output resolution (1k, 2k)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim response from the model proxy
        - image_url (ImageUrlArtifact): First generated image
        - image_url_2 ... image_url_10 (ImageUrlArtifact): Additional images
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    MIN_IMAGES: ClassVar[int] = 1
    MAX_IMAGES: ClassVar[int] = 10
    # xAI accepts up to 5 source images per edit: 'image' plus up to 4 additional ones.
    MAX_SOURCE_IMAGES: ClassVar[int] = 5
    QUALITY_OPTIONS: ClassVar[list[str]] = ["low", "medium", "high"]

    RESOLUTION_OPTIONS: ClassVar[list[str]] = ["1k", "2k"]

    # Migrates values saved before this dropdown stored the provider's own model id.
    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = {
        "Grok Imagine Image": "grok-imagine-image",
        "gtc_grok_imagine_image": "grok-imagine-image",
        # Folded in from GrokImageGeneration's retired DEPRECATED_MODELS dict; this
        # node's own removed _supports_quality check compared against the same value.
        "Grok 2 Image": "grok-imagine-image",
        "grok-2-image-1212": "grok-imagine-image",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit images using Grok image models via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value="grok-imagine-image",
            tooltip="Select the Grok image model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=["grok-imagine-image"],
            default_model="grok-imagine-image",
            deprecated_values=self.LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Prompt for image editing. See the [xAI prompting guide](https://docs.x.ai/developers/model-capabilities/images/prompting-guide) for best practices.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the edits you want to make...",
                allow_output=False,
            )
        )

        self.add_parameter(
            ParameterImage(
                name="image",
                default_value="",
                tooltip="Input image to edit",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
                ui_options={"display_name": "Image"},
            )
        )

        self.add_parameter(
            ParameterList(
                name="additional_images",
                input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
                default_value=[],
                tooltip=f"Optional extra source images sent with 'Image' (up to {self.MAX_SOURCE_IMAGES - 1})",
                allowed_modes={ParameterMode.INPUT},
                max_items=self.MAX_SOURCE_IMAGES - 1,
                ui_options={"display_name": "Additional Images", "expander": True, "hide_property": True},
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterInt(
                name="n",
                default_value=1,
                tooltip="Number of images to generate (1-10)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=1,
                max_val=10,
                slider=True,
            )

            ParameterString(
                name="quality",
                default_value="medium",
                tooltip="Quality of the output image (currently a no-op)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.QUALITY_OPTIONS)},
            )

            ParameterString(
                name="resolution",
                default_value="1k",
                tooltip="Resolution of the generated image (only 1k currently supported)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.RESOLUTION_OPTIONS)},
            )

        self.add_node_element(generation_settings_group)

        # OUTPUTS
        self.add_parameter(
            ParameterDict(
                name="provider_response",
                tooltip="Verbatim response from Griptape model proxy",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        for i in range(1, 11):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.add_parameter(
                ParameterImage(
                    name=param_name,
                    tooltip=f"Generated image {i}",
                    allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                    settable=False,
                    ui_options={"pulse_on_run": True, "hide": i > 1},
                )
            )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="grok_image_edit.jpg",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the image editing result or any errors",
            result_details_placeholder="Editing status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    @staticmethod
    def _has_media_value(value: Any) -> bool:
        if value is None:
            return False
        if hasattr(value, "value"):
            return bool(value.value)
        return bool(value)

    async def _prepare_image_data_uri(self, image_input: Any) -> str | None:
        return await prepare_media_data_uri(image_input, kind="image", node_name=self.name)

    def _get_additional_images(self) -> list[Any]:
        # Connected values bypass after_value_set, so normalize on read as well.
        images = self.get_parameter_list_value("additional_images")
        normalized = normalize_artifact_list(images, ImageUrlArtifact, accepted_types=(ImageArtifact,))
        return [item for item in normalized if self._has_media_value(item)]

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "additional_images" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("additional_images", updated_list)
        return super().after_value_set(parameter, value)

    def _show_image_output_parameters(self, count: int) -> None:
        for i in range(1, 11):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            if i <= count:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

    def _get_api_model_id(self) -> str:
        # Decorate the resolved provider id with the URL-path operation suffix the
        # proxy expects; the catalog declares the bare id (see _get_catalog_model_id).
        return f"{self._get_selected_model_id()}:edit"

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(ValueError("'Prompt' is empty. Describe the edit you want."))

        image_value = self.get_parameter_value("image")
        if not self._has_media_value(image_value):
            exceptions.append(ValueError("Connect an image to 'Image'."))

        additional_count = len(self._get_additional_images())
        if additional_count > self.MAX_SOURCE_IMAGES - 1:
            exceptions.append(
                ValueError(
                    f"'Additional Images' holds {additional_count} images; "
                    f"at most {self.MAX_SOURCE_IMAGES - 1} can be sent with 'Image'."
                )
            )

        n_value = self.get_parameter_value("n")
        if n_value is None or not self.MIN_IMAGES <= int(n_value) <= self.MAX_IMAGES:
            exceptions.append(ValueError(f"'n' must be between {self.MIN_IMAGES} and {self.MAX_IMAGES}."))

        return exceptions if exceptions else None

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        n_value = int(self.get_parameter_value("n") or 1)
        resolution = self.get_parameter_value("resolution") or "1k"
        api_model_id = self._get_selected_model_id()
        source_images = [self.get_parameter_value("image"), *self._get_additional_images()]
        image_data_uris = [await self._prepare_image_data_uri(image) for image in source_images]
        unreadable = [str(idx) for idx, uri in enumerate(image_data_uris, start=1) if not uri]
        if unreadable:
            msg = f"Source image(s) {', '.join(unreadable)} could not be read."
            raise ValueError(msg)

        payload: dict[str, Any] = {
            "model": api_model_id,
            "prompt": prompt,
            "n": n_value,
            "quality": self.get_parameter_value("quality") or "medium",
            "resolution": resolution,
            "response_format": "url",
        }

        if len(image_data_uris) == 1:
            payload["image"] = {"url": image_data_uris[0]}
        else:
            payload["images"] = [{"url": uri} for uri in image_data_uris]

        return payload

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        try:
            hosted = [a for a in await self._hosted_artifacts(generation_id) if a.kind == ArtifactKind.IMAGE]
        except Exception as e:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"The generation finished, but its images could not be listed: {e}",
            )
            return

        if not hosted:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="The generation finished, but no images were hosted.",
            )
            return

        # Slots follow provider order: a failed download leaves its slot empty instead of
        # pulling later images forward, so image_url_N always holds the Nth hosted image.
        saved_by_position = [await self._save_single_image(generation_id, position) for position in range(len(hosted))]

        image_artifacts = [artifact for artifact in saved_by_position if artifact is not None]
        if not image_artifacts:
            self._set_safe_defaults()
            details = "The generation finished, but the image(s) could not be retrieved."
            self._set_status_results(was_successful=False, result_details=details)
            return

        self._show_image_output_parameters(len(saved_by_position))

        for idx, artifact in enumerate(saved_by_position, start=1):
            param_name = "image_url" if idx == 1 else f"image_url_{idx}"
            self.parameter_output_values[param_name] = artifact

        filenames = [artifact.name for artifact in image_artifacts]
        missing = [str(idx) for idx, artifact in enumerate(saved_by_position, start=1) if artifact is None]
        if missing:
            details = (
                f"Edited {len(image_artifacts)} of {len(saved_by_position)} images: {', '.join(filenames)}. "
                f"Image(s) {', '.join(missing)} could not be retrieved; their output slots are empty."
            )
        elif len(image_artifacts) == 1:
            details = f"Image edited successfully and saved as {filenames[0]}."
        else:
            details = f"Edited {len(image_artifacts)} images successfully: {', '.join(filenames)}."

        self._set_status_results(was_successful=True, result_details=details)

    def _set_safe_defaults(self) -> None:
        for i in range(1, 11):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.parameter_output_values[param_name] = None

    async def _save_single_image(self, generation_id: str, position: int) -> ImageUrlArtifact | None:
        try:
            image_bytes = await self._load_generated_media(generation_id, kind=ArtifactKind.IMAGE, position=position)
            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(image_bytes)
            return ImageUrlArtifact(value=saved.location, name=saved.name)
        except Exception as e:
            # A billed generation whose image cannot be retrieved is a failure, not a
            # silent success. Return None so this image is not counted as saved.
            with suppress(Exception):
                logger.warning("%s failed to retrieve image %s: %s", self.name, position, e)
            return None
