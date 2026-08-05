from __future__ import annotations

import logging
from contextlib import suppress
from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["GrokImageEdit"]


class GrokImageEdit(GriptapeProxyNode):
    """Edit images using Grok image models via Griptape model proxy.

    Inputs:
        - model (str): Grok image model to use
        - image (ImageUrlArtifact): Input image to edit (required)
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
    QUALITY_OPTIONS: ClassVar[list[str]] = ["low", "medium", "high"]

    RESOLUTION_OPTIONS: ClassVar[list[str]] = ["1k", "2k"]

    # Migrates values saved before this dropdown stored catalog model keys.
    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = {
        "Grok Imagine Image": "gtc_grok_imagine_image",
        "grok-imagine-image": "gtc_grok_imagine_image",
        # Folded in from GrokImageGeneration's retired DEPRECATED_MODELS dict; this
        # node's own removed _supports_quality check compared against the same value.
        "Grok 2 Image": "gtc_grok_imagine_image",
        "grok-2-image-1212": "gtc_grok_imagine_image",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Edit images using Grok image models via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value="gtc_grok_imagine_image",
            tooltip="Select the Grok image model to use",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        # License-policy dropdown: the component adds Options + refresh Button traits and
        # marks the models the license denies; the proxy base refuses a denied selection.
        self._install_model_access(
            parameter=model_param,
            model_choices=["gtc_grok_imagine_image"],
            default_model="gtc_grok_imagine_image",
            deprecated_values=self.LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Prompt for image editing",
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

    def _extract_image_value(self, image_input: Any) -> str | None:
        if isinstance(image_input, str):
            return image_input

        try:
            if hasattr(image_input, "value"):
                value = getattr(image_input, "value", None)
                if isinstance(value, str):
                    return value

            if hasattr(image_input, "base64"):
                b64 = getattr(image_input, "base64", None)
                if isinstance(b64, str) and b64:
                    return b64
        except Exception:
            return None

        return None

    async def _prepare_image_data_uri(self, image_input: Any) -> str | None:
        if not image_input:
            return None

        image_value = self._extract_image_value(image_input)
        if not image_value:
            return None

        try:
            return await File(image_value).aread_data_uri(fallback_mime="image/png")
        except FileLoadError:
            logger.debug("%s failed to load image value: %s", self.name, image_value)
            return None

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
        return f"{self._provider_model_id_for_selection()}:edit"

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(ValueError(f"{self.name}: Prompt is required for image editing."))

        image_value = self.get_parameter_value("image")
        if not self._has_media_value(image_value):
            exceptions.append(ValueError(f"{self.name}: Image is required for editing."))

        n_value = self.get_parameter_value("n")
        if n_value is None or not self.MIN_IMAGES <= int(n_value) <= self.MAX_IMAGES:
            exceptions.append(ValueError(f"{self.name}: n must be between {self.MIN_IMAGES} and {self.MAX_IMAGES}."))

        return exceptions if exceptions else None

    async def _build_payload(self) -> dict[str, Any]:
        prompt = (self.get_parameter_value("prompt") or "").strip()
        n_value = int(self.get_parameter_value("n") or 1)
        resolution = self.get_parameter_value("resolution") or "1k"
        api_model_id = self._provider_model_id_for_selection()
        image_data_uri = await self._prepare_image_data_uri(self.get_parameter_value("image"))

        payload: dict[str, Any] = {
            "model": api_model_id,
            "prompt": prompt,
            "n": n_value,
            "quality": self.get_parameter_value("quality") or "medium",
            "resolution": resolution,
            "response_format": "url",
        }

        if image_data_uri:
            payload["image"] = {"url": image_data_uri}

        return payload

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        data = result_json.get("data", [])
        if not data:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no image data was found in the response.",
            )
            return

        image_artifacts: list[ImageUrlArtifact] = []
        failed_urls: list[str] = []
        for idx, image_data in enumerate(data):
            image_url = image_data.get("url")
            if not image_url:
                continue

            artifact = await self._save_single_image_from_url(image_url, generation_id, idx)
            if artifact:
                image_artifacts.append(artifact)
            else:
                failed_urls.append(image_url)

        if not image_artifacts:
            self._set_safe_defaults()
            if failed_urls:
                details = (
                    f"{self.name} generation completed upstream but the image(s) could not be retrieved. "
                    f"Provider URL(s) (may be temporary): {', '.join(failed_urls)}"
                )
            else:
                details = f"{self.name} generation completed but no image URLs were found in the response."
            self._set_status_results(was_successful=False, result_details=details)
            return

        self._show_image_output_parameters(len(image_artifacts))

        for idx, artifact in enumerate(image_artifacts, start=1):
            param_name = "image_url" if idx == 1 else f"image_url_{idx}"
            self.parameter_output_values[param_name] = artifact

        filenames = [artifact.name for artifact in image_artifacts]
        if len(image_artifacts) == 1:
            details = f"Image edited successfully and saved as {filenames[0]}."
        else:
            details = f"Edited {len(image_artifacts)} images successfully: {', '.join(filenames)}."

        self._set_status_results(was_successful=True, result_details=details)

    def _set_safe_defaults(self) -> None:
        for i in range(1, 11):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.parameter_output_values[param_name] = None

    async def _save_single_image_from_url(
        self, image_url: str, generation_id: str | None = None, index: int = 0
    ) -> ImageUrlArtifact | None:
        try:
            image_bytes = await File(image_url).aread_bytes()
            if not image_bytes:
                msg = "downloaded image was empty"
                raise ValueError(msg)  # noqa: TRY301

            dest = self._output_file.build_file()
            saved = await dest.awrite_bytes(image_bytes)
            return ImageUrlArtifact(value=saved.location, name=saved.name)
        except Exception as e:
            # A billed generation whose image cannot be retrieved is a failure, not a
            # silent success. Return None so this image is not counted as saved; the
            # caller reports failure and surfaces the provider URL for manual retrieval.
            with suppress(Exception):
                logger.warning("%s failed to retrieve image %s from %s: %s", self.name, index, image_url, e)
            return None
