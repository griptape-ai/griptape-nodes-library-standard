from __future__ import annotations

import base64
import logging
from typing import Any, ClassVar

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["OpenAiImageGeneration"]


class OpenAiImageGeneration(GriptapeProxyNode):
    """Generate images using OpenAI GPT Image models via Griptape model proxy."""

    MODEL_NAME_MAP: ClassVar[dict[str, str]] = {
        "GPT Image 1": "gpt-image-1",
        "GPT Image 1.5": "gpt-image-1.5",
        "GPT Image 2": "gpt-image-2",
    }
    GPT_IMAGE_SIZE_OPTIONS: ClassVar[list[str]] = ["1024x1024", "1024x1536", "1536x1024"]
    QUALITY_OPTIONS: ClassVar[list[str]] = ["low", "medium", "high"]
    BACKGROUND_OPTIONS: ClassVar[list[str]] = ["auto", "opaque", "transparent"]
    MODERATION_OPTIONS: ClassVar[list[str]] = ["auto", "low"]
    OUTPUT_FORMAT_OPTIONS: ClassVar[list[str]] = ["png", "jpeg", "webp"]
    MIN_IMAGES: ClassVar[int] = 1
    MAX_IMAGES: ClassVar[int] = 10
    MAX_PROMPT_LENGTH: ClassVar[int] = 32_000
    DEFAULT_MODEL: ClassVar[str] = "GPT Image 2"
    DEFAULT_OUTPUT_FORMAT: ClassVar[str] = "png"
    DEFAULT_OUTPUT_FILENAME_BASE: ClassVar[str] = "openai_image"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using OpenAI GPT Image models via Griptape model proxy"

        self.add_parameter(
            ParameterString(
                name="model",
                default_value=self.DEFAULT_MODEL,
                tooltip="Select the OpenAI image model to use",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(self.MODEL_NAME_MAP.keys()))},
            )
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                default_value="",
                tooltip="Prompt for image generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
            )
        )

        self.add_parameter(
            ParameterString(
                name="size",
                default_value=self.GPT_IMAGE_SIZE_OPTIONS[0],
                tooltip="Output image size. GPT Image 2 also accepts custom sizes when provided via an input connection.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.GPT_IMAGE_SIZE_OPTIONS)},
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterInt(
                name="n",
                default_value=1,
                tooltip="Number of images to generate (1-10)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=self.MIN_IMAGES,
                max_val=self.MAX_IMAGES,
                slider=True,
            )

            ParameterString(
                name="quality",
                default_value="medium",
                tooltip="Output quality",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.QUALITY_OPTIONS)},
            )

            ParameterString(
                name="background",
                default_value="auto",
                tooltip="Background handling. Transparent backgrounds require PNG or WEBP output.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.BACKGROUND_OPTIONS)},
            )

            ParameterString(
                name="moderation",
                default_value="auto",
                tooltip="Moderation level for generated images",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.MODERATION_OPTIONS)},
            )

            ParameterString(
                name="output_format",
                default_value=self.DEFAULT_OUTPUT_FORMAT,
                tooltip="Output image format",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=self.OUTPUT_FORMAT_OPTIONS)},
            )

            ParameterInt(
                name="output_compression",
                default_value=80,
                tooltip="Compression level for JPEG or WEBP output (0-100)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                min_val=0,
                max_val=100,
                ui_options={"hide": True},
            )

        self.add_node_element(generation_settings_group)

        self.add_parameter(
            ParameterString(
                name="generation_id",
                tooltip="Generation ID from the API",
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

        for i in range(1, self.MAX_IMAGES + 1):
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
            default_filename=self._default_output_filename(self.DEFAULT_OUTPUT_FORMAT),
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )
        self._sync_output_format_visibility(self.get_parameter_value("output_format") or self.DEFAULT_OUTPUT_FORMAT)

    @classmethod
    def _default_output_filename(cls, output_format: str) -> str:
        extension = "jpg" if output_format == "jpeg" else output_format
        return f"{cls.DEFAULT_OUTPUT_FILENAME_BASE}.{extension}"

    def _get_payload_model_id(self) -> str:
        model_name = self.get_parameter_value("model") or self.DEFAULT_MODEL
        return self.MODEL_NAME_MAP.get(model_name, model_name)

    def _get_api_model_id(self) -> str:
        return self._get_payload_model_id()

    def _show_image_output_parameters(self, count: int) -> None:
        for i in range(1, self.MAX_IMAGES + 1):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            if i <= count:
                self.show_parameter_by_name(param_name)
            else:
                self.hide_parameter_by_name(param_name)

    def _sync_output_format_visibility(self, output_format: str) -> None:
        if output_format in {"jpeg", "webp"}:
            self.show_parameter_by_name("output_compression")
        else:
            self.hide_parameter_by_name("output_compression")

    def _sync_output_filename(self, output_format: str) -> None:
        current_value = self.get_parameter_value("output_file")
        default_filenames = {self._default_output_filename(fmt) for fmt in self.OUTPUT_FORMAT_OPTIONS}
        if current_value not in default_filenames:
            return

        updated_value = self._default_output_filename(output_format)
        if current_value == updated_value:
            return

        self.set_parameter_value("output_file", updated_value)
        self.publish_update_to_parameter("output_file", updated_value)

    def after_value_set(self, parameter: Any, value: Any) -> None:
        if parameter.name == "output_format" and isinstance(value, str):
            self._sync_output_format_visibility(value)
            self._sync_output_filename(value)
        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(ValueError(f"{self.name}: Prompt is required for image generation."))
        elif len(prompt) > self.MAX_PROMPT_LENGTH:
            exceptions.append(
                ValueError(
                    f"{self.name}: Prompt must be {self.MAX_PROMPT_LENGTH} characters or fewer for GPT Image generation."
                )
            )

        model_name = self.get_parameter_value("model") or self.DEFAULT_MODEL
        size = (self.get_parameter_value("size") or "").strip()
        if not size:
            exceptions.append(ValueError(f"{self.name}: Size is required for image generation."))
        elif model_name in {"GPT Image 1", "GPT Image 1.5"} and size not in self.GPT_IMAGE_SIZE_OPTIONS:
            valid_sizes = ", ".join(self.GPT_IMAGE_SIZE_OPTIONS)
            exceptions.append(ValueError(f"{self.name}: {model_name} size must be one of: {valid_sizes}."))

        n_value = self.get_parameter_value("n")
        if n_value is None or not self.MIN_IMAGES <= int(n_value) <= self.MAX_IMAGES:
            exceptions.append(ValueError(f"{self.name}: n must be between {self.MIN_IMAGES} and {self.MAX_IMAGES}."))

        output_format = self.get_parameter_value("output_format") or self.DEFAULT_OUTPUT_FORMAT
        background = self.get_parameter_value("background") or "auto"
        if background == "transparent" and output_format not in {"png", "webp"}:
            exceptions.append(
                ValueError(f"{self.name}: Transparent backgrounds require output_format to be png or webp.")
            )

        if output_format in {"jpeg", "webp"}:
            output_compression = self.get_parameter_value("output_compression")
            if output_compression is None or not 0 <= int(output_compression) <= 100:
                exceptions.append(ValueError(f"{self.name}: output_compression must be between 0 and 100."))

        return exceptions if exceptions else None

    async def _build_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._get_payload_model_id(),
            "prompt": (self.get_parameter_value("prompt") or "").strip(),
            "size": (self.get_parameter_value("size") or "").strip(),
            "n": int(self.get_parameter_value("n") or 1),
            "quality": self.get_parameter_value("quality") or "medium",
            "background": self.get_parameter_value("background") or "auto",
            "moderation": self.get_parameter_value("moderation") or "auto",
            "output_format": self.get_parameter_value("output_format") or self.DEFAULT_OUTPUT_FORMAT,
        }

        if payload["output_format"] in {"jpeg", "webp"}:
            payload["output_compression"] = int(self.get_parameter_value("output_compression") or 80)

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
        for idx, image_data in enumerate(data, start=1):
            b64_json = image_data.get("b64_json")
            if not isinstance(b64_json, str) or not b64_json:
                logger.warning("%s response item %s did not include b64_json", self.name, idx)
                continue

            artifact = await self._save_single_image_from_base64(b64_json, index=idx)
            if artifact is not None:
                image_artifacts.append(artifact)

        if not image_artifacts:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"{self.name} generation completed but no decodable GPT image payloads were found.",
            )
            return

        self._show_image_output_parameters(len(image_artifacts))
        for idx, artifact in enumerate(image_artifacts, start=1):
            param_name = "image_url" if idx == 1 else f"image_url_{idx}"
            self.parameter_output_values[param_name] = artifact

        filenames = [artifact.name for artifact in image_artifacts if artifact.name]
        if len(image_artifacts) == 1:
            details = f"Image generated successfully and saved as {filenames[0] if filenames else generation_id}."
        else:
            details = (
                f"Generated {len(image_artifacts)} images successfully: {', '.join(filenames)}."
                if filenames
                else f"Generated {len(image_artifacts)} images successfully."
            )
        self._set_status_results(was_successful=True, result_details=details)

    def _set_safe_defaults(self) -> None:
        self._show_image_output_parameters(1)
        for i in range(1, self.MAX_IMAGES + 1):
            param_name = "image_url" if i == 1 else f"image_url_{i}"
            self.parameter_output_values[param_name] = None

    async def _save_single_image_from_base64(self, b64_json: str, *, index: int) -> ImageUrlArtifact | None:
        try:
            image_bytes = base64.b64decode(b64_json)
            dest = self._output_file.build_file(_index=index)
            saved = await dest.awrite_bytes(image_bytes)
            return ImageUrlArtifact(value=saved.location, name=saved.name)
        except Exception as e:
            logger.warning("%s failed to save generated image %s: %s", self.name, index, e)
            return None
