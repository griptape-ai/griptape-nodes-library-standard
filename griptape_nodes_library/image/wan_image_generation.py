from __future__ import annotations

import logging
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["WanImageGeneration"]

# Model options
MODEL_OPTIONS = ["Wan 2.7 Image", "Wan 2.7 Image Pro"]

MODEL_MAPPING = {
    "Wan 2.7 Image": "wan2.7-image",
    "Wan 2.7 Image Pro": "wan2.7-image-pro",
}

DEFAULT_MODEL = "Wan 2.7 Image"

# Size presets supported by the API
SIZE_PRESETS = ["Custom", "1K", "2K", "4K"]

# Pixel constraints per model
# wan2.7-image: 768*768 to 2048*2048, no 4K
# wan2.7-image-pro: 768*768 to 4096*4096 (4K only for text-to-image)
MIN_DIMENSION = 768
MAX_DIMENSION_STANDARD = 2048
MAX_DIMENSION_PRO = 4096


class WanImageGeneration(GriptapeProxyNode):
    """Generate images using Wan 2.7 models via Griptape model proxy.

    Inputs:
        - model (str): Wan model to use (default: "Wan 2.7 Image")
        - prompt (str): Text description of the desired image
        - negative_prompt (str): Text description of what to avoid
        - size_preset (str): Named resolution preset ("1K", "2K", "4K") or "Custom"
        - width (int): Output width in pixels (used when size_preset is "Custom")
        - height (int): Output height in pixels (used when size_preset is "Custom")
        - n (int): Number of images to generate (1-4, default: 4)
        - thinking_mode (bool): Improves quality at cost of longer generation time (default: True)
        - randomize_seed (bool): If true, randomize the seed on each run
        - seed (int): Random seed for reproducible results

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response
        - IMAGE (ImageUrlArtifact): Generated image
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate images using Wan 2.7 models via Griptape model proxy"

        # Model selection
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                default_value=DEFAULT_MODEL,
                tooltip="Select the Wan model to use",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=MODEL_OPTIONS)},
            )
        )

        # Core parameters
        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the desired image",
                multiline=True,
                placeholder_text="Describe the image you want to generate...",
                allow_output=False,
                ui_options={
                    "display_name": "Prompt",
                },
            )
        )

        self.add_parameter(
            ParameterString(
                name="negative_prompt",
                tooltip="Describe what to avoid in the generated image",
                multiline=True,
                placeholder_text="low quality, blurry, distorted...",
                allow_output=False,
                ui_options={
                    "display_name": "Negative Prompt",
                },
            )
        )

        # Size parameters
        self.add_parameter(
            Parameter(
                name="size_preset",
                input_types=["str"],
                type="str",
                default_value="2K",
                tooltip="Named resolution preset. Select 'Custom' to specify exact width and height. wan2.7-image does not support 4K.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=SIZE_PRESETS)},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=1024,
                tooltip="Output width in pixels (used when size_preset is 'Custom'). Aspect ratio must be between 1:8 and 8:1.",
                allow_output=False,
                min_val=MIN_DIMENSION,
                max_val=MAX_DIMENSION_PRO,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=1024,
                tooltip="Output height in pixels (used when size_preset is 'Custom'). Aspect ratio must be between 1:8 and 8:1.",
                allow_output=False,
                min_val=MIN_DIMENSION,
                max_val=MAX_DIMENSION_PRO,
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterInt(
                name="n",
                default_value=4,
                tooltip="Number of images to generate (1-4). Cost scales with number of images.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=1, max_val=4)},
            )

            ParameterBool(
                name="thinking_mode",
                default_value=True,
                tooltip="Improves image quality at the cost of longer generation time. Only effective for text-to-image without image input.",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )

            self._seed_parameter = SeedParameter(self)
            self._seed_parameter.add_input_parameters(inside_param_group=True)

        self.add_node_element(generation_settings_group)

        # OUTPUTS
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

        self.add_parameter(
            ParameterImage(
                name="IMAGE",
                tooltip="Generated image",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"is_full_width": True, "pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="wan_image.png",
        )
        self._output_file.add_parameter()

        # Status parameters MUST be last
        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    async def _process_generation(self) -> None:
        self._seed_parameter.preprocess()
        await super()._process_generation()

    def _get_api_model_id(self) -> str:
        model = self.get_parameter_value("model") or DEFAULT_MODEL
        return MODEL_MAPPING.get(str(model), str(model))

    def _resolve_size(self) -> str:
        """Resolve the size parameter from preset or custom width/height."""
        size_preset = self.get_parameter_value("size_preset") or "2K"
        if size_preset != "Custom":
            return size_preset

        width = self.get_parameter_value("width") or 1024
        height = self.get_parameter_value("height") or 1024
        return f"{int(width)}*{int(height)}"

    async def _build_payload(self) -> dict[str, Any]:
        prompt = self.get_parameter_value("prompt") or ""
        negative_prompt = self.get_parameter_value("negative_prompt") or ""
        size = self._resolve_size()
        n = self.get_parameter_value("n") or 4
        seed = self._seed_parameter.get_seed()
        thinking_mode = self.get_parameter_value("thinking_mode")

        # Build content array with text prompt
        content = [{"text": prompt}]

        # Build input with messages format
        input_data: dict[str, Any] = {
            "messages": [{"role": "user", "content": content}],
        }

        if negative_prompt:
            input_data["negative_prompt"] = negative_prompt

        payload: dict[str, Any] = {
            "input": input_data,
            "parameters": {
                "n": int(n),
                "size": size,
                "seed": seed,
                "thinking_mode": thinking_mode if thinking_mode is not None else True,
            },
        }

        return payload

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        """Parse the Wan 2.7 image generation result.

        Response structure from proxy:
        {
            "request_id": "...",
            "output": {
                "task_id": "...",
                "task_status": "SUCCEEDED",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"image": "https://...", "type": "image"}
                            ]
                        }
                    }
                ]
            },
            "usage": {...}
        }
        """
        # Extract image URL from output.choices[0].message.content[0].image
        try:
            output = result_json.get("output", result_json)
            choices = output.get("choices", [])
            if not choices:
                # Try top-level choices (in case proxy unwraps output)
                choices = result_json.get("choices", [])
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content", [])
            first_content_item = content[0]
            image_url = first_content_item.get("image")
        except (IndexError, KeyError, TypeError) as e:
            logger.error("Failed to extract image URL from response: %s", e)
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image found in the response.",
            )
            return

        if image_url:
            await self._save_image_from_url(image_url)
        else:
            logger.warning("No image URL found in content")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="Generation completed but no image URL was found in the response.",
            )

    async def _save_image_from_url(self, image_url: str) -> None:
        """Download and save the image from the provided URL."""
        try:
            logger.info("Downloading image from URL")
            image_bytes = await File(image_url).aread_bytes()
            if image_bytes:
                dest = self._output_file.build_file()
                saved = await dest.awrite_bytes(image_bytes)
                self.parameter_output_values["IMAGE"] = ImageUrlArtifact(saved.location)
                logger.info("Saved image as %s", saved.name)
                self._set_status_results(
                    was_successful=True, result_details=f"Image generated successfully and saved as {saved.name}."
                )
            else:
                self.parameter_output_values["IMAGE"] = ImageUrlArtifact(value=image_url)
                self._set_status_results(
                    was_successful=True,
                    result_details="Image generated successfully. Using provider URL (could not download image bytes).",
                )
        except Exception as e:
            logger.error("Failed to save image from URL: %s", e)
            self.parameter_output_values["IMAGE"] = ImageUrlArtifact(value=image_url)
            self._set_status_results(
                was_successful=True,
                result_details=f"Image generated successfully. Using provider URL (could not save to static storage: {e}).",
            )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["IMAGE"] = None
