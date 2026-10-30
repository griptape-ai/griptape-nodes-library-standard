from __future__ import annotations

import logging
from contextlib import suppress
from enum import StrEnum
from typing import Any, ClassVar

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_list

from griptape_nodes_library.media import prepare_media_data_uri
from griptape_nodes_library.proxy import ArtifactKind, GriptapeProxyNode

logger = logging.getLogger("griptape_nodes")

__all__ = ["Flux3ImageGeneration"]

MAX_INPUT_IMAGES = 10
DEFAULT_IMAGE_SIZE = 1024
MIN_DIMENSION_FLUX_3_IMAGE = 256
MIN_DIMENSION_FLUX_3_KLEIN = 64
MAX_IMAGE_DIMENSION = 6256
MAX_TOTAL_PIXELS = 16_822_272
MAX_TOTAL_PIXELS_WITH_REFERENCES = 4_194_304
DEFAULT_STEPS_KLEIN = 16
MAX_STEPS_KLEIN = 16


class Flux3ImageModel(StrEnum):
    FLUX_3_IMAGE = "flux-3-image"
    FLUX_3_IMAGE_KLEIN = "flux-3-image-klein"


class Flux3SizeMode(StrEnum):
    RESOLUTION_TIER = "Resolution Tier"
    CUSTOM_DIMENSIONS = "Custom Dimensions"


class Flux3SafetyTolerance(StrEnum):
    LEAST_RESTRICTIVE = "least restrictive"
    MODERATE = "moderate"
    MOST_RESTRICTIVE = "most restrictive"


MODEL_OPTIONS = [
    Flux3ImageModel.FLUX_3_IMAGE.value,
    Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value,
]
DEFAULT_MODEL = Flux3ImageModel.FLUX_3_IMAGE.value

LEGACY_MODEL_VALUES: dict[str, str] = {
    "FLUX 3 Image": Flux3ImageModel.FLUX_3_IMAGE.value,
    "FLUX 3 Image Klein": Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value,
    "gtc_flux_3_image": Flux3ImageModel.FLUX_3_IMAGE.value,
    "gtc_flux_3_image_klein": Flux3ImageModel.FLUX_3_IMAGE_KLEIN.value,
}

ASPECT_RATIO_OPTIONS = [
    "auto",
    "21:9",
    "2:1",
    "16:9",
    "3:2",
    "7:5",
    "4:3",
    "5:4",
    "1:1",
    "4:5",
    "3:4",
    "5:7",
    "2:3",
    "9:16",
    "1:2",
    "9:21",
]

RESOLUTION_OPTIONS_FLUX_3_IMAGE = ["768sq", "1k", "1.5k", "2k", "4k", "match_input"]
RESOLUTION_OPTIONS_KLEIN = ["768sq", "1k", "1.5k", "2k", "4k"]
SIZE_MODE_OPTIONS = [
    Flux3SizeMode.RESOLUTION_TIER.value,
    Flux3SizeMode.CUSTOM_DIMENSIONS.value,
]
OUTPUT_FORMAT_OPTIONS = ["jpeg", "png"]
SAFETY_TOLERANCE_OPTIONS = [
    Flux3SafetyTolerance.LEAST_RESTRICTIVE.value,
    Flux3SafetyTolerance.MODERATE.value,
    Flux3SafetyTolerance.MOST_RESTRICTIVE.value,
]


class Flux3ImageGeneration(GriptapeProxyNode):
    """Generate and edit images using Black Forest Labs FLUX 3 models via Griptape model proxy.

    Inputs:
        - model (str): FLUX 3 model to use ("flux-3-image" or "flux-3-image-klein")
        - prompt (str): Text description of the desired image
        - input_images (list): Optional reference images (up to 10) for editing or multi-reference composition
        - size_mode (str): Choose between resolution tier ("Resolution Tier") or exact pixel dimensions ("Custom Dimensions")
        - aspect_ratio (str): Output aspect ratio ("auto" or preset ratio)
        - resolution (str): Output resolution tier ("768sq", "1k", "1.5k", "2k", "4k", or "match_input" on flux-3-image)
        - width (int): Custom output width in pixels
        - height (int): Custom output height in pixels
        - grounding (bool): [flux-3-image only] Ground the prompt with web and image search (default: True)
        - safety_tolerance (str): Content moderation preset ("least restrictive", "moderate", or "most restrictive")
        - randomize_seed (bool) / seed (int): [flux-3-image-klein only] Seed controls
        - output_format (str): [flux-3-image-klein only] Output image format ("jpeg" or "png")
        - steps (int): [flux-3-image-klein only] Denoising steps (1-16, default: 16)

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Verbatim provider response from the model proxy
        - image_url (ImageUrlArtifact): Generated image as URL artifact
        - was_successful (bool): Whether the generation succeeded
        - result_details (str): Details about the generation result or error
    """

    LEGACY_MODEL_VALUES: ClassVar[dict[str, str]] = LEGACY_MODEL_VALUES

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate and edit images using FLUX 3 models via Griptape model proxy"

        model_param = ParameterString(
            name="model",
            default_value=DEFAULT_MODEL,
            tooltip="Select the FLUX 3 image model to use",
            allow_output=False,
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_OPTIONS,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            ParameterString(
                name="prompt",
                tooltip="Text description of the desired image or edit instruction",
                multiline=True,
                placeholder_text="Describe the image you want to generate or edit...",
                allow_output=False,
                ui_options={"display_name": "Prompt"},
            )
        )

        self.add_parameter(
            ParameterList(
                name="input_images",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                default_value=[],
                tooltip="Optional reference images (1 to 10) for editing or multi-reference composition",
                allowed_modes={ParameterMode.INPUT},
                ui_options={"expander": True, "display_name": "Input Images", "hide_property": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="size_mode",
                default_value=Flux3SizeMode.RESOLUTION_TIER.value,
                tooltip="Choose output size by resolution tier + aspect ratio or by exact width and height",
                allow_output=False,
                traits={Options(choices=SIZE_MODE_OPTIONS)},
                ui_options={"display_name": "Size Mode"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="aspect_ratio",
                default_value="auto",
                tooltip="Output aspect ratio. 'auto' lets FLUX 3 pick from the prompt or first reference image.",
                allow_output=False,
                traits={Options(choices=ASPECT_RATIO_OPTIONS)},
                ui_options={"display_name": "Aspect Ratio"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolution",
                default_value="1k",
                tooltip="Equal-area resolution tier ('768sq', '1k', '1.5k', '2k', '4k', or 'match_input' for edits on flux-3-image)",
                allow_output=False,
                traits={Options(choices=RESOLUTION_OPTIONS_FLUX_3_IMAGE)},
                ui_options={"display_name": "Resolution"},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="width",
                default_value=DEFAULT_IMAGE_SIZE,
                tooltip="Exact output width in pixels (256-6256 for flux-3-image, 64-6256 for Klein)",
                allow_output=False,
                min_val=MIN_DIMENSION_FLUX_3_KLEIN,
                max_val=MAX_IMAGE_DIMENSION,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="height",
                default_value=DEFAULT_IMAGE_SIZE,
                tooltip="Exact output height in pixels (256-6256 for flux-3-image, 64-6256 for Klein)",
                allow_output=False,
                min_val=MIN_DIMENSION_FLUX_3_KLEIN,
                max_val=MAX_IMAGE_DIMENSION,
                hide=True,
            )
        )

        with ParameterGroup(name="Generation Settings", ui_options={"collapsed": True}) as generation_settings_group:
            ParameterBool(
                name="grounding",
                default_value=True,
                tooltip="Allow FLUX 3 Image to ground the prompt with web and image search",
                allow_output=False,
            )

            ParameterString(
                name="safety_tolerance",
                default_value=Flux3SafetyTolerance.MODERATE.value,
                tooltip="Content moderation strictness level",
                allow_output=False,
                traits={Options(choices=SAFETY_TOLERANCE_OPTIONS)},
            )

            self._seed_parameter = SeedParameter(self)
            self._seed_parameter.add_input_parameters(inside_param_group=True)

            ParameterString(
                name="output_format",
                default_value="jpeg",
                tooltip="Desired format of the output image (Klein only)",
                allow_output=False,
                traits={Options(choices=OUTPUT_FORMAT_OPTIONS)},
                hide=True,
            )

            ParameterInt(
                name="steps",
                default_value=DEFAULT_STEPS_KLEIN,
                tooltip="Number of denoising steps (Klein only, 1-16)",
                allow_output=False,
                slider=True,
                min_val=1,
                max_val=MAX_STEPS_KLEIN,
                hide=True,
            )

        self.add_node_element(generation_settings_group)
        self._sync_model_parameter_visibility()

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

        self.add_parameter(
            ParameterImage(
                name="image_url",
                tooltip="Generated image as URL artifact",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="flux_3_image.png",
        )
        self._output_file.add_parameter()

        self._create_status_parameters(
            result_details_tooltip="Details about the image generation result or any errors",
            result_details_placeholder="Generation status and details will appear here.",
            parameter_group_initially_collapsed=True,
        )

    def _log(self, message: str) -> None:
        with suppress(Exception):
            logger.info(message)

    def _sync_model_parameter_visibility(self) -> None:
        api_model_id = self._get_selected_model_id() or DEFAULT_MODEL
        is_klein = api_model_id == Flux3ImageModel.FLUX_3_IMAGE_KLEIN
        if is_klein:
            self.hide_parameter_by_name("grounding")
            self.show_parameter_by_name("randomize_seed")
            self.show_parameter_by_name("seed")
            self.show_parameter_by_name("output_format")
            self.show_parameter_by_name("steps")
            self._update_option_choices("resolution", RESOLUTION_OPTIONS_KLEIN, "1k")
        else:
            self.show_parameter_by_name("grounding")
            self.hide_parameter_by_name("randomize_seed")
            self.hide_parameter_by_name("seed")
            self.hide_parameter_by_name("output_format")
            self.hide_parameter_by_name("steps")
            current_res = self.get_parameter_value("resolution") or "1k"
            default_res = current_res if current_res in RESOLUTION_OPTIONS_FLUX_3_IMAGE else "1k"
            self._update_option_choices("resolution", RESOLUTION_OPTIONS_FLUX_3_IMAGE, default_res)

    def _sync_size_mode_visibility(self) -> None:
        size_mode = self.get_parameter_value("size_mode") or Flux3SizeMode.RESOLUTION_TIER.value
        if size_mode == Flux3SizeMode.CUSTOM_DIMENSIONS:
            self.hide_parameter_by_name("aspect_ratio")
            self.hide_parameter_by_name("resolution")
            self.show_parameter_by_name("width")
            self.show_parameter_by_name("height")
        else:
            self.show_parameter_by_name("aspect_ratio")
            self.show_parameter_by_name("resolution")
            self.hide_parameter_by_name("width")
            self.hide_parameter_by_name("height")

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        self._seed_parameter.after_value_set(parameter, value)

        if parameter.name == "input_images" and isinstance(value, list):
            updated_list = normalize_artifact_list(value, ImageUrlArtifact, accepted_types=(ImageArtifact,))
            if updated_list != value:
                self.set_parameter_value("input_images", updated_list)

        if parameter.name == "model":
            self._sync_model_parameter_visibility()

        if parameter.name == "size_mode":
            self._sync_size_mode_visibility()

    @staticmethod
    def _parse_safety_tolerance(value: str | None) -> int:
        if not value:
            msg = "Pick a 'safety_tolerance' value."
            raise NodeError(msg)

        match value:
            case Flux3SafetyTolerance.MOST_RESTRICTIVE:
                return 0
            case Flux3SafetyTolerance.MODERATE:
                return 2
            case Flux3SafetyTolerance.LEAST_RESTRICTIVE:
                return 4
            case _:
                msg = f"'safety_tolerance' must be one of {SAFETY_TOLERANCE_OPTIONS}. It's set to {value!r}."
                raise NodeError(msg)

    def _raw_input_images(self) -> list[Any]:
        raw = self.get_parameter_list_value("input_images") or self.parameter_values.get("input_images") or []
        if not isinstance(raw, list):
            raw = [raw] if raw else []
        return [item for item in raw if item is not None and item != ""]

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = super().validate_before_node_run() or []

        prompt = (self.get_parameter_value("prompt") or "").strip()
        if not prompt:
            exceptions.append(NodeError("Enter a prompt in 'Prompt' to generate an image."))

        images = self._raw_input_images()
        if len(images) > MAX_INPUT_IMAGES:
            exceptions.append(
                NodeError(
                    f"'Input Images' accepts at most {MAX_INPUT_IMAGES} reference images. {len(images)} are connected."
                )
            )

        api_model_id = self._get_selected_model_id() or DEFAULT_MODEL
        is_klein = api_model_id == Flux3ImageModel.FLUX_3_IMAGE_KLEIN
        size_mode = self.get_parameter_value("size_mode") or Flux3SizeMode.RESOLUTION_TIER.value

        if size_mode == Flux3SizeMode.CUSTOM_DIMENSIONS:
            width = self.get_parameter_value("width")
            height = self.get_parameter_value("height")
            min_dim = MIN_DIMENSION_FLUX_3_KLEIN if is_klein else MIN_DIMENSION_FLUX_3_IMAGE
            if not isinstance(width, int) or not (min_dim <= width <= MAX_IMAGE_DIMENSION):
                exceptions.append(
                    NodeError(f"'width' must be between {min_dim} and {MAX_IMAGE_DIMENSION}. It's set to {width!r}.")
                )
            if not isinstance(height, int) or not (min_dim <= height <= MAX_IMAGE_DIMENSION):
                exceptions.append(
                    NodeError(f"'height' must be between {min_dim} and {MAX_IMAGE_DIMENSION}. It's set to {height!r}.")
                )
            if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
                pixels = width * height
                max_pixels = MAX_TOTAL_PIXELS_WITH_REFERENCES if (images and not is_klein) else MAX_TOTAL_PIXELS
                if pixels > max_pixels:
                    exceptions.append(
                        NodeError(
                            f"'width' × 'height' ({pixels:,} pixels) exceeds the maximum of {max_pixels:,} pixels."
                        )
                    )
        else:
            resolution = self.get_parameter_value("resolution") or "1k"
            if resolution == "match_input":
                if is_klein:
                    exceptions.append(
                        NodeError("'Resolution' cannot be 'match_input' when 'flux-3-image-klein' is selected.")
                    )
                elif not images:
                    exceptions.append(
                        NodeError(
                            "'Resolution' is set to 'match_input', which requires at least one reference image in 'Input Images'."
                        )
                    )

        return exceptions or None

    async def _build_payload(self) -> dict[str, Any]:
        api_model_id = self._get_selected_model_id() or DEFAULT_MODEL
        is_klein = api_model_id == Flux3ImageModel.FLUX_3_IMAGE_KLEIN

        prompt = (self.get_parameter_value("prompt") or "").strip()
        safety_tolerance = self._parse_safety_tolerance(self.get_parameter_value("safety_tolerance"))
        size_mode = self.get_parameter_value("size_mode") or Flux3SizeMode.RESOLUTION_TIER.value

        payload: dict[str, Any] = {
            "prompt": prompt,
            "safety_tolerance": safety_tolerance,
        }

        if size_mode == Flux3SizeMode.CUSTOM_DIMENSIONS:
            payload["width"] = int(self.get_parameter_value("width") or DEFAULT_IMAGE_SIZE)
            payload["height"] = int(self.get_parameter_value("height") or DEFAULT_IMAGE_SIZE)
        else:
            payload["aspect_ratio"] = self.get_parameter_value("aspect_ratio") or "auto"
            payload["resolution"] = self.get_parameter_value("resolution") or "1k"

        if is_klein:
            self._seed_parameter.preprocess()
            payload["seed"] = self._seed_parameter.get_seed()
            payload["output_format"] = self.get_parameter_value("output_format") or "jpeg"
            payload["steps"] = int(self.get_parameter_value("steps") or DEFAULT_STEPS_KLEIN)
        else:
            grounding = self.get_parameter_value("grounding")
            payload["grounding"] = True if grounding is None else bool(grounding)

        input_images_list = normalize_artifact_list(
            self._raw_input_images(), ImageUrlArtifact, accepted_types=(ImageArtifact,)
        )
        encoded_images: list[str] = []
        for image_input in input_images_list[:MAX_INPUT_IMAGES]:
            data_uri = await prepare_media_data_uri(image_input, kind="image", node_name=self.name)
            if data_uri:
                encoded_images.append(data_uri)

        if encoded_images:
            payload["images"] = encoded_images

        return payload

    async def _parse_result(self, _result_json: dict[str, Any], generation_id: str) -> None:
        await self._save_generated_media(
            generation_id,
            "image_url",
            lambda v, _n: ImageUrlArtifact(v),
            kind=ArtifactKind.IMAGE,
            media_kind="image",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["generation_id"] = ""
        self.parameter_output_values["provider_response"] = None
        self.parameter_output_values["image_url"] = None
