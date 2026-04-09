from __future__ import annotations

import asyncio
import base64
import json
import logging
from contextlib import suppress
from io import BytesIO
from typing import Any

from griptape.artifacts import ImageArtifact, TextArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options
from griptape_nodes.utils.artifact_normalization import normalize_artifact_input, normalize_artifact_list
from PIL import Image

from griptape_nodes_library.griptape_proxy_node import GriptapeProxyNode
from griptape_nodes_library.splat.splat_artifact import SplatUrlArtifact
from griptape_nodes_library.splat.parameter_splat import ParameterSplat
from griptape_nodes_library.three_d.three_d_artifact import ThreeDUrlArtifact
from griptape_nodes_library.three_d.parameter_three_d import ParameterThreeD
from griptape_nodes_library.media.image_url_artifact import ImageUrlArtifact as ImageUrl
from griptape_nodes_library.media.parameter_media import ParameterMedia

logger = logging.getLogger("griptape_nodes")

__all__ = ["WorldLabsWorldGeneration"]

# Model options
MODEL_OPTIONS = ["Marble 1.1 Plus", "Marble 1.1", "Marble 1.0", "Marble 1.0 Draft"]
DEFAULT_MODEL = "Marble 1.1"

# Model ID mapping
MODEL_ID_MAP = {
    "Marble 1.1 Plus": "marble-1.1-plus",
    "Marble 1.1": "marble-1.1",
    "Marble 1.0": "marble-1.0",
    "Marble 1.0 Draft": "marble-1.0-draft",
}

# Input type options
INPUT_TYPE_OPTIONS = ["Text", "Image", "Multi-Image", "Video"]
DEFAULT_INPUT_TYPE = "Text"

# Maximum number of images for multi-image
MAX_MULTI_IMAGES = 8
DEFAULT_MULTI_IMAGES = 4


class WorldLabsWorldGeneration(GriptapeProxyNode):
    """Generate 3D worlds using World Labs Marble via Griptape model proxy.

    Inputs:
        - model (str): Marble model to use (1.1 Plus, 1.1, 1.0, 1.0 Draft)
        - input_type (str): Type of input (Text, Image, Multi-Image, Video)
        - text_prompt (str): Text description or guidance for world generation
        - image (ImageArtifact): Single image input (for Image type)
        - is_panorama (bool): Whether the image is already a panorama (saves cost)
        - images (list): Multiple images (for Multi-Image type, 2-8 images)
        - azimuth_angles (str): Comma-separated azimuth angles in degrees (0=front, 90=right, 180=back, 270=left)
        - enable_reconstruction (bool): Allow up to 8 images with reconstruction mode
        - video (MediaArtifact): Video input (for Video type)
        - disable_recaption (bool): Use text prompt as-is without auto-recaptioning
        - display_name (str): Optional display name for the world
        - randomize_seed (bool): If true, randomize seed on each run
        - seed (int): Random seed for reproducible results
        - tags (str): Comma-separated tags for the world

    Outputs:
        - generation_id (str): Generation ID from the API
        - provider_response (dict): Complete World object from provider
        - world_id (str): World Labs world identifier
        - viewer_url (str): URL to view world in Marble viewer
        - splat_100k (SplatUrlArtifact): 100k resolution Gaussian splat file
        - splat_500k (SplatUrlArtifact): 500k resolution Gaussian splat file
        - splat_full_res (SplatUrlArtifact): Full resolution Gaussian splat file
        - mesh (ThreeDUrlArtifact): Collider mesh in GLB format
        - panorama (ImageUrlArtifact): Panorama image of the world
        - thumbnail (ImageUrlArtifact): Thumbnail image
        - caption (str): AI-generated caption describing the world
        - was_successful (bool): Whether generation succeeded
        - result_details (str): Details about the generation result or error
    """

    SERVICE_NAME = "Griptape"
    API_KEY_NAME = "GT_CLOUD_API_KEY"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "API Nodes"
        self.description = "Generate 3D worlds using World Labs Marble via Griptape model proxy"

        # Model selection
        self.add_parameter(
            ParameterString(
                name="model",
                default_value=DEFAULT_MODEL,
                tooltip="Marble model: 1.1 Plus (variable pricing, larger worlds), 1.1 (standard), 1.0 (previous), 1.0 Draft (fast/cheaper)",
                allow_output=False,
                traits={Options(choices=MODEL_OPTIONS)},
                ui_options={"display_name": "Model"},
            )
        )

        # Input type selector
        self.add_parameter(
            ParameterString(
                name="input_type",
                default_value=DEFAULT_INPUT_TYPE,
                tooltip="Type of input for world generation",
                allow_output=False,
                traits={Options(choices=INPUT_TYPE_OPTIONS)},
                ui_options={"display_name": "Input Type"},
            )
        )

        # Text prompt (always visible, required for text, optional for others)
        self.add_parameter(
            ParameterString(
                name="text_prompt",
                tooltip="Text description for world generation (required for Text input, optional text guidance for Image/Video)",
                multiline=True,
                placeholder_text="Describe the 3D world you want to generate...",
                allow_output=False,
                ui_options={"display_name": "Text Prompt"},
            )
        )

        # Image input (visible when input_type == "Image")
        self.add_parameter(
            ParameterMedia(
                name="image",
                tooltip="Input image for image-to-world generation",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                ui_options={"display_name": "Image Input"},
            )
        )

        # Panorama flag (visible when input_type == "Image")
        self.add_parameter(
            ParameterBool(
                name="is_panorama",
                default_value=False,
                tooltip="Check this if the image is already a panorama (skips pano generation, saves 80 credits)",
                allow_output=False,
                ui_options={"display_name": "Image is Panorama"},
            )
        )

        # Multi-image input (visible when input_type == "Multi-Image")
        self.add_parameter(
            ParameterList(
                name="images",
                input_types=[
                    "ImageArtifact",
                    "ImageUrlArtifact",
                    "str",
                    "list",
                    "list[ImageArtifact]",
                    "list[ImageUrlArtifact]",
                ],
                default_value=[],
                tooltip="Multiple images for multi-view world generation (2-8 images)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"expander": True, "display_name": "Images"},
            )
        )

        # Azimuth angles for multi-image
        self.add_parameter(
            ParameterString(
                name="azimuth_angles",
                default_value="",
                tooltip="Comma-separated azimuth angles in degrees: 0=front, 90=right, 180=back, 270=left. Example: 0,90,180,270",
                allow_output=False,
                ui_options={"display_name": "Azimuth Angles (optional)"},
            )
        )

        # Reconstruction mode for multi-image
        self.add_parameter(
            ParameterBool(
                name="enable_reconstruction",
                default_value=False,
                tooltip="Enable reconstruction mode (allows up to 8 images instead of 4)",
                allow_output=False,
                ui_options={"display_name": "Enable Reconstruction (5-8 images)"},
            )
        )

        # Video input (visible when input_type == "Video")
        self.add_parameter(
            ParameterMedia(
                name="video",
                tooltip="Input video for video-to-world generation (max 100MB)",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                accept_any=True,
                ui_options={"display_name": "Video Input"},
            )
        )

        # Advanced parameters (hidden by default)
        self.add_parameter(
            ParameterBool(
                name="disable_recaption",
                default_value=False,
                tooltip="Use text prompt as-is without auto-recaptioning by the provider",
                allow_output=False,
                hide=True,
                ui_options={"display_name": "Disable Auto-Recaption"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="display_name",
                default_value="",
                tooltip="Optional display name for the world (max 64 characters)",
                allow_output=False,
                hide=True,
                ui_options={"display_name": "Display Name"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="tags",
                default_value="",
                tooltip="Comma-separated tags for the world",
                allow_output=False,
                hide=True,
                ui_options={"display_name": "Tags"},
            )
        )

        # Seed parameter
        self._seed_parameter = SeedParameter(self, 4294967295)
        self._seed_parameter.add_input_parameters()

        # OUTPUTS
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
                tooltip="Complete World object from World Labs",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
                hide=True,
            )
        )

        self.add_parameter(
            ParameterString(
                name="world_id",
                tooltip="World Labs world identifier",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"display_name": "World ID"},
            )
        )

        self.add_parameter(
            ParameterString(
                name="viewer_url",
                tooltip="URL to view the world in Marble viewer",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                ui_options={"display_name": "Viewer URL"},
            )
        )

        # Splat outputs (using ParameterSplat)
        self.add_parameter(
            ParameterSplat(
                name="splat_100k",
                tooltip="100k resolution Gaussian splat file (.spz format)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
                ui_options={"display_name": "Splat (100k)"},
            )
        )

        self.add_parameter(
            ParameterSplat(
                name="splat_500k",
                tooltip="500k resolution Gaussian splat file (.spz format)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
                ui_options={"display_name": "Splat (500k)"},
            )
        )

        self.add_parameter(
            ParameterSplat(
                name="splat_full_res",
                tooltip="Full resolution Gaussian splat file (.spz format)",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
                ui_options={"display_name": "Splat (Full Res)"},
            )
        )

        # Mesh output (using ParameterThreeD)
        self.add_parameter(
            ParameterThreeD(
                name="mesh",
                tooltip="Collider mesh in GLB format",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                pulse_on_run=True,
                ui_options={"display_name": "Mesh (GLB)"},
            )
        )

        # Image outputs (using ParameterMedia)
        self.add_parameter(
            ParameterMedia(
                name="panorama",
                tooltip="Panorama image of the world",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                allow_input=False,
                allow_property=True,
                ui_options={"display_name": "Panorama"},
            )
        )

        self.add_parameter(
            ParameterMedia(
                name="thumbnail",
                tooltip="Thumbnail image",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                allow_input=False,
                allow_property=True,
                ui_options={"display_name": "Thumbnail"},
            )
        )

        # Caption output
        self.add_parameter(
            ParameterString(
                name="caption",
                tooltip="AI-generated caption describing the world",
                allowed_modes={ParameterMode.OUTPUT, ParameterMode.PROPERTY},
                settable=False,
                multiline=True,
                ui_options={"display_name": "Caption"},
            )
        )

        self.add_parameter(
            ParameterBool(
                name="was_successful",
                tooltip="Whether the world generation succeeded",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="result_details",
                tooltip="Details about the generation result or error",
                allow_input=False,
                allow_property=False,
                multiline=True,
            )
        )

    def _get_api_model_id(self) -> str:
        """Map friendly model name to backend model ID."""
        params = self._get_parameters()
        model_name = params.get("model", DEFAULT_MODEL)
        return MODEL_ID_MAP.get(model_name, "marble-1.1")

    def after_value_set(self, parameter: Parameter, old_value: Any, new_value: Any) -> None:
        """Handle parameter changes and update visibility."""
        super().after_value_set(parameter, old_value, new_value)

        if parameter.name == "input_type":
            self._update_input_visibility()

    def _update_input_visibility(self) -> None:
        """Update parameter visibility based on selected input type."""
        params = self._get_parameters()
        input_type = params.get("input_type", DEFAULT_INPUT_TYPE)

        # Image-specific parameters
        self.get_parameter("image").hide = input_type != "Image"
        self.get_parameter("is_panorama").hide = input_type != "Image"

        # Multi-image-specific parameters
        self.get_parameter("images").hide = input_type != "Multi-Image"
        self.get_parameter("azimuth_angles").hide = input_type != "Multi-Image"
        self.get_parameter("enable_reconstruction").hide = input_type != "Multi-Image"

        # Video-specific parameters
        self.get_parameter("video").hide = input_type != "Video"

        # Disable recaption only visible for non-text inputs
        self.get_parameter("disable_recaption").hide = input_type == "Text"

    async def _build_payload(self) -> dict[str, Any]:
        """Build the World Labs request payload."""
        params = self._get_parameters()
        input_type = params.get("input_type", DEFAULT_INPUT_TYPE)

        # Build discriminated union world_prompt based on input type
        world_prompt = await self._build_world_prompt(params, input_type)

        # Build top-level request
        payload: dict[str, Any] = {
            "world_prompt": world_prompt,
            "model": self._get_api_model_id(),
        }

        # Add optional parameters
        seed = params.get("seed")
        if seed is not None:
            payload["seed"] = seed

        display_name = params.get("display_name", "").strip()
        if display_name:
            payload["display_name"] = display_name[:64]  # Max 64 chars

        tags = params.get("tags", "").strip()
        if tags:
            payload["tags"] = [tag.strip() for tag in tags.split(",") if tag.strip()]

        return payload

    async def _build_world_prompt(self, params: dict[str, Any], input_type: str) -> dict[str, Any]:
        """Build the discriminated union world_prompt object."""
        if input_type == "Text":
            return await self._build_text_prompt(params)
        elif input_type == "Image":
            return await self._build_image_prompt(params)
        elif input_type == "Multi-Image":
            return await self._build_multi_image_prompt(params)
        elif input_type == "Video":
            return await self._build_video_prompt(params)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")

    async def _build_text_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build text prompt payload."""
        text_prompt = params.get("text_prompt", "").strip()
        if not text_prompt:
            raise ValueError("Text prompt is required for Text input type")

        prompt = {
            "type": "text",
            "text_prompt": text_prompt,
        }

        if params.get("disable_recaption"):
            prompt["disable_recaption"] = True

        return prompt

    async def _build_image_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build image prompt payload."""
        image = params.get("image")
        if not image:
            raise ValueError("Image is required for Image input type")

        # Convert image to base64 data reference
        image_content = await self._media_to_base64_content(image)

        prompt = {
            "type": "image",
            "image_prompt": image_content,
            "is_pano": params.get("is_panorama", False),
        }

        text_prompt = params.get("text_prompt", "").strip()
        if text_prompt:
            prompt["text_prompt"] = text_prompt

        if params.get("disable_recaption"):
            prompt["disable_recaption"] = True

        return prompt

    async def _build_multi_image_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build multi-image prompt payload."""
        images_list = params.get("images", [])
        if not isinstance(images_list, list):
            images_list = [images_list] if images_list else []

        if len(images_list) < 2:
            raise ValueError("At least 2 images required for Multi-Image input type")

        enable_reconstruction = params.get("enable_reconstruction", False)
        max_images = MAX_MULTI_IMAGES if enable_reconstruction else DEFAULT_MULTI_IMAGES

        if len(images_list) > max_images:
            raise ValueError(
                f"Maximum {max_images} images allowed ({'with' if enable_reconstruction else 'without'} reconstruction mode)"
            )

        # Parse azimuth angles if provided
        azimuth_angles_str = params.get("azimuth_angles", "").strip()
        azimuth_angles = []
        if azimuth_angles_str:
            try:
                azimuth_angles = [float(a.strip()) for a in azimuth_angles_str.split(",")]
            except ValueError:
                raise ValueError(f"Invalid azimuth angles format: {azimuth_angles_str}")

        # Build multi_image_prompt array
        multi_image_prompt = []
        for i, image in enumerate(images_list[:max_images]):
            image_content = await self._media_to_base64_content(image)
            item = {"content": image_content}

            # Add azimuth if provided for this index
            if i < len(azimuth_angles):
                angle = azimuth_angles[i]
                if angle < 0 or angle > 360:
                    raise ValueError(f"Azimuth angle must be between 0 and 360 degrees: {angle}")
                item["azimuth"] = angle

            multi_image_prompt.append(item)

        prompt = {
            "type": "multi-image",
            "multi_image_prompt": multi_image_prompt,
            "reconstruct_images": enable_reconstruction,
        }

        text_prompt = params.get("text_prompt", "").strip()
        if text_prompt:
            prompt["text_prompt"] = text_prompt

        if params.get("disable_recaption"):
            prompt["disable_recaption"] = True

        return prompt

    async def _build_video_prompt(self, params: dict[str, Any]) -> dict[str, Any]:
        """Build video prompt payload."""
        video = params.get("video")
        if not video:
            raise ValueError("Video is required for Video input type")

        # Convert video to base64 data reference
        video_content = await self._media_to_base64_content(video)

        prompt = {
            "type": "video",
            "video_prompt": video_content,
        }

        text_prompt = params.get("text_prompt", "").strip()
        if text_prompt:
            prompt["text_prompt"] = text_prompt

        if params.get("disable_recaption"):
            prompt["disable_recaption"] = True

        return prompt

    async def _media_to_base64_content(self, media_input: Any) -> dict[str, Any]:
        """Convert media artifact to base64 content reference for API."""
        # Get raw bytes
        media_bytes = await self._get_media_bytes(media_input)
        if not media_bytes:
            raise ValueError("Failed to get media bytes")

        # Check size (10MB limit for base64)
        max_size_mb = 10
        if len(media_bytes) > max_size_mb * 1024 * 1024:
            raise ValueError(f"Media file size exceeds {max_size_mb}MB limit for inline data")

        # Detect extension
        extension = await asyncio.to_thread(self._detect_media_extension, media_bytes)

        # Encode to base64
        b64_data = base64.b64encode(media_bytes).decode("utf-8")

        return {
            "source": "data_base64",
            "data_base64": b64_data,
            "extension": extension,
        }

    async def _get_media_bytes(self, media_input: Any) -> bytes | None:
        """Get raw bytes from a media input (image or video)."""
        if not media_input:
            return None

        # Handle artifacts with to_bytes() method
        if hasattr(media_input, "to_bytes"):
            try:
                return media_input.to_bytes()
            except Exception as e:
                self._log(f"Failed to get bytes from artifact: {e}")
                return None

        # Extract string value from various input types
        media_value: str | None = None

        if isinstance(media_input, str):
            media_value = media_input
        elif hasattr(media_input, "value"):
            value = getattr(media_input, "value", None)
            if isinstance(value, str):
                media_value = value

        # Convert string value to bytes
        if media_value:
            return await self._string_to_bytes(media_value)

        return None

    @staticmethod
    def _detect_media_extension(media_bytes: bytes) -> str:
        """Detect media file extension from bytes."""
        try:
            # Try as image first
            with Image.open(BytesIO(media_bytes)) as img:
                image_format = (img.format or "").upper()
                format_map = {
                    "JPEG": "jpg",
                    "JPG": "jpg",
                    "PNG": "png",
                    "WEBP": "webp",
                }
                return format_map.get(image_format, "jpg")
        except Exception:
            # If not an image, check video magic bytes
            if media_bytes[:4] == b"ftyp" or media_bytes[4:8] == b"ftyp":
                return "mp4"
            elif media_bytes[:4] == b"RIFF":
                return "avi"
            elif media_bytes[:3] == b"\x1a\x45\xdf":
                return "mkv"
            # Default to mp4 for videos
            return "mp4"

    async def _string_to_bytes(self, value: str) -> bytes | None:
        """Convert a string (URL, data URI, file path, or base64) to raw bytes."""
        try:
            return await File(value).aread_bytes()
        except FileLoadError as e:
            self._log(f"Failed to load bytes from {value}: {e}")
            return None

    async def _parse_result(self, result_json: dict[str, Any], _generation_id: str) -> None:
        """Parse the World object and populate outputs."""
        try:
            await self._handle_success(result_json)
        except Exception as e:
            self._log(f"Error parsing result: {e}")
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details=f"Failed to parse generation result: {e}",
            )

    async def _handle_success(self, world: dict[str, Any]) -> None:
        """Handle successful world generation result."""
        # Store provider response
        self.parameter_output_values["provider_response"] = world

        # Extract world ID and viewer URL
        world_id = world.get("world_id")
        viewer_url = world.get("world_marble_url")

        if not world_id or not viewer_url:
            self._set_safe_defaults()
            self._set_status_results(
                was_successful=False,
                result_details="World generation completed but world_id or viewer_url is missing",
            )
            return

        self.parameter_output_values["world_id"] = world_id
        self.parameter_output_values["viewer_url"] = viewer_url

        # Parse assets
        assets = world.get("assets") or {}
        await self._parse_assets(assets, world_id)

        # Extract caption
        caption = assets.get("caption", "")
        if caption:
            self.parameter_output_values["caption"] = caption

        self._set_status_results(
            was_successful=True,
            result_details=f"Successfully generated 3D world: {world_id}",
        )

    async def _parse_assets(self, assets: dict[str, Any], world_id: str) -> None:
        """Parse and create artifact outputs from World assets."""
        # Parse splat files
        splats = assets.get("splats") or {}
        spz_urls = splats.get("spz_urls") or {}

        if "100k" in spz_urls:
            self.parameter_output_values["splat_100k"] = SplatUrlArtifact(
                value=spz_urls["100k"], meta={"resolution": "100k", "world_id": world_id}
            )

        if "500k" in spz_urls:
            self.parameter_output_values["splat_500k"] = SplatUrlArtifact(
                value=spz_urls["500k"], meta={"resolution": "500k", "world_id": world_id}
            )

        if "full_res" in spz_urls:
            self.parameter_output_values["splat_full_res"] = SplatUrlArtifact(
                value=spz_urls["full_res"], meta={"resolution": "full_res", "world_id": world_id}
            )

        # Parse mesh
        mesh = assets.get("mesh") or {}
        if mesh.get("collider_mesh_url"):
            self.parameter_output_values["mesh"] = ThreeDUrlArtifact(
                value=mesh["collider_mesh_url"], meta={"format": "glb", "type": "collider_mesh", "world_id": world_id}
            )

        # Parse imagery
        imagery = assets.get("imagery") or {}
        if imagery.get("pano_url"):
            self.parameter_output_values["panorama"] = ImageUrl(
                value=imagery["pano_url"], meta={"type": "panorama", "world_id": world_id}
            )

        # Parse thumbnail
        if assets.get("thumbnail_url"):
            self.parameter_output_values["thumbnail"] = ImageUrl(
                value=assets["thumbnail_url"], meta={"type": "thumbnail", "world_id": world_id}
            )

    def _set_safe_defaults(self) -> None:
        """Clear output parameters on failure."""
        self.parameter_output_values["world_id"] = ""
        self.parameter_output_values["viewer_url"] = ""
        self.parameter_output_values["splat_100k"] = None
        self.parameter_output_values["splat_500k"] = None
        self.parameter_output_values["splat_full_res"] = None
        self.parameter_output_values["mesh"] = None
        self.parameter_output_values["panorama"] = None
        self.parameter_output_values["thumbnail"] = None
        self.parameter_output_values["caption"] = ""
        self.parameter_output_values["provider_response"] = {}
