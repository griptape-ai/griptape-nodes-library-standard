import io
from dataclasses import dataclass
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterGroup,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget
from PIL import Image

from griptape_nodes_library.utils.color_utils import NAMED_COLORS, parse_color_to_rgba
from griptape_nodes_library.utils.file_utils import generate_filename
from griptape_nodes_library.utils.image_utils import (
    load_pil_from_url,
    validate_pil_format,
)

# Constants for magic numbers
NO_ZOOM = 100.0
MAX_ZOOM = 500.0
MIN_ZOOM_FACTOR = 0.1
MAX_ZOOM_FACTOR = 10.0  # Maximum zoom factor to prevent memory issues
MAX_IMAGE_DIMENSION = 32767  # Maximum safe dimension to prevent overflow
# Initial slider max values - should support 8K (8192x4320) and higher resolutions
# These are updated dynamically when an image is loaded, but need to be high enough
# to support common high-res formats before image loading
MAX_WIDTH = MAX_IMAGE_DIMENSION
MAX_HEIGHT = MAX_IMAGE_DIMENSION
ROTATION_MIN = -180.0
ROTATION_MAX = 180.0

# Parameter defaults
DEFAULT_LEFT = 0
DEFAULT_TOP = 0
DEFAULT_WIDTH = 0  # 0 = use full image width
DEFAULT_HEIGHT = 0  # 0 = use full image height
DEFAULT_ZOOM = NO_ZOOM
DEFAULT_ROTATE = 0.0

# Parameters the crop editor widget reads and writes directly
CROP_PARAMETERS = ["left", "top", "width", "height", "zoom", "rotate"]


@dataclass
class CropArea:
    """Represents a crop area with coordinates and dimensions."""

    left: int
    top: int
    right: int
    bottom: int
    center_x: float
    center_y: float

    @property
    def width(self) -> int:
        """Get the width of the crop area."""
        return self.right - self.left

    @property
    def height(self) -> int:
        """Get the height of the crop area."""
        return self.bottom - self.top


class CropImage(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.MAX_WIDTH = MAX_WIDTH
        self.MAX_HEIGHT = MAX_HEIGHT

        self.add_parameter(
            ParameterImage(
                name="input_image",
                default_value=None,
                tooltip="Input image to crop",
                hide_property=True,
            )
        )

        # Interactive crop editor widget — the primary UI for setting crop coordinates. It reads and
        # writes the crop parameters directly, so its own value holds only the image it shows.
        self.add_parameter(
            ParameterDict(
                name="crop_editor",
                default_value={"image_url": "", "img_width": 0, "img_height": 0},
                tooltip="Interactive crop editor — drag to set crop area",
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Widget(
                        name="CropImageEditor",
                        library="Griptape Nodes Library",
                        synced_parameters=CROP_PARAMETERS,
                    )
                },
            )
        )

        with ParameterGroup(name="crop_coordinates", ui_options={"collapsed": True}) as crop_coordinates:
            ParameterInt(
                name="left",
                default_value=DEFAULT_LEFT,
                tooltip="Left edge of crop area in pixels",
                traits={Slider(min_val=0, max_val=self.MAX_WIDTH, soft_limits=True)},
            )

            ParameterInt(
                name="top",
                default_value=DEFAULT_TOP,
                tooltip="Top edge of crop area in pixels",
                traits={Slider(min_val=0, max_val=self.MAX_HEIGHT, soft_limits=True)},
            )

            ParameterInt(
                name="width",
                default_value=DEFAULT_WIDTH,
                tooltip="Width of crop area in pixels (0 = use full width)",
                traits={Slider(min_val=0, max_val=self.MAX_WIDTH, soft_limits=True)},
            )

            ParameterInt(
                name="height",
                default_value=DEFAULT_HEIGHT,
                tooltip="Height of crop area in pixels (0 = use full height)",
                traits={Slider(min_val=0, max_val=self.MAX_HEIGHT, soft_limits=True)},
            )
        self.add_node_element(crop_coordinates)

        with ParameterGroup(name="transform_options", ui_options={"collapsed": True}) as transform_options:
            ParameterFloat(
                name="zoom",
                default_value=DEFAULT_ZOOM,
                tooltip="Zoom percentage (100 = no zoom, 200 = 2x zoom in, 50 = 0.5x zoom out)",
                traits={Slider(min_val=0.0, max_val=MAX_ZOOM)},
            )
            ParameterFloat(
                name="rotate",
                default_value=DEFAULT_ROTATE,
                tooltip="Rotation in degrees (-180 to 180)",
                traits={Slider(min_val=ROTATION_MIN, max_val=ROTATION_MAX)},
            )

        self.add_node_element(transform_options)
        with ParameterGroup(name="output_options", ui_options={"collapsed": True}) as output_options:
            ParameterString(
                name="background_color",
                default_value="#00000000",
                placeholder_text="#00000000",
                tooltip="Background color (RGBA or hex) for transparent areas",
                traits={ColorPicker(format="hexa")},
            )
            ParameterString(
                name="output_format",
                default_value="PNG",
                tooltip="Output format: PNG, JPEG, WEBP",
                traits={Options(choices=["PNG", "JPEG", "WEBP"])},
            )

            ParameterFloat(
                name="output_quality",
                default_value=0.9,
                tooltip="Output quality (0.0 to 1.0) for lossy formats",
            )

        self.add_node_element(output_options)

        # Output parameter
        self.add_parameter(
            ParameterImage(
                name="output",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Cropped output image",
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="crop.png",
        )
        self._output_file.add_parameter()

    # ── Image URL resolution ───────────────────────────────────────────────────

    def _resolve_image_url(self, artifact: Any) -> str:
        """Return a browser-accessible URL for an image artifact."""
        from griptape_nodes.retained_mode.events.static_file_events import (
            CreateStaticFileDownloadUrlFromPathRequest,
            CreateStaticFileDownloadUrlFromPathResultSuccess,
        )

        raw = artifact if isinstance(artifact, str) else (getattr(artifact, "value", "") or "")
        if not raw:
            return ""
        if isinstance(raw, str) and raw.startswith(("http://", "https://", "data:")):
            return raw
        try:
            resolved = File(raw).resolve()
        except Exception:
            resolved = str(raw)
        try:
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception as e:
            logger.warning("%s: could not resolve static file URL for %r: %s", self.name, resolved, e)
        return raw

    # ── Widget ─────────────────────────────────────────────────────────────────

    def _update_widget_image(self, image_url: str, img_width: int, img_height: int) -> None:
        """Give the crop editor the image to show.

        The crop coordinates reach the widget from their own parameters, listed in synced_parameters,
        and the editor marks the ones with an incoming connection as locked.
        """
        new_dict = {"image_url": image_url, "img_width": img_width, "img_height": img_height}
        self.set_parameter_value("crop_editor", new_dict)
        self.publish_update_to_parameter("crop_editor", new_dict)

    # ── Input normalization ────────────────────────────────────────────────────

    def _extract_image_path(self, value: Any) -> str | None:
        """Extract string path/URL from str, ImageUrlArtifact, or similar inputs."""
        if isinstance(value, str):
            return value
        try:
            if hasattr(value, "value"):
                v = getattr(value, "value", None)
                if isinstance(v, str):
                    return v
        except Exception:
            pass
        return None

    # ── Crop logic ─────────────────────────────────────────────────────────────

    def _crop(self) -> None:
        # Get parameters
        params = self._get_crop_parameters()

        path = self._extract_image_path(params["input_artifact"])
        if not path:
            msg = "Connect an image to 'Input Image'."
            raise ValueError(msg)

        # Load image
        try:
            img = load_pil_from_url(path)
        except Exception as e:
            msg = f"Could not load the image connected to 'Input Image': {e}"
            raise ValueError(msg) from e

        # Calculate and apply crop area
        crop_area = self._calculate_crop_area(params, img.size)
        img = self._apply_crop_transformations(img, crop_area, params)

        # Save result
        self._save_cropped_image(img, params)

    def _get_crop_parameters(self) -> dict:
        """Get all crop parameters."""
        return {
            "input_artifact": self.get_parameter_value("input_image"),
            "left": self.get_parameter_value("left"),
            "top": self.get_parameter_value("top"),
            "width": self.get_parameter_value("width"),
            "height": self.get_parameter_value("height"),
            "zoom": self.get_parameter_value("zoom"),
            "rotate": self.get_parameter_value("rotate"),
            "background_color": self.get_parameter_value("background_color"),
            "output_format": self.get_parameter_value("output_format"),
            "output_quality": self.get_parameter_value("output_quality"),
        }

    def _calculate_crop_area(self, params: dict, img_size: tuple[int, int]) -> CropArea:
        """Calculate the crop area with validation."""
        img_width, img_height = img_size
        left, top, width, height = params["left"], params["top"], params["width"], params["height"]

        # Calculate crop coordinates relative to original image
        crop_left = left
        crop_top = top
        crop_width = width if width > 0 else img_width
        crop_height = height if height > 0 else img_height

        # A negative origin keeps the box's far edge, so only the part on the image is cropped
        if width > 0 and crop_left < 0:
            crop_width += crop_left
        if height > 0 and crop_top < 0:
            crop_height += crop_top

        # Ensure crop coordinates are within image bounds; an origin past the edge becomes a 1px edge strip
        crop_left = max(0, min(crop_left, img_width - 1))
        crop_top = max(0, min(crop_top, img_height - 1))

        # Ensure crop dimensions are valid; a box entirely off the image becomes a 1px edge strip
        if width <= 0 or crop_left + crop_width > img_width:
            crop_width = img_width - crop_left
        if height <= 0 or crop_top + crop_height > img_height:
            crop_height = img_height - crop_top
        crop_width = max(crop_width, 1)
        crop_height = max(crop_height, 1)

        # A width/height of 0 is the "use full size" sentinel, so only other mismatches count as adjustments
        width_adjusted = width != 0 and crop_width != width
        height_adjusted = height != 0 and crop_height != height
        if crop_left != left or crop_top != top or width_adjusted or height_adjusted:
            logger.info(
                "%s: crop box (left=%s, top=%s, width=%s, height=%s) adjusted to (%s, %s, %s, %s) for %sx%s image",
                self.name,
                left,
                top,
                width,
                height,
                crop_left,
                crop_top,
                crop_width,
                crop_height,
                img_width,
                img_height,
            )

        # Calculate final crop boundaries
        crop_right = crop_left + crop_width
        crop_bottom = crop_top + crop_height

        # Calculate the center of the crop area
        crop_center_x = (crop_left + crop_right) / 2
        crop_center_y = (crop_top + crop_bottom) / 2

        return CropArea(crop_left, crop_top, crop_right, crop_bottom, crop_center_x, crop_center_y)

    def _apply_crop_transformations(self, img: Image.Image, crop_area: CropArea, params: dict) -> Image.Image:
        """Apply zoom, rotation, and final crop to the image."""
        img_width, img_height = img.size

        # Apply zoom by scaling the crop area
        crop_area = self._apply_zoom_to_crop_area(crop_area, params["zoom"], img_width, img_height)

        # Apply rotation around the center of the crop area
        img = self._apply_rotation_to_image(
            img, params["rotate"], crop_area.center_x, crop_area.center_y, params["background_color"]
        )

        # Apply the final crop (the window)
        img = self._apply_final_crop(img, crop_area.left, crop_area.top, crop_area.right, crop_area.bottom)

        return img

    def _save_cropped_image(self, img: Image.Image, params: dict) -> None:
        """Save the cropped image."""
        # Validate and prepare image for saving
        save_format = params["output_format"].upper()

        # Validate that the save format is supported by PIL
        validate_pil_format(save_format, "output_format")

        output_quality = max(0.0, min(1.0, params["output_quality"]))  # Clamp to 0.0-1.0

        # Convert RGBA to RGB for JPEG format (JPEG doesn't support transparency)
        if save_format == "JPEG" and img.mode == "RGBA":
            # Create black background for transparent areas
            background = Image.new("RGB", img.size, (0, 0, 0))
            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
            img = background

        # Determine save options
        save_options = {}
        if save_format == "JPEG":
            save_options["quality"] = int(output_quality * 100)
            save_options["optimize"] = True
        elif save_format == "WEBP":
            save_options["quality"] = int(output_quality * 100)
            save_options["lossless"] = True

        # Save result using context manager to ensure proper cleanup
        img_data = None
        with io.BytesIO() as img_byte_arr:
            img.save(img_byte_arr, format=save_format, **save_options)
            img_byte_arr.seek(0)
            img_data = img_byte_arr.getvalue()  # Returns a copy, data persists after context exit

        # Verify we have valid data before proceeding
        if img_data is None or len(img_data) == 0:
            msg = f"The cropped image could not be encoded as {save_format}."
            raise RuntimeError(msg)

        dest = self._output_file.build_file()
        saved = dest.write_bytes(img_data)
        self.parameter_output_values["output"] = ImageUrlArtifact(value=saved.location)

    def _get_output_suffix(self, **kwargs) -> str:  # noqa: ARG002
        """Get output filename suffix."""
        return "_crop"

    def _generate_filename(self, extension: str) -> str:
        """Generate a meaningful filename based on workflow and node information."""
        # Get processing suffix
        params = self._get_crop_parameters()
        processing_suffix = self._get_output_suffix(**params)

        # Use the general filename utility but with a custom prefix
        base_filename = generate_filename(
            node_name=self.name,
            suffix=processing_suffix,
            extension=extension,
        )

        # Add the "crop" prefix that this node specifically uses
        return base_filename.replace(f"{self.name}{processing_suffix}", f"crop_{self.name}{processing_suffix}")

    def _apply_zoom_to_crop_area(self, crop_area: CropArea, zoom: float, img_width: int, img_height: int) -> CropArea:
        """Apply zoom by scaling the crop area size."""
        if zoom == NO_ZOOM:
            return crop_area

        # Scale the crop area based on zoom
        scaled_area = self._scale_crop_area(crop_area, zoom)

        # Ensure the scaled area stays within image bounds
        bounded_area = self._clamp_crop_area_to_bounds(scaled_area, img_width, img_height)

        return bounded_area

    def _scale_crop_area(self, crop_area: CropArea, zoom: float) -> CropArea:
        """Scale the crop area size based on zoom factor."""
        zoom_factor = zoom / NO_ZOOM

        # Clamp zoom_factor to prevent division by zero and extreme scaling
        zoom_factor = max(MIN_ZOOM_FACTOR, min(MAX_ZOOM_FACTOR, zoom_factor))

        # Calculate new dimensions with overflow protection
        new_width = int(crop_area.width / zoom_factor)
        new_height = int(crop_area.height / zoom_factor)

        # Clamp dimensions to prevent integer overflow
        new_width = max(1, min(new_width, MAX_IMAGE_DIMENSION))
        new_height = max(1, min(new_height, MAX_IMAGE_DIMENSION))

        # Keep center position, adjust size
        new_left = int(crop_area.center_x - new_width / 2)
        new_top = int(crop_area.center_y - new_height / 2)
        new_right = new_left + new_width
        new_bottom = new_top + new_height

        return CropArea(new_left, new_top, new_right, new_bottom, crop_area.center_x, crop_area.center_y)

    def _clamp_crop_area_to_bounds(self, crop_area: CropArea, img_width: int, img_height: int) -> CropArea:
        """Ensure crop area coordinates are within image bounds."""
        clamped_left = max(0, min(crop_area.left, img_width))
        clamped_right = max(clamped_left, min(crop_area.right, img_width))
        clamped_top = max(0, min(crop_area.top, img_height))
        clamped_bottom = max(clamped_top, min(crop_area.bottom, img_height))

        return CropArea(
            clamped_left, clamped_top, clamped_right, clamped_bottom, crop_area.center_x, crop_area.center_y
        )

    def _apply_rotation_to_image(
        self,
        img: Image.Image,
        rotate: float,
        crop_center_x: float,
        crop_center_y: float,
        background_color: str,
    ) -> Image.Image:
        """Apply rotation around the crop center point."""
        if rotate == 0.0:
            return img

        # Convert background color to RGBA
        bg_color = self._parse_color(background_color)

        # Simply rotate around the crop center point
        img = img.rotate(rotate, center=(crop_center_x, crop_center_y), expand=False, fillcolor=bg_color)

        return img

    def _apply_final_crop(
        self, img: Image.Image, crop_left: int, crop_top: int, crop_right: int, crop_bottom: int
    ) -> Image.Image:
        """Apply the final crop to the image."""
        img_width, img_height = img.size

        # Ensure crop coordinates are within the final image bounds
        crop_left = max(0, min(crop_left, img_width))
        crop_right = max(crop_left, min(crop_right, img_width))
        crop_top = max(0, min(crop_top, img_height))
        crop_bottom = max(crop_top, min(crop_bottom, img_height))

        # Apply the final crop
        if crop_right > crop_left and crop_bottom > crop_top:
            try:
                img = img.crop((crop_left, crop_top, crop_right, crop_bottom))
            except Exception as e:
                msg = f"{self.name}: Final crop failed: {e}. Using image as is."
                logger.warning(msg)
        else:
            msg = f"{self.name}: Invalid final crop coordinates, using image as is"
            logger.warning(msg)

        return img

    def process(self) -> None:
        self._crop()

    def _parse_color(self, color_str: str) -> tuple[int, int, int, int]:
        """Parse color string to RGBA tuple."""
        try:
            return parse_color_to_rgba(color_str)
        except ValueError:
            # Fallback to transparent if color parsing fails
            return NAMED_COLORS["transparent"]

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # ── Input image changed: update sliders + push image URL to widget ───
        if parameter.name == "input_image":
            if not value:
                return super().after_value_set(parameter, value)
            path = self._extract_image_path(value)
            if not path:
                return super().after_value_set(parameter, value)
            try:
                img = load_pil_from_url(path)
            except Exception as e:
                logger.error("%s: Error loading image: %s", self.name, e)
                return super().after_value_set(parameter, value)

            if img.width and img.height:
                top_param = self.get_parameter_by_name("top")
                left_param = self.get_parameter_by_name("left")
                width_param = self.get_parameter_by_name("width")
                height_param = self.get_parameter_by_name("height")
                if top_param:
                    top_param.update_ui_options({"slider": {"max_val": img.height}})
                if left_param:
                    left_param.update_ui_options({"slider": {"max_val": img.width}})
                if width_param:
                    width_param.update_ui_options({"slider": {"max_val": img.width}})
                if height_param:
                    height_param.update_ui_options({"slider": {"max_val": img.height, "min_val": 0}})

                url = self._resolve_image_url(path)
                self._update_widget_image(url, img.width, img.height)

        return super().after_value_set(parameter, value)

    def _validate_image(self) -> list[Exception]:
        """Validate the input image parameter."""
        exceptions = []

        input_artifact = self.get_parameter_value("input_image")
        if not input_artifact:
            msg = "Connect an image to 'Input Image'."
            exceptions.append(Exception(msg))
            return exceptions

        if not self._extract_image_path(input_artifact):
            msg = "'Input Image' could not be resolved to a valid path. Connect a valid image to 'Input Image'."
            exceptions.append(Exception(msg))

        return exceptions

    def _validate_parameters(self) -> list[Exception]:
        """Validate parameter values."""
        exceptions = []

        # Validate zoom parameter
        zoom = self.get_parameter_value("zoom")
        if zoom is not None and (zoom < 0.0 or zoom > MAX_ZOOM):
            msg = f"Zoom must be between 0.0 and {MAX_ZOOM}, got {zoom}"
            exceptions.append(Exception(msg))

        # Validate rotation parameter
        rotate = self.get_parameter_value("rotate")
        if rotate is not None and (rotate < ROTATION_MIN or rotate > ROTATION_MAX):
            msg = f"Rotation must be between {ROTATION_MIN} and {ROTATION_MAX} degrees, got {rotate}"
            exceptions.append(Exception(msg))

        # Validate output quality parameter
        output_quality = self.get_parameter_value("output_quality")
        if output_quality is not None and (output_quality < 0.0 or output_quality > 1.0):
            msg = f"Output quality must be between 0.0 and 1.0, got {output_quality}"
            exceptions.append(Exception(msg))

        # Validate crop size is non-negative; a negative left/top is clamped to the image at run time
        width = self.get_parameter_value("width")
        height = self.get_parameter_value("height")

        if width is not None and width < 0:
            msg = f"Width must be non-negative, got {width}"
            exceptions.append(Exception(msg))

        if height is not None and height < 0:
            msg = f"Height must be non-negative, got {height}"
            exceptions.append(Exception(msg))

        return exceptions

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions = []

        # Validate image
        exceptions.extend(self._validate_image())

        # Validate parameters
        exceptions.extend(self._validate_parameters())

        return exceptions if exceptions else None
