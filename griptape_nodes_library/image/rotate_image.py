import io
import math
from enum import StrEnum
from typing import Any

import cv2
import numpy as np
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image, ImageOps

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.color_utils import parse_color_to_rgba
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_pil_from_url,
)

ANGLE_MIN = -360.0
ANGLE_MAX = 360.0
OPAQUE = 255
# Scales an 8-bit color channel to 16 bits: 255 * 257 == 65535.
EIGHT_TO_SIXTEEN_BIT = 257
EXIF_ORIENTATION = 0x0112
GRAYSCALE_DIMENSIONS = 2
BGR_CHANNELS = 3
# Formats that hold 16 bits per channel, mapped to the extension OpenCV encodes them by.
SIXTEEN_BIT_FORMATS = {"PNG": ".png", "TIFF": ".tiff"}
CUSTOM_ONLY_PARAMETERS = ["angle", "expand_canvas", "background_color"]
PREVIEW_PARAMETERS = {"rotation", *CUSTOM_ONLY_PARAMETERS}


class Rotation(StrEnum):
    CLOCKWISE_90 = "90° clockwise"
    ROTATE_180 = "180°"
    COUNTER_CLOCKWISE_90 = "90° counter-clockwise"
    CUSTOM = "custom angle"


QUARTER_TURNS = {
    Rotation.CLOCKWISE_90: Image.Transpose.ROTATE_270,
    Rotation.ROTATE_180: Image.Transpose.ROTATE_180,
    Rotation.COUNTER_CLOCKWISE_90: Image.Transpose.ROTATE_90,
}

# np.rot90 turns counter-clockwise for positive k.
QUARTER_TURNS_K = {
    Rotation.CLOCKWISE_90: -1,
    Rotation.ROTATE_180: 2,
    Rotation.COUNTER_CLOCKWISE_90: 1,
}

ROTATION_ICONS = {
    Rotation.CLOCKWISE_90: "rotate-cw",
    Rotation.ROTATE_180: "refresh-cw",
    Rotation.COUNTER_CLOCKWISE_90: "rotate-ccw",
    Rotation.CUSTOM: "sliders-horizontal",
}


def is_sixteen_bit(pil_image: Image.Image) -> bool:
    """Whether the file behind an unloaded PIL image stores 16 bits per channel.

    PIL decodes 16-bit RGB and RGBA to 8 bits, so this reads the raw mode it would decode from.
    """
    if (pil_image.format or "").upper() not in SIXTEEN_BIT_FORMATS:
        return False
    if pil_image.mode.startswith("I;16"):
        return True
    for tile in getattr(pil_image, "tile", []):
        rawmode = tile.args if isinstance(tile.args, str) else (tile.args or ("",))[0]
        if isinstance(rawmode, str) and ";16" in rawmode:
            return True
    return False


def exif_transpose_array(pixels: np.ndarray, orientation: int | None) -> np.ndarray:
    """Turn pixels to how an EXIF orientation tag says they are displayed, like ImageOps.exif_transpose."""
    match orientation:
        case None | 1:
            return pixels
        case 2:
            return pixels[:, ::-1]
        case 3:
            return np.rot90(pixels, 2)
        case 4:
            return pixels[::-1]
        case 5:
            return pixels.swapaxes(0, 1)
        case 6:
            return np.rot90(pixels, -1)
        case 7:
            return np.rot90(pixels, 2).swapaxes(0, 1)
        case 8:
            return np.rot90(pixels, 1)
        case _:
            return pixels


def expanded_size(width: int, height: int, angle: float) -> tuple[int, int]:
    """Canvas size that fits an image rotated by angle degrees, rounded the way PIL's rotate(expand=True) does."""
    if angle % 180 == 0:
        return width, height
    if angle % 90 == 0:
        return height, width
    radians = math.radians(angle)
    cos = round(math.cos(radians), 15)
    sin = round(math.sin(radians), 15)
    center_x, center_y = width / 2, height / 2
    xs, ys = [], []
    for x, y in ((0, 0), (width, 0), (width, height), (0, height)):
        xs.append(cos * (x - center_x) - sin * (y - center_y) + center_x)
        ys.append(sin * (x - center_x) + cos * (y - center_y) + center_y)
    return math.ceil(max(xs)) - math.floor(min(xs)), math.ceil(max(ys)) - math.floor(min(ys))


def rotate_array_by_angle(
    pixels: np.ndarray, angle: float, *, expand_canvas: bool, background_color: str
) -> np.ndarray:
    """Rotate 16-bit OpenCV (BGR) pixels clockwise by an arbitrary angle, filling exposed corners."""
    red, green, blue, alpha = (channel * EIGHT_TO_SIXTEEN_BIT for channel in parse_color_to_rgba(background_color))
    channels = 1 if pixels.ndim == GRAYSCALE_DIMENSIONS else pixels.shape[2]

    if channels == BGR_CHANNELS and alpha == OPAQUE * EIGHT_TO_SIXTEEN_BIT:
        border: tuple[int, ...] = (blue, green, red)
    else:
        match channels:
            case 1:
                pixels = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGRA)
            case 3:
                pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2BGRA)
            case 4:
                pass
            case _:
                msg = f"Can't rotate an image with {channels} channels."
                raise ValueError(msg)
        border = (blue, green, red, alpha)

    height, width = pixels.shape[:2]
    # Whole quarter turns that keep every pixel are done losslessly, as PIL does.
    if angle % 180 == 0 or (angle % 90 == 0 and (expand_canvas or width == height)):
        return np.ascontiguousarray(np.rot90(pixels, -round(angle / 90)))

    out_width, out_height = expanded_size(width, height, angle) if expand_canvas else (width, height)

    # Map source pixel centers onto the output, turning clockwise on screen (y points down).
    radians = math.radians(angle)
    cos, sin = math.cos(radians), math.sin(radians)
    src_x, src_y = (width - 1) / 2, (height - 1) / 2
    dst_x, dst_y = (out_width - 1) / 2, (out_height - 1) / 2
    matrix = np.array(
        [
            [cos, -sin, dst_x - cos * src_x + sin * src_y],
            [sin, cos, dst_y - sin * src_x - cos * src_y],
        ]
    )
    return cv2.warpAffine(
        pixels,
        matrix,
        (out_width, out_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def rotate_array(
    pixels: np.ndarray, rotation: str, *, angle: float, expand_canvas: bool, background_color: str
) -> np.ndarray:
    """Rotate 16-bit OpenCV pixels the same way RotateImage rotates PIL images."""
    match rotation:
        case Rotation.CLOCKWISE_90 | Rotation.ROTATE_180 | Rotation.COUNTER_CLOCKWISE_90:
            return np.ascontiguousarray(np.rot90(pixels, QUARTER_TURNS_K[Rotation(rotation)]))
        case Rotation.CUSTOM:
            return rotate_array_by_angle(
                np.ascontiguousarray(pixels),
                angle,
                expand_canvas=expand_canvas,
                background_color=background_color,
            )
        case _:
            msg = f"Unknown rotation: {rotation!r}"
            raise ValueError(msg)


def rotate_sixteen_bit_file(data: bytes, input_format: str, output_format: str, **kwargs) -> bytes:
    """Rotate an encoded 16-bit PNG or TIFF and encode the result as 16-bit output_format."""
    pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if pixels is None or pixels.dtype != np.uint16:
        msg = f"Couldn't read the {input_format} image as 16 bits per channel."
        raise ValueError(msg)

    # Rotate the image as it is displayed, since the saved output drops the EXIF orientation tag.
    # OpenCV already applies a TIFF's orientation when decoding it.
    if input_format.upper() != "TIFF":
        orientation = Image.open(io.BytesIO(data)).getexif().get(EXIF_ORIENTATION)
        pixels = exif_transpose_array(pixels, orientation)

    rotated = rotate_array(
        pixels,
        kwargs.get("rotation") or Rotation.CLOCKWISE_90,
        angle=kwargs.get("angle") or 0.0,
        expand_canvas=kwargs.get("expand_canvas", True),
        background_color=kwargs.get("background_color") or "#00000000",
    )

    ok, buffer = cv2.imencode(SIXTEEN_BIT_FORMATS[output_format.upper()], rotated)
    if not ok:
        msg = f"Couldn't save the rotated image as 16-bit {output_format}."
        raise ValueError(msg)
    return buffer.tobytes()


def image_url(image_value: Any) -> str:
    """URL of an image parameter value, which may be an artifact or its dict form."""
    if isinstance(image_value, dict):
        return dict_to_image_url_artifact(image_value).value
    if hasattr(image_value, "to_dict"):
        return dict_to_image_url_artifact(image_value.to_dict()).value
    return image_value.value


class RotateImage(BaseImageProcessor):
    """Rotate an image by a quarter turn or a custom angle."""

    def _setup_custom_parameters(self) -> None:
        """Setup rotation-specific parameters."""
        with ParameterGroup(name="rotate_settings", ui_options={"collapsed": False}) as rotate_group:
            rotation_parameter = ParameterString(
                name="rotation",
                default_value=Rotation.CLOCKWISE_90,
                tooltip="Quarter turns are lossless. Pick 'custom angle' to rotate by any number of degrees.",
                ui_options={
                    "data": [{"name": rotation.value, "icon": icon} for rotation, icon in ROTATION_ICONS.items()],
                    "dropdown_row_icons": True,
                },
            )
            rotation_parameter.add_trait(Options(choices=[r.value for r in Rotation]))
            self.add_parameter(rotation_parameter)

            self.add_parameter(
                ParameterFloat(
                    name="angle",
                    default_value=0.0,
                    tooltip="Degrees to rotate. Positive turns clockwise, negative counter-clockwise.",
                    traits={Slider(min_val=ANGLE_MIN, max_val=ANGLE_MAX)},
                    hide=True,
                )
            )
            self.add_parameter(
                ParameterBool(
                    name="expand_canvas",
                    default_value=True,
                    tooltip="Grow the canvas to fit the rotated image so no corners are cut off. "
                    "Off keeps the original size.",
                    hide=True,
                )
            )
            self.add_parameter(
                ParameterString(
                    name="background_color",
                    default_value="#00000000",
                    placeholder_text="#00000000",
                    tooltip="Color for the empty corners. Transparent by default; "
                    "formats without transparency, like JPEG, fill it with white.",
                    traits={ColorPicker(format="hexa")},
                    hide=True,
                )
            )

        self.add_node_element(rotate_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Process image automatically when connected or when rotation parameters change."""
        if parameter.name == "rotation":
            if value == Rotation.CUSTOM:
                self.show_parameter_by_name(CUSTOM_ONLY_PARAMETERS)
            else:
                self.hide_parameter_by_name(CUSTOM_ONLY_PARAMETERS)

        if parameter.name == "input_image" and value is not None:
            self._process_image_immediately(value)
        elif parameter.name in PREVIEW_PARAMETERS:
            input_image = self.get_parameter_value("input_image")
            if input_image is not None:
                self._process_image_immediately(input_image)
        return super().after_value_set(parameter, value)

    def _process_image_immediately(self, image_value: Any) -> None:
        """Process image immediately for live preview."""
        try:
            if isinstance(image_value, dict):
                image_artifact = dict_to_image_url_artifact(image_value)
            else:
                image_artifact = image_value

            pil_image = load_pil_from_url(image_artifact.value)
            custom_parameters = self._get_custom_parameters()
            if is_sixteen_bit(pil_image):
                output_artifact = self._rotate_sixteen_bit(pil_image, image_artifact.value, "PNG", **custom_parameters)
            else:
                processed_image = self._process_image(pil_image, **custom_parameters)
                output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _process(self, pil_image: Image.Image, detected_format: str, **kwargs) -> None:
        """Rotate 16-bit images with OpenCV, since PIL would reduce their color to 8 bits."""
        output_format = self._get_output_format(detected_format).upper()
        if not (is_sixteen_bit(pil_image) and output_format in SIXTEEN_BIT_FORMATS):
            super()._process(pil_image, detected_format, **kwargs)
            return

        self.append_value_to_parameter("logs", f"{self._get_processing_description()} at 16 bits per channel\n")
        url = image_url(self.parameter_values.get("input_image"))
        self.parameter_output_values["output"] = self._rotate_sixteen_bit(pil_image, url, output_format, **kwargs)

    def _rotate_sixteen_bit(self, pil_image: Image.Image, url: str, output_format: str, **kwargs) -> ImageUrlArtifact:
        """Rotate a 16-bit image without reducing its bit depth, and save it as 16-bit PNG or TIFF."""
        rotated = rotate_sixteen_bit_file(File(url).read_bytes(), pil_image.format or "", output_format, **kwargs)
        saved = self._output_file.build_file().write_bytes(rotated)
        return ImageUrlArtifact(saved.location)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image rotation"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL image by rotating it."""
        rotation = kwargs.get("rotation") or Rotation.CLOCKWISE_90

        logger.debug(f"{self.name}: Rotating image with rotation={rotation}")

        # Rotate the image as it is displayed, since the saved output drops the EXIF orientation tag.
        pil_image = ImageOps.exif_transpose(pil_image)

        match rotation:
            case Rotation.CLOCKWISE_90 | Rotation.ROTATE_180 | Rotation.COUNTER_CLOCKWISE_90:
                return pil_image.transpose(QUARTER_TURNS[Rotation(rotation)])
            case Rotation.CUSTOM:
                return self._rotate_by_angle(
                    pil_image,
                    angle=kwargs.get("angle") or 0.0,
                    expand_canvas=kwargs.get("expand_canvas", True),
                    background_color=kwargs.get("background_color") or "#00000000",
                )
            case _:
                msg = f"Unknown rotation: {rotation!r}"
                raise ValueError(msg)

    def _rotate_by_angle(
        self, pil_image: Image.Image, angle: float, *, expand_canvas: bool, background_color: str
    ) -> Image.Image:
        """Rotate clockwise by an arbitrary angle, filling exposed corners with the background color."""
        fill = parse_color_to_rgba(background_color)

        if pil_image.mode == "RGB" and fill[3] == OPAQUE:
            fill_color: tuple[int, ...] = fill[:3]
        else:
            if pil_image.mode != "RGBA":
                pil_image = pil_image.convert("RGBA")
            fill_color = fill

        # PIL rotates counter-clockwise for positive angles.
        return pil_image.rotate(-angle, resample=Image.Resampling.BICUBIC, expand=expand_canvas, fillcolor=fill_color)

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate rotation parameters."""
        exceptions = []

        rotation = self.get_parameter_value("rotation")
        valid_choices = [r.value for r in Rotation]
        if rotation is not None and rotation not in valid_choices:
            msg = f"'Rotation' must be one of {valid_choices}, got {rotation!r}."
            exceptions.append(ValueError(msg))

        if rotation == Rotation.CUSTOM:
            background_color = self.get_parameter_value("background_color")
            if background_color:
                try:
                    parse_color_to_rgba(background_color)
                except ValueError:
                    msg = f"'Background Color' {background_color!r} isn't a color. Use a hex value like #00000000."
                    exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get rotation parameters."""
        return {
            "rotation": self.get_parameter_value("rotation"),
            "angle": self.get_parameter_value("angle"),
            "expand_canvas": self.get_parameter_value("expand_canvas"),
            "background_color": self.get_parameter_value("background_color"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        rotation = kwargs.get("rotation") or Rotation.CLOCKWISE_90
        if rotation == Rotation.CUSTOM:
            return f"_rotated_{kwargs.get('angle') or 0.0:g}"
        return f"_rotated_{Rotation(rotation).name.lower()}"
