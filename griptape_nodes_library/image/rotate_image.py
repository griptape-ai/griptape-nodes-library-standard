from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import logger
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from PIL import Image

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.color_utils import parse_color_to_rgba
from griptape_nodes_library.utils.image_utils import (
    dict_to_image_url_artifact,
    load_pil_from_url,
)

ANGLE_MIN = -360.0
ANGLE_MAX = 360.0
OPAQUE = 255
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


class RotateImage(BaseImageProcessor):
    """Rotate an image by a quarter turn or a custom angle."""

    def _setup_custom_parameters(self) -> None:
        """Setup rotation-specific parameters."""
        with ParameterGroup(name="rotate_settings", ui_options={"collapsed": False}) as rotate_group:
            rotation_parameter = ParameterString(
                name="rotation",
                default_value=Rotation.CLOCKWISE_90,
                tooltip="Quarter turns are lossless. Pick 'custom angle' to rotate by any number of degrees.",
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
            processed_image = self._process_image(pil_image, **self._get_custom_parameters())
            output_artifact = self._save_image_artifact(processed_image, "png")

            self.set_parameter_value("output", output_artifact)
            self.publish_update_to_parameter("output", output_artifact)

        except Exception as e:
            logger.warning(f"{self.name}: Live preview failed: {e}")

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image rotation"

    def _process_image(self, pil_image: Image.Image, **kwargs) -> Image.Image:
        """Process the PIL image by rotating it."""
        rotation = kwargs.get("rotation") or Rotation.CLOCKWISE_90

        logger.debug(f"{self.name}: Rotating image with rotation={rotation}")

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
