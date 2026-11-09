from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File
from griptape_nodes.image import GriptapeImage, GriptapeImagePipeline
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.color_picker import ColorPicker
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.image.base_image_processor import BaseImageProcessor
from griptape_nodes_library.utils.color_utils import parse_color_to_rgba
from griptape_nodes_library.utils.image_utils import ImageMode, converted

ANGLE_MIN = -360.0
ANGLE_MAX = 360.0
OPAQUE = 255
RGB_CHANNELS = 3
CUSTOM_ONLY_PARAMETERS = ["angle", "expand_canvas", "background_color"]
PREVIEW_PARAMETERS = {"rotation", *CUSTOM_ONLY_PARAMETERS}
DEFAULT_BACKGROUND = "#00000000"


class Rotation(StrEnum):
    CLOCKWISE_90 = "90° clockwise"
    ROTATE_180 = "180°"
    COUNTER_CLOCKWISE_90 = "90° counter-clockwise"
    CUSTOM = "custom angle"


# Counter-clockwise degrees for each quarter turn
QUARTER_TURNS = {
    Rotation.CLOCKWISE_90: 270,
    Rotation.ROTATE_180: 180,
    Rotation.COUNTER_CLOCKWISE_90: 90,
}

ROTATION_ICONS = {
    Rotation.CLOCKWISE_90: "rotate-cw",
    Rotation.ROTATE_180: "refresh-cw",
    Rotation.COUNTER_CLOCKWISE_90: "rotate-ccw",
    Rotation.CUSTOM: "sliders-horizontal",
}


class RotateImage(BaseImageProcessor):
    """Rotate an image by a quarter turn or a custom angle."""

    def _setup_custom_parameters(self) -> None:
        """Setup rotation-specific parameters."""
        # The preview widget shows the image, so the plain input row would be a second copy of it.
        input_image = self.get_parameter_by_name("input_image")
        if input_image is not None:
            input_image.hide_property = True

        self.add_parameter(
            ParameterDict(
                name="rotation_preview",
                default_value={"image_url": ""},
                tooltip="Preview of the rotation. The image is only rotated when the node runs.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="RotateImagePreview", library="Griptape Nodes Library")},
            )
        )

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
                    default_value=DEFAULT_BACKGROUND,
                    placeholder_text=DEFAULT_BACKGROUND,
                    tooltip="Color for the empty corners. Transparent by default; "
                    "formats without transparency, like JPEG, fill it with white.",
                    traits={ColorPicker(format="hexa")},
                    hide=True,
                )
            )

        self.add_node_element(rotate_group)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Show the rotation in the preview widget. The image itself is only rotated when the node runs."""
        if parameter.name == "rotation":
            if value == Rotation.CUSTOM:
                self.show_parameter_by_name(CUSTOM_ONLY_PARAMETERS)
            else:
                self.hide_parameter_by_name(CUSTOM_ONLY_PARAMETERS)

        if parameter.name == "input_image" or parameter.name in PREVIEW_PARAMETERS:
            self._update_preview(refresh_image=parameter.name == "input_image")
        return super().after_value_set(parameter, value)

    def _update_preview(self, *, refresh_image: bool) -> None:
        """Push the input image and current rotation settings to the preview widget."""
        existing = self.get_parameter_value("rotation_preview") or {}
        if refresh_image:
            image_url = self._resolve_image_url(self.get_parameter_value("input_image"))
        else:
            image_url = existing.get("image_url", "")

        preview = {
            "image_url": image_url,
            "rotation": self.get_parameter_value("rotation") or Rotation.CLOCKWISE_90,
            "angle": self.get_parameter_value("angle") or 0.0,
            "expand_canvas": self.get_parameter_value("expand_canvas") is not False,
            "background_color": self.get_parameter_value("background_color") or DEFAULT_BACKGROUND,
        }
        if preview == existing:
            return
        self.set_parameter_value("rotation_preview", preview)
        self.publish_update_to_parameter("rotation_preview", preview)

    def _resolve_image_url(self, image: Any) -> str:
        """Return a browser-accessible URL for an image artifact or its dict form."""
        raw = image.get("value") if isinstance(image, dict) else getattr(image, "value", image)
        if not raw or not isinstance(raw, str):
            return ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw
        try:
            resolved = File(raw).resolve()
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
        except Exception as e:
            logger.warning(f"{self.name}: could not build a preview URL for {raw!r}: {e}")
            return raw
        if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            return result.url
        return raw

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "image rotation"

    def _process_image(self, image: GriptapeImage, **kwargs) -> GriptapeImage:
        """Process the image by rotating it."""
        rotation = kwargs.get("rotation") or Rotation.CLOCKWISE_90

        logger.debug(f"{self.name}: Rotating image with rotation={rotation}")

        # Rotate the image as it is displayed; this also resets its EXIF orientation tag to upright.
        image = GriptapeImagePipeline(image).exif_transpose().end()

        match rotation:
            case Rotation.CLOCKWISE_90 | Rotation.ROTATE_180 | Rotation.COUNTER_CLOCKWISE_90:
                # Quarter turns with an expanded canvas are lossless
                return GriptapeImagePipeline(image).rotate(QUARTER_TURNS[Rotation(rotation)], expand=True).end()
            case Rotation.CUSTOM:
                return self._rotate_by_angle(
                    image,
                    angle=kwargs.get("angle") or 0.0,
                    expand_canvas=kwargs.get("expand_canvas", True),
                    background_color=kwargs.get("background_color") or "#00000000",
                )
            case _:
                msg = f"Unknown rotation: {rotation!r}"
                raise ValueError(msg)

    def _rotate_by_angle(
        self, image: GriptapeImage, angle: float, *, expand_canvas: bool, background_color: str
    ) -> GriptapeImage:
        """Rotate clockwise by an arbitrary angle, filling exposed corners with the background color."""
        fill = parse_color_to_rgba(background_color)

        if image.channels == RGB_CHANNELS and fill[3] == OPAQUE:
            fill_color: tuple[int, ...] = fill[:3]
        else:
            image = converted(image, ImageMode.RGBA)
            fill_color = fill

        # The pipeline rotates counter-clockwise for positive angles.
        return GriptapeImagePipeline(image).rotate(-angle, expand=expand_canvas, fill=fill_color).end()

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
