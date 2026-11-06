from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.elements.badge import BadgeData
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.image_utils import extract_image_url, load_pil_from_url

DEFAULT_CANVAS_SIZE = 1000
MIN_CANVAS_SIZE = 1
MAX_CANVAS_SIZE = 65_535  # Largest side common image formats (e.g. JPEG) can encode.

ARRAY_ORDER_PRESETS = [
    "x1, y1, x2, y2",
    "x, y, width, height",
    "y1, x1, y2, x2",
    "y, x, height, width",
]
DICT_ORDER_PRESETS = [
    "x, y, width, height",
    "x1, y1, x2, y2",
    "left, top, right, bottom",
    "x, y, w, h",
]
DEFAULT_ARRAY_ORDER = ARRAY_ORDER_PRESETS[0]
DEFAULT_DICT_ORDER = DICT_ORDER_PRESETS[0]

# Accepted order tokens mapped to the coordinate they stand for.
TOKEN_ALIASES = {
    "x1": "x1",
    "y1": "y1",
    "x2": "x2",
    "y2": "y2",
    "x": "x1",
    "y": "y1",
    "left": "x1",
    "top": "y1",
    "right": "x2",
    "bottom": "y2",
    "width": "width",
    "height": "height",
    "w": "width",
    "h": "height",
}
TOKENS_HELP = (
    "Order fields take a comma-separated list of these tokens: "
    "x1, y1, x2, y2 (corners), x, y (top-left corner, same as x1, y1), width, height (also w, h), "
    "left, top, right, bottom (same as x1, y1, x2, y2). "
    "Example: x1, y1, x2, y2. Array values follow the order given; dict keys use the tokens as typed."
)


class ParamName(StrEnum):
    INPUT_IMAGE = "input_image"
    USE_IMAGE_SIZE = "use_image_size"
    CANVAS_WIDTH = "canvas_width"
    CANVAS_HEIGHT = "canvas_height"
    LEFT = "left"
    TOP = "top"
    WIDTH = "width"
    HEIGHT = "height"
    ARRAY_ORDER = "array_order"
    DICT_ORDER = "dict_order"


BOX_PARAM_NAMES = (ParamName.LEFT, ParamName.TOP, ParamName.WIDTH, ParamName.HEIGHT)
PASS_THROUGH_PARAM_NAMES = (
    ParamName.INPUT_IMAGE,
    ParamName.USE_IMAGE_SIZE,
    ParamName.CANVAS_WIDTH,
    ParamName.CANVAS_HEIGHT,
    *BOX_PARAM_NAMES,
)
# Parameters the bounding box editor widget reads and writes directly
EDITOR_PARAMETERS = [name.value for name in (*BOX_PARAM_NAMES, ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT)]


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def as_params(self) -> dict[str, int]:
        """The box in the form of the node's coordinate parameters."""
        return {"left": self.x1, "top": self.y1, "width": self.width, "height": self.height}


@dataclass(frozen=True)
class CanvasSize:
    width: int
    height: int


DEFAULT_BOX = BoundingBox(250, 250, 750, 750)


@dataclass(frozen=True)
class ImageInfo:
    path: str
    width: int
    height: int
    url: str


def parse_order(text: object, label: str) -> list[str]:
    """Split an order string into validated, lowercase tokens."""
    if not isinstance(text, str) or not text.strip():
        msg = f"Attempted to read the {label} order. Failed with an empty value because at least one token is required. {TOKENS_HELP}"
        raise ValueError(msg)

    tokens = [token.strip().lower() for token in text.split(",")]
    seen: set[str] = set()
    for token in tokens:
        if not token:
            msg = f"Attempted to read the {label} order '{text}'. Failed because it has an empty entry between commas. {TOKENS_HELP}"
            raise ValueError(msg)
        if token not in TOKEN_ALIASES:
            msg = f"Attempted to read the {label} order '{text}'. Failed because '{token}' is not a recognised token. {TOKENS_HELP}"
            raise ValueError(msg)
        if token in seen:
            msg = f"Attempted to read the {label} order '{text}'. Failed because '{token}' appears more than once."
            raise ValueError(msg)
        seen.add(token)
    return tokens


def token_value(box: BoundingBox, token: str) -> int:
    # Every TOKEN_ALIASES value is the name of a BoundingBox attribute.
    return getattr(box, TOKEN_ALIASES[token])


def build_array(box: BoundingBox, tokens: list[str]) -> list[int]:
    return [token_value(box, token) for token in tokens]


def build_dict(box: BoundingBox, tokens: list[str]) -> dict[str, int]:
    return {token: token_value(box, token) for token in tokens}


def box_problems(box: BoundingBox, canvas: CanvasSize) -> list[str]:
    """Describe why a box is unusable on the canvas; an empty list means it is valid."""
    problems = []
    if box.width <= 0:
        problems.append(f"width ({box.width}) must be greater than zero")
    if box.height <= 0:
        problems.append(f"height ({box.height}) must be greater than zero")
    if box.x1 < 0 or box.y1 < 0:
        problems.append(f"left ({box.x1}) and top ({box.y1}) must not be negative")
    if box.x2 > canvas.width:
        problems.append(f"box right edge ({box.x2}) must not exceed the canvas width ({canvas.width})")
    if box.y2 > canvas.height:
        problems.append(f"box bottom edge ({box.y2}) must not exceed the canvas height ({canvas.height})")
    return problems


class CreateBoundingBox(DataNode):
    """Draw a single bounding box on a canvas and output it as an array and a dictionary."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # The connected image is loaded once per input_image change, not per read. The cache key is the loaded
        # path; None means nothing is loaded. A failed load is cached too (as a key with no ImageInfo).
        self._image_cache_key: str | None = None
        self._image_cache: ImageInfo | None = None
        self._image_revision = 0

        # Canvas dimensions fed by an incoming connection. A connection is the source of truth for them.
        self._connected_canvas: set[str] = set()

        default_array = build_array(DEFAULT_BOX, parse_order(DEFAULT_ARRAY_ORDER, "array"))
        default_dict = build_dict(DEFAULT_BOX, parse_order(DEFAULT_DICT_ORDER, "dict"))
        default_params = DEFAULT_BOX.as_params()

        self.add_parameter(
            ParameterImage(
                name=ParamName.INPUT_IMAGE.value,
                default_value=None,
                tooltip="Optional image to show behind the box. Enable 'Use Image Size' to fit the canvas to it.",
                hide_property=True,
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterBool(
                name=ParamName.USE_IMAGE_SIZE.value,
                default_value=False,
                tooltip="Use the connected image's size as the canvas dimensions.",
                hide=True,
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterInt(
                name=ParamName.CANVAS_WIDTH.value,
                default_value=DEFAULT_CANVAS_SIZE,
                min_val=MIN_CANVAS_SIZE,
                max_val=MAX_CANVAS_SIZE,
                validate_min_max=True,
                tooltip="Canvas width. The canvas always starts at 0, 0.",
                allow_output=True,
            )
        )
        self.add_parameter(
            ParameterInt(
                name=ParamName.CANVAS_HEIGHT.value,
                default_value=DEFAULT_CANVAS_SIZE,
                min_val=MIN_CANVAS_SIZE,
                max_val=MAX_CANVAS_SIZE,
                validate_min_max=True,
                tooltip="Canvas height. The canvas always starts at 0, 0.",
                allow_output=True,
            )
        )

        # The editor reads and writes the box and canvas parameters directly, so its own value holds only the
        # image it draws behind the box.
        self.add_parameter(
            ParameterDict(
                name="bbox_editor",
                default_value={"image_url": "", "image_revision": 0},
                tooltip="Interactive bounding box editor. Drag to move, resize, or draw a new box.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={
                    Widget(
                        name="BoundingBoxEditor",
                        library="Griptape Nodes Library",
                        synced_parameters=EDITOR_PARAMETERS,
                    )
                },
            )
        )

        with ParameterGroup(name="bounding_box_coordinates", ui_options={"collapsed": False}) as coordinates:
            for name, minimum, maximum, tooltip in (
                (ParamName.LEFT, 0, DEFAULT_CANVAS_SIZE - 1, "Left edge of the box in pixels"),
                (ParamName.TOP, 0, DEFAULT_CANVAS_SIZE - 1, "Top edge of the box in pixels"),
                (ParamName.WIDTH, 1, DEFAULT_CANVAS_SIZE, "Width of the box in pixels"),
                (ParamName.HEIGHT, 1, DEFAULT_CANVAS_SIZE, "Height of the box in pixels"),
            ):
                ParameterInt(
                    name=name.value,
                    default_value=default_params[name],
                    tooltip=tooltip,
                    traits={Slider(min_val=minimum, max_val=maximum, soft_limits=True)},
                    allow_output=True,
                )
        self.add_node_element(coordinates)

        with ParameterGroup(name="output_format", ui_options={"collapsed": True}) as output_format:
            ParameterString(
                name=ParamName.ARRAY_ORDER.value,
                default_value=DEFAULT_ARRAY_ORDER,
                tooltip="Comma-separated tokens for the array output. Choose a preset or enter a custom order.",
                placeholder_text=DEFAULT_ARRAY_ORDER,
                traits={Options(choices=ARRAY_ORDER_PRESETS, allow_custom=True)},
                badge=BadgeData(variant="info", title="Accepted tokens", message=TOKENS_HELP),
                allow_output=False,
            )
            ParameterString(
                name=ParamName.DICT_ORDER.value,
                default_value=DEFAULT_DICT_ORDER,
                tooltip="Comma-separated tokens for dict keys. Choose a preset or enter a custom order.",
                placeholder_text=DEFAULT_DICT_ORDER,
                traits={Options(choices=DICT_ORDER_PRESETS, allow_custom=True)},
                badge=BadgeData(variant="info", title="Accepted tokens", message=TOKENS_HELP),
                allow_output=False,
            )
        self.add_node_element(output_format)

        self.add_parameter(
            Parameter(
                name="bbox_array",
                type="json",
                output_type="json",
                default_value=default_array,
                tooltip="Bounding box coordinates as a JSON array, in the order set by 'Array Order'.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            ParameterDict(
                name="bbox_dict",
                default_value=default_dict,
                tooltip="Bounding box as a dictionary, with keys in the order set by 'Dict Order'. The default works with Add Bounding Boxes.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.parameter_output_values["bbox_array"] = default_array
        self.parameter_output_values["bbox_dict"] = default_dict
        for name in PASS_THROUGH_PARAM_NAMES:
            self.parameter_output_values[name] = self.get_parameter_value(name)

    # ── Reading state ──────────────────────────────────────────────────────────

    def _get_parameter(self, name: str) -> Parameter:
        parameter = self.get_parameter_by_name(name)
        if parameter is None:
            msg = f"Attempted to look up parameter '{name}' on '{self.name}'. Failed because it is not defined."
            raise KeyError(msg)
        return parameter

    def _read_box(self) -> BoundingBox:
        defaults = DEFAULT_BOX.as_params()
        values = {}
        for name in BOX_PARAM_NAMES:
            value = self.get_parameter_value(name)
            if value is None:
                value = defaults[name]
            values[name] = int(value)
        return BoundingBox(
            x1=values[ParamName.LEFT],
            y1=values[ParamName.TOP],
            x2=values[ParamName.LEFT] + values[ParamName.WIDTH],
            y2=values[ParamName.TOP] + values[ParamName.HEIGHT],
        )

    def _read_canvas(self) -> CanvasSize:
        """The canvas size parameters, which _apply_canvas keeps equal to the image size while it is in use."""
        sizes = []
        for name in (ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT):
            value = self.get_parameter_value(name)
            if value is None:
                value = DEFAULT_CANVAS_SIZE
            sizes.append(max(MIN_CANVAS_SIZE, int(value)))
        return CanvasSize(width=sizes[0], height=sizes[1])

    def _use_image_size_info(self) -> ImageInfo | None:
        """The connected image, only when the toggle asks for its size."""
        if not self.get_parameter_value(ParamName.USE_IMAGE_SIZE):
            return None
        return self._image_info()

    # ── Image handling ─────────────────────────────────────────────────────────

    def _image_path(self) -> str | None:
        """Extract the source path or URL of the image from the input_image value."""
        return extract_image_url(self.get_parameter_value(ParamName.INPUT_IMAGE)) or None

    def _resolve_image_url(self, path: str) -> str:
        """Return a browser-accessible URL for an image path."""
        if path.startswith(("http://", "https://", "data:")):
            return path

        try:
            result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=path))
        except Exception as e:
            logger.warning("%s: could not create a static file URL for %r: %s", self.name, path, e)
            return path
        if not isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            logger.warning("%s: could not create a static file URL for %r: %s", self.name, path, result.result_details)
            return path

        return result.url

    def _image_info(self) -> ImageInfo | None:
        """Size and widget URL of the connected image. Loaded once, until input_image changes."""
        path = self._image_path()
        if not path:
            self._forget_image()
            return None
        if self._image_cache_key == path:
            return self._image_cache

        self._image_cache_key = path
        self._image_cache = None
        try:
            image = load_pil_from_url(path)
        except (ValueError, OSError) as e:
            logger.warning("%s: could not load the image connected to 'input_image': %s", self.name, e)
            return None

        if max(image.width, image.height) > MAX_CANVAS_SIZE:
            logger.warning(
                "%s: ignoring the image connected to 'input_image'. Its size %dx%d exceeds the maximum canvas size %d.",
                self.name,
                image.width,
                image.height,
                MAX_CANVAS_SIZE,
            )
            return None

        self._image_cache = ImageInfo(
            path=path,
            width=image.width,
            height=image.height,
            url=self._resolve_image_url(path),
        )
        self._image_revision += 1
        return self._image_cache

    def _forget_image(self) -> None:
        """Drop the cached image so the next read loads it again."""
        self._image_cache_key = None
        self._image_cache = None

    # ── Connection state ───────────────────────────────────────────────────────

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name == ParamName.INPUT_IMAGE:
            # Shown from the connection alone, so it holds on workflow load, where connections are made before any
            # value is set. The engine sets input_image from the source right after this, which loads the image.
            self.show_parameter_by_name(ParamName.USE_IMAGE_SIZE)
        elif target_parameter.name in (ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT):
            self._connected_canvas.add(target_parameter.name)
            if self.get_parameter_value(ParamName.USE_IMAGE_SIZE):
                self.set_parameter_value(ParamName.USE_IMAGE_SIZE, False)
            else:
                self._apply_canvas()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name == ParamName.INPUT_IMAGE:
            # input_image also allows PROPERTY, so the engine leaves the last image in place on disconnect.
            self.hide_parameter_by_name(ParamName.USE_IMAGE_SIZE)
            self._forget_image()
            if self.get_parameter_value(ParamName.INPUT_IMAGE) is None:
                self._refresh_image_state()
            else:
                self.set_parameter_value(ParamName.INPUT_IMAGE, None)
        elif target_parameter.name in (ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT):
            self._connected_canvas.discard(target_parameter.name)
            self._apply_canvas()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # ── Editor image ───────────────────────────────────────────────────────────

    def _push_editor_image(self) -> None:
        """Give the editor the image to draw behind the box.

        The box and canvas size reach the editor from their own parameters, listed in synced_parameters,
        and the editor works out which ones are locked by a connection.
        """
        image_url = ""
        info = self._image_info()
        if info is not None:
            image_url = info.url
        state = {"image_url": image_url, "image_revision": self._image_revision}
        self.set_parameter_value("bbox_editor", state)
        self.publish_update_to_parameter("bbox_editor", state)

    # ── Canvas ─────────────────────────────────────────────────────────────────

    def _refresh_image_state(self) -> None:
        """Show the image in the editor and re-apply the canvas."""
        self._push_editor_image()
        self._apply_canvas()

    def _apply_canvas(self) -> None:
        info = self._use_image_size_info()
        active = info is not None

        for name in (ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT):
            self._get_parameter(name).settable = not (active or name in self._connected_canvas)
        self._get_parameter(ParamName.USE_IMAGE_SIZE).settable = not self._connected_canvas

        if info is not None:
            for name, value in ((ParamName.CANVAS_WIDTH, info.width), (ParamName.CANVAS_HEIGHT, info.height)):
                if self.get_parameter_value(name) != value:
                    self.set_parameter_value(name, value)

        self._refresh_canvas_limits()

    def _refresh_canvas_limits(self) -> None:
        """Fit the coordinate sliders to the canvas, then re-check the outputs against it."""
        canvas = self._read_canvas()
        # A box edge can sit at most one pixel inside the canvas, so left and top stop one short of its size.
        for name, limit in (
            (ParamName.LEFT, canvas.width - 1),
            (ParamName.WIDTH, canvas.width),
            (ParamName.TOP, canvas.height - 1),
            (ParamName.HEIGHT, canvas.height),
        ):
            self._get_parameter(name).update_ui_options({"slider": {"max_val": limit}})

        self._update_outputs()

    # ── Outputs ────────────────────────────────────────────────────────────────

    def _try_parse_order(self, name: str, label: str) -> list[str] | None:
        try:
            return parse_order(self.get_parameter_value(name), label)
        except ValueError:
            # Surfaced to the user by validate_before_node_run; live edits keep the last good output.
            return None

    def _update_outputs(self) -> None:
        """Push both outputs so connected nodes see every edit without a run."""
        box = self._read_box()
        if box_problems(box, self._read_canvas()):
            return

        array_tokens = self._try_parse_order(ParamName.ARRAY_ORDER, "array")
        if array_tokens is not None:
            array_value = build_array(box, array_tokens)
            self.parameter_output_values["bbox_array"] = array_value
            self.set_parameter_value("bbox_array", array_value)
            self.publish_update_to_parameter("bbox_array", array_value)

        dict_tokens = self._try_parse_order(ParamName.DICT_ORDER, "dict")
        if dict_tokens is not None:
            dict_value = build_dict(box, dict_tokens)
            self.parameter_output_values["bbox_dict"] = dict_value
            self.set_parameter_value("bbox_dict", dict_value)
            self.publish_update_to_parameter("bbox_dict", dict_value)

    # ── Node lifecycle ─────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in PASS_THROUGH_PARAM_NAMES:
            self.parameter_output_values[parameter.name] = value
            self.publish_update_to_parameter(parameter.name, value)

        match parameter.name:
            case ParamName.INPUT_IMAGE:
                self._forget_image()
                self._refresh_image_state()
            case ParamName.USE_IMAGE_SIZE:
                self._apply_canvas()
            case ParamName.CANVAS_WIDTH | ParamName.CANVAS_HEIGHT:
                self._refresh_canvas_limits()
            case (
                ParamName.LEFT
                | ParamName.TOP
                | ParamName.WIDTH
                | ParamName.HEIGHT
                | ParamName.ARRAY_ORDER
                | ParamName.DICT_ORDER
            ):
                self._update_outputs()
            case _:
                pass

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        for name in (ParamName.CANVAS_WIDTH, ParamName.CANVAS_HEIGHT):
            size = self.get_parameter_value(name)
            if size is not None and size < MIN_CANVAS_SIZE:
                msg = (
                    f"Attempted to use '{name}' on '{self.name}'. Failed with value {size} "
                    f"because it must be at least {MIN_CANVAS_SIZE}."
                )
                exceptions.append(ValueError(msg))

        problems = box_problems(self._read_box(), self._read_canvas())
        if problems:
            msg = f"Attempted to build the bounding box on '{self.name}'. Failed because {'; '.join(problems)}."
            exceptions.append(ValueError(msg))

        for name, label in ((ParamName.ARRAY_ORDER, "array"), (ParamName.DICT_ORDER, "dict")):
            try:
                parse_order(self.get_parameter_value(name), label)
            except ValueError as e:
                exceptions.append(ValueError(f"{self.name}: {e}"))

        return exceptions or None

    def process(self) -> None:
        for name in PASS_THROUGH_PARAM_NAMES:
            self.parameter_output_values[name] = self.get_parameter_value(name)
        self._update_outputs()
