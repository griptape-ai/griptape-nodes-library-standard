from dataclasses import dataclass
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.elements.badge import BadgeData
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from griptape_nodes.retained_mode.events.connection_events import (
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.static_file_events import (
    CreateStaticFileDownloadUrlFromPathRequest,
    CreateStaticFileDownloadUrlFromPathResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.image_utils import load_pil_from_url

DEFAULT_CANVAS_SIZE = 1000
MIN_CANVAS_SIZE = 1
MAX_CANVAS_SIZE = 2_147_483_647
DEFAULT_BOX = (250, 250, 750, 750)

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

BOX_PARAM_NAMES = ("x1", "y1", "x2", "y2")
INPUT_OUTPUT_PARAM_NAMES = ("input_image", "use_image_size", "canvas_width", "canvas_height", *BOX_PARAM_NAMES)


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


@dataclass(frozen=True)
class CanvasSize:
    width: int
    height: int


@dataclass(frozen=True)
class ImageInfo:
    path: str
    width: int
    height: int
    url: str


def parse_order(text: Any, label: str) -> list[str]:
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
    coordinate = TOKEN_ALIASES[token]
    match coordinate:
        case "x1":
            value = box.x1
        case "y1":
            value = box.y1
        case "x2":
            value = box.x2
        case "y2":
            value = box.y2
        case "width":
            value = box.width
        case "height":
            value = box.height
        case _:
            msg = f"Attempted to resolve token '{token}'. Failed because '{coordinate}' is not a known coordinate."
            raise ValueError(msg)
    return value


def build_array(box: BoundingBox, tokens: list[str]) -> list[int]:
    return [token_value(box, token) for token in tokens]


def build_dict(box: BoundingBox, tokens: list[str]) -> dict[str, int]:
    return {token: token_value(box, token) for token in tokens}


def fit_box_to_canvas(box: BoundingBox, canvas: CanvasSize) -> BoundingBox:
    x1 = min(max(box.x1, 0), canvas.width - 1)
    y1 = min(max(box.y1, 0), canvas.height - 1)
    x2 = min(max(box.x2, x1 + 1), canvas.width)
    y2 = min(max(box.y2, y1 + 1), canvas.height)
    return BoundingBox(x1, y1, x2, y2)


def box_problems(box: BoundingBox, canvas: CanvasSize) -> list[str]:
    """Describe why a box is unusable on the canvas; an empty list means it is valid."""
    problems = []
    if box.x2 <= box.x1:
        problems.append(f"x2 ({box.x2}) must be greater than x1 ({box.x1})")
    if box.y2 <= box.y1:
        problems.append(f"y2 ({box.y2}) must be greater than y1 ({box.y1})")
    if box.x1 < 0 or box.y1 < 0:
        problems.append(f"x1 ({box.x1}) and y1 ({box.y1}) must not be negative")
    if box.x2 > canvas.width:
        problems.append(f"x2 ({box.x2}) must not exceed the canvas width ({canvas.width})")
    if box.y2 > canvas.height:
        problems.append(f"y2 ({box.y2}) must not exceed the canvas height ({canvas.height})")
    return problems


class CreateBoundingBox(DataNode):
    """Draw a single bounding box on a canvas and output it as an array and a dictionary."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._syncing_to_widget = False  # Prevent loop: param -> widget -> param
        self._syncing_to_params = False  # Prevent loop: widget -> param -> widget
        self._image_cache: ImageInfo | None = None

        self.add_parameter(
            ParameterImage(
                name="input_image",
                default_value=None,
                tooltip="Optional image. Enable 'Use Image Size' to fit the canvas to it and show it behind the box.",
                hide_property=True,
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterBool(
                name="use_image_size",
                default_value=False,
                tooltip="Use the connected image's size as the canvas and show the image behind the box.",
                hide=True,
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="canvas_width",
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
                name="canvas_height",
                default_value=DEFAULT_CANVAS_SIZE,
                min_val=MIN_CANVAS_SIZE,
                max_val=MAX_CANVAS_SIZE,
                validate_min_max=True,
                tooltip="Canvas height. The canvas always starts at 0, 0.",
                allow_output=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="bbox_editor",
                default_value={
                    "image_url": "",
                    "img_width": DEFAULT_CANVAS_SIZE,
                    "img_height": DEFAULT_CANVAS_SIZE,
                    "left": DEFAULT_BOX[0],
                    "top": DEFAULT_BOX[1],
                    "width": DEFAULT_BOX[2] - DEFAULT_BOX[0],
                    "height": DEFAULT_BOX[3] - DEFAULT_BOX[1],
                    "locked": [],
                },
                tooltip="Interactive bounding box editor. Drag to move, resize, or draw a new box.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="BoundingBoxEditor", library="Griptape Nodes Library")},
            )
        )

        with ParameterGroup(name="bounding_box_coordinates", ui_options={"collapsed": True}) as coordinates:
            for name, default, tooltip in (
                ("x1", DEFAULT_BOX[0], "Left edge of the box"),
                ("y1", DEFAULT_BOX[1], "Top edge of the box"),
                ("x2", DEFAULT_BOX[2], "Right edge of the box"),
                ("y2", DEFAULT_BOX[3], "Bottom edge of the box"),
            ):
                ParameterInt(
                    name=name,
                    default_value=default,
                    tooltip=tooltip,
                    traits={Slider(min_val=0, max_val=DEFAULT_CANVAS_SIZE, soft_limits=True)},
                    allow_output=True,
                )
        self.add_node_element(coordinates)

        with ParameterGroup(name="output_format", ui_options={"collapsed": True}) as output_format:
            ParameterString(
                name="array_order",
                default_value=DEFAULT_ARRAY_ORDER,
                tooltip="Comma-separated tokens for the array output. Choose a preset or enter a custom order.",
                placeholder_text=DEFAULT_ARRAY_ORDER,
                traits={Options(choices=ARRAY_ORDER_PRESETS, allow_custom=True)},
                badge=BadgeData(variant="info", title="Accepted tokens", message=TOKENS_HELP),
                allow_output=False,
            )
            ParameterString(
                name="dict_order",
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
                default_value=build_array(BoundingBox(*DEFAULT_BOX), parse_order(DEFAULT_ARRAY_ORDER, "array")),
                tooltip="Bounding box coordinates as a JSON array, in the order set by 'Array Order'.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.add_parameter(
            ParameterDict(
                name="bbox_dict",
                default_value=build_dict(BoundingBox(*DEFAULT_BOX), parse_order(DEFAULT_DICT_ORDER, "dict")),
                tooltip="Bounding box as a dictionary, with keys in the order set by 'Dict Order'. The default works with Add Bounding Boxes.",
                allowed_modes={ParameterMode.OUTPUT},
                hide_property=True,
            )
        )
        self.parameter_output_values["bbox_array"] = build_array(
            BoundingBox(*DEFAULT_BOX), parse_order(DEFAULT_ARRAY_ORDER, "array")
        )
        self.parameter_output_values["bbox_dict"] = build_dict(
            BoundingBox(*DEFAULT_BOX), parse_order(DEFAULT_DICT_ORDER, "dict")
        )
        for name in INPUT_OUTPUT_PARAM_NAMES:
            self.parameter_output_values[name] = self.get_parameter_value(name)

    # ── Reading state ──────────────────────────────────────────────────────────

    def _get_parameter(self, name: str) -> Parameter:
        parameter = self.get_parameter_by_name(name)
        if parameter is None:
            msg = f"Attempted to look up parameter '{name}' on '{self.name}'. Failed because it is not defined."
            raise KeyError(msg)
        return parameter

    def _read_box(self) -> BoundingBox:
        values = {}
        for index, name in enumerate(BOX_PARAM_NAMES):
            value = self.get_parameter_value(name)
            if value is None:
                value = DEFAULT_BOX[index]
            values[name] = int(value)
        return BoundingBox(**values)

    def _read_canvas(self) -> CanvasSize:
        info = self._image_info()
        if self._image_size_active() and info is not None:
            return CanvasSize(width=info.width, height=info.height)

        sizes = []
        for name in ("canvas_width", "canvas_height"):
            value = self.get_parameter_value(name)
            if value is None:
                value = DEFAULT_CANVAS_SIZE
            sizes.append(max(MIN_CANVAS_SIZE, int(value)))
        return CanvasSize(width=sizes[0], height=sizes[1])

    def _image_size_active(self) -> bool:
        return bool(self.get_parameter_value("use_image_size")) and self._image_info() is not None

    # ── Image handling ─────────────────────────────────────────────────────────

    def _extract_image_path(self, value: Any) -> str | None:
        if isinstance(value, str):
            return value
        inner = getattr(value, "value", None)
        if isinstance(inner, str):
            return inner
        return None

    def _resolve_image_url(self, path: str) -> str:
        """Return a browser-accessible URL for an image path."""
        if path.startswith(("http://", "https://", "data:")):
            return path
        try:
            resolved = File(path).resolve()
        except FileLoadError as e:
            logger.warning("%s: could not resolve %r to a local file: %s", self.name, path, e)
            resolved = path
        result = GriptapeNodes.handle_request(CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved))
        if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
            return result.url
        logger.warning("%s: could not create a static file URL for %r", self.name, resolved)
        return path

    def _image_info(self) -> ImageInfo | None:
        """Size and widget URL of the connected image, cached per image path."""
        path = self._extract_image_path(self.get_parameter_value("input_image"))
        if not path:
            self._image_cache = None
            return None
        if self._image_cache is not None and self._image_cache.path == path:
            return self._image_cache

        try:
            image = load_pil_from_url(path)
        except (ValueError, OSError) as e:
            logger.warning("%s: could not load the image connected to 'input_image': %s", self.name, e)
            self._image_cache = None
            return None

        self._image_cache = ImageInfo(
            path=path, width=image.width, height=image.height, url=self._resolve_image_url(path)
        )
        return self._image_cache

    # ── Connection state ───────────────────────────────────────────────────────

    def _get_locked_params(self) -> list[str]:
        """Return box param names that have an incoming wire connection."""
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            logger.warning("%s: could not determine incoming box parameter connections", self.name)
            return []
        return [
            connection.target_parameter_name
            for connection in result.incoming_connections
            if connection.target_parameter_name in BOX_PARAM_NAMES
        ]

    def after_incoming_connection(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in BOX_PARAM_NAMES:
            self._refresh_locked_in_widget()
        if target_parameter.name == "input_image":
            self._refresh_image_state()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(self, source_node, source_parameter, target_parameter) -> None:
        if target_parameter.name in BOX_PARAM_NAMES:
            self._refresh_locked_in_widget()
        if target_parameter.name == "input_image":
            self._refresh_image_state()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # ── Widget <-> parameter sync ──────────────────────────────────────────────

    def _build_widget_dict(self, box: BoundingBox, canvas: CanvasSize) -> dict:
        image_url = ""
        info = self._image_info()
        if info is not None:
            image_url = info.url
        return {
            "image_url": image_url,
            "img_width": canvas.width,
            "img_height": canvas.height,
            "left": box.x1,
            "top": box.y1,
            "width": box.width,
            "height": box.height,
            "locked": self._get_locked_params(),
        }

    def _push_widget(self, new_dict: dict) -> None:
        self._syncing_to_widget = True
        try:
            self.set_parameter_value("bbox_editor", new_dict)
            self.publish_update_to_parameter("bbox_editor", new_dict)
        finally:
            self._syncing_to_widget = False

    def _refresh_locked_in_widget(self) -> None:
        existing = self.get_parameter_value("bbox_editor") or {}
        new_locked = self._get_locked_params()
        if set(existing.get("locked") or []) == set(new_locked):
            return
        self._push_widget({**existing, "locked": new_locked})

    def _update_widget_coords(self) -> None:
        """Push the current box to the widget unless it already shows it or the box is mid-edit and invalid."""
        box = self._read_box()
        canvas = self._read_canvas()
        if box_problems(box, canvas):
            return

        existing = self.get_parameter_value("bbox_editor") or {}
        if (
            existing.get("left") == box.x1
            and existing.get("top") == box.y1
            and existing.get("width") == box.width
            and existing.get("height") == box.height
        ):
            return
        self._push_widget({**existing, **self._build_widget_dict(box, canvas)})

    def _set_box_params(self, box: BoundingBox) -> None:
        """Write box coordinates into the params that are not driven by a connection."""
        locked = set(self._get_locked_params())
        self._syncing_to_params = True
        try:
            for name in BOX_PARAM_NAMES:
                value = getattr(box, name)
                if name in locked or self.get_parameter_value(name) == value:
                    continue
                self.set_parameter_value(name, value)
        finally:
            self._syncing_to_params = False

    def _sync_params_from_widget(self, widget_dict: dict) -> None:
        canvas = self._read_canvas()
        left = int(round(widget_dict.get("left") or 0))
        top = int(round(widget_dict.get("top") or 0))
        width = int(round(widget_dict.get("width") or canvas.width))
        height = int(round(widget_dict.get("height") or canvas.height))

        # Publish the widget dict first so the frontend's stored value is current before param publishes re-render it.
        self.publish_update_to_parameter("bbox_editor", widget_dict)
        self._set_box_params(BoundingBox(x1=left, y1=top, x2=left + width, y2=top + height))
        self._update_outputs()

    # ── Canvas ─────────────────────────────────────────────────────────────────

    def _refresh_image_state(self) -> None:
        """Show the image-size toggle only while an image is available, then re-apply the canvas."""
        if self._image_info() is None:
            self.hide_parameter_by_name("use_image_size")
        else:
            self.show_parameter_by_name("use_image_size")
        self._apply_canvas()

    def _apply_canvas(self) -> None:
        active = self._image_size_active()

        canvas_params = [self._get_parameter("canvas_width"), self._get_parameter("canvas_height")]
        for param in canvas_params:
            param.settable = not active

        canvas = self._read_canvas()
        for name, limit in (("x1", canvas.width), ("x2", canvas.width), ("y1", canvas.height), ("y2", canvas.height)):
            self._get_parameter(name).update_ui_options({"slider": {"max_val": limit}})

        box = fit_box_to_canvas(self._read_box(), canvas)
        self._set_box_params(box)
        existing = self.get_parameter_value("bbox_editor") or {}
        self._push_widget({**existing, **self._build_widget_dict(box, canvas)})
        self._update_outputs()

    # ── Outputs ────────────────────────────────────────────────────────────────

    def _try_parse_order(self, name: str, label: str) -> list[str] | None:
        try:
            tokens = parse_order(self.get_parameter_value(name), label)
        except ValueError:
            # Surfaced to the user by validate_before_node_run; live edits keep the last good output.
            return None
        return tokens

    def _update_outputs(self) -> None:
        """Push both outputs so connected nodes see every edit without a run."""
        box = self._read_box()
        if box_problems(box, self._read_canvas()):
            return

        array_tokens = self._try_parse_order("array_order", "array")
        if array_tokens is not None:
            array_value = build_array(box, array_tokens)
            self.parameter_output_values["bbox_array"] = array_value
            self.set_parameter_value("bbox_array", array_value)
            self.publish_update_to_parameter("bbox_array", array_value)

        dict_tokens = self._try_parse_order("dict_order", "dict")
        if dict_tokens is not None:
            dict_value = build_dict(box, dict_tokens)
            self.parameter_output_values["bbox_dict"] = dict_value
            self.set_parameter_value("bbox_dict", dict_value)
            self.publish_update_to_parameter("bbox_dict", dict_value)

    # ── Node lifecycle ─────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name in INPUT_OUTPUT_PARAM_NAMES:
            self.parameter_output_values[parameter.name] = value
            self.publish_update_to_parameter(parameter.name, value)

        match parameter.name:
            case "input_image":
                self._refresh_image_state()
            case "use_image_size":
                self._apply_canvas()
            case "canvas_width" | "canvas_height":
                self._apply_canvas()
            case "bbox_editor":
                if not self._syncing_to_widget and isinstance(value, dict):
                    self._sync_params_from_widget(value)
            case "x1" | "y1" | "x2" | "y2":
                if not self._syncing_to_params:
                    self._update_widget_coords()
                    self._update_outputs()
            case "array_order":
                self._update_outputs()
            case "dict_order":
                self._update_outputs()
            case _:
                pass

        return super().after_value_set(parameter, value)

    def validate_before_node_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        for name in ("canvas_width", "canvas_height"):
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

        for name, label in (("array_order", "array"), ("dict_order", "dict")):
            try:
                parse_order(self.get_parameter_value(name), label)
            except ValueError as e:
                exceptions.append(ValueError(f"{self.name}: {e}"))

        return exceptions or None

    def process(self) -> None:
        for name in INPUT_OUTPUT_PARAM_NAMES:
            self.parameter_output_values[name] = self.get_parameter_value(name)
        self._update_outputs()
