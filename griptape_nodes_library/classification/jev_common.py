from __future__ import annotations

import asyncio
import base64
import json
import math
from io import BytesIO
from typing import TYPE_CHECKING, Any

from griptape.artifacts import ImageArtifact
from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.files.file import File, FileLoadError
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from collections.abc import Iterator

    from griptape_nodes.exe_types.node_types import BaseNode

__all__ = [
    "CONTEXT_INPUT_TYPES",
    "DEFAULT_MODEL",
    "MODEL_CHOICES",
    "QUESTION_KEY",
    "add_context_parameter",
    "add_images_parameter",
    "add_model_group",
    "build_payload",
    "build_state_and_assets",
    "image_reference",
    "to_state",
]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
QUESTION_KEY = "answer"
CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]

# JEV's image limits: each side 256-1568 px, at least 280x280 px in total, and a 32 MiB request body.
IMAGE_MIN_SIDE = 280
IMAGE_MAX_SIDE = 1568
IMAGE_MIN_PIXELS = 280 * 280
IMAGE_JPEG_QUALITY = 95
IMAGE_BACKGROUND = (255, 255, 255)
SIXTEEN_BIT_TO_EIGHT_BIT = 1 / 256
MAX_IMAGE_PAYLOAD_BYTES = 30 * 1024 * 1024


def to_state(text: str | None) -> str | dict | list | None:
    """Convert a context string to JEV state, or None if empty.

    JSON objects and arrays are parsed back out so JEV reads them as structured
    state rather than as raw text.
    """
    text = (text or "").strip()
    if not text:
        return None
    if text[0] in "{[":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return text


def image_reference(index: int) -> str:
    """The text that stands in for the image at `index` (0-based) in the context and question."""
    return f"[image {index + 1}]"


def _fit_to_image_limits(image: Image.Image, reference: str) -> Image.Image:
    """Scale an image into JEV's size limits, keeping its aspect ratio."""
    width, height = image.size
    scale = min(1.0, IMAGE_MAX_SIDE / max(width, height))
    scale = max(scale, IMAGE_MIN_SIDE / min(width, height), math.sqrt(IMAGE_MIN_PIXELS / (width * height)))
    if scale == 1.0:
        return image

    rounding = math.ceil if scale > 1.0 else math.floor
    size = (rounding(width * scale), rounding(height * scale))
    if min(size) < IMAGE_MIN_SIDE or max(size) > IMAGE_MAX_SIDE:
        raise ValueError(
            f"{reference} is {width}x{height} px. JEV needs each side between {IMAGE_MIN_SIDE} and "
            f"{IMAGE_MAX_SIDE} px, which this aspect ratio can't fit. Crop it closer to square."
        )
    return image.resize(size, Image.Resampling.LANCZOS)


async def _read_image_bytes(image: Any, reference: str) -> bytes:
    if isinstance(image, ImageArtifact):
        return image.value
    match image:
        case str():
            location = image
        case dict():
            location = image.get("value")
        case _:
            location = getattr(image, "value", None)
    if not isinstance(location, str) or not location:
        raise ValueError(f"{reference} isn't an image JEV can read. Connect an image.")
    try:
        return await File(location).aread_bytes()
    except FileLoadError as e:
        raise ValueError(f"{reference} couldn't be loaded: {e}") from e


def _to_rgb(image: Image.Image) -> Image.Image:
    """Convert an image to upright 8-bit RGB, flattening transparency onto white."""
    image = ImageOps.exif_transpose(image)
    if image.mode == "I" or image.mode.startswith("I;16"):
        image = image.convert("I").point(lambda value: value * SIXTEEN_BIT_TO_EIGHT_BIT).convert("L")
    elif image.mode == "F":
        image = image.point(lambda value: value * 255).convert("L")
    if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, IMAGE_BACKGROUND)
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        return flattened
    return image.convert("RGB")


def _encode_image(data: bytes, reference: str) -> str:
    """Decode image bytes, fit them to JEV's limits, and return base64 JPEG."""
    try:
        image = Image.open(BytesIO(data))
        image.load()
    except Exception as e:
        raise ValueError(f"{reference} couldn't be read as an image: {e}") from e

    image = _fit_to_image_limits(_to_rgb(image), reference)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=IMAGE_JPEG_QUALITY)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


async def _to_image_asset(image: Any, reference: str) -> dict[str, str]:
    """Load an image input and encode it as a JEV image asset."""
    data = await _read_image_bytes(image, reference)
    data_base64 = await asyncio.to_thread(_encode_image, data, reference)
    return {"type": "image", "format": "jpg", "data_base64": data_base64}


async def build_state_and_assets(node: BaseNode) -> tuple[str | dict | list | None, dict[str, dict[str, str]]]:
    """Read a node's Context and encode its Images as JEV image assets keyed by reference."""
    state = to_state(node.get_parameter_value("context"))
    images = [image for image in node.get_parameter_list_value("images") or [] if image]
    if not images and state is None:
        raise ValueError("'Context' is empty. Connect the text for JEV to read.")

    assets: dict[str, dict[str, str]] = {}
    for index, image in enumerate(images):
        reference = image_reference(index)
        assets[reference] = await _to_image_asset(image, reference)
    return state, assets


def _strings(value: Any) -> Iterator[str]:
    """Yield every string value nested in `value`. Dict keys are labels JEV doesn't substitute into."""
    match value:
        case str():
            yield value
        case dict():
            for item in value.values():
                yield from _strings(item)
        case list():
            for item in value:
                yield from _strings(item)


def build_payload(
    state: str | dict | list | None, question: dict[str, Any], assets: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Build the JEV request body for one question, with image assets when there are any.

    JEV swaps every occurrence of an asset's reference for the image itself, so each
    use sends the image again. Images the context or question already name are left
    out of the state's "images" list; the rest are listed there so every asset is
    referenced at least once.
    """
    if not assets:
        return {"state": state, "questions": {QUESTION_KEY: question}}

    texts = list(_strings([state, question]))
    uses = {reference: sum(text.count(reference) for text in texts) for reference in assets}
    unreferenced = [reference for reference, count in uses.items() if count == 0]
    payload_bytes = sum(len(asset["data_base64"]) * max(uses[ref], 1) for ref, asset in assets.items())
    if payload_bytes > MAX_IMAGE_PAYLOAD_BYTES:
        raise ValueError(
            "The images, counting each time the context or question names one, add up to more than JEV "
            "accepts in one request. Connect fewer images or name each one fewer times."
        )

    image_state: dict[str, Any] = {"images": unreferenced}
    if state is not None:
        image_state = {"context": state, **image_state}
    return {"state": image_state, "questions": {QUESTION_KEY: question}, "assets": assets}


def add_context_parameter(node: BaseNode) -> None:
    """Add the shared Context parameter to a node."""
    node.add_parameter(
        ParameterString(
            name="context",
            display_name="Context",
            tooltip="The text JEV reads. JSON works too. Optional when images are connected.",
            default_value="",
            multiline=True,
            placeholder_text="Text to ask about",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            input_types=CONTEXT_INPUT_TYPES,
        )
    )


def add_images_parameter(node: BaseNode) -> None:
    """Add the shared Images parameter to a node."""
    node.add_parameter(
        ParameterList(
            name="images",
            input_types=["ImageArtifact", "ImageUrlArtifact", "str"],
            default_value=[],
            tooltip="Optional. Images JEV looks at alongside the context. "
            "Refer to them in the question as [image 1], [image 2], and so on.",
            allowed_modes={ParameterMode.INPUT},
            ui_options={"display_name": "Images", "hide_property": True},
        )
    )


def add_model_group(node: BaseNode) -> ModelAccessComponent:
    """Add the collapsed Advanced group with the license-aware model dropdown to a node."""
    with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced_group:
        model_param = ParameterString(
            name="model",
            display_name="Model",
            tooltip="jev-latest is the newest stable model. jev-preview is the newest release.",
            default_value=DEFAULT_MODEL,
            allow_input=False,
            allow_output=False,
        )
    node.add_node_element(advanced_group)
    return ModelAccessComponent(
        node=node,
        parameter=model_param,
        model_choices=MODEL_CHOICES,
        default_model=DEFAULT_MODEL,
    )
