"""Images connected to a JEV node are sent as assets JEV can read."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

import pytest
from griptape.artifacts import ImageArtifact
from PIL import Image

from griptape_nodes_library.classification.jev_ask_yes_no import JevAskYesNo
from griptape_nodes_library.classification.jev_common import IMAGE_MAX_SIDE, IMAGE_MIN_SIDE
from griptape_nodes_library.classification.jev_pick_one import JevPickOne
from griptape_nodes_library.classification.jev_rate import JevRate


def _image(width: int, height: int, mode: str = "RGB") -> ImageArtifact:
    buffer = BytesIO()
    Image.new(mode, (width, height)).save(buffer, format="PNG")
    return ImageArtifact(buffer.getvalue(), format="png", width=width, height=height)


def _decoded_size(asset: dict[str, str]) -> tuple[int, int]:
    return Image.open(BytesIO(base64.b64decode(asset["data_base64"]))).size


def _node(node_class: type, values: dict[str, Any], images: list[Any], monkeypatch: pytest.MonkeyPatch) -> Any:
    node = node_class(name="jev")
    monkeypatch.setattr(node, "get_parameter_value", lambda name: values.get(name))
    monkeypatch.setattr(node, "get_parameter_list_value", lambda name: images if name == "images" else [])
    return node


NODE_VALUES = [
    (JevAskYesNo, {"question": "Is [image 1] in focus?"}),
    (JevPickOne, {"options": ["day", "night"]}),
    (JevRate, {}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("node_class", "values"), NODE_VALUES)
async def test_images_without_context_are_sent_as_assets(
    node_class: type, values: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(node_class, values, [_image(512, 512), _image(640, 480)], monkeypatch)
    if node_class is JevRate:
        monkeypatch.setattr(node, "_rows", lambda: [("level_1", "bad"), ("level_2", "good")])

    payload = await node._build_payload()

    assert payload["state"] == {"images": ["[image 1]", "[image 2]"]}
    assert set(payload["assets"]) == {"[image 1]", "[image 2]"}
    for asset in payload["assets"].values():
        assert asset["type"] == "image"
        assert asset["format"] == "jpg"


@pytest.mark.asyncio
async def test_context_and_images_are_combined_in_state(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {"context": '{"shot": "010"}', "question": "Is [image 1] in focus?"}
    node = _node(JevAskYesNo, values, [_image(512, 512, "RGBA")], monkeypatch)

    payload = await node._build_payload()

    assert payload["state"] == {"context": {"shot": "010"}, "images": ["[image 1]"]}


@pytest.mark.asyncio
async def test_text_only_payload_has_no_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(JevAskYesNo, {"context": "hello", "question": "Is it a greeting?"}, [], monkeypatch)

    payload = await node._build_payload()

    assert payload["state"] == "hello"
    assert "assets" not in payload


@pytest.mark.asyncio
async def test_no_context_and_no_images_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(JevAskYesNo, {"question": "Is it a greeting?"}, [], monkeypatch)

    with pytest.raises(ValueError, match="'Context' is empty"):
        await node._build_payload()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "expected"),
    [
        ((3136, 1568), (IMAGE_MAX_SIDE, 784)),
        ((128, 256), (IMAGE_MIN_SIDE, 512)),
        ((1000, 800), (1000, 800)),
    ],
)
async def test_images_are_scaled_into_jev_limits(
    size: tuple[int, int], expected: tuple[int, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node(JevAskYesNo, {"question": "Is [image 1] dark?"}, [_image(*size)], monkeypatch)

    payload = await node._build_payload()

    assert _decoded_size(payload["assets"]["[image 1]"]) == expected


@pytest.mark.asyncio
async def test_an_aspect_ratio_jev_cant_fit_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node(JevAskYesNo, {"question": "Is [image 1] dark?"}, [_image(4000, 400)], monkeypatch)

    with pytest.raises(ValueError, match=r"\[image 1\] is 4000x400 px"):
        await node._build_payload()


def _decoded(asset: dict[str, str]) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(asset["data_base64"])))


def _artifact(image: Image.Image, **save_kwargs: Any) -> ImageArtifact:
    buffer = BytesIO()
    fmt = save_kwargs.pop("format", "PNG")
    image.save(buffer, format=fmt, **save_kwargs)
    return ImageArtifact(buffer.getvalue(), format=fmt.lower(), width=image.width, height=image.height)


@pytest.mark.asyncio
async def test_sixteen_bit_images_keep_their_tones(monkeypatch: pytest.MonkeyPatch) -> None:
    sixteen_bit = Image.new("I;16", (512, 512), 32768)
    node = _node(JevAskYesNo, {"question": "Is [image 1] dark?"}, [_artifact(sixteen_bit)], monkeypatch)

    payload = await node._build_payload()

    pixel = _decoded(payload["assets"]["[image 1]"]).getpixel((256, 256))
    assert isinstance(pixel, tuple)
    red, green, blue = pixel
    assert 120 <= red <= 136
    assert red == green == blue


@pytest.mark.asyncio
async def test_transparency_is_flattened_onto_white(monkeypatch: pytest.MonkeyPatch) -> None:
    transparent = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    node = _node(JevAskYesNo, {"question": "Is [image 1] dark?"}, [_artifact(transparent)], monkeypatch)

    payload = await node._build_payload()

    pixel = _decoded(payload["assets"]["[image 1]"]).getpixel((256, 256))
    assert isinstance(pixel, tuple)
    assert min(pixel) >= 250  # noqa: PLR2004


@pytest.mark.asyncio
async def test_exif_orientation_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    exif = Image.Exif()
    exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display.
    photo = _artifact(Image.new("RGB", (800, 400)), format="JPEG", exif=exif)
    node = _node(JevAskYesNo, {"question": "Is [image 1] upright?"}, [photo], monkeypatch)

    payload = await node._build_payload()

    assert _decoded(payload["assets"]["[image 1]"]).size == (400, 800)
