"""Tests for the Apply Mask node's output pixels."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

from griptape_nodes_library.image.apply_mask import ApplyMask

IMAGE_COLOR = (200, 120, 40, 255)
MASK_VALUE = 64


@pytest.fixture
def written() -> list[bytes]:
    return []


def make_node(written: list[bytes]) -> ApplyMask:
    node = ApplyMask(name="apply_mask")
    images = {
        "image.png": Image.new("RGBA", (4, 4), IMAGE_COLOR),
        "mask.png": Image.new("RGB", (4, 4), (MASK_VALUE, MASK_VALUE, MASK_VALUE)),
    }
    node.load_pil_from_url = lambda url: images[url]  # type: ignore[method-assign]

    def _write(data: bytes) -> SimpleNamespace:
        written.append(data)
        return SimpleNamespace(location="/tmp/out.png")  # noqa: S108

    node._output_file = MagicMock()
    node._output_file.build_file.return_value.write_bytes.side_effect = _write
    return node


def test_output_is_straight_alpha(written: list[bytes]) -> None:
    """RGB passes through unchanged and the mask only sets alpha."""
    node = make_node(written)
    node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))
    node.set_parameter_value("input_mask", ImageUrlArtifact("mask.png"))

    node.process()

    output = Image.open(BytesIO(written[-1]))
    assert output.mode == "RGBA"
    assert output.getpixel((0, 0)) == (*IMAGE_COLOR[:3], MASK_VALUE)
