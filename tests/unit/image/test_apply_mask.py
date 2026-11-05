"""Tests for the Apply Mask node's output pixels."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.image import GriptapeImage

from griptape_nodes_library.image import apply_mask
from griptape_nodes_library.image.apply_mask import ApplyMask

IMAGE_COLOR = (200, 120, 40, 255)
MASK_VALUE = 64
RGBA_CHANNELS = 4


@pytest.fixture
def written() -> list[bytes]:
    return []


def make_node(written: list[bytes], monkeypatch: pytest.MonkeyPatch) -> ApplyMask:
    node = ApplyMask(name="apply_mask")
    images = {
        "image.png": GriptapeImage.new(4, 4, IMAGE_COLOR),
        "mask.png": GriptapeImage.new(4, 4, (MASK_VALUE, MASK_VALUE, MASK_VALUE)),
    }
    monkeypatch.setattr(apply_mask, "load_image_from_url", lambda url: images[url])

    def _write(data: bytes) -> SimpleNamespace:
        written.append(data)
        return SimpleNamespace(location="/tmp/out.png")  # noqa: S108

    node._output_file = MagicMock()
    node._output_file.build_file.return_value.write_bytes.side_effect = _write
    return node


def test_output_is_straight_alpha(written: list[bytes], monkeypatch: pytest.MonkeyPatch) -> None:
    """RGB passes through unchanged and the mask only sets alpha."""
    node = make_node(written, monkeypatch)
    node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))
    node.set_parameter_value("input_mask", ImageUrlArtifact("mask.png"))

    node.process()

    output = GriptapeImage.open(written[-1]).to_array()
    assert output.shape[2] == RGBA_CHANNELS
    assert tuple(output[0, 0]) == (*IMAGE_COLOR[:3], MASK_VALUE)
