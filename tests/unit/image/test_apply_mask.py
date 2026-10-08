"""Tests for the Apply Mask node's live preview and run behaviour."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

from griptape_nodes_library.image.apply_mask import ApplyMask


def make_node(name: str, load: MagicMock) -> ApplyMask:
    node = ApplyMask(name=name)
    written = SimpleNamespace(location="/tmp/out.png")  # noqa: S108
    node._output_file = MagicMock()
    node._output_file.build_file.return_value.write_bytes.return_value = written
    node.load_pil_from_url = load
    return node


def load_existing(missing: str) -> MagicMock:
    """Load a small image for any URL except `missing`, which raises like an absent file."""

    def load(url: str) -> Image.Image:
        if url == missing:
            msg = f"File not found: {url}"
            raise FileNotFoundError(msg)
        return Image.new("RGBA", (2, 2), (255, 255, 255, 255))

    return MagicMock(side_effect=load)


class TestApplyMask:
    def test_preview_skips_stale_unreadable_input(self) -> None:
        node = make_node("stale", load_existing(missing="stale_mask.png"))
        node.set_parameter_value("input_mask", ImageUrlArtifact("stale_mask.png"))

        node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))  # preview logs, doesn't raise

        assert node.get_parameter_value("output") is None

    def test_failed_preview_clears_earlier_output(self) -> None:
        node = make_node("clear", load_existing(missing="stale_mask.png"))
        node.set_parameter_value("input_mask", ImageUrlArtifact("mask.png"))
        node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))
        assert node.get_parameter_value("output") is not None

        node.set_parameter_value("input_mask", ImageUrlArtifact("stale_mask.png"))

        assert node.get_parameter_value("output") is None
        assert node.parameter_output_values.get("output") is None

    def test_process_runs_once_current_inputs_arrive(self) -> None:
        node = make_node("current", load_existing(missing="stale_mask.png"))
        node.set_parameter_value("input_mask", ImageUrlArtifact("stale_mask.png"))
        node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))
        node.set_parameter_value("input_mask", ImageUrlArtifact("mask.png"))
        writes = cast("MagicMock", node._output_file).build_file.return_value.write_bytes
        writes_before = writes.call_count

        node.process()

        assert writes.call_count == writes_before + 1
        assert node.get_parameter_value("output") is not None

    def test_process_raises_on_unreadable_input(self) -> None:
        node = make_node("raise", load_existing(missing="stale_mask.png"))
        node.set_parameter_value("input_mask", ImageUrlArtifact("stale_mask.png"))
        node.set_parameter_value("input_image", ImageUrlArtifact("image.png"))

        with pytest.raises(FileNotFoundError, match="File not found"):
            node.process()
