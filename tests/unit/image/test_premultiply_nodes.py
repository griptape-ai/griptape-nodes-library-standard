"""Tests for the Premultiply Image and Unpremultiply Image nodes' run behaviour."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from griptape.artifacts import ImageUrlArtifact
from PIL import Image

from griptape_nodes_library.image import base_alpha_conversion
from griptape_nodes_library.image.premultiply_image import PremultiplyImage
from griptape_nodes_library.image.unpremultiply_image import UnpremultiplyImage

NODE_CASES = [
    pytest.param(PremultiplyImage, id="premultiply"),
    pytest.param(UnpremultiplyImage, id="unpremultiply"),
]


@pytest.fixture
def loads(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub image loading in the shared node base and count the calls."""
    load = MagicMock(return_value=Image.new("RGBA", (2, 2), (100, 50, 25, 128)))
    monkeypatch.setattr(base_alpha_conversion, "load_pil_from_url", load)
    return load


def make_node(node_cls: type, name: str) -> PremultiplyImage | UnpremultiplyImage:
    node = node_cls(name=name)
    written = SimpleNamespace(location="/tmp/out.png")  # noqa: S108
    node._output_file = MagicMock()
    node._output_file.build_file.return_value.write_bytes.return_value = written
    return node


@pytest.mark.parametrize("node_cls", NODE_CASES)
class TestRun:
    def test_process_skips_when_preview_already_ran(self, node_cls: type, loads: MagicMock) -> None:
        node = make_node(node_cls, "skip")
        node.set_parameter_value("input_image", ImageUrlArtifact("a.png"))
        assert loads.call_count == 1

        node.process()

        assert loads.call_count == 1
        assert node.get_parameter_value("output") is not None

    def test_changing_invert_reruns(self, node_cls: type, loads: MagicMock) -> None:
        node = make_node(node_cls, "rerun")
        node.set_parameter_value("input_image", ImageUrlArtifact("a.png"))
        node.set_parameter_value("invert", True)  # noqa: FBT003

        assert loads.call_count == 2  # noqa: PLR2004

    def test_process_raises_on_failure(self, node_cls: type, loads: MagicMock) -> None:
        node = make_node(node_cls, "raise")
        loads.side_effect = ValueError("unreadable image")
        node.set_parameter_value("input_image", ImageUrlArtifact("bad.png"))  # preview logs, doesn't raise

        with pytest.raises(ValueError, match="unreadable image"):
            node.process()

    def test_process_without_input_does_nothing(self, node_cls: type, loads: MagicMock) -> None:
        node = make_node(node_cls, "empty")
        node.process()

        assert loads.call_count == 0
        assert node.get_parameter_value("output") is None
