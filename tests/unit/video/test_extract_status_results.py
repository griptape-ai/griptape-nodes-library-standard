from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from griptape_nodes_library.video.extract_audio import ExtractAudio
from griptape_nodes_library.video.extract_last_frame import ExtractLastFrame

if TYPE_CHECKING:
    from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor

NODE_WORKERS = [
    (ExtractAudio, "_process_extract_audio"),
    (ExtractLastFrame, "_process_extract_frame"),
]


def _run(node: BaseVideoProcessor) -> None:
    """Drive the node's async process() to completion, running the yielded work inline."""
    gen = node.process()
    work = next(gen)
    try:
        work()
    except Exception as e:  # noqa: BLE001
        try:
            gen.throw(e)
        except StopIteration:
            return
    else:
        try:
            gen.send(None)
        except StopIteration:
            return


def _make_node(
    monkeypatch: pytest.MonkeyPatch, node_cls: type[BaseVideoProcessor], worker: str, error: Exception | None
):
    node = node_cls(name="Node")
    monkeypatch.setattr(node, "_get_video_input_data", lambda: ("/tmp/input.mp4", "mp4"))

    def fake_worker(*_args, **_kwargs) -> None:
        if error is not None:
            raise error

    monkeypatch.setattr(node, worker, fake_worker)
    return node


@pytest.mark.parametrize(("node_cls", "worker"), NODE_WORKERS)
def test_success_sets_status(monkeypatch: pytest.MonkeyPatch, node_cls: type[BaseVideoProcessor], worker: str) -> None:
    node = _make_node(monkeypatch, node_cls, worker, error=None)

    _run(node)

    assert node.get_parameter_value("was_successful") is True
    assert node.get_parameter_value("result_details").startswith("Successfully extracted")


@pytest.mark.parametrize(("node_cls", "worker"), NODE_WORKERS)
def test_failure_with_failure_output_connected_sets_status_without_raising(
    monkeypatch: pytest.MonkeyPatch, node_cls: type[BaseVideoProcessor], worker: str
) -> None:
    node = _make_node(monkeypatch, node_cls, worker, error=ValueError("No audio stream found in the input video"))
    monkeypatch.setattr(node, "_has_outgoing_connections", lambda _param: True)

    _run(node)

    assert node.get_parameter_value("was_successful") is False
    assert "No audio stream found in the input video" in node.get_parameter_value("result_details")


@pytest.mark.parametrize(("node_cls", "worker"), NODE_WORKERS)
def test_failure_without_failure_output_raises_and_sets_status(
    monkeypatch: pytest.MonkeyPatch, node_cls: type[BaseVideoProcessor], worker: str
) -> None:
    node = _make_node(monkeypatch, node_cls, worker, error=ValueError("boom"))
    monkeypatch.setattr(node, "_has_outgoing_connections", lambda _param: False)

    with pytest.raises(ValueError, match="boom"):
        _run(node)

    assert node.get_parameter_value("was_successful") is False
    assert "boom" in node.get_parameter_value("result_details")
