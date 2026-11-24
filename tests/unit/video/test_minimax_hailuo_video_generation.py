from __future__ import annotations

from typing import Any

import pytest

from griptape_nodes_library.video.minimax_hailuo_video_generation import MinimaxHailuoVideoGeneration

FAST = "MiniMax-Hailuo-2.3-Fast"
FRAME = "data:image/png;base64,AAAA"


def _node(**values: Any) -> MinimaxHailuoVideoGeneration:
    node = MinimaxHailuoVideoGeneration(name="Hailuo")
    node.set_parameter_value("prompt", "a cat")
    for name, value in values.items():
        node.set_parameter_value(name, value)
    return node


def _choices(node: MinimaxHailuoVideoGeneration, name: str) -> list[Any]:
    return list(node.get_parameter_by_name(name).ui_options["simple_dropdown"])  # type: ignore[union-attr]


def _errors(node: MinimaxHailuoVideoGeneration) -> list[str]:
    return [str(e) for e in node._input_errors(node._get_parameters())]


class TestModelChange:
    def test_fast_resets_resolution_to_its_default(self) -> None:
        node = _node(model_id=FAST)

        assert node.get_parameter_value("resolution") == "720p"

    def test_fast_with_default_resolution_passes_checks(self) -> None:
        node = _node(model_id=FAST, first_frame_image=FRAME)

        assert _errors(node) == []

    def test_valid_resolution_is_kept(self) -> None:
        node = _node(resolution="1080p", model_id=FAST)

        assert node.get_parameter_value("resolution") == "1080p"

    def test_leaving_fast_resets_720p(self) -> None:
        node = _node(model_id=FAST)
        node.set_parameter_value("model_id", "MiniMax-Hailuo-2.3")

        assert node.get_parameter_value("resolution") == "768p"

    def test_unsupported_duration_still_resets_resolution(self) -> None:
        node = _node(duration=10, model_id=FAST)

        assert node.get_parameter_value("duration") == 10
        assert node.get_parameter_value("resolution") == "720p"

    def test_choices_are_not_narrowed(self) -> None:
        node = _node(model_id=FAST)

        assert _choices(node, "duration") == [6, 10]
        assert _choices(node, "resolution") == ["720p", "768p", "1080p"]


class TestInputErrors:
    @pytest.mark.parametrize("resolution", ["720p", "768p", "1080p"])
    def test_fast_rejects_10s(self, resolution: str) -> None:
        node = _node(model_id=FAST, first_frame_image=FRAME, duration=10, resolution=resolution)

        assert _errors(node) == [f"Model {FAST} does not support a duration of 10s. Valid durations: 6s."]

    def test_unsupported_resolution_is_rejected(self) -> None:
        node = _node(duration=10, resolution="1080p")

        assert _errors(node) == [
            "Model MiniMax-Hailuo-2.3 does not support the combination of duration 10s and resolution 1080p. "
            "Valid resolutions for 10s: 768p."
        ]

    def test_saved_fast_720p_loads_and_passes(self) -> None:
        node = MinimaxHailuoVideoGeneration(name="Hailuo")
        for name, value in {
            "prompt": "a cat",
            "model_id": FAST,
            "resolution": "720p",
            "first_frame_image": FRAME,
        }.items():
            node.set_parameter_value(name, value, initial_setup=True)

        assert node.get_parameter_value("resolution") == "720p"
        assert _errors(node) == []

    def test_reported_before_run(self) -> None:
        node = _node(model_id=FAST)
        node.set_parameter_value("prompt", "")

        errors = [str(e) for e in node.validate_before_node_run() or []]

        assert "'prompt' is empty. Describe the video you want." in errors
        assert any("Connect one to 'First Frame Image'" in e for e in errors)

    @pytest.mark.asyncio
    async def test_build_payload_still_refuses(self) -> None:
        node = _node(model_id=FAST)

        with pytest.raises(ValueError, match="First Frame Image"):
            await node._build_payload()
