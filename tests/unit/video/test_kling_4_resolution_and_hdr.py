"""Kling 4.0's 4k resolution and HDR option, on the shared helpers and on each Kling node.

Kling accepts ``settings.resolution: "4k"`` for Kling 4.0 (not Flash) from 2026-11-19, and
``settings.dynamic_range: "hdr"`` at 1080p and 4k only, for an extra per-second fee. SDR is Kling's
default, so only an HDR request sends the field.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.traits.options import Options

from griptape_nodes_library.video import kling_4
from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.kling_text_to_video_generation import KlingTextToVideoGeneration

NODE_CLASSES = [KlingTextToVideoGeneration, KlingImageToVideoGeneration, KlingOmniVideoGeneration]


def _set(node: BaseNode, name: str, value: Any) -> None:
    node.set_parameter_value(name, value)
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    node.after_value_set(parameter, value)


def _is_hidden(node: BaseNode, name: str) -> bool:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    return bool(parameter.ui_options.get("hide"))


def _choices(node: BaseNode, name: str) -> list[str]:
    parameter = node.get_parameter_by_name(name)
    assert parameter is not None
    (options,) = parameter.find_elements_by_type(Options)
    return list(options.choices)


class TestHelpers:
    def test_4k_is_a_kling_4_0_resolution_not_a_flash_one(self) -> None:
        assert kling_4.RESOLUTION_4K in kling_4.KLING_4_CAPABILITIES[kling_4.KLING_4_0].resolutions
        assert kling_4.KLING_4_CAPABILITIES[kling_4.KLING_4_0_FLASH].resolutions == [kling_4.RESOLUTION_720P]

    def test_4k_is_a_valid_resolution_but_the_default_stays_1080p(self) -> None:
        assert kling_4.resolution_error(kling_4.KLING_4_0, "4k") is None
        assert kling_4.resolution_error(kling_4.KLING_4_0_FLASH, "4k") is not None
        assert kling_4.KLING_4_CAPABILITIES[kling_4.KLING_4_0].default_resolution == kling_4.RESOLUTION_1080P

    def test_flash_falls_back_to_720p_when_a_4k_choice_carries_over(self) -> None:
        assert kling_4.resolution_choice(kling_4.KLING_4_0_FLASH, "4k") == kling_4.RESOLUTION_720P
        assert kling_4.resolution_choice(kling_4.KLING_4_0, "4k") == "4k"

    @pytest.mark.parametrize(
        ("model_id", "resolution", "expected"),
        [
            (kling_4.KLING_4_0, "720p", False),
            (kling_4.KLING_4_0, "1080p", True),
            (kling_4.KLING_4_0, "4k", True),
            (kling_4.KLING_4_0_FLASH, "720p", False),
        ],
    )
    def test_hdr_is_offered_at_1080p_and_4k_on_kling_4_0_only(
        self, model_id: str, resolution: str, expected: bool
    ) -> None:
        assert kling_4.supports_hdr(model_id, resolution) is expected

    def test_dynamic_range_choice_keeps_hdr_only_where_it_is_offered(self) -> None:
        assert kling_4.dynamic_range_choice(kling_4.KLING_4_0, "1080p", "hdr") == "hdr"
        assert kling_4.dynamic_range_choice(kling_4.KLING_4_0, "4k", "hdr") == "hdr"
        assert kling_4.dynamic_range_choice(kling_4.KLING_4_0, "720p", "hdr") == "sdr"
        assert kling_4.dynamic_range_choice(kling_4.KLING_4_0_FLASH, "720p", "hdr") == "sdr"
        assert kling_4.dynamic_range_choice(kling_4.KLING_4_0, "4k", "sdr") == "sdr"

    def test_settings_send_dynamic_range_only_for_hdr(self) -> None:
        hdr = kling_4.build_settings(resolution="4k", duration=5, dynamic_range="hdr")
        assert hdr == {"resolution": "4k", "duration": 5, "dynamic_range": "hdr"}
        for sdr in ("sdr", None, ""):
            settings = kling_4.build_settings(resolution="4k", duration=5, dynamic_range=sdr)
            assert "dynamic_range" not in settings

    @pytest.mark.parametrize(
        ("model_id", "resolution", "dynamic_range", "fragment"),
        [
            (kling_4.KLING_4_0, "720p", "hdr", "supports HDR only at 1080p, 4k"),
            (kling_4.KLING_4_0_FLASH, "720p", "hdr", "does not support HDR"),
            (kling_4.KLING_4_0, "4k", "hdr10", "Unknown dynamic range"),
        ],
    )
    def test_validation_rejects_hdr_where_it_is_not_offered(
        self, model_id: str, resolution: str, dynamic_range: str, fragment: str
    ) -> None:
        errors = kling_4.validate_common(
            model_id, prompt="a ball", resolution=resolution, duration=5, dynamic_range=dynamic_range
        )
        assert any(fragment in str(error) for error in errors)

    @pytest.mark.parametrize("dynamic_range", [None, "", "sdr", "hdr"])
    def test_validation_accepts_sdr_and_hdr_at_4k(self, dynamic_range: str | None) -> None:
        errors = kling_4.validate_common(
            kling_4.KLING_4_0, prompt="a ball", resolution="4k", duration=5, dynamic_range=dynamic_range
        )
        assert errors == []


@pytest.mark.parametrize("node_class", NODE_CLASSES)
class TestNodes:
    def _node(self, node_class: type[BaseNode]) -> BaseNode:
        node = node_class(name="KlingNode")
        _set(node, "model_name", kling_4.KLING_4_0)
        return node

    def test_resolution_offers_4k_for_kling_4_0_and_only_720p_for_flash(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        assert _choices(node, "resolution") == ["720p", "1080p", "4k"]

        _set(node, "model_name", kling_4.KLING_4_0_FLASH)
        assert _choices(node, "resolution") == ["720p"]

    def test_dynamic_range_shows_only_where_hdr_is_offered(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        assert node.get_parameter_value("resolution") == "1080p"
        assert not _is_hidden(node, "dynamic_range")

        _set(node, "resolution", "720p")
        assert _is_hidden(node, "dynamic_range")

        _set(node, "resolution", "4k")
        assert not _is_hidden(node, "dynamic_range")

    def test_dynamic_range_resets_to_sdr_when_hdr_stops_being_offered(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        _set(node, "dynamic_range", "hdr")
        assert node.get_parameter_value("dynamic_range") == "hdr"

        _set(node, "resolution", "720p")

        assert node.get_parameter_value("dynamic_range") == "sdr"

    def test_dynamic_range_resets_and_hides_for_flash(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        _set(node, "dynamic_range", "hdr")

        _set(node, "model_name", kling_4.KLING_4_0_FLASH)

        assert node.get_parameter_value("dynamic_range") == "sdr"
        assert _is_hidden(node, "dynamic_range")

    def test_dynamic_range_is_hidden_for_models_before_kling_4(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        _set(node, "dynamic_range", "hdr")

        _set(node, "model_name", "kling-v3" if node_class is not KlingOmniVideoGeneration else "kling-v3-omni")

        assert _is_hidden(node, "dynamic_range")
        assert node.get_parameter_value("dynamic_range") == "sdr"

    def test_hdr_at_720p_through_a_connected_input_fails_validation(self, node_class: type[BaseNode]) -> None:
        node = self._node(node_class)
        node.set_parameter_value("prompt", "a red ball")
        node.set_parameter_value("resolution", "720p")
        node.set_parameter_value("dynamic_range", "hdr")

        errors = node.validate_before_node_run()

        assert errors is not None
        assert any("supports HDR only at" in str(error) for error in errors)
