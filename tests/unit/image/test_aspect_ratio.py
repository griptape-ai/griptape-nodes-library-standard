"""Tests for AspectRatio's presets."""

from __future__ import annotations

from math import gcd

import pytest

from griptape_nodes_library.image.aspect_ratio import ASPECT_RATIO_PRESETS, LEGACY_PRESET_NAMES, AspectRatio


@pytest.fixture
def node() -> AspectRatio:
    return AspectRatio(name="test_aspect_ratio")


def _ratio_presets() -> dict[str, tuple[int, int]]:
    return {
        name: (value[2], value[3])
        for name, value in ASPECT_RATIO_PRESETS.items()
        if value is not None and value[0] is None and value[2] is not None and value[3] is not None
    }


@pytest.mark.parametrize(("name", "ratio"), _ratio_presets().items())
def test_ratio_preset_label_matches_shape(name: str, ratio: tuple[int, int]) -> None:
    aspect_width, aspect_height = ratio
    if aspect_width > aspect_height:
        assert name.endswith(" landscape")
    elif aspect_width < aspect_height:
        assert name.endswith(" portrait")
    else:
        assert name.endswith(" square")


def test_ratio_presets_are_unique() -> None:
    reduced = [
        (aspect_width // gcd(aspect_width, aspect_height), aspect_height // gcd(aspect_width, aspect_height))
        for aspect_width, aspect_height in _ratio_presets().values()
    ]
    assert len(reduced) == len(set(reduced))


def test_legacy_names_point_at_current_presets() -> None:
    for legacy_name, current_name in LEGACY_PRESET_NAMES.items():
        assert legacy_name not in ASPECT_RATIO_PRESETS
        assert current_name in ASPECT_RATIO_PRESETS


@pytest.mark.parametrize(("legacy_name", "current_name"), LEGACY_PRESET_NAMES.items())
def test_saved_legacy_preset_loads_as_current(node: AspectRatio, legacy_name: str, current_name: str) -> None:
    node.set_parameter_value("preset", legacy_name, initial_setup=True)
    assert node.get_parameter_value("preset") == current_name


def test_selecting_portrait_preset_gives_portrait_dimensions(node: AspectRatio) -> None:
    node.set_parameter_value("preset", "3:4 portrait")
    assert node.get_parameter_value("width") == 768
    assert node.get_parameter_value("height") == 1024
