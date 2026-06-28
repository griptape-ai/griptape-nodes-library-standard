from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter

from griptape_nodes_library.video.kling_image_to_video_generation import KlingImageToVideoGeneration
from griptape_nodes_library.video.kling_omni_video_generation import KlingOmniVideoGeneration
from griptape_nodes_library.video.kling_text_to_video_generation import KlingTextToVideoGeneration


def _parameter_by_name(node: Any, parameter_name: str) -> Parameter:
    return next(parameter for parameter in node.parameters if parameter.name == parameter_name)


def _mode_choices(node: Any) -> list[str]:
    return list(_parameter_by_name(node, "mode").ui_options["simple_dropdown"])


def test_kling_text_v3_adds_4k_mode_choice() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")

    assert _mode_choices(node) == ["std", "pro", "4k"]

    node.set_parameter_value("mode", "4k")
    node.set_parameter_value("model_name", "Kling v2.6")

    assert _mode_choices(node) == ["pro"]
    assert node.get_parameter_value("mode") == "pro"


def test_kling_text_non_v3_models_do_not_accept_4k_mode() -> None:
    node = KlingTextToVideoGeneration(name="KlingText")
    node.set_parameter_value("model_name", "Kling v2 Master")
    node.set_parameter_value("mode", "4k")

    assert _mode_choices(node) == ["std", "pro"]
    assert node.get_parameter_value("mode") == "std"


def test_kling_image_v3_adds_4k_mode_choice() -> None:
    node = KlingImageToVideoGeneration(name="KlingImage")

    assert _mode_choices(node) == ["std", "pro", "4k"]

    node.set_parameter_value("mode", "4k")
    node.set_parameter_value("model_name", "Kling v2.6")

    assert _mode_choices(node) == ["pro"]
    assert node.get_parameter_value("mode") == "pro"


def test_kling_omni_4k_mode_requires_no_reference_video() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")

    assert _mode_choices(node) == ["std", "pro", "4k"]

    node.set_parameter_value("mode", "4k")
    node.set_parameter_value("reference_video", "https://example.com/reference.mp4")

    assert _mode_choices(node) == ["std", "pro"]
    assert node.get_parameter_value("mode") == "pro"

    node.set_parameter_value("reference_video", "")

    assert _mode_choices(node) == ["std", "pro", "4k"]


def test_kling_omni_base_model_does_not_offer_4k_mode() -> None:
    node = KlingOmniVideoGeneration(name="KlingOmni")
    node.set_parameter_value("model_name", "Kling Omni")
    node.set_parameter_value("mode", "4k")

    assert _mode_choices(node) == ["std", "pro"]
    assert node.get_parameter_value("mode") == "std"
