from __future__ import annotations

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.video.flux_3_video_edit import Flux3VideoEdit
from griptape_nodes_library.video.flux_3_video_generation import Flux3VideoGeneration, Flux3VideoMode
from griptape_nodes_library.video.flux_3_video_upscale import Flux3VideoUpscale


def test_flux_3_video_validation_rules() -> None:
    node = Flux3VideoGeneration(name="Flux3Video")
    node.set_parameter_value("prompt", "Ocean waves")
    node.set_parameter_value("draft", True)
    node.set_parameter_value("resolution", "fhd")

    errors = node.validate_before_node_run()
    assert errors is not None
    assert any("only supports 'hd'" in str(e) for e in errors)

    node.set_parameter_value("draft", False)
    node.set_parameter_value("resolution", "hd")
    node.set_parameter_value("mode", Flux3VideoMode.I2V.value)
    errors_i2v = node.validate_before_node_run()
    assert errors_i2v is not None
    assert any("Keyframes" in str(e) for e in errors_i2v)

    node.set_parameter_value("mode", Flux3VideoMode.V2V.value)
    node.set_parameter_value("duration", "20")
    errors_v2v = node.validate_before_node_run()
    assert errors_v2v is not None
    assert any("between 5 and 15" in str(e) for e in errors_v2v)
    assert any("Start Video" in str(e) for e in errors_v2v)


@pytest.mark.asyncio
async def test_flux_3_video_build_payload_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_prepare_image(image_input: object, **_kwargs: object) -> str:  # noqa: ARG001
        return "data:image/png;base64,FRAME"

    monkeypatch.setattr(
        "griptape_nodes_library.video.flux_3_video_generation.prepare_media_data_uri",
        fake_prepare_image,
    )

    node = Flux3VideoGeneration(name="Flux3Video")
    node.set_parameter_value("mode", Flux3VideoMode.T2V.value)
    node.set_parameter_value("prompt", "A lantern-lit alley")
    node.set_parameter_value("duration", "8")
    node.set_parameter_value("resolution", "hd")
    node.set_parameter_value("draft", True)

    t2v_payload = await node._build_payload()
    assert t2v_payload == {
        "mode": "t2v",
        "prompt": "A lantern-lit alley",
        "aspect_ratio": "auto",
        "duration": 8,
        "resolution": "hd",
        "generate_audio": True,
        "draft": True,
        "safety_tolerance": 2,
    }

    node.set_parameter_value("mode", Flux3VideoMode.I2V.value)
    node.set_parameter_value("keyframes", [ImageUrlArtifact("https://example.com/frame1.png")])
    i2v_payload = await node._build_payload()
    assert i2v_payload["mode"] == "i2v"
    assert i2v_payload["keyframes"] == "data:image/png;base64,FRAME"

    node.set_parameter_value("mode", Flux3VideoMode.V2V.value)
    node.set_parameter_value("start_video", VideoUrlArtifact("data:video/mp4;base64,CLIP"))
    v2v_payload = await node._build_payload()
    assert v2v_payload["mode"] == "v2v"
    assert v2v_payload["start_video"] == "data:video/mp4;base64,CLIP"


@pytest.mark.asyncio
async def test_flux_3_video_edit_validation_and_payload() -> None:
    node = Flux3VideoEdit(name="Flux3VideoEdit")
    node._probe_duration = lambda _v: 5.0  # type: ignore[method-assign]

    errors = node.validate_before_node_run()
    assert errors is not None
    assert len(errors) == 2

    node.set_parameter_value("prompt", "Add snowfall")
    node.set_parameter_value("video", "data:video/mp4;base64,CLIP")
    assert node.validate_before_node_run() is None

    payload = await node._build_payload()
    assert payload == {
        "video": "data:video/mp4;base64,CLIP",
        "prompt": "Add snowfall",
        "safety_tolerance": 2,
    }

    with pytest.raises(NodeError, match=r"requires \.mp4"):
        await node._prepare_video_data_uri("data:video/quicktime;base64,CLIP")


@pytest.mark.asyncio
async def test_flux_3_video_upscale_validation_and_payload() -> None:
    node = Flux3VideoUpscale(name="Flux3VideoUpscale")
    node._probe_duration = lambda _v: 5.0  # type: ignore[method-assign]

    errors = node.validate_before_node_run()
    assert errors is not None

    node.set_parameter_value("input_video", "data:video/mp4;base64,CLIP")
    node.set_parameter_value("upscale_factor", 1.5)
    node.set_parameter_value("creativity", "precise")
    node.set_parameter_value("prompt", "Sharp edges")
    assert node.validate_before_node_run() is None

    payload = await node._build_payload()
    assert payload == {
        "input_video": "data:video/mp4;base64,CLIP",
        "upscale_factor": 1.5,
        "creativity": 0,
        "prompt": "Sharp edges",
        "safety_tolerance": 2,
    }
