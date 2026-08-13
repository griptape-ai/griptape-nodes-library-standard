from __future__ import annotations

import pytest

from griptape_nodes_library.video.omnihuman_subject_detection import OmnihumanSubjectDetection
from griptape_nodes_library.video.omnihuman_subject_recognition import OmnihumanSubjectRecognition

NODE_CLASSES = [OmnihumanSubjectDetection, OmnihumanSubjectRecognition]


@pytest.mark.parametrize("node_class", NODE_CLASSES)
def test_validate_before_node_run_reports_missing_image(node_class: type) -> None:
    """An unset ``image_url`` must be reported as a missing input before the run
    starts, rather than surfacing as an AttributeError from the payload builder.
    """
    node = node_class(name="Subject")

    exceptions = node.validate_before_node_run()

    assert exceptions, "an unset image must block the run"
    message = str(exceptions[0])
    assert "requires an input image" in message
    assert "Subject" in message, "the message must name the node"
    assert "NoneType" not in message


@pytest.mark.parametrize("node_class", NODE_CLASSES)
def test_validate_before_node_run_passes_with_image(node_class: type) -> None:
    node = node_class(name="Subject")
    node.set_parameter_value("image_url", "https://public.example/source.jpg")

    assert node.validate_before_node_run() is None


@pytest.mark.parametrize("node_class", NODE_CLASSES)
@pytest.mark.asyncio
async def test_build_payload_raises_readable_error_without_image(node_class: type) -> None:
    """``_build_payload`` is the second line of defence: reached directly (e.g. via
    the auto-detect path), it must still name the missing parameter rather than
    calling ``.startswith`` on ``None``.
    """
    node = node_class(name="Subject")

    with pytest.raises(ValueError, match="requires an input image") as excinfo:
        await node._build_payload()

    assert "startswith" not in str(excinfo.value)


@pytest.mark.parametrize("node_class", NODE_CLASSES)
@pytest.mark.asyncio
async def test_build_payload_uses_public_url(node_class: type, monkeypatch: pytest.MonkeyPatch) -> None:
    node = node_class(name="Subject")
    node.set_parameter_value("image_url", "{inputs}/source.jpg")
    monkeypatch.setattr(
        node._public_image_url_parameter,
        "get_public_url_for_parameter",
        lambda: "https://public.example/source.jpg",
    )

    payload = await node._build_payload()

    assert payload["image_url"] == "https://public.example/source.jpg"
