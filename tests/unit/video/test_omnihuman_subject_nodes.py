from __future__ import annotations

import json

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
    assert "An input image is required" in message
    assert "'Image URL'" in message
    assert "Subject" not in message, "the editor names the node, so the message must not"
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

    with pytest.raises(ValueError, match="An input image is required") as excinfo:
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


_DETECTION_RESP_DATA = {
    "status": 1,
    "object_detection_result": {"mask": {"url": ["https://public.example/mask-0.png"]}},
}


@pytest.mark.parametrize(
    "result_json",
    [
        pytest.param({"data": {"status": "done", "resp_data": json.dumps(_DETECTION_RESP_DATA)}}, id="unwrapped"),
        pytest.param(
            {"provider_response": {"data": {"status": "done", "resp_data": json.dumps(_DETECTION_RESP_DATA)}}},
            id="provider_response",
        ),
        pytest.param(
            {
                "provider_response": json.dumps(
                    {"data": {"status": "done", "resp_data": json.dumps(_DETECTION_RESP_DATA)}}
                )
            },
            id="provider_response_string",
        ),
        pytest.param({"data": {"status": "done", "resp_data": _DETECTION_RESP_DATA}}, id="resp_data_dict"),
    ],
)
@pytest.mark.asyncio
async def test_detection_parse_result_reads_masks(result_json: dict) -> None:
    node = OmnihumanSubjectDetection(name="Subject")

    await node._parse_result(result_json, "gen-1")

    assert node.parameter_output_values["contains_subject"] is True
    assert node.parameter_output_values["mask_image_urls"] == ["https://public.example/mask-0.png"]
    assert node.parameter_output_values["was_successful"] is True


@pytest.mark.parametrize(
    "result_json",
    [
        pytest.param({"data": {"status": "done"}}, id="no_resp_data"),
        pytest.param({"provider_response": {"data": {"status": "done"}}}, id="provider_response_no_resp_data"),
    ],
)
@pytest.mark.asyncio
async def test_detection_parse_result_without_resp_data_reports_no_subject(result_json: dict) -> None:
    node = OmnihumanSubjectDetection(name="Subject")

    await node._parse_result(result_json, "gen-1")

    assert node.parameter_output_values["contains_subject"] is False
    assert node.parameter_output_values["mask_image_urls"] == []
    assert node.parameter_output_values["was_successful"] is True


@pytest.mark.parametrize(
    "result_json",
    [
        pytest.param({"data": {"status": "expired"}}, id="unwrapped"),
        pytest.param({"provider_response": {"data": {"status": "not_found"}}}, id="provider_response"),
        pytest.param({}, id="empty"),
        pytest.param({"data": {"resp_data": json.dumps(_DETECTION_RESP_DATA)}}, id="missing_status"),
        pytest.param({"data": {"status": "", "resp_data": json.dumps(_DETECTION_RESP_DATA)}}, id="empty_status"),
    ],
)
@pytest.mark.asyncio
async def test_detection_parse_result_reports_unfinished_status(result_json: dict) -> None:
    node = OmnihumanSubjectDetection(name="Subject")

    await node._parse_result(result_json, "gen-1")

    assert node.parameter_output_values["was_successful"] is False
    assert "did not finish" in node.parameter_output_values["result_details"]
    assert node.parameter_output_values["mask_image_urls"] == []


@pytest.mark.asyncio
async def test_detection_parse_result_reports_unparseable_provider_response() -> None:
    node = OmnihumanSubjectDetection(name="Subject")

    await node._parse_result({"provider_response": "not json"}, "gen-1")

    assert node.parameter_output_values["was_successful"] is False
    assert "could not parse provider_response" in node.parameter_output_values["result_details"]
    assert node.parameter_output_values["mask_image_urls"] == []
