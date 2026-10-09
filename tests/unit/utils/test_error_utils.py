from __future__ import annotations

import json
from typing import Any

import pytest
import requests
from griptape.artifacts import ErrorArtifact, TextArtifact
from griptape_nodes.exe_types.core_types import NodeError

from griptape_nodes_library.utils.error_utils import try_throw_error


def _http_error(
    status: int, *, json_body: dict[str, Any] | None = None, text: str = "", request_id: str | None = None
) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    if request_id is not None:
        response.headers["x-request-id"] = request_id
    if json_body is not None:
        response._content = json.dumps(json_body).encode()
    else:
        response._content = text.encode()
    return requests.HTTPError(f"{status} Client Error", response=response)


class _SdkError(Exception):
    """Shaped like an OpenAI or Anthropic SDK error: no readable response, a parsed body."""

    def __init__(self, body: dict[str, Any]) -> None:
        super().__init__("Error code: 400")
        self.status_code = 400
        self.request_id = "req_123"
        self.body = body


def test_non_error_output_does_not_raise() -> None:
    try_throw_error(TextArtifact("fine"))


def test_provider_message_becomes_the_message_and_body_is_attached() -> None:
    body = {"error": {"message": "Invalid model 'gpt-9'.", "type": "invalid_request_error", "param": "model"}}
    exc = _http_error(400, json_body=body, request_id="req_abc")

    with pytest.raises(NodeError) as raised:
        try_throw_error(ErrorArtifact("400 Client Error", exception=exc))

    assert str(raised.value) == "Invalid model 'gpt-9'."
    assert raised.value.response == body
    assert raised.value.fields == {"status_code": 400, "parameter": "model", "request_id": "req_abc"}
    assert raised.value.__cause__ is exc


def test_sdk_error_body_and_request_id_are_used() -> None:
    exc = _SdkError({"message": "Prompt is too long.", "code": "context_length_exceeded"})

    with pytest.raises(NodeError) as raised:
        try_throw_error(ErrorArtifact("Error code: 400", exception=exc))

    assert str(raised.value) == "Prompt is too long."
    assert raised.value.fields == {
        "status_code": 400,
        "error_code": "context_length_exceeded",
        "request_id": "req_123",
    }


def test_non_json_body_falls_back_to_its_text() -> None:
    exc = _http_error(502, text="Bad Gateway")

    with pytest.raises(NodeError, match="Agent run failed: Bad Gateway") as raised:
        try_throw_error(ErrorArtifact("502 Server Error", exception=exc))

    assert raised.value.response is None
    assert raised.value.fields == {"status_code": 502}


def test_error_without_exception_uses_the_artifact_value() -> None:
    with pytest.raises(NodeError, match="Agent run failed: something broke"):
        try_throw_error(ErrorArtifact("something broke"))


_OPENAI_BODY = {
    "error": {
        "message": "Unsupported value: 'temperature' does not support 0.1 with this model.",
        "type": "invalid_request_error",
        "param": "temperature",
        "code": "unsupported_value",
    }
}


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(json.dumps({"error": f"Error code: 400 - {_OPENAI_BODY!r}"}), id="json-error-string"),
        pytest.param(json.dumps({"detail": f"Error code: 400 - {_OPENAI_BODY!r}"}), id="json-detail-string"),
        pytest.param(f"Error code: 400 - {_OPENAI_BODY!r}", id="plain-text"),
        pytest.param(json.dumps({"error": f'"Error code: 400 - {_OPENAI_BODY!r}"'}), id="quoted"),
    ],
)
def test_provider_body_griptape_cloud_passed_on_as_text_is_unwrapped(text: str) -> None:
    exc = _http_error(400, text=text)

    with pytest.raises(NodeError) as raised:
        try_throw_error(ErrorArtifact(str(exc), exception=exc))

    assert str(raised.value) == "Unsupported value: 'temperature' does not support 0.1 with this model."
    assert raised.value.response == _OPENAI_BODY
    assert raised.value.fields == {"status_code": 400, "error_code": "unsupported_value", "parameter": "temperature"}


def test_node_error_on_the_artifact_is_raised_as_it_is() -> None:
    original = NodeError("The stream failed.", response={"event": "error"})

    with pytest.raises(NodeError) as raised:
        try_throw_error(ErrorArtifact("The stream failed.", exception=original))

    assert raised.value is original
