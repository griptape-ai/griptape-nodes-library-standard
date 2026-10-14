"""Tests for ``SplitVideo`` timecode parsing through the LLM."""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo

import griptape_nodes_library.video.split_video as split_video_module
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model, text_model
from griptape_nodes_library.video.split_video import SplitVideo

LLM_REPLY = "00:00:00:00-00:00:01:00|Intro:\n00:00:01:00-00:00:02:00|Segment 2:"


@pytest.fixture
def node(monkeypatch: pytest.MonkeyPatch) -> SplitVideo:
    monkeypatch.setattr(split_video_module, "resolve_cloud_api_key", lambda: "key")
    return SplitVideo(name="split")


def test_timecodes_are_parsed_from_model_reply(node: SplitVideo) -> None:
    prompts: list[str] = []

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        request = messages[-1]
        assert isinstance(request, ModelRequest)
        prompts.extend(str(p.content) for p in request.parts if isinstance(p, UserPromptPart))
        return ModelResponse(parts=[TextPart(LLM_REPLY)])

    with override_model(fake_model(respond)):
        segments = node._parse_timecodes("intro 0-1s, second 1-2s", 24.0, drop_frame=False)

    assert "intro 0-1s, second 1-2s" in prompts[0]
    assert [s.title for s in segments] == ["Intro:", "Segment 2:"]


def test_missing_credential_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(split_video_module, "resolve_cloud_api_key", lambda: "")
    node = SplitVideo(name="split")

    with pytest.raises(ValueError, match="Griptape"):
        node._parse_timecodes_with_agent("0-1s")


def test_empty_reply_is_reported(node: SplitVideo) -> None:
    with override_model(text_model("")), pytest.raises(ValueError, match="Agent failed to parse timecodes"):
        node._parse_timecodes_with_agent("0-1s")
