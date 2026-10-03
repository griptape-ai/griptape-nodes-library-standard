"""Direct-provider budget gating and usage metering.

Only a call that goes straight to a provider is budget-checked and reported; a call through
the Griptape proxy is budgeted server-side and must not be counted twice.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape.drivers.image_generation.griptape_cloud import GriptapeCloudImageGenerationDriver
from griptape.drivers.prompt.anthropic import AnthropicPromptDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.drivers.prompt.grok import GrokPromptDriver
from griptape.drivers.prompt.ollama import OllamaPromptDriver
from griptape.drivers.prompt.openai import OpenAiChatPromptDriver
from griptape.events import EventBus, FinishPromptEvent
from griptape_nodes.exe_types.node_types import BaseNode

import griptape_nodes_library.utils.direct_provider_usage as usage_module
from griptape_nodes_library.utils.direct_provider_usage import (
    cost_micro_usd,
    is_direct_provider,
    provider_name,
    require_driver_access,
)


class _StubNode(BaseNode):
    def process(self) -> None:
        return


@pytest.fixture
def node() -> _StubNode:
    return _StubNode(name="StubNode")


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Record which gate ran and every usage report, without touching the engine."""
    seen: list[tuple[str, dict[str, Any]]] = []
    monkeypatch.setattr(
        usage_module,
        "require_model_invocation_sync",
        lambda _node, model, **kw: seen.append(("permission", {"model": model, **kw})),
    )

    def _access(_node: Any, model: str, **kw: Any) -> str:
        seen.append(("access", {"model": model, **kw}))
        return "corr-1"

    monkeypatch.setattr(usage_module, "require_model_access_sync", _access)
    monkeypatch.setattr(usage_module, "report_model_usage_sync", lambda _node, **kw: seen.append(("report", kw)))
    return seen


def _openai() -> OpenAiChatPromptDriver:
    return OpenAiChatPromptDriver(model="gpt-4o", api_key="k")


def _finish(model: str, input_tokens: int | None, output_tokens: int | None) -> None:
    EventBus.publish_event(
        FinishPromptEvent(model=model, result="", input_token_count=input_tokens, output_token_count=output_tokens)
    )


@pytest.mark.parametrize(
    ("driver", "direct"),
    [
        (GriptapeCloudPromptDriver(model="gpt-4o", api_key="k"), False),
        (GriptapeCloudImageGenerationDriver(model="gpt-image-1", api_key="k"), False),
        (OllamaPromptDriver(model="llama3"), False),
        (OpenAiChatPromptDriver(model="gpt-4o", api_key="k"), True),
        (AnthropicPromptDriver(model="claude-sonnet-4-6", api_key="k"), True),
    ],
)
def test_is_direct_provider(driver: Any, *, direct: bool) -> None:
    assert is_direct_provider(driver) is direct


def test_grok_is_not_mistaken_for_openai() -> None:
    """GrokPromptDriver subclasses OpenAiChatPromptDriver."""
    assert provider_name(GrokPromptDriver(model="grok-3-beta", api_key="k")) == "xai"


def test_cost_is_micro_usd_per_token() -> None:
    # gpt-4o: $2.50 / $10.00 per 1M tokens.
    assert cost_micro_usd("openai", "gpt-4o", 1000, 500) == 7500


def test_cost_is_none_for_an_unpriced_model() -> None:
    assert cost_micro_usd("openai", "my-finetune", 1000, 500) is None
    assert cost_micro_usd(None, "gpt-4o", 1000, 500) is None


def test_cloud_driver_gets_permission_only(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    meter = require_driver_access(node, GriptapeCloudPromptDriver(model="gpt-4o", api_key="k"))
    meter.run(lambda: _finish("gpt-4o", 1000, 500))

    assert [name for name, _ in calls] == ["permission"]


def test_direct_driver_is_checked_and_reported(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    meter = require_driver_access(node, _openai(), purpose="prompt enhancement")

    def _call() -> str:
        # Two prompt rounds (e.g. a tool call and the answer) are summed.
        _finish("gpt-4o", 600, 200)
        _finish("gpt-4o", 400, 300)
        return "done"

    assert meter.run(_call) == "done"

    assert calls == [
        ("access", {"model": "gpt-4o", "purpose": "prompt enhancement"}),
        (
            "report",
            {"declared_cost_micro_usd": 7500, "provider": "openai", "model": "gpt-4o", "correlation_id": "corr-1"},
        ),
    ]


def test_other_models_are_not_counted(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    """A tool may run its own sub-agent on another model; only this driver's calls are metered."""
    meter = require_driver_access(node, _openai())

    def _call() -> None:
        _finish("gpt-4o", 1000, 500)
        _finish("gpt-4.1", 99_999, 99_999)

    meter.run(_call)

    assert calls[-1][1]["declared_cost_micro_usd"] == 7500


def test_failed_call_still_reports_tokens_spent(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    meter = require_driver_access(node, _openai())

    def _call() -> None:
        _finish("gpt-4o", 1000, 500)
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError, match="tool failed"):
        meter.run(_call)

    assert calls[-1][0] == "report"


def test_unpriced_model_is_checked_but_not_reported(
    node: _StubNode, calls: list[tuple[str, dict[str, Any]]], caplog: pytest.LogCaptureFixture
) -> None:
    meter = require_driver_access(node, OpenAiChatPromptDriver(model="my-finetune", api_key="k"))

    with caplog.at_level("WARNING", logger="griptape_nodes"):
        meter.run(lambda: _finish("my-finetune", 1000, 500))

    assert [name for name, _ in calls] == ["access"]
    assert "no price for 'my-finetune'" in caplog.text


def test_call_without_token_usage_is_not_reported(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    meter = require_driver_access(node, _openai())

    meter.run(lambda: None)

    assert [name for name, _ in calls] == ["access"]


def test_listener_is_removed_after_the_call(node: _StubNode, calls: list[tuple[str, dict[str, Any]]]) -> None:
    before = list(EventBus.event_listeners)
    require_driver_access(node, _openai()).run(lambda: None)

    assert EventBus.event_listeners == before
