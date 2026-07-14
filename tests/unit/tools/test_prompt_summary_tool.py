"""Gating boundary for `PromptSummary`'s prompt driver.

`PromptSummary.process()` builds a `PromptSummaryEngine` around a caller-
supplied driver, which calls `driver.run(...)` directly whenever the
summarization tool is actually invoked by an agent -- well after `process()`
has already returned. `BaseTool._gate_prompt_driver` wraps that `run` so the
permission layer sees (and can deny) every real call at the moment it
happens, not just once back when the node ran.
"""

from __future__ import annotations

from typing import Any

import pytest
from griptape_nodes.retained_mode.events.model_events import DeclareModelInvocationResultFailure
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.tools.prompt_summary_tool import PromptSummary
from griptape_nodes_library.utils.model_invocation import resolve_catalog_model_id


class _FakeTokenizer:
    """Just enough of `BaseTokenizer` for the summary engine's chunker to build."""

    max_input_tokens = 100_000


class _RecordingDriver:
    """Minimal prompt-driver stand-in: records whether the real (unwrapped)
    call happened, without making any network call.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.tokenizer = _FakeTokenizer()
        self.call_count = 0

    def run(self, prompt_input: Any) -> str:
        self.call_count += 1
        return "ran"


def _make_node(driver: _RecordingDriver | None = None) -> tuple[PromptSummary, _RecordingDriver]:
    node = PromptSummary(name="PromptSummary")
    if driver is None:
        driver = _RecordingDriver(model="gpt-4o")
    node.parameter_values["prompt_driver"] = driver
    node.process()
    return node, driver


def test_missing_prompt_driver_raises_instead_of_producing_a_tool() -> None:
    node = PromptSummary(name="PromptSummary")

    with pytest.raises(ValueError, match="Prompt driver is required"):
        node.process()


def test_process_hands_the_gated_driver_to_the_summary_engine() -> None:
    node, driver = _make_node()
    tool = node.parameter_output_values["tool"]

    assert tool.prompt_summary_engine.prompt_driver is driver


def test_denied_invocation_blocks_the_real_driver_call(monkeypatch: pytest.MonkeyPatch) -> None:
    _node, driver = _make_node()

    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="denied for test"),
    )

    with pytest.raises(RuntimeError, match="denied for test"):
        driver.run("prompt stack")

    assert driver.call_count == 0


def test_permitted_invocation_reaches_the_real_driver_call() -> None:
    """No policy denies the call in this environment, so the engine clears it
    by default and the wrapped `run` falls through to the real driver call.
    """
    _node, driver = _make_node()

    result = driver.run("prompt stack")

    assert result == "ran"
    assert driver.call_count == 1


def test_gating_is_not_stacked_across_repeated_process_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """A driver connected in from an upstream node can be reused across runs.

    Re-running `process()` against the same driver instance must not wrap it
    in a second layer -- otherwise each real call would declare twice.
    """
    driver = _RecordingDriver(model="gpt-4o")
    node, _driver = _make_node(driver)
    node.process()  # re-gate the same already-gated driver

    declare_calls = 0
    real_handle_request = GriptapeNodes.handle_request

    def _counting_handle_request(request: Any) -> Any:
        nonlocal declare_calls
        declare_calls += 1
        return real_handle_request(request)

    monkeypatch.setattr(GriptapeNodes, "handle_request", _counting_handle_request)

    driver.run("prompt stack")

    assert declare_calls == 1
    assert driver.call_count == 1


def test_default_driver_model_resolves_to_a_stable_catalog_key() -> None:
    """See the equivalent test in `test_extraction_tool.py` for why the
    resolution is proven against an already-registered node type rather than
    `PromptSummary` itself: `PromptSummary` is not currently registered in
    `griptape_nodes_library.json`, so it has no `model_usage` declaration of
    its own. This proves the provider model id string a connected driver
    would carry (e.g. "gpt-4o") resolves to a real, stable catalog key.
    """
    node = PromptSummary(name="PromptSummary")
    node.metadata["library"] = "Griptape Nodes Library"
    node.metadata["node_type"] = "OpenAiPrompt"

    assert resolve_catalog_model_id(node, "gpt-4o") == "gtc_gpt_4o"
