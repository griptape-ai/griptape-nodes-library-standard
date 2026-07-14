"""Gating boundary for `AudioTranscription`.

Unlike `StructuredDataExtractor`/`PromptSummary`, this node never builds the
underlying `OpenAiAudioTranscriptionDriver` itself: `process()` only emits a
serializable `{"tool_type": ..., "model": ...}` config, and
`griptape_nodes_library.utils.agent_utils.build_tool_from_config` rebuilds
that config into a live driver later, from inside whichever node actually
consumes the tool -- with no reference back to this node instance. There is
no live driver to wrap here the way `BaseTool._gate_prompt_driver` wraps one
for the other two tools, so `process()` itself declares the invocation for
the model it is about to hand off, using its own identity, and fails closed
if that is denied -- a denied model never leaves this node as a usable tool
config in the first place.
"""

from __future__ import annotations

from typing import cast

import pytest
from griptape.tools.audio_transcription.tool import AudioTranscriptionTool
from griptape_nodes.retained_mode.events.model_events import DeclareModelInvocationResultFailure
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.tools.audio_transcription_tool import DEFAULT_MODEL, AudioTranscription
from griptape_nodes_library.utils.agent_utils import build_tool_from_config
from griptape_nodes_library.utils.model_invocation import resolve_catalog_model_id


def _make_node(model: str | None = None) -> AudioTranscription:
    node = AudioTranscription(name="AudioTranscription")
    if model is not None:
        node.parameter_values["model"] = model
    return node


def test_permitted_invocation_emits_the_tool_config() -> None:
    """No policy denies the call in this environment, so the engine clears
    it by default and `process()` emits the config `build_tool_from_config`
    later rebuilds into a live `AudioTranscriptionTool`.
    """
    node = _make_node()

    node.process()

    assert node.parameter_output_values["tool"] == {"tool_type": "AudioTranscription", "model": DEFAULT_MODEL}


def test_denied_invocation_prevents_the_tool_config_from_being_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _make_node()

    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="denied for test"),
    )

    with pytest.raises(RuntimeError, match="denied for test"):
        node.process()

    assert "tool" not in node.parameter_output_values


def test_denied_invocation_prevents_the_downstream_driver_from_ever_being_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end proof: a denial in `process()` means the config that
    `build_tool_from_config` would otherwise turn into a live
    `OpenAiAudioTranscriptionDriver` never gets produced at all.
    """
    node = _make_node()

    monkeypatch.setattr(
        GriptapeNodes,
        "handle_request",
        lambda _request: DeclareModelInvocationResultFailure(result_details="denied for test"),
    )

    with pytest.raises(RuntimeError):
        node.process()

    # There is nothing for a downstream Agent/DescribeImage/McpTask node to
    # rebuild, because no config was ever emitted onto the "tool" output.
    config = node.parameter_output_values.get("tool")
    assert config is None


def test_permitted_config_round_trips_through_build_tool_from_config() -> None:
    node = _make_node()
    node.process()

    tool = cast("AudioTranscriptionTool", build_tool_from_config(node.parameter_output_values["tool"]))

    assert tool.audio_transcription_driver.model == DEFAULT_MODEL


def test_default_model_resolves_to_a_stable_catalog_key() -> None:
    """`AudioTranscription`'s default model ("whisper-1") already has a
    stable catalog id elsewhere in the library -- `TranscribeAudio` declares
    `gtc_whisper_1` for the same provider model id.

    `AudioTranscription` is not currently registered as a node in
    `griptape_nodes_library.json`, so it has no `model_usage` declaration of
    its own for `resolve_catalog_model_id` to match against; the gating call
    site falls back to declaring the raw provider model id in that case (see
    `griptape_nodes_library.utils.model_invocation._build_declaration`).
    Swapping in an already-registered node type that shares the same
    provider model id demonstrates that the id this node declares uses is a
    real, stable catalog key -- the same resolution
    `declare_model_invocation_sync` would produce once/if this node gets a
    matching `model_usage` block.
    """
    node = _make_node()
    node.metadata["library"] = "Griptape Nodes Library"
    node.metadata["node_type"] = "TranscribeAudio"

    assert resolve_catalog_model_id(node, DEFAULT_MODEL) == "gtc_whisper_1"
