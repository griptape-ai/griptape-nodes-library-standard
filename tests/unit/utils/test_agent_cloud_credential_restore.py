"""Griptape Cloud credential survival across the Agent wire format.

``api_key`` on every ``GriptapeCloud*`` driver is not serializable, so a chained agent's
credential does not survive ``to_dict()``/``from_dict()`` and is silently re-read from
``GT_CLOUD_API_KEY``. ``unwrap_agent`` re-resolves it (License first) before the caller
deserializes; these tests pin that behaviour and the failure modes it removes.
"""

from __future__ import annotations

import copy

import pytest

import griptape_nodes_library.utils.agent_utils as agent_utils
from griptape_nodes_library.utils.agent_utils import unwrap_agent, wrap_agent

_LICENSE = "header.payload.signature"
"""A Griptape Nodes License is a JWT: three dot-separated segments."""

_OTHER_ORG_KEY = "gt-key-for-a-different-org"
"""Stands in for the stale ``GT_CLOUD_API_KEY`` that produced the 402."""


def _stub_resolved_credential(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    """Stub the credential resolver, since License precedence is covered elsewhere."""
    monkeypatch.setattr(agent_utils, "resolve_cloud_api_key", lambda: value)


def _cloud_agent_dict(*, model: str = "gpt-4.1") -> dict:
    """A serialized agent whose prompt driver is a Griptape Cloud driver.

    Shaped like real ``Agent.to_dict()`` output: no ``api_key`` key at all, because
    griptape strips it.
    """
    return {
        "type": "Agent",
        "tasks": [
            {
                "type": "PromptTask",
                "prompt_driver": {"type": "GriptapeCloudPromptDriver", "model": model, "stream": True},
            }
        ],
    }


def _ollama_agent_dict() -> dict:
    """A serialized agent with no Griptape Cloud driver anywhere in it."""
    return {
        "type": "Agent",
        "tasks": [{"type": "PromptTask", "prompt_driver": {"type": "OllamaPromptDriver", "model": "llama3"}}],
    }


def test_cloud_driver_gets_the_resolved_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """The License-first credential lands on the driver dict before from_dict() runs."""
    _stub_resolved_credential(monkeypatch, _LICENSE)
    wrapper = wrap_agent(_cloud_agent_dict(), [], [])

    agent_core_dict, _, _ = unwrap_agent(wrapper)

    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE


def test_stale_env_key_is_overwritten_not_preferred(monkeypatch: pytest.MonkeyPatch) -> None:
    """The regression itself: a credential already on the wire loses to a fresh resolve.

    An upstream node running with a different credential (or an older library version)
    can leave a wrong key in the dict. Trusting it is what authenticated the downstream
    agent as the wrong org.
    """
    _stub_resolved_credential(monkeypatch, _LICENSE)
    agent_dict = _cloud_agent_dict()
    agent_dict["tasks"][0]["prompt_driver"]["api_key"] = _OTHER_ORG_KEY

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(agent_dict, [], []))

    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE


def test_missing_credential_is_reported_from_the_unwrap(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credential + a Cloud driver raises the user-facing message, not a bare 401.

    The unwrap is the only place that knows a Cloud driver is about to be rebuilt with
    nothing to authenticate it: a connected agent turns off the Agent node's own
    credential check, and several consumers of the wire format define none at all.
    """
    _stub_resolved_credential(monkeypatch, "")

    with pytest.raises(KeyError) as excinfo:
        unwrap_agent(wrap_agent(_cloud_agent_dict(), [], []))

    # Names both credentials, so a license-only user is not sent after an API key.
    message = str(excinfo.value)
    assert "license" in message.lower()
    assert "GT_CLOUD_API_KEY" in message


def test_missing_credential_does_not_raise_for_readers(monkeypatch: pytest.MonkeyPatch) -> None:
    """``require_credential=False`` keeps the read-only paths working with no credential.

    Displaying, clearing, and editing memory never send a request, so they must not fail
    for want of a credential they do not use. ``""`` is still injected explicitly:
    leaving the key absent lets attrs evaluate its ``os.environ[...]`` default and raise
    ``KeyError`` on rebuild.
    """
    _stub_resolved_credential(monkeypatch, "")

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(_cloud_agent_dict(), [], []), require_credential=False)

    driver = agent_core_dict["tasks"][0]["prompt_driver"]
    assert "api_key" in driver
    assert driver["api_key"] == ""


def test_missing_credential_does_not_raise_without_a_cloud_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent with no Cloud driver needs no Cloud credential.

    A BYOK or Ollama agent authenticates through its own provider config, so requiring
    a Griptape Cloud credential to unwrap it would block a chain that never touches
    Griptape Cloud.
    """
    _stub_resolved_credential(monkeypatch, "")

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(_ollama_agent_dict(), [], []))

    assert agent_core_dict["tasks"][0]["prompt_driver"] == {"type": "OllamaPromptDriver", "model": "llama3"}


def test_no_credential_is_resolved_when_no_cloud_driver_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The secrets lookup is skipped entirely when the walk finds no Cloud driver.

    ``unwrap_agent`` also runs on read-only, UI-frequency paths (the memory nodes), so a
    non-Cloud agent must not pay a credential resolve on every unwrap.
    """
    calls = 0

    def _counting_resolve() -> str:
        nonlocal calls
        calls += 1
        return _LICENSE

    monkeypatch.setattr(agent_utils, "resolve_cloud_api_key", _counting_resolve)

    unwrap_agent(wrap_agent(_ollama_agent_dict(), [], []))

    assert calls == 0


def test_no_cloud_driver_returns_the_input_dict_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No Cloud driver means no copy either — nothing needs repairing."""
    _stub_resolved_credential(monkeypatch, _LICENSE)
    wrapper = wrap_agent(_ollama_agent_dict(), [], [])

    agent_core_dict, _, _ = unwrap_agent(wrapper)

    assert agent_core_dict is wrapper["agent"]


def test_non_cloud_driver_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Cloud credential must never be written onto a BYOK provider's driver.

    Non-GTC drivers are repaired by ``restore_provider_driver`` from the wrapper's
    ``provider`` blob; injecting a Griptape Cloud credential here would hand the user's
    License to a third-party endpoint.
    """
    _stub_resolved_credential(monkeypatch, _LICENSE)
    agent_dict = {
        "type": "Agent",
        "tasks": [{"type": "PromptTask", "prompt_driver": {"type": "OpenAiChatPromptDriver", "model": "gpt-4.1"}}],
    }

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(agent_dict, [], []))

    assert "api_key" not in agent_core_dict["tasks"][0]["prompt_driver"]


def test_cloud_memory_driver_alone_still_triggers_the_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-Cloud prompt driver plus a Cloud memory driver is still a Cloud agent.

    The walk, not the prompt driver's type, decides whether a credential is needed.
    """
    _stub_resolved_credential(monkeypatch, _LICENSE)
    agent_dict = _ollama_agent_dict()
    agent_dict["conversation_memory"] = {
        "type": "ConversationMemory",
        "conversation_memory_driver": {"type": "GriptapeCloudConversationMemoryDriver", "alias": "thread-alias"},
    }

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(agent_dict, [], []))

    assert agent_core_dict["conversation_memory"]["conversation_memory_driver"]["api_key"] == _LICENSE
    assert "api_key" not in agent_core_dict["tasks"][0]["prompt_driver"]


def test_legacy_raw_agent_dict_is_also_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflows saved before the wrapper format pass the raw griptape dict through."""
    _stub_resolved_credential(monkeypatch, _LICENSE)

    agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(_cloud_agent_dict())

    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE
    assert tool_configs == []
    assert ruleset_configs == []


def test_non_dict_and_taskless_input_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Odd input is tolerated rather than raising."""
    _stub_resolved_credential(monkeypatch, _LICENSE)

    assert unwrap_agent(None) == ({}, [], [])  # type: ignore[arg-type]
    assert unwrap_agent({"agent": {}, "tools": []}) == ({}, [], [])
    # A task with no prompt_driver at all (e.g. a non-PromptTask first task).
    agent_core_dict, _, _ = unwrap_agent({"agent": {"tasks": [{"type": "ToolkitTask"}]}, "tools": []})
    assert agent_core_dict["tasks"][0] == {"type": "ToolkitTask"}


def test_caller_dict_is_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    """The credential must not be written back into the node's parameter value.

    ``unwrap_agent`` receives the upstream node's own ``agent`` parameter value, and a
    saved workflow pickles that value verbatim into the workflow file. Repairing it in
    place would persist a License JWT to disk, and would leak it into the wrapper the
    node re-emits downstream.
    """
    _stub_resolved_credential(monkeypatch, _LICENSE)
    wrapper = wrap_agent(_cloud_agent_dict(), [], [])
    before = copy.deepcopy(wrapper)

    agent_core_dict, _, _ = unwrap_agent(wrapper)

    assert wrapper == before
    assert "api_key" not in wrapper["agent"]["tasks"][0]["prompt_driver"]
    # ...while the returned copy does carry it.
    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE


def test_legacy_raw_dict_caller_is_not_mutated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same guarantee on the legacy raw-dict path."""
    _stub_resolved_credential(monkeypatch, _LICENSE)
    agent_dict = _cloud_agent_dict()
    before = copy.deepcopy(agent_dict)

    unwrap_agent(agent_dict)

    assert agent_dict == before


def test_nested_cloud_drivers_are_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every ``GriptapeCloud*`` driver is repaired, not just ``tasks[].prompt_driver``.

    They all declare ``api_key`` the same unserializable way, so a Cloud
    conversation-memory driver hits the identical 401/402/KeyError on rebuild.
    """
    _stub_resolved_credential(monkeypatch, _LICENSE)
    agent_dict = _cloud_agent_dict()
    agent_dict["conversation_memory"] = {
        "type": "ConversationMemory",
        "conversation_memory_driver": {"type": "GriptapeCloudConversationMemoryDriver", "alias": "thread-alias"},
    }

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(agent_dict, [], []))

    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE
    assert agent_core_dict["conversation_memory"]["conversation_memory_driver"]["api_key"] == _LICENSE


def test_round_trip_through_griptape_uses_restored_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end against real griptape: the rebuilt driver carries the License.

    Guards the actual bug rather than our dict manipulation — including that attrs never
    evaluates its environment-reading default, with ``GT_CLOUD_API_KEY`` deliberately
    pointing at another org.
    """
    from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
    from griptape.structures import Agent as GtAgent

    monkeypatch.setenv("GT_CLOUD_API_KEY", _OTHER_ORG_KEY)
    _stub_resolved_credential(monkeypatch, _LICENSE)

    upstream = GtAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key=_LICENSE, stream=True))
    wrapper = wrap_agent(upstream.to_dict(), [], [])

    agent_core_dict, _, _ = unwrap_agent(wrapper)
    rebuilt = GtAgent().from_dict(agent_core_dict)

    assert rebuilt.tasks[0].prompt_driver.api_key == _LICENSE


def test_round_trip_succeeds_with_no_env_key_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Deleting GT_CLOUD_API_KEY must no longer turn the 402 into a KeyError."""
    from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
    from griptape.structures import Agent as GtAgent

    monkeypatch.setenv("GT_CLOUD_API_KEY", "set-while-building-upstream")
    upstream = GtAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key=_LICENSE, stream=True))
    wrapper = wrap_agent(upstream.to_dict(), [], [])

    monkeypatch.delenv("GT_CLOUD_API_KEY")
    _stub_resolved_credential(monkeypatch, _LICENSE)

    rebuilt = GtAgent().from_dict(unwrap_agent(wrapper)[0])

    assert rebuilt.tasks[0].prompt_driver.api_key == _LICENSE


def test_reader_round_trip_survives_a_missing_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """A memory node can still rebuild the agent with no credential set anywhere.

    Against real griptape, with ``GT_CLOUD_API_KEY`` unset: the ``""`` injection is what
    keeps attrs from evaluating its environment default, so the read-only path neither
    raises nor needs a credential.
    """
    from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
    from griptape.structures import Agent as GtAgent

    monkeypatch.setenv("GT_CLOUD_API_KEY", "set-while-building-upstream")
    upstream = GtAgent(prompt_driver=GriptapeCloudPromptDriver(model="gpt-4.1", api_key=_LICENSE, stream=True))
    wrapper = wrap_agent(upstream.to_dict(), [], [])

    monkeypatch.delenv("GT_CLOUD_API_KEY")
    _stub_resolved_credential(monkeypatch, "")

    agent_core_dict, _, _ = unwrap_agent(wrapper, require_credential=False)
    rebuilt = GtAgent().from_dict(agent_core_dict)

    assert rebuilt.tasks[0].prompt_driver.api_key == ""
