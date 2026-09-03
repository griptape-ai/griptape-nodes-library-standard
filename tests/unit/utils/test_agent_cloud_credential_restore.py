"""Griptape Cloud credential survival across the Agent wire format.

``GriptapeCloudPromptDriver.api_key`` is not serializable, so a chained agent's
credential does not survive ``to_dict()``/``from_dict()`` and is silently re-read from
``GT_CLOUD_API_KEY``. ``unwrap_agent`` re-resolves it (License first) before the caller
deserializes; these tests pin that behaviour and the failure modes it removes.
"""

from __future__ import annotations

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


def test_empty_credential_is_still_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """No credential resolves to ``""``, which must still be set explicitly.

    Leaving the key absent lets attrs evaluate its ``os.environ[...]`` default, which
    raises ``KeyError`` when nothing is set — the failure seen after deleting the API
    key. ``""`` defers the error to ``validate_before_workflow_run``, where the
    user-facing missing-credential message is produced.
    """
    _stub_resolved_credential(monkeypatch, "")

    agent_core_dict, _, _ = unwrap_agent(wrap_agent(_cloud_agent_dict(), [], []))

    driver = agent_core_dict["tasks"][0]["prompt_driver"]
    assert "api_key" in driver
    assert driver["api_key"] == ""


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


def test_legacy_raw_agent_dict_is_also_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Workflows saved before the wrapper format pass the raw griptape dict through."""
    _stub_resolved_credential(monkeypatch, _LICENSE)

    agent_core_dict, tool_configs, ruleset_configs = unwrap_agent(_cloud_agent_dict())

    assert agent_core_dict["tasks"][0]["prompt_driver"]["api_key"] == _LICENSE
    assert tool_configs == []
    assert ruleset_configs == []


def test_non_dict_and_taskless_input_do_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """unwrap_agent's existing tolerance for odd input is preserved."""
    _stub_resolved_credential(monkeypatch, _LICENSE)

    assert unwrap_agent(None) == ({}, [], [])  # type: ignore[arg-type]
    assert unwrap_agent({"agent": {}, "tools": []}) == ({}, [], [])
    # A task with no prompt_driver at all (e.g. a non-PromptTask first task).
    agent_core_dict, _, _ = unwrap_agent({"agent": {"tasks": [{"type": "ToolkitTask"}]}, "tools": []})
    assert agent_core_dict["tasks"][0] == {"type": "ToolkitTask"}


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
