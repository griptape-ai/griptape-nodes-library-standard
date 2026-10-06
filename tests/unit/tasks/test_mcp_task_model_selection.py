"""Tests for `MCPTaskNode` model selection: wiring, model-config dispatch, and policy gates.

These tests cover the node's own wiring and the two gates around the selection.
The components' own behaviour is covered by the engine suite, and the shared
manifest invariants by `tests/unit/test_model_catalog_consistency.py` and
`tests/unit/test_legacy_model_value_migration.py`.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any, cast

import pytest
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.events.agent_events import ProviderConfig
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial, CheckpointFailure
from griptape_nodes.traits.button import Button
from griptape_nodes.traits.options import Options

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.tasks.mcp_task import DEFAULT_MODEL, MCPTaskNode

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

LIBRARY_NAME = "Griptape Nodes Library"


def _create_node() -> MCPTaskNode:
    """Create the node through the library so its metadata carries `library` / `node_type`.

    `ModelAccessComponent`'s policy query and `resolve_catalog_model_id` both read
    those two metadata keys to resolve the node's declared models; a bare
    `MCPTaskNode(name=...)` construction does not set them.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    node: BaseNode = library.create_node(node_type="MCPTaskNode", name="MCP Task")
    return cast("MCPTaskNode", node)


@pytest.fixture
def mcp_task_node(monkeypatch: pytest.MonkeyPatch) -> MCPTaskNode:
    monkeypatch.setenv("GT_CLOUD_API_KEY", "test-key")
    return _create_node()


def _denial(detail: str) -> CheckpointDenial:
    return CheckpointDenial(failures=(CheckpointFailure(detail=detail),))


def _node_module(node: MCPTaskNode) -> Any:
    """The module object the library actually registered this node's class from.

    The library loader reimports each node's module from its file path, so patching
    a name on the `griptape_nodes_library.tasks.mcp_task` imported here would miss
    the module instance the running node resolves its helpers through.
    """
    return sys.modules[type(node).__module__]


class TestDropdownWiring:
    def test_model_param_wired_to_model_access_component(self, mcp_task_node: MCPTaskNode) -> None:
        assert isinstance(mcp_task_node._model_access, ModelAccessComponent)

        model_param = mcp_task_node.get_parameter_by_name("model")
        assert model_param is not None
        # The component installs the Options dropdown and a refresh Button.
        assert model_param.find_elements_by_type(Options)
        assert model_param.find_elements_by_type(Button)
        # Per-row decoration the frontend uses to flag denied models.
        ui_options = model_param.ui_options
        assert ui_options.get("dropdown_row_icons") is True
        assert ui_options.get("dropdown_row_subtitles") is True
        assert ui_options["data"]

    def test_provider_param_offers_choices(self, mcp_task_node: MCPTaskNode) -> None:
        provider_param = mcp_task_node.get_parameter_by_name("model_provider")
        assert provider_param is not None
        assert provider_param.find_elements_by_type(Options)
        assert provider_param.find_elements_by_type(Button)

    def test_dropdown_offers_more_than_the_default(self, mcp_task_node: MCPTaskNode) -> None:
        """The dropdown must offer more than one model.

        A dropdown that offers only one model leaves a caller whose license denies
        that model with no alternative to select.
        """
        choices = mcp_task_node._model_access.model_choices
        assert DEFAULT_MODEL in choices
        assert len(choices) > 1


class TestModelConfigFollowsSelection:
    def test_selected_cloud_model_reaches_the_config(self, mcp_task_node: MCPTaskNode) -> None:
        mcp_task_node.set_parameter_value("model", "claude-opus-5")

        config = mcp_task_node._create_model_config()

        assert config.provider == ModelProvider.GRIPTAPE_CLOUD
        assert config.model == "claude-opus-5"

    def test_unrecognized_stored_model_falls_back_to_the_default(self, mcp_task_node: MCPTaskNode) -> None:
        """A value outside the catalog would resolve to no catalog key and fail closed.

        `ModelAccessComponent`'s `Options` trait accepts legacy values, so a stored
        selection can be one the dropdown no longer offers. Falling back keeps the
        node runnable instead of failing the license gate on an unresolvable id.
        """
        model_param = mcp_task_node.get_parameter_by_name("model")
        assert model_param is not None
        # Bypass the parameter's converters/Options, as a hand-edited saved workflow would.
        mcp_task_node.parameter_values["model"] = "some-retired-model"

        config = mcp_task_node._create_model_config()

        assert config.provider == ModelProvider.GRIPTAPE_CLOUD
        assert config.model == DEFAULT_MODEL

    def test_connected_model_config_is_used_as_is(self, mcp_task_node: MCPTaskNode) -> None:
        """A connected Prompt Model Config *is* the model, so the dropdown is bypassed."""
        connected = ModelConfig(provider=ModelProvider.OPENAI, model="gpt-4o")
        mcp_task_node.parameter_values["model"] = connected.to_wire()

        assert mcp_task_node._create_model_config() == connected

    def test_third_party_provider_builds_that_providers_config(
        self, mcp_task_node: MCPTaskNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-Cloud provider's model goes to that provider's own endpoint, not Cloud's."""
        provider = ProviderConfig(
            name="my-lmstudio",
            type="lmstudio",
            model="",
            base_url="http://localhost:1234/v1",
            api_key_secret_name="LMSTUDIO_KEY",  # noqa: S106
        )
        monkeypatch.setattr(mcp_task_node._provider_selection, "_fetch_providers", lambda: [provider])
        mcp_task_node.parameter_values["model_provider"] = "my-lmstudio"
        mcp_task_node.parameter_values["model"] = "local-model-7b"

        config = mcp_task_node._create_model_config()

        assert config.provider == ModelProvider.LMSTUDIO
        assert config.model == "local-model-7b"
        assert config.base_url == "http://localhost:1234/v1"
        assert config.api_key_secret == "LMSTUDIO_KEY"  # noqa: S105

    def test_connected_agent_supplies_its_own_model(self, mcp_task_node: MCPTaskNode) -> None:
        agent_model = ModelConfig(provider=ModelProvider.ANTHROPIC, model="claude-x")
        mcp_task_node.set_parameter_value("agent", AgentState(model=agent_model).to_wire())

        setup = mcp_task_node._setup_agent()

        assert setup is not None
        assert setup[1] == agent_model


class TestSelectionGate:
    def test_denied_selection_reports_failure_and_never_connects(
        self, mcp_task_node: MCPTaskNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A denied selection stops the node before the MCP server connection.

        `SuccessFailureNode` reports through its status parameters rather than
        raising, so the denial routes down the Failed control output with the
        policy's own reason attached.
        """
        mcp_task_node.set_parameter_value("prompt", "hello")
        monkeypatch.setattr(
            mcp_task_node._model_access,
            "selection_denial",
            lambda: _denial("Using models from this provider is not permitted under your license."),
        )

        toolset_builds: list[Any] = []
        monkeypatch.setattr(
            MCPTaskNode,
            "_build_toolsets",
            lambda self, state, config, name: toolset_builds.append((name, config)),  # noqa: ARG005
        )

        generator = mcp_task_node.process()
        assert generator is not None
        with pytest.raises(StopIteration):
            next(generator)

        assert toolset_builds == []
        assert mcp_task_node._execution_succeeded is False
        result_details = mcp_task_node.get_parameter_value("result_details")
        assert "not permitted under your license" in result_details

    def test_connected_agent_skips_the_selection_gate(
        self, mcp_task_node: MCPTaskNode, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A connected agent brings its own model, so the node's dropdown is stale.

        The parameter is hidden rather than cleared when an agent is connected, so
        gating on its leftover value would deny a run the connected agent's own
        model is permitted for. The INVOKE_MODEL gate still covers the real model.
        """
        mcp_task_node.set_parameter_value("prompt", "hello")
        mcp_task_node.set_parameter_value("agent", {"agent": {}, "tools": []})
        monkeypatch.setattr(mcp_task_node._model_access, "selection_denial", lambda: _denial("denied for test"))
        monkeypatch.setattr(_node_module(mcp_task_node), "get_server_config", lambda _name: {"transport": "stdio"})

        generator = mcp_task_node.process()
        assert generator is not None
        # Reaching the first yield -- the execution thunk -- means the selection gate did
        # not fire: it sits above the server-config lookup in `process`, and firing would
        # have returned before yielding anything. The thunk itself is left uncalled; it
        # would open a real MCP connection.
        assert callable(next(generator))

        result_details = mcp_task_node.get_parameter_value("result_details")
        assert "denied for test" not in result_details
