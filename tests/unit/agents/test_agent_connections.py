"""Tests for the Agent node's connection hooks."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.traits.options import Options

from griptape_nodes_library.agents.agent import Agent


def test_connect_disconnect_reconnect_prompt_model_config_round_trip(
    agent_node: Agent, prompt_model_config_source: DataNode
) -> None:
    """Grouping reroutes the connection: connect -> disconnect -> reconnect.

    The full round-trip must not raise, and the ``model`` parameter must end back
    in the connected state (no ``Options`` trait, type mirroring the source).
    """
    model_param = agent_node.get_parameter_by_name("model")
    assert model_param is not None
    # A fresh model parameter carries an Options dropdown.
    assert model_param.find_elements_by_type(Options)

    source_param = prompt_model_config_source.get_parameter_by_name("prompt_model_config")
    assert source_param is not None

    # 1) Connect: the Agent strips the Options trait so the param acts as a pure input.
    agent_node.after_incoming_connection(prompt_model_config_source, source_param, model_param)
    assert not model_param.find_elements_by_type(Options)
    assert model_param.type == "Prompt Model Config"

    # 2) Disconnect (first half of the group reroute): must restore the dropdown.
    agent_node.after_incoming_connection_removed(prompt_model_config_source, source_param, model_param)
    assert model_param.find_elements_by_type(Options)
    assert model_param.type == "str"

    # 3) Reconnect through the group proxy (second half of the reroute).
    agent_node.after_incoming_connection(prompt_model_config_source, source_param, model_param)
    assert not model_param.find_elements_by_type(Options)
    assert model_param.type == "Prompt Model Config"


def test_connect_when_options_trait_already_absent_does_not_raise(
    agent_node: Agent, prompt_model_config_source: DataNode
) -> None:
    """Defensive: after_incoming_connection must tolerate a model param with no
    Options trait instead of doing find_elements_by_type(Options)[0] and raising
    IndexError. Simulates the trait having been removed by some other/future path.
    """
    model_param = agent_node.get_parameter_by_name("model")
    assert model_param is not None
    source_param = prompt_model_config_source.get_parameter_by_name("prompt_model_config")
    assert source_param is not None

    # Put the param in the (unexpected) state where its Options trait is already gone.
    model_param.remove_trait(model_param.find_elements_by_type(Options)[0])
    assert not model_param.find_elements_by_type(Options)

    # Connecting must be a safe no-op on the missing trait, not an IndexError.
    agent_node.after_incoming_connection(prompt_model_config_source, source_param, model_param)
    assert not model_param.find_elements_by_type(Options)
    assert model_param.type == "Prompt Model Config"


@pytest.fixture
def prompt_model_config_source() -> DataNode:
    """A stand-in prompt driver node exposing a ``prompt_model_config`` output.

    The Agent's ``model`` connection hooks never read the source node itself, so a
    lightweight ``DataNode`` avoids the network calls a real prompt-driver node
    makes in ``__init__``.
    """
    source_node = DataNode(name="Griptape Cloud Prompt")
    source_node.add_parameter(
        Parameter(
            name="prompt_model_config",
            type="Prompt Model Config",
            output_type="Prompt Model Config",
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Prompt model configuration output.",
        )
    )
    return source_node
