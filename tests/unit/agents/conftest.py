"""Shared fixtures for Agent node unit tests."""

from __future__ import annotations

import pytest

from griptape_nodes_library.agents.agent import Agent


@pytest.fixture
def agent_node() -> Agent:
    return Agent(name="Agent")
