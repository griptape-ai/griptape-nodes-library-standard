"""Tests for FileOutputSettings."""

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.filesystem.file_output_settings import FileOutputSettings


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> FileOutputSettings:  # noqa: ARG001
    return FileOutputSettings(name="test_file_output_settings")


@pytest.mark.parametrize("parameter_name", ["filename", "macro"])
def test_macro_tokens_are_not_substituted_as_workflow_variables(node: FileOutputSettings, parameter_name: str) -> None:
    """Substituting before a run would resolve {outputs} and drop unset tokens like {sub_dirs?:/}."""
    parameter = node.get_parameter_by_name(parameter_name)

    assert parameter is not None
    assert parameter.allow_variable_substitution is False
