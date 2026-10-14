"""Tool nodes emit serializable `{"tool_type": ...}` configs that `build_toolsets` can turn into toolsets."""

from __future__ import annotations

import pytest

from griptape_nodes_library.llm.tools import build_toolsets
from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.tools.calculator_tool import Calculator
from griptape_nodes_library.tools.date_time_tool import DateTime
from griptape_nodes_library.tools.tool_list import ToolList
from griptape_nodes_library.tools.web_scraper_tool import WebScraper


@pytest.mark.parametrize(
    ("node_class", "tool_type"),
    [(Calculator, "Calculator"), (DateTime, "DateTime"), (WebScraper, "WebScraper")],
)
def test_tool_node_emits_a_buildable_config(node_class: type[BaseTool], tool_type: str) -> None:
    node = node_class(name="tool")

    node.process()

    config = node.parameter_output_values["tool"]
    assert config["tool_type"] == tool_type
    assert len(build_toolsets([config])) == 1


def test_base_tool_emits_no_tool() -> None:
    node = BaseTool(name="tool")

    node.process()

    assert node.parameter_output_values["tool"] is None


def test_base_tool_keeps_off_prompt_parameter_for_saved_workflows() -> None:
    node = BaseTool(name="tool")

    param = node.get_parameter_by_name("off_prompt")

    assert param is not None
    assert param.default_value is False


def test_tool_list_collects_connected_tools() -> None:
    node = ToolList(name="tools")
    node.parameter_values["tool_1"] = {"tool_type": "Calculator"}
    node.parameter_values["tool_3"] = {"tool_type": "DateTime"}

    node.process()

    assert node.parameter_output_values["tool_list"] == [{"tool_type": "Calculator"}, {"tool_type": "DateTime"}]
