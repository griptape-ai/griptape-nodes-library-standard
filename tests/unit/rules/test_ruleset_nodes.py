"""Ruleset nodes emit plain wire dicts that `rulesets_from_inputs` accepts."""

from __future__ import annotations

from griptape_nodes_library.llm.rulesets import rulesets_from_inputs
from griptape_nodes_library.rules.create_ruleset import Ruleset
from griptape_nodes_library.rules.ruleset_list import RulesetList


def test_ruleset_node_splits_rules_on_blank_lines() -> None:
    node = Ruleset(name="ruleset")
    node.set_parameter_value("name", "Tone")
    node.set_parameter_value("rules", "Be kind\n\nBe brief")

    node.process()

    assert node.parameter_output_values["ruleset"] == {"name": "Tone", "rules": ["Be kind", "Be brief"]}


def test_ruleset_list_output_is_accepted_by_agent_inputs() -> None:
    node = RulesetList(name="rulesets")
    node.parameter_values["ruleset_1"] = {"name": "A", "rules": ["one"]}
    node.parameter_values["ruleset_4"] = {"name": "B", "rules": ["two"]}

    node.process()

    combined = node.parameter_output_values["rulesets"]
    assert rulesets_from_inputs([combined, "plain text"]) == [
        {"name": "A", "rules": ["one"]},
        {"name": "B", "rules": ["two"]},
        {"name": "behavior_1", "rules": ["plain text"]},
    ]
