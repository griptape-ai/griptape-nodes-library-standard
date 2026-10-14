from __future__ import annotations

from typing import Any


def ruleset_to_config(ruleset: Any) -> dict | None:
    """Normalize a ruleset value to its wire dict. Accepts dicts and legacy objects with `name`/`rules`."""
    if isinstance(ruleset, dict):
        if "name" not in ruleset:
            return None
        return {"name": ruleset["name"], "rules": [_rule_text(r) for r in ruleset.get("rules", [])]}
    name = getattr(ruleset, "name", None)
    rules = getattr(ruleset, "rules", None)
    if name is None or rules is None:
        return None
    return {"name": name, "rules": [str(getattr(r, "value", r)) for r in rules]}


def _rule_text(rule: Any) -> str:
    # Griptape `Rule.to_dict()` is `{"type": "Rule", "value": ...}`.
    if isinstance(rule, dict) and "value" in rule:
        return str(rule["value"])
    return str(rule)


def rulesets_from_inputs(values: list[Any]) -> list[dict]:
    """Normalize a node's `rulesets` list input.

    Plain strings become single-rule rulesets named `behavior_1`, `behavior_2`, ...
    Lists are flattened (a `RulesetList` output).
    """
    configs: list[dict] = []
    counter = 0
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if isinstance(item, str):
                if not item.strip():
                    continue
                counter += 1
                configs.append({"name": f"behavior_{counter}", "rules": [item.strip()]})
                continue
            config = ruleset_to_config(item)
            if config:
                configs.append(config)
    return configs


def render_rulesets(rulesets: list[dict]) -> str:
    if not rulesets:
        return ""
    lines = ["When responding, always use rules from the following rulesets.", ""]
    for ruleset in rulesets:
        name = ruleset.get("name", "")
        lines.append(f"Ruleset name: {name}")
        lines.append(f'"{name}" rules:')
        for index, rule in enumerate(ruleset.get("rules", []), start=1):
            lines.append(f"Rule #{index}")
            lines.append(str(rule))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
