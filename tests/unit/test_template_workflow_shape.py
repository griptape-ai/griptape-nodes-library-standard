import ast
import json
import pickle
import tomllib
from pathlib import Path
from typing import Any

import pytest

TEMPLATES_DIR = Path(__file__).parents[2] / "workflows" / "templates"
START_FLOW_NODE_TYPE = "StartFlow"


def _read_header(source: str) -> dict[str, Any]:
    """Parse the `# /// script` header block into a dict."""
    lines = source.splitlines()
    start = lines.index("# /// script") + 1
    end = lines.index("# ///", start)
    toml_text = "\n".join(line.removeprefix("#").removeprefix(" ") for line in lines[start:end])
    return tomllib.loads(toml_text)


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((kw.value for kw in call.keywords if kw.arg == name), None)


def _str_constant(expr: ast.expr | None) -> str:
    assert isinstance(expr, ast.Constant)
    assert isinstance(expr.value, str)
    return expr.value


def _calls_named(tree: ast.AST, name: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name
    ]


def _unique_values(tree: ast.Module) -> dict[str, Any]:
    """Evaluate the module's `top_level_unique_values_dict` of pickled literals."""
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "top_level_unique_values_dict"
            and isinstance(stmt.value, ast.Dict)
        ):
            values = {}
            for key, value in zip(stmt.value.keys, stmt.value.values, strict=True):
                assert isinstance(value, ast.Call)
                values[_str_constant(key)] = pickle.loads(ast.literal_eval(value.args[0]))  # noqa: S301
            return values
    return {}


def _start_flow_set_values(tree: ast.Module) -> dict[str, dict[str, Any]]:
    """Map each Start Flow node's name to the parameter values the template sets on it."""
    node_names_by_var: dict[str, str] = {}
    for stmt in ast.walk(tree):
        if not (isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)):
            continue
        for call in _calls_named(stmt.value, "CreateNodeRequest"):
            node_type = _keyword(call, "node_type")
            if isinstance(node_type, ast.Constant) and node_type.value == START_FLOW_NODE_TYPE:
                node_names_by_var[stmt.targets[0].id] = _str_constant(_keyword(call, "node_name"))

    unique_values = _unique_values(tree)
    set_values: dict[str, dict[str, Any]] = {name: {} for name in node_names_by_var.values()}
    for call in _calls_named(tree, "SetParameterValueRequest"):
        match _keyword(call, "node_name"):
            case ast.Name(id=node_var) if node_var in node_names_by_var:
                node_name = node_names_by_var[node_var]
            case ast.Constant(value=node_name) if node_name in set_values:
                pass
            case _:
                continue
        parameter_name = _str_constant(_keyword(call, "parameter_name"))
        value = _keyword(call, "value")
        assert isinstance(value, ast.Subscript)
        set_values[node_name][parameter_name] = unique_values[_str_constant(value.slice)]
    return set_values


def _templates_with_shape() -> list[Path]:
    return [
        path
        for path in sorted(TEMPLATES_DIR.glob("*.py"))
        if not path.name.startswith("__")
        and "workflow_shape" in _read_header(path.read_text())["tool"]["griptape-nodes"]
    ]


def test_templates_with_workflow_shape_are_found() -> None:
    assert _templates_with_shape(), f"No templates with a workflow_shape found in {TEMPLATES_DIR}"


@pytest.mark.parametrize("template_path", _templates_with_shape(), ids=lambda path: path.name)
def test_workflow_shape_defaults_match_start_flow_values(template_path: Path) -> None:
    """A Workflow node loading a template takes its input defaults from the header's workflow_shape."""
    source = template_path.read_text()
    shape = json.loads(_read_header(source)["tool"]["griptape-nodes"]["workflow_shape"])
    set_values = _start_flow_set_values(ast.parse(source))
    assert set_values, f"No {START_FLOW_NODE_TYPE} node found in {template_path.name}"

    mismatches = {
        f"{node_name}.{parameter_name}": {
            "shape_default": shape["inputs"][node_name][parameter_name]["default_value"],
            "start_flow_value": value,
        }
        for node_name, values in set_values.items()
        for parameter_name, value in values.items()
        if shape["inputs"][node_name][parameter_name]["default_value"] != value
    }

    assert not mismatches
