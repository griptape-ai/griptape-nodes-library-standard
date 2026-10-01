"""Check that each template's workflow_shape header matches the values its Start Flow sets.

A Workflow Node builds its parameters from the `default_value` entries in the header and
pushes them into the subflow's Start Flow when it runs. A header default that differs from
the value the template sets on its Start Flow therefore replaces the author's value.

Engines before the fix for griptape-nodes-engine#5698 write the parameter's declared default
into the header rather than its set value, so re-saving a template on one of them brings the
mismatch back. The templates are read statically, so the result does not depend on the
installed engine version.
"""

import ast
import io
import json
import pickle
import re
import tomllib
from pathlib import Path
from typing import Any

import pytest

TEMPLATES_DIR = Path(__file__).parents[2] / "workflows" / "templates"
SHAPE_HEADER = re.compile(r"^# workflow_shape = (.*)$", re.MULTILINE)
UNIQUE_VALUES_NAME = "top_level_unique_values_dict"
CONTROL_TYPE = "parametercontroltype"


class _PrimitiveUnpickler(pickle.Unpickler):
    """Load only builtin values. Anything needing a class lookup is an object the header cannot hold."""

    def find_class(self, module: str, name: str) -> Any:
        msg = f"{module}.{name} is not a primitive value"
        raise pickle.UnpicklingError(msg)


class _NotPrimitive:
    """Marker for a stored value that is not plain JSON data."""


NOT_PRIMITIVE = _NotPrimitive()


def _templates_with_shape() -> list[Path]:
    return [
        path
        for path in sorted(TEMPLATES_DIR.glob("*.py"))
        if not path.name.startswith("__") and SHAPE_HEADER.search(path.read_text(encoding="utf-8"))
    ]


@pytest.mark.parametrize("template_path", _templates_with_shape(), ids=lambda path: path.stem)
def test_shape_input_defaults_match_start_flow_values(template_path: Path) -> None:
    source = template_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    shape = _read_shape(source)
    set_values = _read_set_input_values(tree)

    mismatches = []
    for node_name, parameters in shape["inputs"].items():
        for parameter_name, definition in parameters.items():
            if definition.get("type") == CONTROL_TYPE:
                continue
            key = (node_name, parameter_name)
            if key not in set_values:
                continue
            value = set_values[key]
            if not _is_json_value(value):
                continue
            if definition.get("default_value") != value:
                mismatches.append(
                    f"{node_name}.{parameter_name}: header has {definition.get('default_value')!r}, "
                    f"Start Flow sets {value!r}"
                )

    assert not mismatches, "workflow_shape defaults differ from the Start Flow values:\n" + "\n".join(mismatches)


def _read_shape(source: str) -> dict[str, Any]:
    match = SHAPE_HEADER.search(source)
    assert match is not None
    encoded = tomllib.loads(f"workflow_shape = {match.group(1)}")["workflow_shape"]
    return json.loads(encoded)


def _read_set_input_values(tree: ast.Module) -> dict[tuple[str, str], Any]:
    """Map (node name, parameter name) to the input value each SetParameterValueRequest stores."""
    unique_values = _read_unique_values(tree)
    node_names = _read_node_names(tree)

    set_values = {}
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "SetParameterValueRequest":
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        is_output = keywords.get("is_output")
        if isinstance(is_output, ast.Constant) and is_output.value:
            continue
        node_ref = keywords["node_name"]
        parameter = keywords["parameter_name"]
        value_ref = keywords["value"]
        if not isinstance(node_ref, ast.Name) or not isinstance(parameter, ast.Constant):
            continue
        if not isinstance(value_ref, ast.Subscript) or not isinstance(value_ref.slice, ast.Constant):
            continue
        value_key = value_ref.slice.value
        if not isinstance(value_key, str):
            continue
        set_values[(node_names[node_ref.id], parameter.value)] = unique_values[value_key]
    return set_values


def _read_unique_values(tree: ast.Module) -> dict[str, Any]:
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not isinstance(statement.value, ast.Dict):
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id != UNIQUE_VALUES_NAME:
            continue
        values = {}
        for key, value in zip(statement.value.keys, statement.value.values, strict=True):
            assert isinstance(key, ast.Constant)
            assert isinstance(value, ast.Call)
            payload = value.args[0]
            assert isinstance(payload, ast.Constant)
            assert isinstance(payload.value, bytes)
            values[key.value] = _unpickle_primitive(payload.value)
        return values
    return {}


def _read_node_names(tree: ast.Module) -> dict[str, str]:
    """Map each `nodeN_name` variable to the node name passed to its CreateNodeRequest."""
    names = {}
    for statement in ast.walk(tree):
        if not isinstance(statement, ast.Assign) or not isinstance(statement.targets[0], ast.Name):
            continue
        for call in ast.walk(statement.value):
            if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
                continue
            if call.func.id != "CreateNodeRequest":
                continue
            for keyword in call.keywords:
                if keyword.arg == "node_name" and isinstance(keyword.value, ast.Constant):
                    names[statement.targets[0].id] = keyword.value.value
    return names


def _unpickle_primitive(payload: bytes) -> Any:
    try:
        return _PrimitiveUnpickler(io.BytesIO(payload)).load()
    except pickle.UnpicklingError:
        return NOT_PRIMITIVE


def _is_json_value(value: Any) -> bool:
    if value is NOT_PRIMITIVE:
        return False
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        return False
    return True
