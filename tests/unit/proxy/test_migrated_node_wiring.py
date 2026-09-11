"""Static check that every migrated node's `_save_generated_media` / `_load_generated_media`

call site names a parameter the node actually declares and an artifact kind that
actually exists.

Driving `_parse_result` end to end would need a provider-specific response shape
per node, so the smallest check that still catches a wrong `kind=` or a typo'd
`output_param` (both invisible to the rest of the suite, since they only ever
show up against a live provider) is a source-level one: every call is found by
walking the module's AST, and every literal it passes is checked, rather than
executed.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from griptape_nodes_library.proxy.hosted_artifacts import ArtifactKind

LIBRARY_ROOT = Path(__file__).resolve().parents[2].parent / "griptape_nodes_library"

# `_save_generated_media` / `_load_generated_media` calls with a literal `output_param`
# or `kind=` to check statically.
_CHECKED_METHODS = ("_save_generated_media", "_load_generated_media")


def _migrated_modules() -> list[Path]:
    """Every module under image/video/audio/three_d that calls the hosted-media helpers.

    Excludes rodin_2_3d_generation.py and world_labs_world_generation.py, which pair
    hosted artifacts to files positionally instead and have their own dedicated tests.
    """
    modules = []
    for sub in ("image", "video", "audio", "three_d"):
        for path in sorted((LIBRARY_ROOT / sub).glob("*.py")):
            text = path.read_text()
            if any(f"{method}(" in text for method in _CHECKED_METHODS):
                modules.append(path)
    return modules


def _declared_parameter_names(tree: ast.Module) -> set[str]:
    """Every `name="..."` passed to a `Parameter*(...)` call in the module."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        func = node.func
        func_name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if not func_name or not func_name.startswith("Parameter"):
            continue

        for keyword in node.keywords:
            if (
                keyword.arg == "name"
                and isinstance(keyword.value, ast.Constant)
                and isinstance(keyword.value.value, str)
            ):
                names.add(keyword.value.value)

    return names


@dataclass
class _HostedMediaCall:
    lineno: int
    method: str
    output_param: str | None
    kind_member: str | None
    kind_is_checkable: bool


def _hosted_media_calls(tree: ast.Module) -> list[_HostedMediaCall]:
    """Every `self._save_generated_media(...)` / `self._load_generated_media(...)` call."""
    calls: list[_HostedMediaCall] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        method = node.func.attr
        if method not in _CHECKED_METHODS:
            continue

        output_param: str | None = None
        if method == "_save_generated_media" and len(node.args) >= 2:
            second_arg = node.args[1]
            if isinstance(second_arg, ast.Constant) and isinstance(second_arg.value, str):
                output_param = second_arg.value

        kind_member: str | None = None
        kind_is_checkable = False
        for keyword in node.keywords:
            if keyword.arg != "kind":
                continue
            value = keyword.value
            if (
                isinstance(value, ast.Attribute)
                and isinstance(value.value, ast.Name)
                and value.value.id == "ArtifactKind"
            ):
                kind_is_checkable = True
                kind_member = value.attr

        calls.append(
            _HostedMediaCall(
                lineno=node.lineno,
                method=method,
                output_param=output_param,
                kind_member=kind_member,
                kind_is_checkable=kind_is_checkable,
            )
        )

    return calls


MIGRATED_MODULE_PATHS = _migrated_modules()


@pytest.mark.parametrize("module_path", MIGRATED_MODULE_PATHS, ids=[path.stem for path in MIGRATED_MODULE_PATHS])
def test_hosted_media_calls_name_real_parameters_and_kinds(module_path: Path) -> None:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    calls = _hosted_media_calls(tree)
    assert calls, f"{module_path.name} was selected as migrated but has no _save/_load_generated_media call"

    declared_params = _declared_parameter_names(tree)

    for call in calls:
        if call.output_param is not None:
            assert call.output_param in declared_params, (
                f"{module_path.name}:{call.lineno} calls {call.method} with output_param="
                f"{call.output_param!r}, which is not a parameter name declared in this module "
                f"(declared: {sorted(declared_params)})"
            )

        if call.kind_is_checkable:
            assert hasattr(ArtifactKind, call.kind_member or ""), (
                f"{module_path.name}:{call.lineno} calls {call.method} with kind=ArtifactKind."
                f"{call.kind_member}, which is not a real ArtifactKind member"
            )
