"""Rewrite strings inside the pickled values of a saved workflow file.

Saved workflows store parameter values as `pickle.loads(b"...")` literals, so paths
captured when a fixture was generated cannot be swapped with plain text replacement.
"""

from __future__ import annotations

import ast
import pickle
from typing import Any

import attrs


def _replace(value: Any, replacements: dict[str, str]) -> Any:
    match value:
        case str():
            for old, new in replacements.items():
                value = value.replace(old, new)
            return value
        case dict():
            return {_replace(k, replacements): _replace(v, replacements) for k, v in value.items()}
        case list():
            return [_replace(v, replacements) for v in value]
        case tuple():
            return tuple(_replace(v, replacements) for v in value)
        case _ if attrs.has(type(value)):
            for field in attrs.fields(type(value)):
                current = getattr(value, field.name)
                updated = _replace(current, replacements)
                if updated is not current:
                    object.__setattr__(value, field.name, updated)
            return value
        case _:
            return value


def rewrite_pickled_strings(source: str, replacements: dict[str, str]) -> str:
    """Return `source` with every string inside its `pickle.loads(b"...")` literals rewritten."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    edits: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "loads"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pickle"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, bytes)
        ):
            continue
        arg = node.args[0]
        original = arg.value
        assert isinstance(original, bytes)
        if not any(old.encode() in original for old in replacements):
            continue
        updated = pickle.dumps(_replace(pickle.loads(original), replacements), protocol=4)  # noqa: S301
        start = offsets[arg.lineno - 1] + arg.col_offset
        end = offsets[arg.end_lineno - 1] + arg.end_col_offset  # type: ignore[operator]
        edits.append((start, end, repr(updated)))

    for start, end, text in sorted(edits, reverse=True):
        source = source[:start] + text + source[end:]
    return source
