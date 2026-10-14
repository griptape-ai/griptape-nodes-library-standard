"""Parameters of every node the pydantic-ai migration touched match main (d19c834).

Saved workflows address parameters by name and restore values into them, so names,
order, types, modes, serializability, and defaults must not drift. The snapshot was
captured on main; `model`-like defaults come from engine catalogs and are excluded.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest
from griptape_nodes.exe_types.core_types import BaseNodeElement, Parameter

SNAPSHOT = json.loads((Path(__file__).parent / "fixtures" / "main_parameter_shapes.json").read_text())
VOLATILE = {"model", "model_provider", "mcp_server_name", "message"}
FIELDS = (
    "type",
    "input_types",
    "output_type",
    "serializable",
    "settable",
    "mode_allowed_input",
    "mode_allowed_property",
    "mode_allowed_output",
    "parent_group_name",
)


@pytest.fixture(autouse=True)
def _offline_constructors(monkeypatch: pytest.MonkeyPatch) -> None:
    models = SimpleNamespace(data=[SimpleNamespace(id="gpt-4.1"), SimpleNamespace(id="gpt-4o")])
    monkeypatch.setattr(
        openai, "Client", lambda *_a, **_k: SimpleNamespace(models=SimpleNamespace(list=lambda: models))
    )
    monkeypatch.setattr(
        httpx,
        "get",
        lambda url, *_a, **_k: httpx.Response(
            200, json={"models": [], "buckets": []}, request=httpx.Request("GET", url)
        ),
    )


def _shape(element: BaseNodeElement, path: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in element.children:
        name = getattr(child, "name", None)
        if isinstance(child, Parameter):
            d = child.to_dict()
            row = {"name": name, "element": type(child).__name__, "path": path, **{k: d.get(k) for k in FIELDS}}
            if name not in VOLATILE:
                row["default_value"] = d.get("default_value")
            rows.append(json.loads(json.dumps(row, default=repr)))
        elif not str(name).startswith("BaseNodeElement_"):
            rows.append({"name": name, "element": type(child).__name__, "path": path})
        rows.extend(_shape(child, f"{path}/{name}"))
    return rows


@pytest.mark.parametrize("class_name", sorted(SNAPSHOT))
def test_parameters_match_main(class_name: str) -> None:
    entry = SNAPSHOT[class_name]
    module = importlib.import_module(entry["file"].removesuffix(".py").replace("/", "."))
    node = getattr(module, class_name)(name=class_name)
    assert _shape(node.root_ui_element) == entry["elements"]
