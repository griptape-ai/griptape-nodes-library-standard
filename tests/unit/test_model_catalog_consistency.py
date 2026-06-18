"""Keep the `model_catalog` declarations in sync with the model lists the
library actually serves.

Nodes do not read the catalog at runtime: the catalog is the contract the
library exposes to the platform (policy, key support, UI grouping), and the
node Python is the implementation. This test is the guard that keeps the two
from drifting, so the catalog stays trustworthy without any node depending on
the library registry to decide its behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from griptape_nodes_library.config.prompt.cloud_models import MODEL_CHOICES_ARGS

LIBRARY_JSON = Path(__file__).parents[2] / "griptape_nodes_library.json"


def _load_library() -> dict[str, Any]:
    return json.loads(LIBRARY_JSON.read_text())


def _provider_model_id_by_catalog_id(library: dict[str, Any]) -> dict[str, str]:
    """Map every catalog model id to its provider_model_id, across all providers."""
    catalog = next(d for d in library["metadata"]["declarations"] if d["type"] == "model_catalog")
    return {
        model_id: model["provider_model_id"]
        for provider in catalog["providers"].values()
        for model_id, model in provider["models"].items()
    }


def _model_usage_ids(library: dict[str, Any], class_name: str) -> list[str]:
    node = next(n for n in library["nodes"] if n["class_name"] == class_name)
    usage = next(d for d in node["metadata"]["declarations"] if d["type"] == "model_usage")
    return usage["model_ids"]


@pytest.mark.parametrize("class_name", ["Agent", "GriptapeCloudPrompt"])
def test_chat_node_model_usage_matches_static_choices(class_name: str) -> None:
    """The chat model list in Python and the models the node declares must agree.

    Each node's `model_usage` ids resolve (through the catalog) to the same
    ordered provider model ids the node serves from `MODEL_CHOICES_ARGS`. A
    mismatch means the manifest and the code drifted and one side needs updating.
    """
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    declared = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, class_name)]
    served = [model["name"] for model in MODEL_CHOICES_ARGS]

    assert declared == served
