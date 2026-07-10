"""Keep the `model_catalog` declarations in sync with the model lists the
library actually serves.

The catalog is the contract the library exposes to the platform (policy, key
support, UI grouping). The chat nodes carry their model lists statically in
Python, while the proxy base node resolves a runtime selection back to its
stable catalog key from the library registry. Either way the catalog and the
served models must agree, and these tests are the guard that keeps them from
drifting.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from griptape_nodes_library.config.image.griptape_cloud_image_driver import (
    MODEL_CHOICES as GRIPTAPE_CLOUD_IMAGE_MODEL_CHOICES,
)
from griptape_nodes_library.config.image.grok_image_driver import MODEL_CHOICES as GROK_IMAGE_MODEL_CHOICES
from griptape_nodes_library.config.image.openai_image_driver import MODEL_CHOICES as OPENAI_IMAGE_MODEL_CHOICES
from griptape_nodes_library.config.prompt.cloud_models import MODEL_CHOICES_ARGS
from griptape_nodes_library.config.prompt.cohere_prompt import MODEL_CHOICES as COHERE_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.grok_prompt import MODEL_CHOICES as GROK_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.groq_prompt import MODEL_CHOICES as GROQ_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.nim_prompt import MODEL_CHOICES as NIM_PROMPT_MODEL_CHOICES
from griptape_nodes_library.image.create_image import MODEL_CHOICES as GENERATE_IMAGE_MODEL_CHOICES

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


def _nodes_with_model_usage(library: dict[str, Any]) -> list[str]:
    """Class names of every node that declares `model_usage`."""
    return [
        node["class_name"]
        for node in library["nodes"]
        if any(d.get("type") == "model_usage" for d in node.get("metadata", {}).get("declarations", []))
    ]


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


@pytest.mark.parametrize(
    ("class_name", "expected_provider_model_ids"),
    [
        ("GenerateImage", GENERATE_IMAGE_MODEL_CHOICES),
        ("GriptapeCloudImage", GRIPTAPE_CLOUD_IMAGE_MODEL_CHOICES),
        ("OpenAiImage", OPENAI_IMAGE_MODEL_CHOICES),
        ("GrokImage", GROK_IMAGE_MODEL_CHOICES),
        ("GrokPrompt", GROK_PROMPT_MODEL_CHOICES),
        ("CoherePrompt", COHERE_PROMPT_MODEL_CHOICES),
        ("GroqPrompt", GROQ_PROMPT_MODEL_CHOICES),
        ("NimPrompt", NIM_PROMPT_MODEL_CHOICES),
    ],
)
def test_model_selection_node_usage_matches_static_choices(
    class_name: str, expected_provider_model_ids: list[str]
) -> None:
    """The model list in Python and the models the node declares must agree.

    Each node's `model_usage` ids resolve (through the catalog) to the same
    ordered provider model ids the node's `MODEL_CHOICES` constant serves. A
    mismatch means the manifest and the code drifted and one side needs updating.
    """
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    declared = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, class_name)]

    assert declared == expected_provider_model_ids


def test_declared_models_resolve_uniquely_per_node() -> None:
    """Each node's declared models map one-to-one to provider model ids.

    The proxy base node resolves a selected provider model id back to its
    stable catalog key by matching within the node's own declared models
    (`griptape_nodes_library.utils.model_invocation.resolve_catalog_model_id`).
    That match is unambiguous only when a node does not declare two catalog
    ids that share a `provider_model_id`; a duplicate would make the runtime
    resolver fail closed. Guard the manifest against introducing one.
    """
    library = _load_library()
    provider_model_id_by_catalog_id = _provider_model_id_by_catalog_id(library)

    duplicates: dict[str, list[str]] = {}
    for class_name in _nodes_with_model_usage(library):
        wire_ids = [provider_model_id_by_catalog_id[model_id] for model_id in _model_usage_ids(library, class_name)]
        repeated = sorted({wire_id for wire_id in wire_ids if wire_ids.count(wire_id) > 1})
        if repeated:
            duplicates[class_name] = repeated

    assert not duplicates, f"nodes declare catalog ids that share a provider_model_id: {duplicates}"
