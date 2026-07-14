"""Keep the `model_catalog` declarations in sync with the model lists the
library actually serves.

The catalog is the contract the library exposes to the platform (policy, key
support, UI grouping). The chat nodes carry their model lists statically in
Python, while the proxy base node resolves a runtime selection back to its
stable catalog key from the library registry. Either way the catalog and the
served models must agree, and these tests are the guard that keeps them from
drifting.

A local-runtime provider like Ollama is the exception: it declares no models
of its own, and the node that uses it (`OllamaPrompt`) references the whole
provider via `model_provider_usage` instead of a fixed `model_usage` list.
There is no static choice list to compare against there, so those tests check
the shape of that declaration and the provider metadata instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.node_library.library_declarations import find_model_catalog, resolve_node_models
from griptape_nodes.node_library.library_registry import LibrarySchema
from griptape_nodes.node_library.library_validation import validate_library_declarations

from griptape_nodes_library.config.image.griptape_cloud_image_driver import (
    MODEL_CHOICES as GRIPTAPE_CLOUD_IMAGE_MODEL_CHOICES,
)
from griptape_nodes_library.config.image.grok_image_driver import MODEL_CHOICES as GROK_IMAGE_MODEL_CHOICES
from griptape_nodes_library.config.image.openai_image_driver import MODEL_CHOICES as OPENAI_IMAGE_MODEL_CHOICES
from griptape_nodes_library.config.prompt.amazon_bedrock_prompt import MODEL_CHOICES as AMAZON_BEDROCK_MODEL_CHOICES
from griptape_nodes_library.config.prompt.anthropic_prompt import MODEL_CHOICES as ANTHROPIC_MODEL_CHOICES
from griptape_nodes_library.config.prompt.cloud_models import MODEL_CHOICES_ARGS
from griptape_nodes_library.config.prompt.cohere_prompt import MODEL_CHOICES as COHERE_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.grok_prompt import MODEL_CHOICES as GROK_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.groq_prompt import MODEL_CHOICES as GROQ_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.nim_prompt import MODEL_CHOICES as NIM_PROMPT_MODEL_CHOICES
from griptape_nodes_library.config.prompt.openai_prompt import MODEL_CHOICES as OPENAI_MODEL_CHOICES
from griptape_nodes_library.image.create_image import MODEL_CHOICES as GENERATE_IMAGE_MODEL_CHOICES
from griptape_nodes_library.image.describe_image import GTC_VISION_MODEL_CHOICES as DESCRIBE_IMAGE_MODEL_CHOICES

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


def _model_provider_usage_ids(library: dict[str, Any], class_name: str) -> list[str]:
    node = next(n for n in library["nodes"] if n["class_name"] == class_name)
    usage = next(d for d in node["metadata"]["declarations"] if d["type"] == "model_provider_usage")
    return usage["provider_ids"]


def _provider(library: dict[str, Any], provider_id: str) -> dict[str, Any]:
    catalog = next(d for d in library["metadata"]["declarations"] if d["type"] == "model_catalog")
    return catalog["providers"][provider_id]


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
        ("DescribeImage", DESCRIBE_IMAGE_MODEL_CHOICES),
        ("OpenAiPrompt", OPENAI_MODEL_CHOICES),
        ("AnthropicPrompt", ANTHROPIC_MODEL_CHOICES),
        ("GrokPrompt", GROK_PROMPT_MODEL_CHOICES),
        ("CoherePrompt", COHERE_PROMPT_MODEL_CHOICES),
        ("GroqPrompt", GROQ_PROMPT_MODEL_CHOICES),
        ("NimPrompt", NIM_PROMPT_MODEL_CHOICES),
        ("AmazonBedrockPrompt", AMAZON_BEDROCK_MODEL_CHOICES),
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


def test_ollama_provider_declares_no_models_and_requires_no_key() -> None:
    """The `ollama` provider documents the local-runtime, no-enumerated-models case.

    `OllamaPrompt` discovers its models dynamically from whatever the user has
    pulled into their local Ollama server, so the provider carries no static
    model list. `KeySupport` docs the provider-level `key_support` as the
    fallback for exactly this shape of provider, and `OllamaPrompt` requires no
    API key at runtime (unlike every other prompt config node), so the
    provider-level value must be `NO_KEY_REQUIRED`.
    """
    library = _load_library()
    provider = _provider(library, "ollama")

    assert provider["models"] == {}
    assert provider["key_support"] == "NO_KEY_REQUIRED"


def test_ollama_prompt_declares_provider_usage_for_ollama() -> None:
    """`OllamaPrompt` references the whole `ollama` provider, not specific models.

    A node that enumerates every model a provider offers at runtime (rather
    than a fixed set) declares `model_provider_usage`, not `model_usage`. Guard
    that `OllamaPrompt` uses that declaration shape and points at the `ollama`
    provider.
    """
    library = _load_library()

    assert _model_provider_usage_ids(library, "OllamaPrompt") == ["ollama"]


def test_ollama_provider_usage_resolves_to_no_catalog_models() -> None:
    """Resolving `OllamaPrompt`'s declared usage against the catalog yields nothing.

    `OllamaPrompt` builds its own model dropdown by querying the local Ollama
    server (`_get_available_models`) rather than reading the catalog.
    Declaring `model_provider_usage` against a provider with no enumerated
    models is how that split is represented: `resolve_node_models` -- the same
    function the engine calls to build a node's model list from the catalog --
    must resolve it to an empty list, confirming the catalog stays inert for
    this node instead of silently feeding it models it doesn't actually offer.
    """
    library = _load_library()
    schema = LibrarySchema.model_validate(library)
    catalog = find_model_catalog(schema.metadata.declarations)
    assert catalog is not None

    node = next(n for n in schema.nodes if n.class_name == "OllamaPrompt")
    resolved = resolve_node_models(catalog, node.metadata.declarations)

    assert resolved == []


def test_full_manifest_validates_against_engine_schema() -> None:
    """The manifest as a whole must satisfy the engine's own schema and cross-reference checks.

    `LibrarySchema.model_validate` enforces shape (discriminated unions,
    required fields); `validate_library_declarations` enforces cross-references
    (every `model_usage` / `model_provider_usage` entry resolves against the
    `model_catalog`, no duplicate model ids). This is the same path the engine
    takes when it loads the library, so a manifest that passes here loads
    cleanly at runtime too.
    """
    library = _load_library()

    schema = LibrarySchema.model_validate(library)
    problems = validate_library_declarations(schema)

    assert problems == []
