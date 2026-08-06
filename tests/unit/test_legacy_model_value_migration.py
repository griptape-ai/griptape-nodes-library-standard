"""Regression coverage for `deprecated_values` tables losing entries during migration.

An audit found that several nodes' pre-existing deprecated-model tables (e.g.
`DEPRECATED_MODELS`, `MODEL_NAME_MAP`) were dropped, or dropped entries, when
their model dropdowns were migrated to store catalog keys via
`ModelAccessComponent`'s `deprecated_values=`. A saved workflow holding one of
those older values would silently snap to the dropdown's first choice on load
instead of migrating to its intended replacement.

These tests cover two things:

- Every node type that installs a model-access dropdown with a
  `deprecated_values` table: every legacy key it declares migrates, on
  assignment, to a canonical value that is actually one of the dropdown's
  offered choices.
- A fixed, real-world set of historical values (the ones this audit found
  missing) migrate correctly, checked against a small explicit table rather
  than each node's live one, so a future refactor that drops them again fails
  this test loudly instead of shrinking the set it's checked against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
import requests
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import LibraryRegistry

if TYPE_CHECKING:
    from griptape_nodes.exe_types.core_types import Parameter
    from griptape_nodes.exe_types.node_types import BaseNode

LIBRARY_NAME = "Griptape Nodes Library"

# Every node type that installs a model-access dropdown with a `deprecated_values`
# table, discovered by constructing each node type declared in the library and
# checking for one. `TopazImageEnhance` is deliberately excluded: its `operation`
# dropdown is a plain switch (denoise / enhance / sharpen), not a license-gated
# model choice, so it never installs `ModelAccessComponent` (see
# `tests/unit/test_proxy_model_access.py::test_topaz_image_enhance_operation_has_no_model_access`).
NODE_TYPES_WITH_MODEL_ACCESS: list[str] = [
    "Agent",
    "AnthropicPrompt",
    "Askulator",
    "CoherePrompt",
    "DateAndTime",
    "DescribeImage",
    "ElevenLabsTextToSpeechGeneration",
    "EvaluateTextResult",
    "Flux2ImageGeneration",
    "FluxImageGeneration",
    "GenerateImage",
    "GoogleImageGeneration",
    "GriptapeCloudImage",
    "GriptapeCloudPrompt",
    "GrokImage",
    "GrokImageEdit",
    "GrokImageGeneration",
    "GrokPrompt",
    "GrokVideoEdit",
    "GrokVideoGeneration",
    "GroqPrompt",
    "KlingImageToVideoGeneration",
    "KlingOmniVideoGeneration",
    "KlingTextToVideoGeneration",
    "LTXAudioToVideoGeneration",
    "LTXImageToVideoGeneration",
    "LTXTextToVideoGeneration",
    "LTXVideoExtend",
    "LTXVideoRetake",
    "LTXVideoToVideoHDR",
    "MinimaxHailuoVideoGeneration",
    "NimPrompt",
    "OmnihumanSubjectDetection",
    "OmnihumanSubjectRecognition",
    "OmnihumanVideoGeneration",
    "OpenAiImage",
    "OpenAiImageGeneration",
    "QwenImageEdit",
    "QwenImageGeneration",
    "ScrapeWeb",
    "SearchWeb",
    "Seedance20VideoGeneration",
    "SeedanceVideoGeneration",
    "SeedreamImageGeneration",
    "SoraVideoGeneration",
    "SummarizeText",
    "TranscribeAudio",
    "Veo3VideoGeneration",
    "WanAnimateGeneration",
    "WanImageGeneration",
    "WanImageToVideoGeneration",
    "WanReferenceToVideoGeneration",
    "WanTextToVideoGeneration",
    "WorldLabsWorldGeneration",
]

# A fixed set of historical values this audit found missing from the nodes'
# migrated `deprecated_values` tables, keyed by node type. Checked separately
# from `NODE_TYPES_WITH_MODEL_ACCESS` (which reads each node's live table) so
# dropping one of these again fails this test rather than silently shrinking
# the set the parametrized test above checks against.
EXPLICIT_LEGACY_VALUES: dict[str, dict[str, str]] = {
    "AnthropicPrompt": {
        "claude-3-5-sonnet-20241022": "claude-sonnet-4-6",
        "claude-3-5-sonnet-20240620": "claude-sonnet-4-6",
        "claude-3-5-haiku-20241022": "claude-haiku-4-5",
        "claude-3-opus-20240229": "claude-opus-4-7",
        "claude-3-sonnet-20240229": "claude-sonnet-4-6",
        "claude-3-haiku-20240307": "claude-haiku-4-5",
        "claude-3-7-sonnet-20250219": "claude-sonnet-4-6",
        "claude-sonnet-4-20250514": "claude-sonnet-4-6",
        "claude-opus-4-20250514": "claude-opus-4-7",
        "claude-3-7-sonnet-latest": "claude-sonnet-4-6",
        "claude-3-5-sonnet-latest": "claude-sonnet-4-6",
        "claude-3-5-opus-latest": "claude-opus-4-7",
        "claude-3-5-haiku-latest": "claude-haiku-4-5",
        "claude-haiku-4-5-20251001": "claude-haiku-4-5",
        "claude-opus-4-6": "claude-opus-4-7",
    },
    "GriptapeCloudPrompt": {
        "claude-3-7-sonnet": "claude-sonnet-4-6",
        "claude-3-5-haiku": "claude-haiku-4-5",
        "claude-sonnet-4-20250514": "claude-sonnet-4-6",
        "amazon.titan-text-premier-v1": "claude-sonnet-4-6",
        "gpt-4.5-preview": "gpt-4.1",
        "o1-mini": "o3-mini",
        "gemini-2.0-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-preview-05-20": "gemini-2.5-flash",
        "gemini-2.5-pro-preview-06-05": "gemini-2.5-pro",
        "gemini-3-pro": "gemini-3.1-pro",
        "gemini-3-pro-preview": "gemini-3.1-pro",
    },
    "GriptapeCloudImage": {
        "dall-e-3": "gpt-image-1-mini",
    },
    "GrokImageGeneration": {
        "Grok 2 Image": "grok-imagine-image",
        "grok-2-image-1212": "grok-imagine-image",
    },
    "GrokImageEdit": {
        "Grok 2 Image": "grok-imagine-image",
        "grok-2-image-1212": "grok-imagine-image",
    },
    "SeedreamImageGeneration": {
        "Seedream 3.0 T2I": "seedream-5-0-260128",
        "seedream-3-0-t2i-250415": "seedream-5-0-260128",
        "Seedream 3.0 I2I": "seedream-4-0-250828",
        "seededit-3-0-i2i-250628": "seedream-4-0-250828",
        "seedream-4.5": "seedream-4-5-251128",
    },
    "GoogleImageGeneration": {
        "nano-banana-3-pro": "gemini-3-pro-image",
    },
}


def _create_node(node_type: str) -> BaseNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    The engine's model-access query reads those two metadata keys to resolve a
    node's declared models; a bare `NodeClass(name=...)` construction does not
    set them and would leave the dropdown undecorated. Mirrors
    `tests/unit/test_proxy_model_access.py::_create_node`.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


class _FakeCloudModelsResponse:
    """Stand-in for the `requests.Response` `GriptapeCloudPrompt._list_models` reads."""

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, list[dict[str, Any]]]:
        return {"models": [{"model_name": "gpt-4.1-mini", "default": True}]}


@pytest.fixture(autouse=True)
def _stub_griptape_cloud_model_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """`GriptapeCloudPrompt.__init__` calls `_list_models`, which hits Griptape Cloud's
    API over HTTP; stub `requests.get` so constructing the node never depends on network
    access. The library loader reimports each node's module from its file path, so
    patching the `GriptapeCloudPrompt` class imported here would miss the module
    instance the library actually registers; `requests.get` is shared regardless.
    """
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: _FakeCloudModelsResponse())  # noqa: ARG005


def _model_access_component(node: BaseNode) -> ModelAccessComponent:
    """Resolve the node's `ModelAccessComponent`, however it holds one.

    Config-driver and proxy nodes wrap it in a `ModelDropdownAccess` (reached
    via `.component`); `Agent` and a handful of task/control nodes assign the
    `ModelAccessComponent` to `node._model_access` directly. Cast to `Any` since
    `_model_access` isn't part of `BaseNode`'s declared surface, and its type
    differs across the node's many possible base classes.
    """
    access = cast("Any", node)._model_access
    return access.component if hasattr(access, "component") else access


def _offered_choice_names(parameter: Parameter) -> set[str]:
    """The dropdown's currently offered rows, i.e. `model_choices`, not `deprecated_values`.

    `ModelAccessComponent._build_ui_options` builds `ui_options["data"]` from
    `model_choices` alone, so this is the set a migrated legacy value must land in.
    """
    return {row["name"] for row in parameter.ui_options["data"]}


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_every_legacy_value_migrates_to_an_offered_choice(node_type: str) -> None:
    node = _create_node(node_type)
    component = _model_access_component(node)
    assert component is not None, f"{node_type} does not install a model-access dropdown"

    deprecated_values = cast("Any", component)._deprecated_values
    assert deprecated_values, f"{node_type} declares no deprecated_values to check"

    parameter = cast("Any", component)._parameter
    offered_names = _offered_choice_names(parameter)

    for legacy_value, canonical_value in deprecated_values.items():
        node.set_parameter_value(parameter.name, legacy_value)
        migrated = node.get_parameter_value(parameter.name)
        assert migrated == canonical_value, (
            f"{node_type}: legacy value {legacy_value!r} migrated to {migrated!r}, expected {canonical_value!r}"
        )
        assert canonical_value in offered_names, (
            f"{node_type}: migration target {canonical_value!r} for legacy value {legacy_value!r} "
            "is not one of the dropdown's offered choices"
        )


@pytest.mark.parametrize("node_type", sorted(EXPLICIT_LEGACY_VALUES))
def test_specific_historical_values_migrate(node_type: str) -> None:
    """Guards the exact set of legacy values this audit found dropped during migration.

    Checked against the hardcoded `EXPLICIT_LEGACY_VALUES` above rather than
    each node's live `deprecated_values` table (as
    `test_every_legacy_value_migrates_to_an_offered_choice` does), so removing
    one of these entries from a node's table fails this test instead of just
    shrinking what the parametrized test iterates over.
    """
    node = _create_node(node_type)
    component = _model_access_component(node)
    assert component is not None, f"{node_type} does not install a model-access dropdown"

    parameter = cast("Any", component)._parameter
    offered_names = _offered_choice_names(parameter)

    for legacy_value, canonical_value in EXPLICIT_LEGACY_VALUES[node_type].items():
        node.set_parameter_value(parameter.name, legacy_value)
        migrated = node.get_parameter_value(parameter.name)
        assert migrated == canonical_value, (
            f"{node_type}: legacy value {legacy_value!r} migrated to {migrated!r}, expected {canonical_value!r}"
        )
        assert canonical_value in offered_names
