"""Every model dropdown stores a provider model id the node actually declares.

`ModelAccessComponent` resolves a dropdown value to the catalog `model_id` that
license policy gates on by matching it against the `provider_model_id` of the
node's declared catalog models. A choice that matches none of them resolves to
nothing, which fails closed at run time: the artist gets "license policy could
not be evaluated" on a model that is really just misdeclared.

These are invariants rather than a pinned snapshot of every value, so adding a
model to a dropdown does not require updating this file. The exact per-node
values are covered by `test_legacy_model_value_migration.py`, which walks each
node's `deprecated_values` table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
from griptape_nodes.node_library.library_registry import get_declared_models
from test_legacy_model_value_migration import (
    NODE_TYPES_WITH_MODEL_ACCESS,
    _create_node,
    _model_access_component,
    _offered_choice_names,
)

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

MANIFEST_PATH = Path(__file__).resolve().parents[2] / "griptape_nodes_library.json"

# Node types that can't be constructed in the unit-test environment, so the
# all-nodes sweep below skips them. Neither installs a model-access dropdown.
_UNCONSTRUCTIBLE_NODE_TYPES = frozenset({"OpenAiPrompt", "SubflowWorkflowNode"})


def _declared_provider_model_ids(node: BaseNode) -> set[str]:
    """The provider model ids of every catalog model this node's manifest entry declares."""
    return {
        resolved.model.provider_model_id
        for resolved in get_declared_models(node)
        if resolved.model.provider_model_id is not None
    }


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_every_offered_choice_is_a_declared_provider_model_id(node_type: str) -> None:
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    declared = _declared_provider_model_ids(node)
    assert declared, f"{node_type} declares no catalog models, so its dropdown cannot be gated"

    for choice in _offered_choice_names(parameter):
        assert choice in declared, (
            f"{node_type}: dropdown offers {choice!r}, which is not the provider_model_id of any "
            "catalog model this node declares. It would fail closed at run time."
        )


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_default_value_is_an_offered_choice(node_type: str) -> None:
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    offered = _offered_choice_names(parameter)
    assert parameter.default_value in offered, (
        f"{node_type}: default_value {parameter.default_value!r} is not one of the offered choices {sorted(offered)}"
    )


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_no_choice_is_a_catalog_key(node_type: str) -> None:
    """A catalog key left in a dropdown is the specific regression this reversal undoes.

    Catalog keys and provider ids are disjoint across this library's catalog, so
    a choice matching a declared `model_id` means that node was missed.
    """
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    declared_catalog_keys = {resolved.model_id for resolved in get_declared_models(node)}
    leaked = sorted(_offered_choice_names(parameter) & declared_catalog_keys)
    assert not leaked, f"{node_type}: dropdown still offers catalog key(s) {leaked} instead of provider model ids"


def _all_manifest_node_types() -> list[str]:
    manifest = json.loads(MANIFEST_PATH.read_text())
    return sorted(
        node["class_name"] for node in manifest["nodes"] if node["class_name"] not in _UNCONSTRUCTIBLE_NODE_TYPES
    )


def test_every_model_dropdown_offers_only_declared_models() -> None:
    """Sweep every node in the manifest, not just `NODE_TYPES_WITH_MODEL_ACCESS`.

    That list is curated for nodes with a `deprecated_values` table, so a new node
    with no legacy values is not on it, and the per-node checks above never see it.
    Such a node can ship a dropdown whose choice matches no declared catalog model,
    and every run of it fails closed. This catches that whatever the node's history.
    """
    problems: list[str] = []
    for node_type in _all_manifest_node_types():
        node = _create_node(node_type)
        component = getattr(node, "_model_access", None)
        if component is None:
            continue
        declared = _declared_provider_model_ids(node)
        parameter = cast("Any", component)._parameter
        undeclared = sorted(_offered_choice_names(parameter) - declared)
        if undeclared:
            problems.append(f"{node_type}: {undeclared}")

    assert not problems, (
        "These dropdowns offer choices that match no catalog model the node declares, so they "
        f"fail closed at run time: {problems}"
    )
