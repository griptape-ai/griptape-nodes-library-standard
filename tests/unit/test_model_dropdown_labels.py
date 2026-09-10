"""Every model dropdown offers a readable label without changing what it stores.

`ModelAccessComponent` decorates each row with the catalog's `display_name` as
`label`, while `name` stays the `provider_model_id`. The two must not trade
places. A label that reached the stored value would reintroduce exactly what
every node's `LEGACY_MODEL_VALUES` table exists to migrate away from, and it
would corrupt the provenance metadata the engine embeds in generated PNGs, which
reads the parameter's stored value.

These are invariants rather than a pinned table of names, so adding a model or
renaming one in the catalog does not require updating this file. That the names
themselves resolve is covered by `test_model_dropdown_vocabulary.py`: every
offered choice is a declared `provider_model_id`, and every catalog `Model`
declares a `display_name`, so label coverage follows from a guarantee the suite
already enforces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from test_legacy_model_value_migration import (
    NODE_TYPES_WITH_MODEL_ACCESS,
    _create_node,
    _model_access_component,
    _stub_griptape_cloud_model_list,  # noqa: F401  (autouse fixture GriptapeCloudPrompt needs)
)

if TYPE_CHECKING:
    from griptape_nodes.exe_types.core_types import Parameter


def _dropdown_rows(parameter: Parameter) -> list[dict[str, str]]:
    """The dropdown's offered rows, straight off `ui_options`."""
    return parameter.ui_options["data"]


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_every_offered_choice_has_a_readable_label(node_type: str) -> None:
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    rows = _dropdown_rows(parameter)
    assert rows, f"{node_type}: dropdown offers no rows at all"

    for row in rows:
        assert row.get("label"), (
            f"{node_type}: choice {row['name']!r} has no label, so the dropdown shows a raw provider "
            "id. Its catalog entry should supply a display_name."
        )


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_row_name_is_the_stored_value_not_the_label(node_type: str) -> None:
    """`name` pairs the row to `Options.choices`; a label there would change what is stored."""
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    offered = set(component.model_choices)
    for row in _dropdown_rows(parameter):
        assert row["name"] in offered, (
            f"{node_type}: row name {row['name']!r} is not one of model_choices. A display label has "
            "leaked into the position the parameter stores and the proxy receives."
        )


@pytest.mark.parametrize("node_type", NODE_TYPES_WITH_MODEL_ACCESS)
def test_selected_value_is_never_a_label(node_type: str) -> None:
    """The stored value stays a provider id, which is what provenance metadata records.

    The engine writes every INPUT/PROPERTY parameter into generated PNGs as
    `gtn_param_<name>`, reading `get_parameter_value`. A studio identifying the
    exact model behind a render months later depends on that being the dated
    provider id rather than a name that gets re-cased or reused.
    """
    node = _create_node(node_type)
    component = _model_access_component(node)
    parameter = cast("Any", component)._parameter

    stored = node.get_parameter_value(parameter.name)
    assert stored in set(component.model_choices), f"{node_type}: stored value {stored!r} is not one of model_choices"
