"""Unit tests for the library's model-dropdown access wiring.

`ModelDropdownAccess` bundles the engine's `ModelAccessComponent` with the
parameter it decorates. These tests stub the engine's access query so the
decoration it drives, the runtime denial lookup, and the `deprecated_values`
migration are all exercised against known verdicts. The component's own
behavior (badges, refresh button, fail-closed snapshot) is covered in the
engine test suite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.events.access_events import (
    ModelAccessVerdict,
    QueryModelAccessForNodeRequest,
    QueryModelAccessForNodeResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial, CheckpointFailure
from griptape_nodes.traits.button import Button
from griptape_nodes.traits.options import Options

from griptape_nodes_library.utils.model_access import ModelDropdownAccess

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload

# The dropdown stores provider model ids; the catalog keys the engine gates on stay
# `gtc_*`. Kept deliberately distinct so a test cannot pass by conflating them.
# `kling-v2-6` is first so that a migration landing on `kling-v3` cannot be
# confused with `Options` snapping an unknown value to `choices[0]`.
MODEL_CHOICES = ["kling-v2-6", "kling-v3"]
CATALOG_ID_BY_CHOICE = {"kling-v3": "gtc_kling_v3", "kling-v2-6": "gtc_kling_v2_6"}
DENIAL = CheckpointDenial(failures=(CheckpointFailure(detail="Kling v3 is not in your plan."),))


class _ModelNode(DataNode):
    """Minimal node to hang a model parameter off of."""

    def process(self) -> None:
        return None


@pytest.fixture
def stub_access_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every access query with `kling-v3` denied and `kling-v2-6` allowed."""
    original_handle_request = GriptapeNodes.handle_request

    def handle_request(request: RequestPayload, **kwargs: Any) -> ResultPayload:
        if not isinstance(request, QueryModelAccessForNodeRequest):
            return original_handle_request(request, **kwargs)
        choices = MODEL_CHOICES
        if request.candidate_model_ids is not None:
            catalog_ids = set(request.candidate_model_ids)
            choices = [c for c in MODEL_CHOICES if CATALOG_ID_BY_CHOICE[c] in catalog_ids]
        return QueryModelAccessForNodeResultSuccess(
            result_details="stubbed access query",
            verdicts=[
                ModelAccessVerdict(
                    model_id=CATALOG_ID_BY_CHOICE[choice],
                    provider_model_id=choice,
                    denial=DENIAL if choice == "kling-v3" else None,
                )
                for choice in choices
            ],
        )

    monkeypatch.setattr(GriptapeNodes, "handle_request", handle_request)


def _build_node_with_dropdown(*, default_model: str) -> tuple[_ModelNode, Parameter]:
    node = _ModelNode(name="ModelNode")
    parameter = Parameter(
        name="model_name",
        type="str",
        default_value=default_model,
        tooltip="Model to use",
    )
    node.add_parameter(parameter)
    node.set_parameter_value("model_name", default_model)
    return node, parameter


@pytest.mark.usefixtures("stub_access_query")
def test_dropdown_access_installs_traits_and_reports_parameter_name() -> None:
    node, parameter = _build_node_with_dropdown(default_model="kling-v2-6")
    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="kling-v2-6",
    )

    assert parameter.find_elements_by_type(Options)
    assert parameter.find_elements_by_type(Button)
    assert access.parameter_name == "model_name"


@pytest.mark.usefixtures("stub_access_query")
def test_on_value_set_ignores_other_parameters() -> None:
    """Badge tracking is scoped to the dropdown the component owns."""
    node, parameter = _build_node_with_dropdown(default_model="kling-v2-6")
    other = Parameter(name="prompt", type="str", default_value="", tooltip="Prompt")
    node.add_parameter(other)
    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="kling-v2-6",
    )

    access.on_value_set(other, "kling-v3")
    assert parameter.get_badge() is None

    access.on_value_set(parameter, "kling-v3")
    assert parameter.get_badge() is not None


@pytest.mark.usefixtures("stub_access_query")
def test_selection_denial_and_raise_read_the_current_selection() -> None:
    node, parameter = _build_node_with_dropdown(default_model="kling-v2-6")
    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="kling-v2-6",
    )

    assert access.selection_denial() is None

    node.set_parameter_value("model_name", "kling-v3")
    denial = access.selection_denial()
    assert denial is not None
    assert "not in your plan" in denial.reason()
    with pytest.raises(RuntimeError, match="not permitted"):
        access.raise_if_selection_denied()


@pytest.mark.usefixtures("stub_access_query")
def test_deprecated_values_migrate_a_legacy_assignment_to_its_canonical_key() -> None:
    """A legacy value the parameter stored before this convention migrates on assignment.

    The migration target is deliberately NOT ``choices[0]``: ``Options`` rewrites
    any value outside ``choices`` to ``choices[0]``, so a target of ``choices[0]``
    would pass whether the value migrated or was simply snapped.
    """
    node, parameter = _build_node_with_dropdown(default_model="kling-v2-6")
    ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="kling-v2-6",
        deprecated_values={"Kling v3": "kling-v3"},
    )

    node.set_parameter_value("model_name", "Kling v3")
    assert node.get_parameter_value("model_name") == "kling-v3"
