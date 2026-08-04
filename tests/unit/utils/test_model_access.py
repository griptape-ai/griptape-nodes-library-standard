"""Unit tests for the library's model-dropdown access wiring.

`MappedModelAccessComponent` exists for dropdowns whose choices are display
labels rather than the catalog's `provider_model_id`. These tests stub the
engine's access query so the mapping, the decoration it drives, and the runtime
denial lookup are all exercised against known verdicts. The component's own
behavior (badges, refresh button, fail-closed snapshot) is covered in the engine
test suite.
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

from griptape_nodes_library.utils.model_access import MappedModelAccessComponent, ModelDropdownAccess

if TYPE_CHECKING:
    from griptape_nodes.retained_mode.events.base_events import RequestPayload, ResultPayload

# Dropdown labels this library's proxy nodes show, and the provider ids they map to.
PROVIDER_MODEL_ID_BY_CHOICE = {
    "Kling v3.0": "kling-v3",
    "Kling v2.6": "kling-v2-6",
}
MODEL_CHOICES = list(PROVIDER_MODEL_ID_BY_CHOICE)
CATALOG_ID_BY_PROVIDER_MODEL_ID = {
    "kling-v3": "gtc_kling_v3",
    "kling-v2-6": "gtc_kling_v2_6",
}
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
        provider_model_ids = list(CATALOG_ID_BY_PROVIDER_MODEL_ID)
        if request.candidate_model_ids is not None:
            provider_model_ids = [
                provider_model_id
                for provider_model_id, catalog_id in CATALOG_ID_BY_PROVIDER_MODEL_ID.items()
                if catalog_id in request.candidate_model_ids
            ]
        return QueryModelAccessForNodeResultSuccess(
            result_details="stubbed access query",
            verdicts=[
                ModelAccessVerdict(
                    model_id=CATALOG_ID_BY_PROVIDER_MODEL_ID[provider_model_id],
                    provider_model_id=provider_model_id,
                    denial=DENIAL if provider_model_id == "kling-v3" else None,
                )
                for provider_model_id in provider_model_ids
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
def test_denied_label_is_decorated_and_queried_by_label() -> None:
    """A denied provider id flags the dropdown row keyed by its display label."""
    node, parameter = _build_node_with_dropdown(default_model="Kling v2.6")
    component = MappedModelAccessComponent(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="Kling v2.6",
        provider_model_id_by_choice=PROVIDER_MODEL_ID_BY_CHOICE,
    )

    rows = {row["name"]: row for row in parameter.ui_options["data"]}
    assert rows["Kling v3.0"].get("icon") == "shield-off"
    assert "icon" not in rows["Kling v2.6"]

    # The runtime query resolves the label to its catalog id and returns the denial.
    assert component.query_for_denial("Kling v3.0") is not None
    assert component.query_for_denial("Kling v2.6") is None
    # Provider ids still resolve, so a raw-id value saved in an older workflow works.
    assert component.query_for_denial("kling-v3") is not None


@pytest.mark.usefixtures("stub_access_query")
def test_denied_default_label_relocates_to_a_permitted_choice() -> None:
    """A denied default moves the stored value to a permitted label, not a provider id."""
    node, parameter = _build_node_with_dropdown(default_model="Kling v3.0")
    MappedModelAccessComponent(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="Kling v3.0",
        provider_model_id_by_choice=PROVIDER_MODEL_ID_BY_CHOICE,
    )

    assert node.get_parameter_value("model_name") == "Kling v2.6"


@pytest.mark.usefixtures("stub_access_query")
def test_dropdown_access_installs_traits_and_reads_the_selection() -> None:
    node, parameter = _build_node_with_dropdown(default_model="Kling v2.6")
    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="Kling v2.6",
        provider_model_id_by_choice=PROVIDER_MODEL_ID_BY_CHOICE,
    )

    assert isinstance(access.component, MappedModelAccessComponent)
    assert parameter.find_elements_by_type(Options)
    assert parameter.find_elements_by_type(Button)
    assert access.parameter_name == "model_name"

    assert access.selection_denial() is None
    node.set_parameter_value("model_name", "Kling v3.0")
    denial = access.selection_denial()
    assert denial is not None
    assert "not in your plan" in denial.reason()
    with pytest.raises(RuntimeError, match="not permitted"):
        access.raise_if_selection_denied()


@pytest.mark.usefixtures("stub_access_query")
def test_dropdown_access_without_a_mapping_uses_the_engine_component() -> None:
    """Provider-id dropdowns need no mapping, so they get the engine component itself."""
    node = _ModelNode(name="ModelNode")
    parameter = Parameter(name="model", type="str", default_value="kling-v2-6", tooltip="Model to use")
    node.add_parameter(parameter)
    node.set_parameter_value("model", "kling-v2-6")

    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=list(CATALOG_ID_BY_PROVIDER_MODEL_ID),
        default_model="kling-v2-6",
    )

    assert not isinstance(access.component, MappedModelAccessComponent)
    node.set_parameter_value("model", "kling-v3")
    assert access.selection_denial() is not None


@pytest.mark.usefixtures("stub_access_query")
def test_on_value_set_ignores_other_parameters() -> None:
    """Badge tracking is scoped to the dropdown the component owns."""
    node, parameter = _build_node_with_dropdown(default_model="Kling v2.6")
    other = Parameter(name="prompt", type="str", default_value="", tooltip="Prompt")
    node.add_parameter(other)
    access = ModelDropdownAccess(
        node=node,
        parameter=parameter,
        model_choices=MODEL_CHOICES,
        default_model="Kling v2.6",
        provider_model_id_by_choice=PROVIDER_MODEL_ID_BY_CHOICE,
    )

    access.on_value_set(other, "Kling v3.0")
    assert parameter.get_badge() is None

    access.on_value_set(parameter, "Kling v3.0")
    assert parameter.get_badge() is not None
