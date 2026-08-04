"""License-policy wiring for the library's model-selection dropdowns.

The engine's ``ModelAccessComponent`` decorates a model dropdown with the
caller's entitlement, badges a denied selection, and answers "is this selection
permitted right now?" at run time. It keys every lookup on the dropdown value,
which it expects to be the catalog's ``provider_model_id``.

Many nodes here instead offer artist-facing labels (``"Kling v3.0"``,
``"Seedream 4.5"``) and translate to the provider id when they build their
request. Those labels are the values the parameter stores, so they cannot be
swapped for provider ids: ``Options`` converts any value outside ``choices`` to
``choices[0]`` before ``before_value_set`` runs, so a rename would silently
repoint every saved workflow at the first model in the list.
``MappedModelAccessComponent`` closes that gap by keying the component's
snapshot under the label as well, so decoration, badges, and runtime queries all
work off the value the parameter actually holds.

``ModelDropdownAccess`` is what node base classes hold: the component plus the
parameter it owns, so a base class can forward value changes and query the
current selection without every node repeating that bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent

if TYPE_CHECKING:
    from griptape_nodes.exe_types.core_types import Parameter
    from griptape_nodes.exe_types.node_types import BaseNode
    from griptape_nodes.exe_types.param_components.model_access_component import _AccessSnapshot
    from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial

__all__ = ["MappedModelAccessComponent", "ModelDropdownAccess"]


class MappedModelAccessComponent(ModelAccessComponent):
    """A model-access component for a dropdown whose choices are display labels.

    ``provider_model_id_by_choice`` maps each dropdown choice to the
    ``provider_model_id`` the library catalog declares for it -- the same id the
    node sends upstream. Choices missing from the map keep the base class's
    behavior and resolve as provider ids, so a dropdown that mixes labels and
    raw ids still works.
    """

    def __init__(
        self,
        *,
        node: BaseNode,
        parameter: Parameter,
        model_choices: list[str],
        default_model: str,
        provider_model_id_by_choice: dict[str, str],
    ) -> None:
        # Assigned before the base constructor because that constructor fetches
        # the first snapshot, which `_fetch_snapshot` re-keys through this map.
        self._provider_model_id_by_choice = dict(provider_model_id_by_choice)
        super().__init__(
            node=node,
            parameter=parameter,
            model_choices=model_choices,
            default_model=default_model,
        )

    def _fetch_snapshot(self) -> _AccessSnapshot:
        """Index the engine's verdicts under the dropdown labels as well as the provider ids."""
        snapshot = super()._fetch_snapshot()
        for choice, provider_model_id in self._provider_model_id_by_choice.items():
            catalog_id = snapshot.catalog_id_by_provider_id.get(provider_model_id)
            if catalog_id is not None:
                snapshot.catalog_id_by_provider_id[choice] = catalog_id
            denial = snapshot.denial_by_provider_id.get(provider_model_id)
            if denial is not None:
                snapshot.denial_by_provider_id[choice] = denial
        return snapshot


class ModelDropdownAccess:
    """One node's license-filtered model dropdown.

    Owns the ``ModelAccessComponent`` and remembers which parameter it decorates
    so a node base class can forward ``after_value_set`` and query the current
    selection generically. Nodes reach the component itself through
    ``component`` when they need its dropdown-level API (``model_choices``,
    ``pick_permitted_default``, ``reinstall_options``).
    """

    def __init__(
        self,
        *,
        node: BaseNode,
        parameter: Parameter,
        model_choices: list[str],
        default_model: str,
        provider_model_id_by_choice: dict[str, str] | None = None,
    ) -> None:
        self._node = node
        self._parameter_name = parameter.name
        self.component: ModelAccessComponent
        if provider_model_id_by_choice:
            self.component = MappedModelAccessComponent(
                node=node,
                parameter=parameter,
                model_choices=model_choices,
                default_model=default_model,
                provider_model_id_by_choice=provider_model_id_by_choice,
            )
        else:
            self.component = ModelAccessComponent(
                node=node,
                parameter=parameter,
                model_choices=model_choices,
                default_model=default_model,
            )

    @property
    def parameter_name(self) -> str:
        """Name of the parameter this dropdown decorates."""
        return self._parameter_name

    def on_value_set(self, parameter: Parameter, value: Any) -> None:
        """Track the badge for this dropdown; ignore every other parameter on the node."""
        if parameter.name == self._parameter_name:
            self.component.on_value_changed(value)

    def selection_denial(self) -> CheckpointDenial | None:
        """The current selection's verdict, or ``None`` when it is permitted."""
        return self.component.query_for_denial(self._node.get_parameter_value(self._parameter_name))

    def raise_if_selection_denied(self) -> None:
        """Raise ``RuntimeError`` when the current selection is not permitted."""
        self.component.raise_if_denied(self._node.get_parameter_value(self._parameter_name))
