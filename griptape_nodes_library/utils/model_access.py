"""License-policy wiring for the library's model-selection dropdowns.

A dropdown stores the provider's own model id (e.g. ``"seedream-4-5-251128"``) --
the id a node already needs in order to build its upstream request -- and the
engine's ``ModelAccessComponent`` resolves it to the catalog model key that
license policy gates on. That keeps the catalog out of node code entirely.
``deprecated_values`` covers a parameter that stored something else before it
adopted this convention (an artist-facing label, a catalog key): the mapping
migrates a legacy stored value to its canonical provider model id, and is never
offered as a fresh selection.

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
    from griptape_nodes.retained_mode.managers.authorization_checkpoint import CheckpointDenial

__all__ = ["ModelDropdownAccess"]


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
        deprecated_values: dict[str, str] | None = None,
    ) -> None:
        self._node = node
        self._parameter_name = parameter.name
        self.component = ModelAccessComponent(
            node=node,
            parameter=parameter,
            model_choices=model_choices,
            default_model=default_model,
            deprecated_values=deprecated_values,
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
