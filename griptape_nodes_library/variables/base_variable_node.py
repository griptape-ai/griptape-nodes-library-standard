from typing import ClassVar

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter
from griptape_nodes.exe_types.node_types import (
    ControlNode,
    NodeDependencies,
    NodeResolutionState,
    VariableAccess,
    VariableReference,
)
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

from griptape_nodes_library.variables.variable_utils import (
    create_advanced_parameter_group,
    list_variables,
    scope_string_to_variable_scope,
)

SCOPE_PARAM_NAME = "scope"


class BaseVariableNode(ControlNode):
    """Shared plumbing for nodes that act on named workflow variables.

    Subclasses build their name parameter and register it with ``_add_variable_name_parameter``,
    then call ``_add_scope_parameter`` after their other parameters so the Advanced group sits last.
    ``VARIABLE_ACCESS`` declares how the node touches the variable for save-time serialization.
    Override ``_is_stale`` to make the node re-run when the engine-side variable changes.
    """

    VARIABLE_ACCESS: ClassVar[VariableAccess]

    variable_name_param: Parameter
    scope_param: Parameter

    def _add_variable_name_parameter(self, parameter: Parameter, *, allow_custom: bool = False) -> None:
        """Attach the variable dropdown and refresh button to ``parameter`` and add it to the node."""
        self.variable_name_param = parameter
        choices = self._variable_name_choices(self._get_variable_names())
        parameter.add_trait(Options(choices=choices, allow_custom=allow_custom))
        parameter.add_trait(
            Button(
                icon="list-restart",
                size="icon",
                variant="secondary",
                on_click=self._refresh_variable_names,
            )
        )
        self.add_parameter(parameter)

    def _add_scope_parameter(self) -> None:
        # Advanced parameters group (collapsed by default)
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

    def _get_scope(self) -> VariableScope:
        # Read by name: the dropdown is populated in __init__ before the scope parameter exists.
        scope_str = self.get_parameter_value(SCOPE_PARAM_NAME)
        return scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL

    def _get_variable_names(self) -> list[str]:
        return list_variables(node_name=self.name, scope=self._get_scope())

    def _variable_name_choices(self, names: list[str]) -> list[str]:
        """Map the visible variable names to dropdown choices. Override to add extra entries."""
        return names

    def _refresh_variable_names(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        param_name = self.variable_name_param.name
        choices = self._variable_name_choices(self._get_variable_names())
        current = self.get_parameter_value(param_name)
        self._update_option_choices(param=param_name, choices=choices, default=choices[0] if choices else "")
        if isinstance(current, str) and current in choices:
            self.set_parameter_value(param_name, current)
        return None

    def _resolve_variable_names(self) -> list[str]:
        """Return the variable names this node targets. Must not raise; it runs at save time."""
        name = self.get_parameter_value(self.variable_name_param.name)
        return [name] if isinstance(name, str) and name else []

    def get_node_dependencies(self) -> NodeDependencies | None:
        """Declare the variables this node touches so they survive serialization.

        Reads the current value of the name parameter via ``get_parameter_value``. If the parameter
        is driven by an incoming connection, this returns the last propagated value (or ``None`` if
        nothing has propagated yet). No declaration is emitted for empty/None names.
        """
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()

        names = self._resolve_variable_names()
        if names:
            scope = self._get_scope()
            for name in names:
                deps.variable_references.add(VariableReference(name=name, scope=scope, access=self.VARIABLE_ACCESS))

        return deps

    def _is_stale(self) -> bool:
        """Return True when the engine-side variable no longer matches what this node last produced.

        Exceptions are treated as stale, so implementations can let lookups of missing variables raise.
        """
        return False

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to treat it as volatile (unresolved whenever ``_is_stale``)."""
        if self._state == NodeResolutionState.RESOLVED:
            try:
                if self._is_stale():
                    return NodeResolutionState.UNRESOLVED
            except (LookupError, RuntimeError, TypeError, ValueError):
                # Variable or flow may not exist yet; assume unresolved.
                return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        # Have to override the setter if we override the getter.
        self._state = new_state
