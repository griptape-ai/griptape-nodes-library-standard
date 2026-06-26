from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode, ParameterTypeBuiltin
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
    get_variable,
    list_variables,
    scope_string_to_variable_scope,
)


class GetVariable(ControlNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.variable_name_param = Parameter(
            name="variable_name",
            type="str",
            allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT, ParameterMode.PROPERTY},
            tooltip="Name of the variable to retrieve",
        )
        available_names = self._get_variable_names()
        self.variable_name_param.add_trait(Options(choices=available_names))
        self.variable_name_param.add_trait(
            Button(
                icon="list-restart",
                size="icon",
                variant="secondary",
                on_click=self._refresh_variable_names,
            )
        )
        self.add_parameter(self.variable_name_param)

        self.value_param = Parameter(
            name="value",
            type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="The value of the workflow variable",
        )
        self.add_parameter(self.value_param)

        # Advanced parameters group (collapsed by default)
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

    def _get_variable_names(self) -> list[str]:
        scope_str = self.get_parameter_value("scope")
        scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL
        return list_variables(node_name=self.name, scope=scope)

    def _refresh_variable_names(self, button: Button, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult | None:  # noqa: ARG002
        names = self._get_variable_names()
        current = self.get_parameter_value("variable_name")
        self._update_option_choices(param="variable_name", choices=names, default=names[0] if names else "")
        if current and current in names:
            self.set_parameter_value("variable_name", current)
        return None

    def process(self) -> None:
        variable_name = self.get_parameter_value(self.variable_name_param.name)
        scope_str = self.get_parameter_value(self.scope_param.name)

        # Convert scope string to VariableScope enum
        scope = scope_string_to_variable_scope(scope_str)

        # This can throw if the variable doesn't exist.
        variable = get_variable(node_name=self.name, variable_name=variable_name, scope=scope)

        var_value = variable.value

        self.set_parameter_value(self.value_param.name, var_value)

        variable_name = self.get_parameter_value(self.variable_name_param.name)

        # Set the output values.
        self.parameter_output_values[self.variable_name_param.name] = variable_name
        self.parameter_output_values[self.value_param.name] = var_value

    def get_node_dependencies(self) -> NodeDependencies | None:
        """Declare the variable this node reads so it survives serialization.

        Access is READ: ``process()`` only calls ``GetVariableRequest``; it never writes.

        Reads the current value of ``variable_name`` via ``get_parameter_value`` — if the parameter
        is driven by an incoming connection, this returns the last propagated value (or ``None`` if
        nothing has propagated yet). No declaration is emitted for empty/None names.
        """
        deps = super().get_node_dependencies()
        if deps is None:
            deps = NodeDependencies()

        variable_name = self.get_parameter_value(self.variable_name_param.name)
        if isinstance(variable_name, str) and variable_name:
            scope_str = self.get_parameter_value(self.scope_param.name)
            scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL
            deps.variable_references.add(VariableReference(name=variable_name, scope=scope, access=VariableAccess.READ))

        return deps

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to treat it as volatile (if the value has changed, mark as unresolved)."""
        if self._state == NodeResolutionState.RESOLVED:
            variable_name = self.get_parameter_value(self.variable_name_param.name)
            scope_str = self.get_parameter_value(self.scope_param.name)

            # Convert scope string to VariableScope enum
            scope = scope_string_to_variable_scope(scope_str)

            # This can throw if the variable doesn't exist.
            try:
                variable = get_variable(node_name=self.name, variable_name=variable_name, scope=scope)

                var_value = variable.value
                our_value = self.get_parameter_value(self.value_param.name)

                # Mark ourselves as unresolved if the value is different than what we last emitted.
                if var_value != our_value:
                    return NodeResolutionState.UNRESOLVED
            except LookupError:
                # Variable may not have been created yet; assume unresolved.
                return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        # Have to override the setter if we override the getter.
        self._state = new_state
