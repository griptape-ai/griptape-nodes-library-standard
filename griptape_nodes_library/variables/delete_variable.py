from typing import Any

from griptape_nodes.exe_types.core_types import NodeMessageResult, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode, NodeResolutionState
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload
from griptape_nodes.traits.options import Options

from griptape_nodes_library.variables.variable_utils import (
    create_advanced_parameter_group,
    delete_variable,
    has_variable,
    list_variables,
    scope_string_to_variable_scope,
)


class DeleteVariable(ControlNode):
    """Delete one or more workflow variables.

    ``variable_names`` accepts a single name (picked from the dropdown or connected as a str)
    or a list of names. Missing variables are skipped unless ``fail_if_missing`` is enabled.
    Read-only variables are refused by the engine and fail the node.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.variable_names_param = Parameter(
            name="variable_names",
            type="str",
            input_types=["str", "list"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Name of the variable to delete, or a list of names",
        )
        available_names = self._get_variable_names()
        # allow_custom keeps connected lists and not-yet-created names from being coerced to the first choice.
        self.variable_names_param.add_trait(Options(choices=available_names, allow_custom=True))
        self.variable_names_param.add_trait(
            Button(
                icon="list-restart",
                size="icon",
                variant="secondary",
                on_click=self._refresh_variable_names,
            )
        )
        self.add_parameter(self.variable_names_param)

        self.fail_if_missing_param = Parameter(
            name="fail_if_missing",
            type="bool",
            default_value=False,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="Fail the node if a named variable does not exist. When off, missing variables are skipped.",
        )
        self.add_parameter(self.fail_if_missing_param)

        self.deleted_names_param = Parameter(
            name="deleted_names",
            type="list",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="Names of the variables that were deleted",
        )
        self.add_parameter(self.deleted_names_param)

        # Advanced parameters group (collapsed by default)
        advanced = create_advanced_parameter_group()
        self.scope_param = advanced.scope_param
        self.add_node_element(advanced.parameter_group)

    def _get_variable_names(self) -> list[str]:
        scope_str = self.get_parameter_value("scope")
        scope = scope_string_to_variable_scope(scope_str) if scope_str else VariableScope.HIERARCHICAL
        return list_variables(node_name=self.name, scope=scope)

    def _refresh_variable_names(
        self, button: Button, button_details: ButtonDetailsMessagePayload
    ) -> NodeMessageResult | None:  # noqa: ARG002
        names = self._get_variable_names()
        current = self.get_parameter_value(self.variable_names_param.name)
        self._update_option_choices(
            param=self.variable_names_param.name, choices=names, default=names[0] if names else ""
        )
        if isinstance(current, str) and current in names:
            self.set_parameter_value(self.variable_names_param.name, current)
        return None

    def _requested_names(self) -> list[str]:
        """Normalize ``variable_names`` to a de-duplicated list, preserving order and dropping blanks."""
        value = self.get_parameter_value(self.variable_names_param.name)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            msg = f"{self.name}: variable_names must be a string or a list of strings, got {type(value).__name__}."
            raise TypeError(msg)

        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"{self.name}: every entry in variable_names must be a string, got {type(item).__name__}."
                raise TypeError(msg)
            stripped = item.strip()
            if stripped and stripped not in names:
                names.append(stripped)
        return names

    def process(self) -> None:
        names = self._requested_names()
        fail_if_missing = bool(self.get_parameter_value(self.fail_if_missing_param.name))
        scope = scope_string_to_variable_scope(self.get_parameter_value(self.scope_param.name))

        deleted: list[str] = []
        errors: list[str] = []
        for variable_name in names:
            if not has_variable(node_name=self.name, variable_name=variable_name, scope=scope):
                if fail_if_missing:
                    errors.append(f"Variable '{variable_name}' does not exist.")
                continue
            try:
                delete_variable(node_name=self.name, variable_name=variable_name, scope=scope)
            except RuntimeError as e:
                errors.append(str(e))
                continue
            deleted.append(variable_name)

        self.set_parameter_value(self.deleted_names_param.name, deleted)
        self.parameter_output_values[self.deleted_names_param.name] = deleted

        if errors:
            msg = f"{self.name}: deleted {len(deleted)} of {len(names)} variable(s). " + " ".join(errors)
            raise RuntimeError(msg)

    @property
    def state(self) -> NodeResolutionState:
        """Overrides BaseNode.state @property to treat it as volatile (re-run if any named variable exists again)."""
        if self._state == NodeResolutionState.RESOLVED:
            try:
                names = self._requested_names()
                scope = scope_string_to_variable_scope(self.get_parameter_value(self.scope_param.name))
                if any(has_variable(node_name=self.name, variable_name=n, scope=scope) for n in names):
                    return NodeResolutionState.UNRESOLVED
            except (RuntimeError, TypeError, ValueError):
                return NodeResolutionState.UNRESOLVED
        return super().state

    @state.setter
    def state(self, new_state: NodeResolutionState) -> None:
        # Have to override the setter if we override the getter.
        self._state = new_state
