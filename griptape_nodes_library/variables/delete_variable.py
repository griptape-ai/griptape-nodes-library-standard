from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode

from griptape_nodes_library.variables.base_variable_node import BaseVariableNode
from griptape_nodes_library.variables.variable_utils import delete_variable, has_variable


class DeleteVariable(BaseVariableNode):
    """Delete one or more workflow variables.

    ``variable_names`` accepts a single name (picked from the dropdown or connected as a str)
    or a list of names. Missing variables are skipped unless ``fail_if_missing`` is enabled.
    Read-only variables are refused by the engine and fail the node.

    Declares no variable dependency: declaring WRITE would persist the variable this node exists
    to remove. A variable named only by this node is therefore not saved with the workflow.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # allow_custom keeps connected lists and not-yet-created names from being coerced to the first choice.
        self._add_variable_name_parameter(
            Parameter(
                name="variable_names",
                type="str",
                input_types=["str", "list"],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Name of the variable to delete, or a list of names",
            ),
            allow_custom=True,
        )

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

        self._add_scope_parameter()

    def _requested_names(self) -> list[str]:
        """Normalize ``variable_names`` to a de-duplicated list, preserving order and dropping blanks."""
        value = self.get_parameter_value(self.variable_name_param.name)
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            msg = f"'variable_names' must be a string or a list of strings, got {type(value).__name__}."
            raise TypeError(msg)

        names: list[str] = []
        for item in value:
            if not isinstance(item, str):
                msg = f"Every entry in 'variable_names' must be a string, got {type(item).__name__}."
                raise TypeError(msg)
            stripped = item.strip()
            if stripped and stripped not in names:
                names.append(stripped)
        return names

    def process(self) -> None:
        names = self._requested_names()
        fail_if_missing = bool(self.get_parameter_value(self.fail_if_missing_param.name))
        scope = self._get_scope()

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
            msg = f"Deleted {len(deleted)} of {len(names)} variable(s). " + " ".join(errors)
            raise RuntimeError(msg)

    def _is_stale(self) -> bool:
        # Re-run if any named variable exists again.
        scope = self._get_scope()
        return any(has_variable(node_name=self.name, variable_name=n, scope=scope) for n in self._requested_names())
