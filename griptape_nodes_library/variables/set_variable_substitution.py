from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.retained_mode.events.workflow_events import (
    SetVariableSubstitutionEnabledRequest,
    SetVariableSubstitutionEnabledResultFailure,
    SetVariableSubstitutionEnabledResultNotAlteredSuccess,
    SetVariableSubstitutionEnabledResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class SetVariableSubstitution(SuccessFailureNode):
    """Enable or disable inline variable substitution for the current workflow.

    When enabled, {VARIABLE_NAME} tokens in parameter values are replaced with
    the corresponding variable's value at execution time. The setting is
    workflow-scoped and baked into the saved workflow file.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.enabled_param = Parameter(
            name="enabled",
            type="bool",
            default_value=True,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip=(
                "When True, {VARIABLE_NAME} tokens in parameter values are replaced with "
                "the variable's value at execution time. Scoped to the current workflow."
            ),
        )
        self.add_parameter(self.enabled_param)

        self._create_status_parameters(
            result_details_tooltip="Details about the variable substitution setting.",
            result_details_placeholder="Results will appear here after the node runs.",
        )

    def process(self) -> None:
        self._clear_execution_status()
        try:
            enabled = bool(self.get_parameter_value(self.enabled_param.name))
            result = GriptapeNodes.handle_request(SetVariableSubstitutionEnabledRequest(enabled=enabled))
            if isinstance(result, SetVariableSubstitutionEnabledResultFailure):
                msg = result.result_details or "Failed to set variable substitution state."
                raise RuntimeError(msg)  # noqa: TRY301
            if not isinstance(
                result,
                SetVariableSubstitutionEnabledResultSuccess | SetVariableSubstitutionEnabledResultNotAlteredSuccess,
            ):
                msg = f"Unexpected result type: {type(result).__name__}"
                raise RuntimeError(msg)  # noqa: TRY301
            state = "enabled" if enabled else "disabled"
            self._set_status_results(was_successful=True, result_details=f"Variable substitution {state}.")
        except Exception as exc:
            self._set_status_results(was_successful=False, result_details=str(exc))
            self._handle_failure_exception(exc)
