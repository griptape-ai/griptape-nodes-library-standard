from typing import Any

from griptape_nodes.common.macro_parser.core import ParsedMacro
from griptape_nodes.common.macro_parser.exceptions import MacroResolutionError, MacroSyntaxError
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes


class ResolveMacroTemplate(SuccessFailureNode):
    """Resolve a macro template string using a dictionary of variable values."""

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            ParameterString(
                name="template",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value="",
                multiline=True,
                placeholder_text="Hello, {name}! You have {count} messages.",
                tooltip=(
                    "The macro template string. Reference variables with braces "
                    "(e.g. '{name}'). Optional variables use '{name?}'."
                ),
            )
        )

        self.add_parameter(
            Parameter(
                name="variables",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="dict",
                default_value={},
                tooltip=(
                    "Dictionary of variable values. Each key is the variable name "
                    "used in the template; each value is the string (or int) to "
                    "substitute in."
                ),
            )
        )

        self.add_parameter(
            ParameterString(
                name="resolved_string",
                allowed_modes={ParameterMode.OUTPUT},
                allow_input=False,
                allow_property=False,
                multiline=True,
                placeholder_text="The resolved string will appear here.",
                tooltip="The template string with all variables substituted.",
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the macro resolution result.",
            result_details_placeholder="Details on the resolution attempt will be presented here.",
        )

    def process(self) -> None:
        self._clear_execution_status()

        template: str = self.get_parameter_value("template")
        raw_variables = self.get_parameter_value("variables")

        variables = self._coerce_variables(raw_variables)

        try:
            parsed = ParsedMacro(template=template)
        except MacroSyntaxError as err:
            self._report_failure(f"Failed to parse template. {err}", exception=err)
            return

        try:
            resolved = parsed.resolve(variables, GriptapeNodes.SecretsManager())
        except MacroResolutionError as err:
            self._report_failure(f"Failed to resolve template. {err}", exception=err)
            return

        self.set_parameter_value("resolved_string", resolved)
        self.parameter_output_values["resolved_string"] = resolved
        self._set_status_results(
            was_successful=True,
            result_details=f"SUCCESS: Resolved template to: {resolved}",
        )

    def _coerce_variables(self, raw_variables: Any) -> dict[str, str | int]:
        """Normalize the variables dict so ParsedMacro.resolve accepts it.

        Keys become strings; ``str`` and ``int`` values pass through (keeping
        format specs like ``{count:03}`` working). Everything else is
        ``str()``-coerced.
        """

        coerced: dict[str, str | int] = {}
        for key, value in raw_variables.items():
            str_key = str(key)
            # bool is a subclass of int; render it as "True"/"False" instead of 1/0.
            if isinstance(value, bool):
                coerced[str_key] = str(value)
            elif isinstance(value, (str, int)):
                coerced[str_key] = value
            else:
                coerced[str_key] = str(value)
        return coerced

    def _report_failure(self, message: str, exception: Exception | None = None) -> None:
        self.set_parameter_value("resolved_string", "")
        self.parameter_output_values["resolved_string"] = ""
        self._set_status_results(was_successful=False, result_details=f"FAILURE: {message}")
        if exception is not None:
            self._handle_failure_exception(exception)
