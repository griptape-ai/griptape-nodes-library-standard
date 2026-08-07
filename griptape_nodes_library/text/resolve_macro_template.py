from enum import StrEnum
from typing import Any

from griptape_nodes.common.macro_parser.core import ParsedMacro
from griptape_nodes.common.macro_parser.exceptions import MacroResolutionError, MacroSyntaxError
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.workflow_events import (
    SetVariableSubstitutionEnabledRequest,
    SetVariableSubstitutionEnabledResultFailure,
)
from griptape_nodes.retained_mode.variable_types import VariableScope
from griptape_nodes.traits.options import Options

from griptape_nodes_library.variables.variable_utils import get_variables, scope_string_to_variable_scope


class VariableSource(StrEnum):
    """Where the node looks up template values, and in what precedence order.

    The value drives branching in process(), so the members are ordered from
    "dictionary only" through to "Variables win".
    """

    DICTIONARY_ONLY = "Dictionary only"
    DICTIONARY_THEN_VARIABLES = "Dictionary, then Variables"
    VARIABLES_THEN_DICTIONARY = "Variables, then dictionary"


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
                    "The macro template string. The macro language is described here: https://docs.griptapenodes.com/en/stable/guides/projects/macros/"
                ),
            )
        )

        self.add_parameter(
            ParameterDict(
                name="variables",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                default_value=None,
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

        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}) as advanced_group:
            variable_source = Parameter(
                name="variable_source",
                type="str",
                default_value=VariableSource.DICTIONARY_ONLY.value,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip=(
                    "Where template values come from. "
                    "Variables = the workflow's Variables system (see Create/Set/Get/Has Variable nodes)."
                ),
            )
            variable_source.add_trait(Options(choices=[source.value for source in VariableSource]))

            variable_search_scope = Parameter(
                name="variable_search_scope",
                type="str",
                default_value=VariableScope.HIERARCHICAL.value,
                allowed_modes={ParameterMode.PROPERTY},
                tooltip="Which Variables layers to search when looking up template values.",
                ui_options={"hide": True},
            )
            variable_search_scope.add_trait(
                Options(
                    choices=[
                        VariableScope.HIERARCHICAL.value,
                        VariableScope.CURRENT_FLOW_ONLY.value,
                        VariableScope.PROJECT_ONLY.value,
                        VariableScope.HIERARCHICAL_FROM_PROJECT.value,
                        VariableScope.GLOBAL_ONLY.value,
                    ]
                )
            )
        self.add_node_element(advanced_group)

        self._create_status_parameters(
            result_details_tooltip="Details about the macro resolution result.",
            result_details_placeholder="Details on the resolution attempt will be presented here.",
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # The scope only matters when we actually consult the Variables system.
        if parameter.name == "variable_source":
            if value == VariableSource.DICTIONARY_ONLY.value:
                self.hide_parameter_by_name("variable_search_scope")
            else:
                self.show_parameter_by_name("variable_search_scope")
        return super().after_value_set(parameter, value)

    def process(self) -> None:

        # Disable variable substitution because that conflicts with this node's purpose
        variable_substitution = SetVariableSubstitutionEnabledRequest(enabled=False)

        # Import lazily to avoid circular import issues during node initialization
        from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

        variable_substitution_status = GriptapeNodes.handle_request(variable_substitution)

        if isinstance(variable_substitution_status, SetVariableSubstitutionEnabledResultFailure):
            self._report_failure(
                "Failed to disable variable substitution. A common cause of this could be no active workflow context.",
            )
            return

        self._clear_execution_status()

        template: str = self.get_parameter_value("template")
        raw_variables = self.get_parameter_value("variables")
        source = VariableSource(self.get_parameter_value("variable_source"))

        coerced = self._coerce_variables(raw_variables)

        try:
            parsed = ParsedMacro(template=template)
        except MacroSyntaxError as err:
            self._report_failure(f"Failed to parse template. {err}", exception=err)
            return

        try:
            variables = self._build_variables(parsed, coerced, source)
        except LookupError as err:
            self._report_failure(
                f"Attempted to look up template variables in the workflow's Variables system. Failed due to: {err}",
                exception=err,
            )
            return

        try:
            resolved = parsed.resolve(variables)
        except MacroResolutionError as err:
            self._report_failure(f"Failed to resolve template. {err}", exception=err)
            return

        self.set_parameter_value("resolved_string", resolved)
        self.parameter_output_values["resolved_string"] = resolved
        self._set_status_results(
            was_successful=True,
            result_details=f"SUCCESS: Resolved template to: {resolved}",
        )

    def _build_variables(
        self,
        parsed: ParsedMacro,
        coerced: dict[str, str | int],
        source: VariableSource,
    ) -> dict[str, str | int]:
        """Merge the coerced dictionary with the Variables system per the chosen precedence.

        The dictionary is never mutated; a copy is returned when values from the
        Variables system are merged in.

        Raises:
            LookupError: If the Variables probe itself could not run.
        """
        if source is VariableSource.DICTIONARY_ONLY:
            return coerced

        scope = scope_string_to_variable_scope(self.get_parameter_value("variable_search_scope"))
        template_names = {variable.name for variable in parsed.get_variables()}

        if source is VariableSource.DICTIONARY_THEN_VARIABLES:
            missing_names = [name for name in template_names if name not in coerced]
            variable_values = self._get_normalized_variables(missing_names, scope)
            return {**variable_values, **coerced}

        variable_values = self._get_normalized_variables(list(template_names), scope)
        return {**coerced, **variable_values}

    def _get_normalized_variables(self, names: list[str], scope: VariableScope) -> dict[str, str | int]:
        """Read selected workflow Variables and normalize their values for macro resolution."""
        if not names:
            return {}

        variables = get_variables(self.name, names, scope)
        return {name: self._normalize_variable_value(value) for name, value in variables.items()}

    def _coerce_variables(self, raw_variables: Any) -> dict[str, str | int]:
        """Normalize the variables dict so ParsedMacro.resolve accepts it.

        Keys become strings; str and int values pass through (keeping
        format specs like {count:03} working). Everything else is
        str()-coerced.
        """
        return {str(key): self._normalize_variable_value(value) for key, value in raw_variables.items()}

    def _normalize_variable_value(self, value: Any) -> str | int:
        """Coerce a single value to what ParsedMacro.resolve accepts (str or int)."""
        # bool is a subclass of int; render it as "True"/"False" instead of 1/0.
        if isinstance(value, bool):
            return str(value)
        if isinstance(value, (str, int)):
            return value
        return str(value)

    def _report_failure(self, message: str, exception: Exception | None = None) -> None:
        self.set_parameter_value("resolved_string", "")
        self.parameter_output_values["resolved_string"] = ""
        self._set_status_results(was_successful=False, result_details=f"FAILURE: {message}")
        if exception is not None:
            self._handle_failure_exception(exception)
