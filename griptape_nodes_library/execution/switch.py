from __future__ import annotations

import fnmatch
from io import StringIO
from typing import Any

import jmespath
from griptape_nodes.exe_types.core_types import (
    ControlParameterOutput,
    NodeError,
    Parameter,
    ParameterList,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_types import BaseNode, ControlNode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from ruamel.yaml import YAML

from griptape_nodes_library.classification.row_outputs import RowOutputsMixin

__all__ = ["Switch", "matches", "value_text"]

MAX_CASES = 255
DEFAULT_OUTPUT = "default"
VALUES_PARAM = "values"
DEFAULT_RESULT = "default_result"
RESULT_PARAM = "result"
TRUE_TEXT = "true"
FALSE_TEXT = "false"
WILDCARD_CHARS = "*?"


def value_text(value: Any) -> str:
    """The text a case row is compared against. Bools read 'true'/'false', whole numbers drop the '.0'."""
    match value:
        case None:
            return ""
        case bool():
            return TRUE_TEXT if value else FALSE_TEXT
        case float() if value.is_integer():
            return str(int(value))
        case _:
            return str(value)


def matches(value: Any, pattern: str, *, case_sensitive: bool, patterns: bool = True) -> bool:
    """Whether a case row's text matches the value.

    With `patterns` on, a row with '*' (any run of text) or '?' (one character) is a wildcard
    pattern. Any other row, or any row with `patterns` off, must match the whole value.
    """
    pattern = pattern.strip()
    is_wildcard = patterns and any(char in pattern for char in WILDCARD_CHARS)
    # An exact row on a number compares numerically, so a row of '3.0' matches 3.
    if not is_wildcard and isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return float(pattern) == value
        except ValueError:
            return False

    text = value_text(value)
    if not case_sensitive:
        text, pattern = text.casefold(), pattern.casefold()
    if is_wildcard:
        # Escape '[' so only * and ? are special. fnmatchcase, because fnmatch folds case by operating system.
        return fnmatch.fnmatchcase(text, pattern.replace("[", "[[]"))
    return text == pattern


class Switch(RowOutputsMixin, ControlNode):
    """Send the flow down the exit whose case matches a value.

    Each case row gets its own flow exit. The first row, from the top, that matches the value
    wins. If none match, the flow takes Default. With 'Pattern matching' on, a row with '*' (any
    text) or '?' (one character) is a pattern, so '*fx' matches 'fx' and 'cfx'. Any other row must
    match the whole value.

    When 'Switch on' is a dictionary, JSON or YAML, set 'Path' to the field to switch on, using a
    JMESPath expression such as 'department' or 'details.size.label'.

    Each case can also have an optional input, a row in 'Optional inputs', for the data to pass on when that case matches. Type text
    into it, or connect anything. The matching case's data comes out of 'Result'. Every connected
    value is computed, matching or not, so wire cheap data here and use the exits to run expensive
    nodes only for the case that matched.

    Inputs:
        - value (any, 'Switch on'): What to switch on. Text, a number, a bool, or structured data with a path.
        - path (str): A JMESPath expression that picks the field out of structured data.
        - case_sensitive (bool): Whether 'Fx' and 'fx' differ.
        - pattern_matching (bool): Whether '*' and '?' in a case row are patterns. Turn off to match them literally.
        - cases (list[str]): One row per case. Use '*' and '?' for patterns. Bools match 'true' and 'false'.
        - values (list, 'Optional inputs'): The data for each case, one row per case row, kept in step with 'cases'.
        - default_result (any, 'Default input'): The data when no case matches.

    Outputs:
        - One flow exit per case row, and Default for when none match.
        - result (any): The data of the case that matched, or the default input.
    """

    ROWS_PARAM = "cases"
    # Not "case_": the Case sensitive parameter starts with that.
    OUTPUT_PREFIX = "switch_case_"
    TRAILING_OUTPUTS = (DEFAULT_OUTPUT,)

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._route: str | None = None

        # The case exits and Default replace the single Flow Out.
        self.remove_parameter_element(self.control_parameter_out)

        self.value_param = Parameter(
            name="value",
            display_name="Switch on",
            type=ParameterTypeBuiltin.ANY.value,
            input_types=[ParameterTypeBuiltin.ANY.value],
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            tooltip="What to switch on: text, a number, a bool, or a dictionary, JSON or YAML with a Path.",
        )
        self.add_parameter(self.value_param)

        self.path_param = ParameterString(
            name="path",
            display_name="Path",
            tooltip="For a dictionary, JSON or YAML value: the field to switch on, as a JMESPath expression "
            "such as 'department' or 'details.size.label'. Leave empty to use the value itself.",
            default_value="",
            placeholder_text="ex: department, details.size.label",
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.path_param.set_badge(
            variant="help",
            title="JMESPath Syntax",
            message="`department` — a field\n`details.size.label` — a nested field\n`items[0].name` — an array item\n\n"
            "[JMESPath Docs](https://jmespath.org/)",
        )
        self.add_parameter(self.path_param)

        self.case_sensitive_param = ParameterBool(
            name="case_sensitive",
            display_name="Case sensitive",
            tooltip="When off, 'Fx' and 'fx' match each other.",
            default_value=False,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.case_sensitive_param)

        self.pattern_matching_param = ParameterBool(
            name="pattern_matching",
            display_name="Pattern matching",
            tooltip="When on, '*' in a case matches any text and '?' matches one character, so '*fx' matches "
            "'fx' and 'cfx'. Turn off to match '*' and '?' as written.",
            default_value=True,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.pattern_matching_param)

        self.add_parameter(
            ParameterList(
                name="cases",
                display_name="Cases",
                tooltip="One case per row. The first row that matches the value, from the top, sets the exit. "
                "With Pattern matching on, '*' matches any text and '?' one character, so '*fx' matches 'fx' and 'cfx'. "
                "A bool matches 'true' or 'false'. Each case gets its own flow exit.",
                type="str",
                ui_options={"placeholder_text": "Text to match"},
                max_items=MAX_CASES,
            )
        )

        self.add_parameter(
            ParameterList(
                name=VALUES_PARAM,
                display_name="Optional inputs",
                tooltip="An optional input per case: the data to pass on when that case matches, one row per case row, in the same order. "
                "Type text, or connect anything. 'Result' carries the matching case's data. Everything "
                "connected here is computed whether or not its case matches.",
                type=ParameterTypeBuiltin.ANY.value,
                input_types=[ParameterTypeBuiltin.ANY.value],
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            Parameter(
                name=DEFAULT_RESULT,
                display_name="Default input",
                type=ParameterTypeBuiltin.ANY.value,
                input_types=[ParameterTypeBuiltin.ANY.value],
                default_value=None,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="The data for 'Result' when no case matches. Optional.",
            )
        )

        self.add_parameter(
            Parameter(
                name=RESULT_PARAM,
                display_name="Result",
                output_type=ParameterTypeBuiltin.ALL.value,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The data of the case that matched, or the default result.",
            )
        )

        self.add_parameter(
            ControlParameterOutput(
                name=DEFAULT_OUTPUT,
                display_name="Default",
                tooltip="Taken when no case matches.",
            )
        )

    @staticmethod
    def _data_row_name(case_row_name: str) -> str:
        """The 'Optional inputs' row that goes with a case row, keyed by the case row's ID."""
        return f"{VALUES_PARAM}_ParameterListUniqueParamID_{case_row_name.rsplit('_', 1)[-1]}"

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        # A row added or deleted in 'Optional inputs' by hand is put right: it has to match 'Cases'.
        if parameter.name == VALUES_PARAM:
            self._sync_row_outputs()
        return super().after_value_set(parameter, value)

    def _sync_row_companions(self) -> None:
        """Give every case that has text a row in 'Optional inputs', in the same order, named after its case row's ID.

        Like the exits, a row is made only once its case has text, so a saved Switch that is loading, with
        case rows that have no value yet, never gets a second copy of rows it is about to restore. A row is
        removed only when its case row is gone.
        """
        cases = self.get_parameter_by_name(self.ROWS_PARAM)
        values = self.get_parameter_by_name(VALUES_PARAM)
        if not isinstance(cases, ParameterList) or not isinstance(values, ParameterList):
            return

        case_rows = cases.get_child_parameters()
        case_data_names = {row.name: self._data_row_name(row.name) for row in case_rows}
        for child in values.get_child_parameters():
            if child.name not in case_data_names.values():
                self._delete_input_connections(child.name)
                values.remove_child(child)

        have = {child.name for child in values.get_child_parameters()}
        texts = {row.name: (self.parameter_values.get(row.name) or "").strip() for row in case_rows}
        for row in case_rows:
            name = case_data_names[row.name]
            if texts[row.name] and name not in have:
                values.add_child_parameter().name = name

        have = {child.name: child for child in values.get_child_parameters()}
        position = 0
        for row in case_rows:
            child = have.get(case_data_names[row.name])
            if child is None:
                continue
            current = values.get_child_parameters().index(child)
            if current != position:
                values.move_child(current, position)
            position += 1
            # A case with no text yet keeps the label it has: a loading Switch's rows get their text later.
            label = texts[row.name]
            if label and child.display_name != label:
                child.display_name = label
                child.tooltip = f"The data for 'Result' when the value matches '{label}'."

    def _row_output_label(self, index: int, text: str) -> str:  # noqa: ARG002
        return text.strip()

    def _row_output_tooltip(self, label: str) -> str:
        return f"Taken when the value matches '{label}'."

    def after_incoming_connection(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if target_parameter is self.value_param:
            self._show_options_for(source_parameter.output_type)
            if source_parameter.output_type == ParameterTypeBuiltin.BOOL:
                self._seed_bool_cases()

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,  # noqa: ARG002
        target_parameter: Parameter,
    ) -> None:
        if target_parameter is self.value_param:
            self._show_options_for(ParameterTypeBuiltin.ANY.value)

    def _seed_bool_cases(self) -> None:
        """Give a Switch with no case rows a 'true' and a 'false' row, the only values a bool can have.

        Any existing row stops it, even one with no text yet. A saved workflow restores its connections
        before it sets its values, so a loading Switch has its rows but not their text, and seeding then
        would add a second pair on every reload.
        """
        rows = self.get_parameter_by_name(self.ROWS_PARAM)
        if not isinstance(rows, ParameterList) or rows.get_child_parameters():
            return
        for text in (TRUE_TEXT, FALSE_TEXT):
            self.set_parameter_value(rows.add_child_parameter().name, text)

    def _show_options_for(self, value_type: str) -> None:
        """Show only the options that mean something for the connected type."""
        match value_type:
            case ParameterTypeBuiltin.BOOL | ParameterTypeBuiltin.INT | ParameterTypeBuiltin.FLOAT:
                show_path, show_text_options = False, False
            case ParameterTypeBuiltin.STR:
                show_path, show_text_options = False, True
            case _:
                # json, dict, yaml, any and everything else: the field inside could be anything.
                show_path, show_text_options = True, True
        for param, show in (
            (self.path_param, show_path),
            (self.case_sensitive_param, show_text_options),
            (self.pattern_matching_param, show_text_options),
        ):
            if show:
                self.show_parameter_by_name(param.name)
            else:
                self.hide_parameter_by_name(param.name)

    def _selected_value(self) -> Any:
        """The value to switch on: 'Switch on', or the field 'Path' picks out of it."""
        value = self.get_parameter_value("value")
        path = "" if self.path_param.hide else (self.get_parameter_value("path") or "").strip()
        if not path:
            if value is None:
                raise NodeError("'Switch on' is empty. Connect a value to it, or type one.")
            return value

        data = value
        if isinstance(value, str):
            try:
                data = YAML(typ="safe").load(StringIO(value))
            except Exception as e:
                raise NodeError(
                    "'Switch on' isn't valid JSON or YAML, so 'Path' can't read it. "
                    "Fix the text, or clear 'Path' to switch on the text itself."
                ) from e
        try:
            found = jmespath.search(path, data)
        except ValueError as e:
            raise NodeError(
                f"'{path}' in 'Path' isn't a valid JMESPath expression: {e}. "
                "Use a field name like 'department', or a nested one like 'details.size.label'."
            ) from e
        if found is None:
            keys = f" The top-level fields are: {', '.join(map(str, data))}." if isinstance(data, dict) else ""
            raise NodeError(f"'Switch on' has nothing at '{path}', or it's null. Check 'Path'.{keys}")
        return found

    def process(self) -> None:
        self._route = None
        self.parameter_output_values.pop(RESULT_PARAM, None)
        selected = self._selected_value()
        if isinstance(selected, dict | list):
            kind = "a dictionary" if isinstance(selected, dict) else "a list"
            raise NodeError(
                f"The value to switch on is {kind}. Set 'Path' to one field inside it, such as 'department'."
            )

        # A hidden option is one that means nothing for the connected type, so it has no effect. Its
        # value is kept, in case the type changes back.
        case_sensitive = not self.case_sensitive_param.hide and bool(self.get_parameter_value("case_sensitive"))
        patterns = self.pattern_matching_param.hide or self.get_parameter_value("pattern_matching") is not False
        self._route = DEFAULT_OUTPUT
        result_name = DEFAULT_RESULT
        for output_name, text in self._rows():
            if matches(selected, text, case_sensitive=case_sensitive, patterns=patterns):
                self._route = output_name
                result_name = self._data_row_name(output_name)
                break
        self.parameter_output_values[RESULT_PARAM] = self.get_parameter_value(result_name)

    def get_next_control_output(self) -> Parameter | None:
        if self._route is None:
            return None
        return self.get_parameter_by_name(self._route)
