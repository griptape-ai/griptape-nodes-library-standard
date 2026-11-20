from __future__ import annotations

import fnmatch
from enum import StrEnum
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
from griptape_nodes.traits.options import Options
from ruamel.yaml import YAML

from griptape_nodes_library.classification.row_outputs import RowOutputsMixin

__all__ = ["MatchMode", "Switch", "matches", "value_text"]

MAX_CASES = 255
DEFAULT_OUTPUT = "default"
TRUE_TEXT = "true"
FALSE_TEXT = "false"


class MatchMode(StrEnum):
    EQUALS = "Equals"
    WILDCARD = "Wildcard"


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


def matches(value: Any, pattern: str, mode: MatchMode, *, case_sensitive: bool) -> bool:
    """Whether a case row's text matches the value."""
    pattern = pattern.strip()
    # Equals on a number compares numerically, so a row of '3.0' matches 3.
    if mode == MatchMode.EQUALS and isinstance(value, int | float) and not isinstance(value, bool):
        try:
            return float(pattern) == value
        except ValueError:
            return False

    text = value_text(value)
    if not case_sensitive:
        text, pattern = text.casefold(), pattern.casefold()
    match mode:
        case MatchMode.EQUALS:
            return text == pattern
        case MatchMode.WILDCARD:
            # fnmatchcase, because fnmatch folds case by operating system.
            return fnmatch.fnmatchcase(text, pattern)
        case _:
            msg = f"Unknown match mode: {mode!r}"
            raise ValueError(msg)


class Switch(RowOutputsMixin, ControlNode):
    """Send the flow down the exit whose case matches a value.

    Each case row gets its own flow exit. The first row, from the top, that matches the value
    wins. If none match, the flow takes Default.

    When 'Value' is a dictionary, JSON or YAML, set 'Path' to the field to switch on, using a
    JMESPath expression such as 'department' or 'details.size.label'.

    Inputs:
        - value (any): What to switch on. Text, a number, a bool, or structured data with a path.
        - path (str): A JMESPath expression that picks the field out of structured data.
        - match_mode (str): Equals, or Wildcard where '*' matches any text and '?' one character.
        - case_sensitive (bool): Whether 'Fx' and 'fx' differ.
        - cases (list[str]): One row per case. Bools match 'true' and 'false'.

    Outputs:
        - One flow exit per case row, and Default for when none match.
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
            display_name="Value",
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

        self.match_mode_param = ParameterString(
            name="match_mode",
            display_name="Match",
            tooltip="Equals needs the whole value to match. Wildcard also accepts '*' for any run of text "
            "and '?' for one character, so '*fx' matches 'fx' and 'cfx'.",
            default_value=MatchMode.EQUALS,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(MatchMode))},
        )
        self.add_parameter(self.match_mode_param)

        self.case_sensitive_param = ParameterBool(
            name="case_sensitive",
            display_name="Case sensitive",
            tooltip="When off, 'Fx' and 'fx' match each other.",
            default_value=False,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(self.case_sensitive_param)

        self.add_parameter(
            ParameterList(
                name="cases",
                display_name="Cases",
                tooltip="One case per row. The first row that matches the value, from the top, sets the exit. "
                "A bool matches 'true' or 'false'. Each case gets its own flow exit.",
                type="str",
                ui_options={"placeholder_text": "Text to match"},
                max_items=MAX_CASES,
            )
        )

        self.add_parameter(
            ControlParameterOutput(
                name=DEFAULT_OUTPUT,
                display_name="Default",
                tooltip="Taken when no case matches.",
            )
        )

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

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,  # noqa: ARG002
        source_parameter: Parameter,  # noqa: ARG002
        target_parameter: Parameter,
    ) -> None:
        if target_parameter is self.value_param:
            self._show_options_for(ParameterTypeBuiltin.ANY.value)

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
            (self.match_mode_param, show_text_options),
            (self.case_sensitive_param, show_text_options),
        ):
            if show:
                self.show_parameter_by_name(param.name)
            else:
                self.hide_parameter_by_name(param.name)

    def _selected_value(self) -> Any:
        """The value to switch on: 'Value', or the field 'Path' picks out of it."""
        value = self.get_parameter_value("value")
        path = (self.get_parameter_value("path") or "").strip()
        if not path:
            if value is None:
                raise NodeError("'Value' is empty. Connect a value to it, or type one.")
            return value

        data = value
        if isinstance(value, str):
            try:
                data = YAML(typ="safe").load(StringIO(value))
            except Exception as e:
                raise NodeError(
                    "'Value' isn't valid JSON or YAML, so 'Path' can't read it. "
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
            raise NodeError(f"'Value' has nothing at '{path}', or it's null. Check 'Path'.{keys}")
        return found

    def process(self) -> None:
        self._route = None
        selected = self._selected_value()
        if isinstance(selected, dict | list):
            kind = "a dictionary" if isinstance(selected, dict) else "a list"
            raise NodeError(
                f"The value to switch on is {kind}. Set 'Path' to one field inside it, such as 'department'."
            )

        mode = MatchMode(self.get_parameter_value("match_mode") or MatchMode.EQUALS)
        case_sensitive = bool(self.get_parameter_value("case_sensitive"))
        self._route = DEFAULT_OUTPUT
        for output_name, text in self._rows():
            if matches(selected, text, mode, case_sensitive=case_sensitive):
                self._route = output_name
                return

    def get_next_control_output(self) -> Parameter | None:
        if self._route is None:
            return None
        return self.get_parameter_by_name(self._route)
