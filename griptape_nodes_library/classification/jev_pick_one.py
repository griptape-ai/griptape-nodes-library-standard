from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.classification.jev_common import (
    DEFAULT_MODEL,
    QUESTION_KEY,
    add_context_parameter,
    add_model_group,
    to_state,
)
from griptape_nodes_library.classification.row_outputs import RowOutputsMixin, parse_row
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevPickOne"]

MAX_OPTIONS = 255


class JevPickOne(RowOutputsMixin, GriptapeProxyNode):
    """Pick the option that best fits some text using TypeSafe JEV via Griptape Cloud.

    Add options as 'Label' or 'Label: description' rows. Each option gets its own
    flow output. JEV routes the flow to the option it picks.

    Inputs:
        - context (str): The text JEV reads to pick an option.
        - question (str): Optional guidance on what JEV should decide.
        - options (list[str]): Two or more options as 'Label' or 'Label: description'.
        - model (str): JEV model alias.

    Outputs:
        - One flow output per option (dynamic, added as you fill in the list).
        - choice (str): The label of the option JEV picked.
        - description (str): The description of the picked option.
        - confidence (float): How sure JEV is, from 0 to 1.
        - probabilities (json): JEV's probability for every option, keyed by label.
    """

    ROWS_PARAM = "options"
    OUTPUT_PREFIX = "option_"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # SuccessFailureNode adds exec_out (Succeeded) and failure (Failed).
        # Remove them — our dynamic option outputs handle control flow instead.
        self.remove_parameter_element(self.control_parameter_out)
        self.remove_parameter_element(self.failure_output)

        add_context_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question",
                display_name="Question",
                tooltip="Optional. What JEV should decide, for example 'Which department should handle this?'",
                default_value="",
                placeholder_text="Which option fits the text best?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.options = ParameterList(
            name="options",
            display_name="Options",
            tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what "
            "the option means. Each option gets its own flow output.",
            type="str",
            ui_options={"placeholder_text": "label: description"},
            max_items=MAX_OPTIONS,
        )
        self.add_parameter(self.options)

        self.add_parameter(
            ParameterString(
                name="choice",
                display_name="Choice",
                tooltip="The label of the option JEV picked.",
                allow_input=False,
                allow_property=False,
                placeholder_text="The label JEV picked.",
            )
        )

        self.add_parameter(
            ParameterString(
                name="description",
                display_name="Description",
                tooltip="The description of the option JEV picked. Empty if the option had no description.",
                allow_input=False,
                allow_property=False,
                placeholder_text="The description of the picked option.",
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="confidence",
                display_name="Confidence",
                tooltip="How sure JEV is of its pick, from 0 to 1. "
                "Low values mean the text could fit another option too.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterJson(
                name="probabilities",
                display_name="Probabilities",
                tooltip="JEV's probability for every option, keyed by label. They add up to about 1.",
                allow_input=False,
                allow_property=False,
                placeholder_text="JEV's probability for every option, keyed by label.",
            )
        )

        add_model_group(self)

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors.",
            result_details_placeholder="JEV result will appear here.",
        )

    def _get_api_model_id(self) -> str:
        return self.get_parameter_value("model") or DEFAULT_MODEL

    def _row_output_label(self, index: int, text: str) -> str:  # noqa: ARG002
        return parse_row(text)[0]

    def _criteria(self) -> dict[str, str | None]:
        criteria: dict[str, str | None] = {}
        for row in self.get_parameter_value("options") or []:
            label, description = parse_row(row)
            if not label:
                continue
            if label in criteria:
                raise ValueError(
                    f"{self.name}: '{label}' appears more than once in Options. Each label must be unique."
                )
            criteria[label] = description
        if len(criteria) < 2:  # noqa: PLR2004
            raise ValueError(f"{self.name}: Options needs at least two options for JEV to pick from.")
        return criteria

    async def _build_payload(self) -> dict[str, Any]:
        state = to_state(self.get_parameter_value("context"))
        if state is None:
            raise ValueError(f"{self.name}: Context is empty.")

        criteria = self._criteria()
        question = (self.get_parameter_value("question") or "").strip()

        choice_q: dict[str, Any] = {
            "type": "choice",
            "criteria": {k: v or k for k, v in criteria.items()},
        }
        if question:
            choice_q["instructions"] = question

        return {"state": state, "questions": {QUESTION_KEY: choice_q}}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
        answer_data = (result_json.get("answers") or {}).get(QUESTION_KEY) or {}
        picked = answer_data.get("choice")
        if picked is None:
            self._set_safe_defaults()
            raise RuntimeError(f"{self.name}: No choice found in JEV response.")

        criteria = self._criteria()
        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        self.parameter_output_values["probabilities"] = dict(answer_data.get("probabilities") or {})
        self.parameter_output_values["description"] = criteria.get(picked) or ""
        self.parameter_output_values["choice"] = picked
        self._set_status_results(was_successful=True, result_details=f"JEV picked '{picked}'.")

    def _set_safe_defaults(self) -> None:
        for key in ("choice", "description", "confidence", "probabilities"):
            self.parameter_output_values.pop(key, None)

    def get_next_control_output(self) -> Parameter | None:
        picked = self.parameter_output_values.get("choice")
        if picked is None:
            return None
        for param in self._row_output_params():
            if param.display_name == picked:
                return param
        return None
