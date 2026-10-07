from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.classification.jev_common import (
    QUESTION_KEY,
    add_context_parameter,
    add_model_group,
    to_state,
)
from griptape_nodes_library.classification.row_outputs import RowOutputsMixin, parse_row
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevRate"]

MAX_LEVELS = 10


class JevRate(RowOutputsMixin, GriptapeProxyNode):
    """Rate text against levels you describe using TypeSafe JEV via Griptape Cloud.

    Define levels lowest-first. Each level gets its own flow output.
    JEV routes the flow to the level it scores.

    Inputs:
        - context (str): The text JEV reads to rate.
        - question (str): Optional guidance on what JEV should rate.
        - levels (list[str]): Two to ten levels, lowest first. Use 'Label: description'
          to name a level or just a plain description.
        - model (str): JEV model alias.

    Outputs:
        - One flow output per level (dynamic, added as you fill in the list).
        - score (float): JEV's score, from 1 to the number of levels.
        - level (int): The score rounded to the nearest level, starting at 1.
        - level_description (str): The description of the level JEV scored.
        - confidence (float): How sure JEV is, from 0 to 1.
        - probabilities (json): JEV's probability for every level, keyed by level number.
    """

    ROWS_PARAM = "levels"
    OUTPUT_PREFIX = "rate_level_"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._route: str | None = None

        # SuccessFailureNode adds exec_out (Succeeded) and failure (Failed).
        # Remove them — our dynamic level outputs handle control flow instead.
        self.remove_parameter_element(self.control_parameter_out)
        self.remove_parameter_element(self.failure_output)

        add_context_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question",
                display_name="Question",
                tooltip="Optional. What JEV should rate, "
                "for example 'How urgent is this message?'",
                default_value="",
                placeholder_text="How ... is the text?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterList(
                name="levels",
                display_name="Levels",
                tooltip="One level per row, lowest first, up to 10. Describe the situation at each level. "
                "To name a level, put a label before a colon: 'Minor: broken but a workaround exists'. "
                "Each level gets its own flow output.",
                type="str",
                ui_options={"placeholder_text": "Describe this level"},
                max_items=MAX_LEVELS,
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="score",
                display_name="Score",
                tooltip="JEV's score, from 1 to the number of levels. "
                "It weighs every level by its probability, so it can fall between levels.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterInt(
                name="level",
                display_name="Level",
                tooltip="The score rounded to the nearest level, as a whole number starting at 1.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterString(
                name="level_description",
                display_name="Level Description",
                tooltip="The description of the level JEV scored.",
                allow_input=False,
                allow_property=False,
                placeholder_text="The description of the level.",
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="confidence",
                display_name="Confidence",
                tooltip="How sure JEV is of its score, from 0 to 1. "
                "Low values mean JEV spread its probability across levels.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterJson(
                name="probabilities",
                display_name="Probabilities",
                tooltip="JEV's probability for every level, keyed by level number (starting at 1).",
                allow_input=False,
                allow_property=False,
                placeholder_text="JEV's probability for every level, keyed by level number.",
            )
        )

        add_model_group(self)

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors.",
            result_details_placeholder="JEV result will appear here.",
        )

    def _get_api_model_id(self) -> str:
        return self.get_parameter_value("model") or "jev-latest"

    def _row_output_label(self, index: int, text: str) -> str:
        label, description = parse_row(text)
        return label if description else f"Level {index + 1}"

    def _row_output_tooltip(self, label: str) -> str:
        return f"Taken when the score rounds to {label}."

    async def _build_payload(self) -> dict[str, Any]:
        state = to_state(self.get_parameter_value("context"))
        if state is None:
            raise ValueError(f"{self.name}: Context is empty.")

        rows = self._rows()
        descriptions = [parse_row(text)[1] or text for _, text in rows]
        if len(descriptions) < 2:  # noqa: PLR2004
            raise ValueError(f"{self.name}: Levels needs at least two levels for JEV to rate against.")
        if len(descriptions) > MAX_LEVELS:
            raise ValueError(
                f"{self.name}: Levels has {len(descriptions)} levels. JEV accepts up to {MAX_LEVELS}."
            )

        question = (self.get_parameter_value("question") or "").strip()
        score_q: dict[str, Any] = {"type": "score", "criteria": descriptions}
        if question:
            score_q["instructions"] = question

        return {"state": state, "questions": {QUESTION_KEY: score_q}}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
        self._route = None
        answer_data = (result_json.get("answers") or {}).get(QUESTION_KEY) or {}
        raw_score = answer_data.get("score")
        if raw_score is None:
            self._set_safe_defaults()
            raise RuntimeError(f"{self.name}: No score found in JEV response.")

        rows = self._rows()
        jev_score = float(raw_score)
        # Round half-up; JEV numbers from 0, display from 1.
        index = min(int(jev_score + 0.5), len(rows) - 1)
        output_name, text = rows[index]
        description = parse_row(text)[1] or text

        self.parameter_output_values["score"] = jev_score + 1
        self.parameter_output_values["level"] = index + 1
        self.parameter_output_values["level_description"] = description
        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        raw_probs = answer_data.get("probabilities") or {}
        self.parameter_output_values["probabilities"] = {str(int(k) + 1): v for k, v in raw_probs.items()}
        self._route = output_name
        self._set_status_results(was_successful=True, result_details=f"JEV scored level {index + 1}.")

    def _set_safe_defaults(self) -> None:
        self._route = None
        for key in ("score", "level", "level_description", "confidence", "probabilities"):
            self.parameter_output_values.pop(key, None)

    def get_next_control_output(self) -> Parameter | None:
        if self._route is None:
            return None
        return self.get_parameter_by_name(self._route)
