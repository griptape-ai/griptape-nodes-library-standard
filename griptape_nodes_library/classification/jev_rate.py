from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from griptape_nodes.exe_types.core_types import ControlParameterInput, Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.classification.row_outputs import RowOutputsMixin, parse_row

logger = logging.getLogger(__name__)

__all__ = ["JevRate"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
API_KEY_NAME = "TYPESAFE_API_KEY"
MAX_LEVELS = 10

CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]
QUESTION_KEY = "answer"


class JevRate(RowOutputsMixin):
    """Rate text against levels you describe using TypeSafe JEV.

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
        - score (float): JEV's score, from 1 to the number of levels. Can fall between levels.
        - level (int): The score rounded to the nearest level, starting at 1.
        - level_description (str): The description of the level JEV scored.
        - confidence (float): How sure JEV is, from 0 to 1.
        - probabilities (dict): JEV's probability for every level, keyed by level number.
    """

    ROWS_PARAM = "levels"
    OUTPUT_PREFIX = "rate_level_"

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)
        self._route: str | None = None

        self.add_parameter(ControlParameterInput(tooltip="Run this node", name="exec_in"))

        self.add_parameter(
            ParameterString(
                name="context",
                display_name="Context",
                tooltip="The text JEV reads to rate. JSON works too.",
                default_value="",
                multiline=True,
                placeholder_text="Text to rate",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=CONTEXT_INPUT_TYPES,
            )
        )

        self.add_parameter(
            ParameterString(
                name="question",
                display_name="Question",
                tooltip="Optional. What JEV should rate, for example 'How urgent is this message?'",
                default_value="",
                placeholder_text="How ... is the text?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self.add_parameter(
            ParameterList(
                name="levels",
                display_name="Levels",
                tooltip="One level per row, lowest first, up to 10. Describe the situation at each level, "
                "like 'Broken, but a workaround exists'. To name a level, put a label before a colon: "
                "'Minor: broken, but a workaround exists'. Each level gets its own flow output.",
                type="str",
                ui_options={"placeholder_text": "Describe this level"},
                max_items=MAX_LEVELS,
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="score",
                display_name="Score",
                tooltip="JEV's score, from 1 to the number of levels. It weighs every level by its "
                "probability, so it can fall between levels, like 1.7.",
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
                tooltip="How sure JEV is of its score, from 0 to 1. Low values mean JEV spread its "
                "probability across levels.",
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

        with ParameterGroup(name="Advanced", ui_options={"collapsed": True}):
            ParameterString(
                name="model",
                display_name="Model",
                tooltip="jev-latest is the newest stable model. jev-preview is the newest release.",
                default_value=DEFAULT_MODEL,
                allow_input=False,
                allow_output=False,
            )

    def _row_output_label(self, index: int, text: str) -> str:
        label, description = parse_row(text)
        return label if description else f"Level {index + 1}"

    def _row_output_tooltip(self, label: str) -> str:
        return f"Taken when the score rounds to {label}."

    def _level_rows(self) -> list[tuple[str, str]]:
        """Return (output_name, description) for each non-empty level row."""
        rows_param = self.get_parameter_by_name("levels")
        if not isinstance(rows_param, ParameterList):
            return []
        result = []
        for i, row in enumerate(rows_param.get_child_parameters()):
            text = (self.get_parameter_value(row.name) or "").strip()
            if not text:
                continue
            _, description = parse_row(text)
            output_name = self.OUTPUT_PREFIX + row.name.rsplit("_", 1)[-1]
            result.append((output_name, description or text))
        return result

    def validate_before_node_run(self) -> list[Exception] | None:
        if not GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME, should_error_on_not_found=False):
            return [
                ValueError(
                    f"{self.name}: {API_KEY_NAME} is not set. "
                    "Add it in Settings > API Keys & Secrets. Get a key at https://console.typesafe.ai/keys"
                )
            ]
        return None

    def process(self) -> AsyncResult[None]:
        self._route = None
        yield lambda: self._ask()

    def _ask(self) -> None:
        context = (self.get_parameter_value("context") or "").strip()
        if not context:
            raise ValueError(f"{self.name}: Context is empty.")

        state: str | dict | list = context
        if context[0] in "{[":
            try:
                state = json.loads(context)
            except json.JSONDecodeError:
                pass

        level_rows = self._level_rows()
        descriptions = [desc for _, desc in level_rows]
        if len(descriptions) < 2:  # noqa: PLR2004
            raise ValueError(f"{self.name}: Levels needs at least two levels for JEV to rate against.")
        if len(descriptions) > MAX_LEVELS:
            raise ValueError(f"{self.name}: Levels has {len(descriptions)} levels. JEV accepts up to {MAX_LEVELS}.")

        question = (self.get_parameter_value("question") or "").strip()
        score_q: dict[str, Any] = {"type": "score", "criteria": descriptions}
        if question:
            score_q["instructions"] = question

        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME, should_error_on_not_found=False)
        model = self.get_parameter_value("model") or DEFAULT_MODEL

        response = httpx.post(
            "https://api.typesafe.ai/v1/systemone",
            json={"model": model, "state": state, "questions": {QUESTION_KEY: score_q}},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        answer_data = (body.get("answers") or {}).get(QUESTION_KEY) or {}
        raw_score = answer_data.get("score")
        if raw_score is None:
            raise RuntimeError(f"{self.name}: No score found in JEV response.")

        jev_score = float(raw_score)
        # Round half-up: 1.5 → 2 rather than to-even. JEV numbers from 0; display from 1.
        index = min(int(jev_score + 0.5), len(level_rows) - 1)
        output_name, description = level_rows[index]

        self.parameter_output_values["score"] = jev_score + 1
        self.parameter_output_values["level"] = index + 1
        self.parameter_output_values["level_description"] = description
        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        raw_probs = answer_data.get("probabilities") or {}
        self.parameter_output_values["probabilities"] = {str(int(k) + 1): v for k, v in raw_probs.items()}
        self._route = output_name

    def get_next_control_output(self) -> Parameter | None:
        if self._route is None:
            return None
        return self.get_parameter_by_name(self._route)
