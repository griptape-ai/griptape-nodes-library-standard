from __future__ import annotations

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevRate"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
MAX_LEVELS = 10

CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]

QUESTION_KEY = "answer"


def _parse_level_row(text: str) -> tuple[str, str]:
    """Split 'Label: description' into (label, description). Returns ('', text) when no colon."""
    if ":" in text:
        label, _, description = text.partition(":")
        return label.strip(), description.strip()
    return "", text.strip()


class JevRate(GriptapeProxyNode):
    """Rate text against levels you describe using TypeSafe JEV via the Griptape Cloud proxy.

    Define levels lowest-first. JEV assigns a score and the score rounds to a level.
    Because JEV weighs every level by probability, the score can fall between levels.

    Inputs:
        - context (str): The text JEV reads to rate.
        - question (str): Optional guidance on what JEV should rate.
        - levels (list[str]): Two to ten levels, lowest first. Use 'Label: description'
          to name a level or just a plain description.
        - model (str): JEV model alias.

    Outputs:
        - score (float): JEV's score, from 1 to the number of levels. Can fall between levels.
        - level (int): The score rounded to the nearest level, starting at 1.
        - level_description (str): The description of the level JEV scored.
        - confidence (float): How sure JEV is, from 0 to 1.
        - probabilities (dict): JEV's probability for every level, keyed by level number.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "classification"
        self.description = "Rate text against levels you describe using TypeSafe JEV"

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
                "'Minor: broken, but a workaround exists'.",
                type="str",
                ui_options={"placeholder_text": "Describe this level"},
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        model_param = ParameterString(
            name="model",
            display_name="Model",
            tooltip="jev-latest is the newest stable model. jev-preview is the newest release.",
            default_value=DEFAULT_MODEL,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
        )

        self.add_parameter(
            ParameterFloat(
                name="score",
                display_name="Score",
                tooltip="JEV's score, from 1 to the number of levels. It weighs every level by its "
                "probability, so it can fall between levels, like 1.7.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="level",
                display_name="Level",
                tooltip="The score rounded to the nearest level, as a whole number starting at 1.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterString(
                name="level_description",
                display_name="Level Description",
                tooltip="The description of the level JEV scored.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="The description of the level.",
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="confidence",
                display_name="Confidence",
                tooltip="How sure JEV is of its score, from 0 to 1. Low values mean JEV spread its "
                "probability across levels.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="probabilities",
                display_name="Probabilities",
                tooltip="JEV's probability for every level, keyed by level number (starting at 1).",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors",
            result_details_placeholder="JEV status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _level_descriptions(self) -> list[str]:
        """Read Levels rows into a list of plain descriptions for the JEV criteria array."""
        rows = self.get_parameter_value("levels") or []
        descriptions = []
        for row in rows:
            _, description = _parse_level_row(row)
            if description:
                descriptions.append(description)
        return descriptions

    def _level_rows(self) -> list[tuple[str, str]]:
        """Read Levels rows as (label, description) pairs."""
        rows = self.get_parameter_value("levels") or []
        result = []
        for i, row in enumerate(rows):
            label, description = _parse_level_row(row)
            result.append((label or f"Level {i + 1}", description or row.strip()))
        return result

    async def _build_payload(self) -> dict[str, Any]:
        context = (self.get_parameter_value("context") or "").strip()
        if not context:
            msg = f"{self.name}: Context is empty. Connect or type the text to rate."
            raise ValueError(msg)

        state: str | dict | list = context
        if context[0] in "{[":
            try:
                state = json.loads(context)
            except json.JSONDecodeError:
                pass

        descriptions = self._level_descriptions()
        if len(descriptions) < 2:  # noqa: PLR2004
            msg = f"{self.name}: Levels needs at least two levels for JEV to rate against."
            raise ValueError(msg)
        if len(descriptions) > MAX_LEVELS:
            msg = f"{self.name}: Levels has {len(descriptions)} levels. JEV accepts up to {MAX_LEVELS}."
            raise ValueError(msg)

        question = (self.get_parameter_value("question") or "").strip()

        score: dict[str, Any] = {"type": "score", "criteria": descriptions}
        if question:
            score["instructions"] = question

        return {"state": state, "questions": {QUESTION_KEY: score}}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
        answers = result_json.get("answers") or {}
        answer_data = answers.get(QUESTION_KEY) or {}
        raw_score = answer_data.get("score")

        if raw_score is None:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details="No score found in JEV response.")
            return

        level_rows = self._level_rows()
        # JEV scores from 0 internally; display from 1.
        jev_score = float(raw_score)
        # Round half-up: 1.5 → 2 rather than to-even.
        index = min(int(jev_score + 0.5), len(level_rows) - 1)
        display_score = jev_score + 1
        display_level = index + 1
        _, description = level_rows[index]

        self.parameter_output_values["score"] = display_score
        self.parameter_output_values["level"] = display_level
        self.parameter_output_values["level_description"] = description
        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        # Re-key probabilities from 0-based to 1-based to match displayed levels.
        raw_probs = answer_data.get("probabilities") or {}
        self.parameter_output_values["probabilities"] = {
            str(int(k) + 1): v for k, v in raw_probs.items()
        }
        self._set_status_results(
            was_successful=True,
            result_details=f"Level {display_level}: {description} (score {display_score:.1f})",
        )

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["score"] = None
        self.parameter_output_values["level"] = None
        self.parameter_output_values["level_description"] = None
        self.parameter_output_values["confidence"] = None
        self.parameter_output_values["probabilities"] = None
