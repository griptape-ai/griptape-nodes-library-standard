from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from griptape_nodes.exe_types.core_types import ControlParameterInput, Parameter, ParameterGroup, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.classification.row_outputs import RowOutputsMixin, parse_row

logger = logging.getLogger(__name__)

__all__ = ["JevPickOne"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
API_KEY_NAME = "TYPESAFE_API_KEY"
MAX_OPTIONS = 255

CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]
QUESTION_KEY = "answer"


class JevPickOne(RowOutputsMixin):
    """Pick the option that best fits some text using TypeSafe JEV.

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
        - probabilities (dict): JEV's probability for every option, keyed by label.
    """

    ROWS_PARAM = "options"
    OUTPUT_PREFIX = "option_"

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(ControlParameterInput(tooltip="Run this node", name="exec_in"))

        self.add_parameter(
            ParameterString(
                name="context",
                display_name="Context",
                tooltip="The text JEV reads to pick an option. JSON works too.",
                default_value="",
                multiline=True,
                placeholder_text="Text to classify",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=CONTEXT_INPUT_TYPES,
            )
        )

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
                tooltip="How sure JEV is of its pick, from 0 to 1. Low values mean the text could fit "
                "another option too.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="probabilities",
                display_name="Probabilities",
                tooltip="JEV's probability for every option, keyed by label. They add up to about 1.",
                allow_input=False,
                allow_property=False,
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
        return parse_row(text)[0]

    def _criteria(self) -> dict[str, str | None]:
        criteria: dict[str, str | None] = {}
        for row in self.get_parameter_value("options") or []:
            label, description = parse_row(row)
            if not label:
                continue
            if label in criteria:
                raise ValueError(f"{self.name}: '{label}' appears more than once in Options. Each label must be unique.")
            criteria[label] = description
        if len(criteria) < 2:  # noqa: PLR2004
            raise ValueError(f"{self.name}: Options needs at least two options for JEV to pick from.")
        return criteria

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
        self.parameter_output_values.pop("choice", None)
        self.parameter_output_values.pop("description", None)
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

        criteria = self._criteria()
        question = (self.get_parameter_value("question") or "").strip()

        choice_q: dict[str, Any] = {
            "type": "choice",
            "criteria": {k: v for k, v in criteria.items() if v is not None} or {k: k for k in criteria},
        }
        if question:
            choice_q["instructions"] = question

        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME, should_error_on_not_found=False)
        model = self.get_parameter_value("model") or DEFAULT_MODEL

        response = httpx.post(
            "https://api.typesafe.ai/v1/systemone",
            json={"model": model, "state": state, "questions": {QUESTION_KEY: choice_q}},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        answer_data = (body.get("answers") or {}).get(QUESTION_KEY) or {}
        picked = answer_data.get("choice")
        if picked is None:
            raise RuntimeError(f"{self.name}: No choice found in JEV response.")

        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        self.parameter_output_values["probabilities"] = dict(answer_data.get("probabilities") or {})
        self.parameter_output_values["description"] = criteria.get(picked) or ""
        self.parameter_output_values["choice"] = picked

    def get_next_control_output(self) -> Parameter | None:
        picked = self.parameter_output_values.get("choice")
        if picked is None:
            return None
        for param in self._row_output_params():
            if param.display_name == picked:
                return param
        return None
