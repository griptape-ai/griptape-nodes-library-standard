from __future__ import annotations

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevPickOne"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]

CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]

QUESTION_KEY = "answer"


def _parse_option_row(text: str) -> tuple[str, str]:
    """Split 'Label: description' into (label, description). Returns (text, '') when no colon."""
    if ":" in text:
        label, _, description = text.partition(":")
        return label.strip(), description.strip()
    return text.strip(), ""


class JevPickOne(GriptapeProxyNode):
    """Pick the option that best fits some text using TypeSafe JEV via the Griptape Cloud proxy.

    Add options as 'Label' or 'Label: description' rows. JEV picks the one that fits
    the context best and returns its label, description, confidence, and probabilities.

    Inputs:
        - context (str): The text JEV reads to pick an option.
        - question (str): Optional guidance on what JEV should decide.
        - options (list[str]): Two or more options as 'Label' or 'Label: description'.
        - model (str): JEV model alias.

    Outputs:
        - choice (str): The label of the option JEV picked.
        - description (str): The description of the picked option (empty if none given).
        - confidence (float): How sure JEV is, from 0 to 1.
        - probabilities (dict): JEV's probability for every option, keyed by label.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "classification"
        self.description = "Pick the option that best fits some text using TypeSafe JEV"

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

        self.add_parameter(
            ParameterList(
                name="options",
                display_name="Options",
                tooltip="One option per row. Write a short label, or 'Label: description' to tell JEV what "
                "the option means. Needs at least two options.",
                type="str",
                ui_options={"placeholder_text": "label: description"},
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
            ParameterString(
                name="choice",
                display_name="Choice",
                tooltip="The label of the option JEV picked.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="The label JEV picked.",
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterString(
                name="description",
                display_name="Description",
                tooltip="The description of the option JEV picked. Empty if the option had no description.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="The description of the picked option.",
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="confidence",
                display_name="Confidence",
                tooltip="How sure JEV is of its pick, from 0 to 1. Low values mean the text could fit "
                "another option too.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="probabilities",
                display_name="Probabilities",
                tooltip="JEV's probability for every option, keyed by label. They add up to about 1.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors",
            result_details_placeholder="JEV status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    def _build_criteria(self) -> dict[str, str | None]:
        """Read Options into JEV criteria: labels mapped to descriptions (or None)."""
        criteria: dict[str, str | None] = {}
        for row in self.get_parameter_value("options") or []:
            label, description = _parse_option_row(row)
            if not label:
                continue
            if label in criteria:
                msg = f"{self.name}: '{label}' appears more than once in Options. Each label must be unique."
                raise ValueError(msg)
            criteria[label] = description or None
        if len(criteria) < 2:  # noqa: PLR2004
            msg = f"{self.name}: Options needs at least two options for JEV to pick from."
            raise ValueError(msg)
        return criteria

    async def _build_payload(self) -> dict[str, Any]:
        context = (self.get_parameter_value("context") or "").strip()
        if not context:
            msg = f"{self.name}: Context is empty. Connect or type the text to classify."
            raise ValueError(msg)

        state: str | dict | list = context
        if context[0] in "{[":
            try:
                state = json.loads(context)
            except json.JSONDecodeError:
                pass

        criteria = self._build_criteria()
        question = (self.get_parameter_value("question") or "").strip()

        choice: dict[str, Any] = {
            "type": "choice",
            "criteria": {k: v for k, v in criteria.items() if v is not None} if any(v for v in criteria.values()) else {k: k for k in criteria},
        }
        if question:
            choice["instructions"] = question

        return {"state": state, "questions": {QUESTION_KEY: choice}}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
        answers = result_json.get("answers") or {}
        answer_data = answers.get(QUESTION_KEY) or {}
        picked = answer_data.get("choice")

        if picked is None:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details="No choice found in JEV response.")
            return

        # Recover the description from the options the user entered.
        options_criteria: dict[str, str | None] = {}
        for row in self.get_parameter_value("options") or []:
            label, description = _parse_option_row(row)
            if label:
                options_criteria[label] = description or None

        self.parameter_output_values["choice"] = picked
        self.parameter_output_values["description"] = options_criteria.get(picked) or ""
        self.parameter_output_values["confidence"] = float(answer_data.get("confidence", 0.0))
        self.parameter_output_values["probabilities"] = dict(answer_data.get("probabilities") or {})
        self._set_status_results(was_successful=True, result_details=f"Picked: {picked}")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["choice"] = None
        self.parameter_output_values["description"] = None
        self.parameter_output_values["confidence"] = None
        self.parameter_output_values["probabilities"] = None
