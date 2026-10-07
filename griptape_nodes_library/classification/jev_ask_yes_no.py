from __future__ import annotations

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevAskYesNo"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]

# JEV reads text only; image/audio/video outputs cannot connect to Context.
CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]

QUESTION_KEY = "answer"


class JevAskYesNo(GriptapeProxyNode):
    """Ask a yes/no question about text using TypeSafe JEV via the Griptape Cloud proxy.

    JEV reads the context and returns a probability that the answer is yes.
    The threshold controls when the answer flips from No to Yes.

    Inputs:
        - context (str): The text JEV reads to answer the question.
        - question (str): A yes/no question about the context.
        - yes_means (str): Optional description of what should count as Yes.
        - no_means (str): Optional description of what should count as No.
        - threshold (float): Minimum probability to answer Yes (default 0.5).
        - model (str): JEV model alias.

    Outputs:
        - answer (bool): True when the probability is at or above the threshold.
        - probability (float): JEV's probability that the answer is Yes (0–1).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.category = "classification"
        self.description = "Ask a yes/no question about text using TypeSafe JEV"

        self.add_parameter(
            ParameterString(
                name="context",
                display_name="Context",
                tooltip="The text JEV reads to answer the question. JSON works too. "
                "JEV accepts text only — to ask about an image, describe it first.",
                default_value="",
                multiline=True,
                placeholder_text="Text to ask about",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                input_types=CONTEXT_INPUT_TYPES,
            )
        )

        self.add_parameter(
            ParameterString(
                name="question",
                display_name="Question",
                tooltip="A yes/no question about the context. Ask one narrow thing, "
                "for example 'Does the customer ask for a refund?'",
                default_value="",
                multiline=True,
                placeholder_text="Does the text ...?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        with ParameterGroup(name="Define Yes and No (optional)", ui_options={"collapsed": True}) as criteria_group:
            ParameterString(
                name="yes_means",
                display_name="Yes means",
                tooltip="Optional. Describe what should count as Yes. Most questions don't need this.",
                default_value="",
                multiline=True,
                placeholder_text="What should count as Yes?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterString(
                name="no_means",
                display_name="No means",
                tooltip="Optional. Describe what should count as No. Most questions don't need this.",
                default_value="",
                multiline=True,
                placeholder_text="What should count as No?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        self.add_node_element(criteria_group)

        self.add_parameter(
            ParameterFloat(
                name="threshold",
                display_name="Say Yes at or above",
                tooltip="Answer Yes when the probability is at least this value. "
                "Raise it to require more certainty; lower it to accept weaker signals.",
                default_value=0.5,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=1.0)},
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
            ParameterBool(
                name="answer",
                display_name="Answer",
                tooltip="True when the probability of Yes is at or above the threshold.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="probability",
                display_name="Probability",
                tooltip="JEV's probability that the answer is Yes, from 0 to 1. "
                "Values near 0.5 mean yes and no are about equally likely.",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors",
            result_details_placeholder="JEV status will appear here...",
            parameter_group_initially_collapsed=True,
        )

    async def _build_payload(self) -> dict[str, Any]:
        context = (self.get_parameter_value("context") or "").strip()
        if not context:
            msg = f"{self.name}: Context is empty. Connect or type the text to ask about."
            raise ValueError(msg)

        question = (self.get_parameter_value("question") or "").strip()
        if not question:
            msg = f"{self.name}: Question is empty."
            raise ValueError(msg)

        # Parse JSON objects/arrays back out — JEV reads structured state better
        # than the same data serialised as a string.
        state: str | dict | list = context
        if context[0] in "{[":
            try:
                state = json.loads(context)
            except json.JSONDecodeError:
                pass

        noul: dict[str, Any] = {"type": "noul", "instructions": question}
        criteria: dict[str, str] = {}
        if yes_means := (self.get_parameter_value("yes_means") or "").strip():
            criteria["true"] = yes_means
        if no_means := (self.get_parameter_value("no_means") or "").strip():
            criteria["false"] = no_means
        if criteria:
            noul["criteria"] = criteria

        return {"state": state, "questions": {QUESTION_KEY: noul}}

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:  # noqa: ARG002
        answers = result_json.get("answers") or {}
        answer_data = answers.get(QUESTION_KEY) or {}
        noul_value = answer_data.get("noul")

        if noul_value is None:
            self._set_safe_defaults()
            self._set_status_results(was_successful=False, result_details="No answer found in JEV response.")
            return

        probability = float(noul_value)
        threshold = self.get_parameter_value("threshold") or 0.5
        answer = probability >= threshold

        self.parameter_output_values["probability"] = probability
        self.parameter_output_values["answer"] = answer
        self._set_status_results(was_successful=True, result_details=f"Answer: {'Yes' if answer else 'No'} (probability {probability:.2f})")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values["answer"] = None
        self.parameter_output_values["probability"] = None
