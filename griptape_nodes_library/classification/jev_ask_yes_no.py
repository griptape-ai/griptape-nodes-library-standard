from __future__ import annotations

import json
import logging
from typing import Any

from griptape_nodes.exe_types.core_types import ControlParameterInput, ControlParameterOutput, Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.node_types import AsyncResult, BaseNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.slider import Slider

logger = logging.getLogger(__name__)

__all__ = ["JevAskYesNo"]

MODEL_CHOICES = ["jev-latest", "jev-preview"]
DEFAULT_MODEL = MODEL_CHOICES[0]
API_KEY_NAME = "TYPESAFE_API_KEY"

CONTEXT_INPUT_TYPES = ["str", "json", "dict", "list", "TextArtifact", "JsonArtifact"]
QUESTION_KEY = "answer"


class JevAskYesNo(BaseNode):
    """Ask a yes/no question about text using TypeSafe JEV.

    JEV reads the context and returns a probability that the answer is yes.
    The flow routes to the Yes or No output based on the threshold.

    Inputs:
        - context (str): The text JEV reads to answer the question.
        - question (str): A yes/no question about the context.
        - yes_means (str): Optional description of what counts as Yes.
        - no_means (str): Optional description of what counts as No.
        - threshold (float): Minimum probability to answer Yes (default 0.5).
        - model (str): JEV model alias.

    Outputs:
        - yes / no: Flow outputs routed based on the answer.
        - answer (bool): True when the probability is at or above the threshold.
        - probability (float): JEV's probability that the answer is Yes (0–1).
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(ControlParameterInput(tooltip="Run this node", name="exec_in"))
        self.add_parameter(
            ControlParameterOutput(
                name="yes",
                display_name="Yes",
                tooltip="Taken when the probability of yes is at or above the threshold.",
            )
        )
        self.add_parameter(
            ControlParameterOutput(
                name="no",
                display_name="No",
                tooltip="Taken when the probability of yes is below the threshold.",
            )
        )

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

        self.add_parameter(
            ParameterBool(
                name="answer",
                display_name="Answer",
                tooltip="True when the probability of Yes is at or above the threshold.",
                allow_input=False,
                allow_property=False,
            )
        )

        self.add_parameter(
            ParameterFloat(
                name="probability",
                display_name="Probability",
                tooltip="JEV's probability that the answer is Yes, from 0 to 1. "
                "Values near 0.5 mean yes and no are about equally likely.",
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
        self.parameter_output_values.pop("answer", None)
        yield lambda: self._ask()

    def _ask(self) -> None:
        import httpx

        context = (self.get_parameter_value("context") or "").strip()
        if not context:
            raise ValueError(f"{self.name}: Context is empty.")

        question = (self.get_parameter_value("question") or "").strip()
        if not question:
            raise ValueError(f"{self.name}: Question is empty.")

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

        api_key = GriptapeNodes.SecretsManager().get_secret(API_KEY_NAME, should_error_on_not_found=False)
        model = self.get_parameter_value("model") or DEFAULT_MODEL

        response = httpx.post(
            f"https://api.typesafe.ai/v1/systemone",
            json={"model": model, "state": state, "questions": {QUESTION_KEY: noul}},
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()
        body = response.json()

        answer_data = (body.get("answers") or {}).get(QUESTION_KEY) or {}
        noul_value = answer_data.get("noul")
        if noul_value is None:
            raise RuntimeError(f"{self.name}: No answer found in JEV response.")

        probability = float(noul_value)
        threshold = self.get_parameter_value("threshold") or 0.5
        answer = probability >= threshold

        self.parameter_output_values["probability"] = probability
        self.parameter_output_values["answer"] = answer

    def get_next_control_output(self) -> Parameter | None:
        if "answer" not in self.parameter_output_values:
            return None
        return self.get_parameter_by_name("yes" if self.parameter_output_values["answer"] else "no")
