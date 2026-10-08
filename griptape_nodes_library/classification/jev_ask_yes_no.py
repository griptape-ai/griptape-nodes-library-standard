from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import ControlParameterOutput, Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.classification.jev_common import (
    QUESTION_KEY,
    add_context_parameter,
    add_model_group,
    to_state,
)
from griptape_nodes_library.proxy import GriptapeProxyNode

logger = logging.getLogger(__name__)

__all__ = ["JevAskYesNo"]


class JevAskYesNo(GriptapeProxyNode):
    """Ask a yes/no question about text using TypeSafe JEV via Griptape Cloud.

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
        - probability (float): JEV's probability that the answer is Yes (0-1).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Yes and No replace Succeeded; Failed stays, below them.
        self.remove_parameter_element(self.control_parameter_out)

        self.add_parameter(
            ControlParameterOutput(
                name="yes",
                display_name="Yes",
                tooltip="Taken when the probability of Yes is at or above the threshold.",
            )
        )
        self.add_parameter(
            ControlParameterOutput(
                name="no",
                display_name="No",
                tooltip="Taken when the probability of Yes is below the threshold.",
            )
        )
        self.root_ui_element.remove_child(self.failure_output)
        self.root_ui_element.add_child(self.failure_output)

        add_context_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question",
                display_name="Question",
                tooltip="A yes/no question about the context. "
                "Ask one narrow thing, for example 'Does the customer ask for a refund?'",
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
                tooltip="Optional. Describe what should count as Yes.",
                default_value="",
                multiline=True,
                placeholder_text="What should count as Yes?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
            ParameterString(
                name="no_means",
                display_name="No means",
                tooltip="Optional. Describe what should count as No.",
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
                tooltip="JEV's probability that the answer is Yes, from 0 to 1.",
                allow_input=False,
                allow_property=False,
            )
        )

        self._model_access = add_model_group(self)

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors.",
            result_details_placeholder="JEV result will appear here.",
        )

    async def _build_payload(self) -> dict[str, Any]:
        state = to_state(self.get_parameter_value("context"))
        if state is None:
            raise ValueError(f"{self.name}: Context is empty.")

        question = (self.get_parameter_value("question") or "").strip()
        if not question:
            raise ValueError(f"{self.name}: Question is empty.")

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
        answer_data = (result_json.get("answers") or {}).get(QUESTION_KEY) or {}
        noul_value = answer_data.get("noul")
        if noul_value is None:
            self._set_safe_defaults()
            raise RuntimeError(f"{self.name}: No answer found in JEV response.")

        probability = float(noul_value)
        threshold = self.get_parameter_value("threshold")
        if threshold is None:
            threshold = 0.5
        self.parameter_output_values["probability"] = probability
        self.parameter_output_values["answer"] = probability >= threshold
        self._set_status_results(was_successful=True, result_details="JEV answered.")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values.pop("answer", None)
        self.parameter_output_values.pop("probability", None)

    def get_next_control_output(self) -> Parameter | None:
        if self._execution_succeeded is False and not self.lock:
            return self.failure_output
        if "answer" not in self.parameter_output_values:
            return None
        return self.get_parameter_by_name("yes" if self.parameter_output_values["answer"] else "no")
