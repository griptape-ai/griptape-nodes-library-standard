from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.classification.jev_common import (
    add_question_key_parameter,
    add_question_output,
    question_key,
)
from griptape_nodes_library.classification.jev_questions import QuestionKind, make_spec, yes_no_question

__all__ = ["JevYesNoQuestion"]


class JevYesNoQuestion(DataNode):
    """Describe a yes/no question for a JEV Ask node to ask alongside others.

    This node makes no JEV call. Connect its Question output to a JEV Ask node's Questions list.

    Inputs:
        - key (str): Name for this question's answer. Defaults to the node's name.
        - question (str): A yes/no question about the context.
        - yes_means (str): Optional description of what counts as Yes.
        - no_means (str): Optional description of what counts as No.
        - threshold (float): Minimum probability to answer Yes (default 0.5).
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        add_question_key_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question_text",
                display_name="Question",
                tooltip="A yes/no question about the context.",
                default_value="",
                multiline=True,
                placeholder_text="Is this a truck?",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        with ParameterGroup(name="Criteria", ui_options={"collapsed": True}) as criteria_group:
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
                tooltip="Answer Yes when the probability is at least this value.",
                default_value=0.5,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Slider(min_val=0.0, max_val=1.0)},
            )
        )

        add_question_output(self)

    def process(self) -> None:
        question = yes_no_question(
            self.get_parameter_value("question_text"),
            self.get_parameter_value("yes_means"),
            self.get_parameter_value("no_means"),
        )
        threshold = self.get_parameter_value("threshold")
        self.parameter_output_values["question"] = make_spec(
            QuestionKind.YES_NO,
            question_key(self),
            question,
            threshold=0.5 if threshold is None else threshold,
        )
