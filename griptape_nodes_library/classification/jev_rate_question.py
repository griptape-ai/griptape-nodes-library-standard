from __future__ import annotations

from typing import Any

from griptape_nodes.exe_types.core_types import ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

from griptape_nodes_library.classification.jev_common import (
    add_question_key_parameter,
    add_question_output,
    question_key,
)
from griptape_nodes_library.classification.jev_questions import (
    MAX_LEVELS,
    QuestionKind,
    make_spec,
    parse_levels,
    score_question,
)

__all__ = ["JevRateQuestion"]


class JevRateQuestion(DataNode):
    """Describe a rating question for a Questions (JEV) node to ask alongside others.

    This node makes no JEV call. Connect its Question output to a Questions (JEV) node's Questions list.

    Inputs:
        - key (str): Name for this question's answer. Defaults to the node's name.
        - question_text (str): Optional. What JEV should rate.
        - levels (list[str]): Two to ten levels, lowest first, as 'Label: description' or a description.

    Outputs:
        - question (JevQuestion): The question, for a Questions (JEV) node.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        add_question_key_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question_text",
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
                tooltip="One level per row, lowest first, up to 10. Describe the situation at each level. "
                "To name a level, put a label before a colon: 'Minor: broken but a workaround exists'.",
                type="str",
                ui_options={"placeholder_text": "Describe this level"},
                max_items=MAX_LEVELS,
            )
        )

        add_question_output(self)

    def process(self) -> None:
        descriptions, labels = parse_levels(self.get_parameter_value("levels") or [])
        question = score_question(self.get_parameter_value("question_text"), descriptions)
        self.parameter_output_values["question"] = make_spec(
            QuestionKind.SCORE, question_key(self), question, labels=labels, descriptions=descriptions
        )
