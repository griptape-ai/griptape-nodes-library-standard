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
from griptape_nodes_library.classification.jev_questions import QuestionKind, choice_question, make_spec, parse_options

__all__ = ["JevPickOneQuestion"]

MAX_OPTIONS = 255


class JevPickOneQuestion(DataNode):
    """Describe a pick-one question for a Questions (JEV) node to ask alongside others.

    This node makes no JEV call. Connect its Question output to a Questions (JEV) node's Questions list.

    Inputs:
        - key (str): Name for this question's answer. Defaults to the node's name.
        - question (str): Optional. What JEV should decide.
        - options (list[str]): Two or more options as 'Label' or 'Label: description'.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        add_question_key_parameter(self)

        self.add_parameter(
            ParameterString(
                name="question_text",
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
                "the option means.",
                type="str",
                ui_options={"placeholder_text": "label: description"},
                max_items=MAX_OPTIONS,
            )
        )

        add_question_output(self)

    def process(self) -> None:
        criteria = parse_options(self.get_parameter_value("options") or [])
        question = choice_question(self.get_parameter_value("question_text"), criteria)
        self.parameter_output_values["question"] = make_spec(QuestionKind.CHOICE, question_key(self), question)
