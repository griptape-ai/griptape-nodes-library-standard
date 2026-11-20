from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import NodeError, ParameterList, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson

from griptape_nodes_library.classification.jev_common import (
    add_context_parameter,
    add_images_parameter,
    add_model_group,
    build_batch_payload,
    build_state_and_assets,
)
from griptape_nodes_library.classification.jev_questions import QUESTION_TYPE, read_answer
from griptape_nodes_library.proxy import GriptapeProxyNode
from griptape_nodes_library.utils.node_error_utils import error_fields, error_response

logger = logging.getLogger(__name__)

__all__ = ["JevAsk"]


class JevAsk(GriptapeProxyNode):
    """Ask several questions about the same text and images in one TypeSafe JEV call.

    Build each question with a JEV Yes/No Question, Pick One Question or Rate Question node and
    connect them to the Questions list. JEV reads the context once, so it is billed once.

    Inputs:
        - context (str): The text JEV reads.
        - images (list): Optional images JEV looks at, referred to as [image 1], [image 2], ...
        - questions (list): Question builder nodes, one per question.
        - model (str): JEV model alias.

    Outputs:
        - answers (json): Each question's answer under its name: a bool for yes/no, the label for
          pick one and the level number for rating.
        - details (json): Each question's full result, with probabilities and confidence.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        add_context_parameter(self)
        add_images_parameter(self)

        self.add_parameter(
            ParameterList(
                name="questions",
                display_name="Questions",
                tooltip="Connect JEV Yes/No Question, Pick One Question and Rate Question nodes, one per question.",
                input_types=[QUESTION_TYPE],
                default_value=[],
                allowed_modes={ParameterMode.INPUT},
                ui_options={"display_name": "Questions", "hide_property": True},
            )
        )

        self.add_parameter(
            ParameterJson(
                name="answers",
                display_name="Answers",
                tooltip="Each question's answer under its name.",
                allow_input=False,
                allow_property=False,
                placeholder_text="Answers keyed by question name.",
            )
        )

        self.add_parameter(
            ParameterJson(
                name="details",
                display_name="Details",
                tooltip="Each question's full result, with probabilities and confidence.",
                allow_input=False,
                allow_property=False,
                placeholder_text="Details keyed by question name.",
            )
        )

        self._model_access = add_model_group(self)

        self._create_status_parameters(
            result_details_tooltip="Details about the JEV result or any errors.",
            result_details_placeholder="JEV result will appear here.",
        )

    def _specs(self) -> dict[str, dict[str, Any]]:
        specs: dict[str, dict[str, Any]] = {}
        for spec in self.get_parameter_list_value("questions") or []:
            if not spec:
                continue
            if spec["key"] in specs:
                raise ValueError(f"Two questions are named '{spec['key']}'. Give each question a unique Name.")
            specs[spec["key"]] = spec
        if not specs:
            raise ValueError("'Questions' is empty. Connect at least one question node.")
        return specs

    async def _build_payload(self) -> dict[str, Any]:
        state, assets = await build_state_and_assets(self)
        questions = {key: spec["question"] for key, spec in self._specs().items()}
        return build_batch_payload(state, questions, assets)

    async def _parse_result(self, result_json: dict[str, Any], generation_id: str) -> None:
        all_answers = result_json.get("answers") or {}
        answers: dict[str, Any] = {}
        details: dict[str, Any] = {}
        for key, spec in self._specs().items():
            read = read_answer(spec, all_answers.get(key) or {})
            if read is None:
                self._set_safe_defaults()
                raise NodeError(
                    f"JEV's response didn't include an answer for '{key}'. Run the node again.",
                    fields=error_fields(generation_id=generation_id),
                    response=error_response(result_json),
                )
            answers[key], details[key] = read

        self.parameter_output_values["answers"] = answers
        self.parameter_output_values["details"] = details
        self._set_status_results(was_successful=True, result_details=f"JEV answered {len(answers)} questions.")

    def _set_safe_defaults(self) -> None:
        self.parameter_output_values.pop("answers", None)
        self.parameter_output_values.pop("details", None)
