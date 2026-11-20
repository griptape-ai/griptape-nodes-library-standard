from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    NodeError,
    NodeMessageResult,
    Parameter,
    ParameterButtonGroup,
    ParameterList,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import NODE_DEFAULT_SIZE
from griptape_nodes.exe_types.param_types.parameter_button import ParameterButton
from griptape_nodes.exe_types.param_types.parameter_json import ParameterJson
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
    ListConnectionsForNodeRequest,
    ListConnectionsForNodeResultSuccess,
)
from griptape_nodes.retained_mode.events.node_events import (
    CreateNodeRequest,
    CreateNodeResultSuccess,
    DeleteNodeRequest,
    GetFlowForNodeRequest,
    GetFlowForNodeResultSuccess,
    GetNodeMetadataRequest,
    GetNodeMetadataResultSuccess,
    SetNodeMetadataRequest,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.button import Button, ButtonDetailsMessagePayload

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

# Pixels of clear space between a new question node and its neighbours.
QUESTION_NODE_GAP = 40


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
          pick one and rating.
        - details (json): Each question's full result, with probabilities and confidence. Ratings
          also give the level number and score.
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

        with ParameterButtonGroup(name="add_question") as add_question_group:
            for node_type, label in (
                ("JevYesNoQuestion", "+ Yes/No"),
                ("JevPickOneQuestion", "+ Pick One"),
                ("JevRateQuestion", "+ Rate"),
            ):
                ParameterButton(
                    name=f"add_{node_type}",
                    label=label,
                    tooltip=f"Add a {label[2:]} question node and connect it to Questions.",
                    variant="secondary",
                    full_width=False,
                    on_click=self._add_question_handler(node_type),
                )
        self.add_node_element(add_question_group)

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

    def _add_question_handler(self, node_type: str) -> Button.OnClickCallback:
        def handler(_button: Button, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult:
            return self._add_question(node_type, button_details)

        return handler

    def _add_question(self, node_type: str, button_details: ButtonDetailsMessagePayload) -> NodeMessageResult:
        """Create a question node beside this one and connect it to a new row of Questions."""
        flow_result = GriptapeNodes.handle_request(GetFlowForNodeRequest(node_name=self.name))
        if not isinstance(flow_result, GetFlowForNodeResultSuccess):
            return NodeMessageResult(
                success=False,
                details=f"Couldn't find this node's flow: {flow_result.result_details}",
                response=button_details,
            )

        questions = self.get_parameter_by_name("questions")
        assert isinstance(questions, ParameterList)
        row_names = {row.name for row in questions.find_elements_by_type(Parameter, find_recursively=False)}
        existing = self._connected_question_nodes(row_names)
        created = GriptapeNodes.handle_request(
            CreateNodeRequest(
                node_type=node_type,
                override_parent_flow_name=flow_result.flow_name,
                parent_group_name=self.parent_group.name if self.parent_group else None,
            )
        )
        if not isinstance(created, CreateNodeResultSuccess):
            return NodeMessageResult(
                success=False,
                details=f"Couldn't add the question node: {created.result_details}",
                response=button_details,
            )

        self._place_question_node(created.node_name, existing)
        row = questions.add_child_parameter()
        connected = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=created.node_name,
                source_parameter_name="question",
                target_node_name=self.name,
                target_parameter_name=row.name,
            )
        )
        if not isinstance(connected, CreateConnectionResultSuccess):
            GriptapeNodes.handle_request(DeleteNodeRequest(node_name=created.node_name))
            return NodeMessageResult(
                success=False,
                details=f"Couldn't connect the question node: {connected.result_details}",
                response=button_details,
            )

        return NodeMessageResult(
            success=True,
            details=f"Added {created.node_name}.",
            response=button_details,
            altered_workflow_state=True,
        )

    @staticmethod
    def _node_box(node_name: str) -> tuple[float, float, float, float] | None:
        """The node's (x, y, width, height) from its metadata, or None if it can't be read."""
        result = GriptapeNodes.handle_request(GetNodeMetadataRequest(node_name=node_name))
        if not isinstance(result, GetNodeMetadataResultSuccess):
            return None
        position = result.metadata.get("position") or {"x": 0, "y": 0}
        size = result.metadata.get("size") or NODE_DEFAULT_SIZE
        return position["x"], position["y"], size["width"], size["height"]

    def _connected_question_nodes(self, row_names: set[str]) -> list[str]:
        """Names of the nodes connected to the Questions rows."""
        result = GriptapeNodes.handle_request(ListConnectionsForNodeRequest(node_name=self.name))
        if not isinstance(result, ListConnectionsForNodeResultSuccess):
            return []
        return [c.source_node_name for c in result.incoming_connections if c.target_parameter_name in row_names]

    def _place_question_node(self, node_name: str, existing: list[str]) -> None:
        """Put a new question node left of this one, below the lowest question node already connected."""
        mine = self._node_box(node_name)
        here = self._node_box(self.name)
        if mine is None or here is None:
            return
        x = here[0] - mine[2] - QUESTION_NODE_GAP
        boxes = [box for name in existing if (box := self._node_box(name)) is not None]
        y = max(top + height for _, top, _, height in boxes) + QUESTION_NODE_GAP if boxes else here[1]
        GriptapeNodes.handle_request(
            SetNodeMetadataRequest(node_name=node_name, metadata={"position": {"x": x, "y": y}})
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
