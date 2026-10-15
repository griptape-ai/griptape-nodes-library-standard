"""Unit tests for RetryGroupNode."""

from __future__ import annotations

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.retained_mode.engine import Engine
from griptape_nodes.retained_mode.events.connection_events import CreateConnectionRequest
from griptape_nodes.retained_mode.events.execution_events import StartFlowRequest
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest
from griptape_nodes.retained_mode.events.parameter_events import SetParameterValueRequest


@pytest.fixture
def engine(griptape_nodes: Engine) -> Engine:
    griptape_nodes.context_manager.push_workflow(workflow_name="retry_group_test")
    griptape_nodes.handle_request(
        CreateFlowRequest(parent_flow_name=None, flow_name="ControlFlow_1", set_as_new_context=True)
    )
    return griptape_nodes


class TestRetry:
    def test_failed_attempt_runs_the_body_again(self, engine: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
        # Nothing in the body reads the group's iteration, as in a retried API call, so the
        # second attempt only differs if the group re-runs its nodes. The stubbed node reports
        # 0 on its first run and 1 on its second, so the check fails once and then succeeds.
        _create_node(engine, "TextInput", "Start", text="0")
        _create_node(engine, "RetryGroupNode", "Retry", max_iterations=3)
        _create_node(engine, "ToInteger", "Attempt", group="Retry")
        _create_node(engine, "CompareNumbers", "Check", group="Retry", evaluate="A >= B", B=1)
        _connect(engine, "Start", "exec_out", "Retry", "group_exec_in")
        _connect(engine, "Start", "text", "Attempt", "from")
        _connect(engine, "Attempt", "output", "Check", "A")
        _connect(engine, "Check", "True", "Retry", "done")
        _connect(engine, "Check", "False", "Retry", "continue_loop")
        runs = []

        def count_runs(node: BaseNode) -> None:
            node.parameter_output_values["output"] = len(runs)
            runs.append(node.name)

        monkeypatch.setattr(type(engine.node_manager.get_node_by_name("Attempt")), "process", count_runs)

        result = engine.handle_request(StartFlowRequest(flow_name="ControlFlow_1"))

        assert result.succeeded(), result.result_details
        retry = engine.node_manager.get_node_by_name("Retry")
        assert len(runs) == 2  # noqa: PLR2004
        assert retry.parameter_output_values["was_successful"] is True
        assert retry.parameter_output_values["total_iterations"] == 2  # noqa: PLR2004


def _create_node(engine: Engine, node_type: str, name: str, group: str | None = None, **values: str | int) -> None:
    result = engine.handle_request(CreateNodeRequest(node_type=node_type, node_name=name, parent_group_name=group))
    assert result.succeeded(), result.result_details
    for parameter_name, value in values.items():
        result = engine.handle_request(
            SetParameterValueRequest(node_name=name, parameter_name=parameter_name, value=value)
        )
        assert result.succeeded(), result.result_details


def _connect(engine: Engine, source: str, source_parameter: str, target: str, target_parameter: str) -> None:
    result = engine.handle_request(
        CreateConnectionRequest(
            source_node_name=source,
            source_parameter_name=source_parameter,
            target_node_name=target,
            target_parameter_name=target_parameter,
        )
    )
    assert result.succeeded(), result.result_details
