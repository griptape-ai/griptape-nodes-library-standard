"""Tests for MathExpression node."""

import math
import signal
import uuid
from collections.abc import Generator
from pathlib import Path

import pytest
from griptape_nodes.retained_mode.events.connection_events import (
    CreateConnectionRequest,
    CreateConnectionResultSuccess,
)
from griptape_nodes.retained_mode.events.flow_events import CreateFlowRequest, CreateFlowResultSuccess
from griptape_nodes.retained_mode.events.node_events import CreateNodeRequest, CreateNodeResultSuccess
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.number.math_expression import MathExpression

EVALUATION_TIMEOUT_SECONDS = 5
LIBRARY_NAME = "Griptape Nodes Library"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> MathExpression:  # noqa: ARG001
    node = MathExpression(name="test_math_expression")
    node.parameter_values["num_variables"] = 3
    node.parameter_values["a"] = 2.0
    node.parameter_values["b"] = 3.0
    node.parameter_values["c"] = 4.0
    return node


@pytest.fixture
def evaluation_timeout() -> Generator[None, None, None]:
    """Fail instead of hanging if an expression never returns."""

    def _on_timeout(_signum: int, _frame: object) -> None:
        msg = f"Expression evaluation exceeded {EVALUATION_TIMEOUT_SECONDS}s"
        raise TimeoutError(msg)

    previous_handler = signal.signal(signal.SIGALRM, _on_timeout)
    signal.alarm(EVALUATION_TIMEOUT_SECONDS)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


class TestMathExpressionEvaluation:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("a + b * c", 14.0),
            ("sin(a) + cos(b)", round(math.sin(2.0) + math.cos(3.0), 6)),
            ("sum(a, b, c)", 9.0),
            ("2a", 4.0),
            ("a(b + c)", 14.0),
            ("round(a / b) * b", 3.0),
            ("max(a, b, c) - min(a, b, c)", 2.0),
            ("pow(a, b) + abs(-c) + sqrt(c)", 14.0),
            ("radians(180)", round(math.pi, 6)),
        ],
    )
    def test_math_expressions_evaluate(self, node: MathExpression, expression: str, expected: float) -> None:
        assert node._evaluate_expression(expression) == expected

    def test_rand_stays_within_range(self, node: MathExpression) -> None:
        result = node._evaluate_expression("rand(a, b)")
        assert 2.0 <= result <= 3.0  # noqa: PLR2004


class TestMathExpressionSandbox:
    def test_open_is_not_available(self, node: MathExpression, tmp_path: Path) -> None:
        secret = tmp_path / "secret.txt"
        secret.write_text("x" * 42)

        result = node._evaluate_expression(f"len(open({str(secret)!r}).read())")

        assert result == 0.0
        assert "open" not in node._create_interpreter().symtable

    @pytest.mark.parametrize("name", ["open", "type", "dir", "len", "getattr", "eval", "exec", "__import__"])
    def test_python_builtins_are_not_exposed(self, node: MathExpression, name: str) -> None:
        assert name not in node._create_interpreter().symtable

    @pytest.mark.usefixtures("evaluation_timeout")
    @pytest.mark.parametrize(
        "expression",
        [
            "while 1:pass",
            "for x in [1, 2]: a",
            "[x for x in [1, 2]]",
            "(lambda: 1)()",
            "def f(): pass",
        ],
    )
    def test_statements_are_rejected(self, node: MathExpression, expression: str) -> None:
        assert node._evaluate_expression(expression) == 0.0


class TestMathExpressionBoolOutput:
    @pytest.mark.parametrize(
        ("expression", "expected"),
        [
            ("a < b", True),
            ("a > b", False),
            ("a + b == 5", True),
            ("a < b and b < c", True),
            ("a > b or b > c", False),
            ("not a > b", True),
            ("a - a", False),
            ("a * b", True),
            ("nan", False),
            ("a / 0", False),
            ("not valid (", False),
            ("", False),
        ],
    )
    def test_bool_output(self, node: MathExpression, expression: str, *, expected: bool) -> None:
        node.parameter_values["output_type"] = "bool"
        assert node._evaluate_expression(expression) is expected

    def test_float_output_still_gives_numbers_for_comparisons(self, node: MathExpression) -> None:
        assert node._evaluate_expression("a < b") == 1.0


class TestMathExpressionControlFlow:
    def test_has_execution_pins(self, node: MathExpression) -> None:
        assert node.get_parameter_by_name("exec_in") is not None
        assert node.get_parameter_by_name("exec_out") is not None


class TestMathExpressionResultConnections:
    @pytest.fixture
    def flow_name(self, griptape_nodes: GriptapeNodes) -> str:  # noqa: ARG002
        suffix = uuid.uuid4().hex[:8]
        GriptapeNodes.ContextManager().push_workflow(workflow_name=f"math_expression_{suffix}")
        result = GriptapeNodes.handle_request(
            CreateFlowRequest(parent_flow_name=None, flow_name=f"MathExpression_{suffix}", set_as_new_context=True)
        )
        assert isinstance(result, CreateFlowResultSuccess), result
        return result.flow_name

    def _create(self, node_type: str, flow_name: str) -> str:
        result = GriptapeNodes.handle_request(
            CreateNodeRequest(
                node_type=node_type,
                specific_library_name=LIBRARY_NAME,
                node_name=f"{node_type}_{uuid.uuid4().hex[:8]}",
                override_parent_flow_name=flow_name,
            )
        )
        assert isinstance(result, CreateNodeResultSuccess), result
        return result.node_name

    def _connect(self, source_node: str, target_node: str, target_param: str) -> None:
        result = GriptapeNodes.handle_request(
            CreateConnectionRequest(
                source_node_name=source_node,
                source_parameter_name="result",
                target_node_name=target_node,
                target_parameter_name=target_param,
            )
        )
        assert isinstance(result, CreateConnectionResultSuccess), result

    def _targets(self, source_node: str) -> set[str]:
        connections = GriptapeNodes.FlowManager().get_connections()
        ids = connections.outgoing_index.get(source_node, {}).get("result", [])
        return {connections.connections[i].target_node.name for i in ids}

    def test_switching_type_drops_only_incompatible_connections(self, flow_name: str) -> None:
        expression = self._create("MathExpression", flow_name)
        if_else = self._create("IfElse", flow_name)
        math_node = self._create("Math", flow_name)
        node = GriptapeNodes.NodeManager().get_node_by_name(expression)

        node.set_parameter_value("output_type", "bool")
        self._connect(expression, if_else, "evaluate")
        self._connect(expression, math_node, "A")
        assert self._targets(expression) == {if_else, math_node}

        node.set_parameter_value("output_type", "float")

        assert self._targets(expression) == {math_node}

    def test_bool_output_switches_result_type_and_hides_precision(self, flow_name: str) -> None:
        node = GriptapeNodes.NodeManager().get_node_by_name(self._create("MathExpression", flow_name))
        node.set_parameter_value("output_type", "bool")
        result = node.get_parameter_by_name("result")
        precision = node.get_parameter_by_name("precision")
        assert result is not None
        assert precision is not None
        assert result.output_type == "bool"
        assert precision.hide is True

        node.set_parameter_value("output_type", "float")
        assert result.output_type == "float"
        assert precision.hide is False
