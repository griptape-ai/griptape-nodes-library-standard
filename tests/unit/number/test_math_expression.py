"""Tests for MathExpression node."""

import math
import signal
from collections.abc import Generator
from pathlib import Path

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.number.math_expression import MathExpression

EVALUATION_TIMEOUT_SECONDS = 5


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
