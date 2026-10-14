"""License-policy gating for the task/text node family.

Two independent checkpoints protect a model invocation:

  - `OFFER_MODEL` gates the dropdown a `ModelAccessComponent`-backed node
    exposes. `raise_if_denied()` re-queries it at the top of `process()`, so a
    selection that was permitted when the node was constructed but has since
    been denied stops the node before it builds anything.
  - `INVOKE_MODEL` gates the actual model call via
    `declare_model_invocation_sync`, dispatched immediately before the
    network call regardless of whether the model came from a dropdown or is
    hardcoded. This is the fail-closed backstop even for nodes with no
    dropdown at all.

These tests register a policy hook directly on `GriptapeNodes.EventManager()`
(the same extension point the app's license policy uses) and prove each
checkpoint actually stops the corresponding node -- not just that a
declaration is dispatched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import pytest
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.retained_mode.managers.authorization_checkpoint import (
    AuthorizationCheckpoint,
    CheckpointAction,
    CheckpointDenial,
    CheckpointFailure,
)
from pydantic_ai.messages import ModelResponse, TextPart

from griptape_nodes_library.llm.agent_state import AgentState
from griptape_nodes_library.llm.model_config import ModelConfig, ModelProvider
from griptape_nodes_library.llm.models import override_model
from griptape_nodes_library.llm.testing import fake_model
from griptape_nodes_library.tasks.mcp_task import MCPTaskNode
from griptape_nodes_library.text.random_text import RandomText

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from pydantic_ai.messages import ModelMessage
    from pydantic_ai.models.function import AgentInfo

LIBRARY_NAME = "Griptape Nodes Library"

type AuthorizationHook = Callable[[AuthorizationCheckpoint], CheckpointDenial | None]


def _create_node(node_type: str) -> BaseNode:
    """Create a node through the library so its metadata carries `library` / `node_type`.

    `resolve_catalog_model_id` (and therefore `declare_model_invocation_sync`)
    reads those two metadata keys to resolve a node's declared models; a bare
    `NodeClass(name=...)` construction (as many other unit tests use) does not
    set them and would silently fall back to the raw provider model id.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return library.create_node(node_type=node_type, name=node_type)


def _first_yielded_thunk(node: BaseNode) -> Callable[[], Any]:
    """Advance a generator-based `process()` to its first `yield` and return the thunk.

    Raises whatever `process()` raises before its first `yield` (e.g. a
    `raise_if_denied()` gate) -- callers asserting on that use `pytest.raises`
    around this call directly.
    """
    generator = node.process()
    assert generator is not None
    return next(generator)


def _deny_hook(action: CheckpointAction, subject_id: str) -> AuthorizationHook:
    """Build a hook that denies one action against one catalog model id, else allows."""

    def hook(checkpoint: AuthorizationCheckpoint) -> CheckpointDenial | None:
        if checkpoint.action == action and checkpoint.subject_id == subject_id:
            return CheckpointDenial(failures=(CheckpointFailure(detail="denied for test"),))
        return None

    return hook


@pytest.fixture
def authorization_hook() -> Iterator[Callable[[AuthorizationHook], None]]:
    """Register an authorization hook for the test, guaranteed removed afterward."""
    registered: list[AuthorizationHook] = []

    def register(hook: AuthorizationHook) -> None:
        GriptapeNodes.EventManager().add_authorization_hook(hook)
        registered.append(hook)

    try:
        yield register
    finally:
        for hook in registered:
            GriptapeNodes.EventManager().remove_authorization_hook(hook)


class ModelCalls:
    """A scripted model that counts how many times the node reached it."""

    def __init__(self, text: str = "ok") -> None:
        self.count = 0
        self.model = fake_model(self._respond)
        self._text = text

    def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:  # noqa: ARG002
        self.count += 1
        return ModelResponse(parts=[TextPart(self._text)])


class TestInvokeModelDenialStopsSharedBaseRunSite:
    """`BaseTask._process` is the shared model call site for the three
    subclasses that don't override it (DateAndTime, SearchWeb, SummarizeText).
    """

    @pytest.mark.parametrize(
        ("node_type", "prompt_param", "default_catalog_id"),
        [
            ("DateAndTime", "prompt", "gtc_gpt_4_1_mini"),
            ("SearchWeb", "prompt", "gtc_gpt_4_1_mini"),
            ("SummarizeText", "prompt", "gtc_gpt_4_1_nano"),
        ],
    )
    def test_denied_invocation_raises_before_model_call(
        self,
        node_type: str,
        prompt_param: str,
        default_catalog_id: str,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = _create_node(node_type)
        node.set_parameter_value(prompt_param, "hello")
        calls = ModelCalls()
        authorization_hook(_deny_hook(CheckpointAction.INVOKE_MODEL, default_catalog_id))

        thunk = _first_yielded_thunk(node)

        with override_model(calls.model), pytest.raises(RuntimeError, match="denied"):
            thunk()

        assert calls.count == 0

    @pytest.mark.parametrize(
        ("node_type", "prompt_param"),
        [
            ("DateAndTime", "prompt"),
            ("SearchWeb", "prompt"),
            ("SummarizeText", "prompt"),
        ],
    )
    def test_allowed_invocation_still_runs(self, node_type: str, prompt_param: str) -> None:
        """Sanity check: with no denying hook registered, the same node reaches the model."""
        node = _create_node(node_type)
        node.set_parameter_value(prompt_param, "hello")
        calls = ModelCalls()

        thunk = _first_yielded_thunk(node)
        with override_model(calls.model):
            thunk()

        assert calls.count == 1


class TestInvokeModelDenialStopsOwnCallSites:
    """Nodes that bypass `BaseTask._process` declare at their own call site instead."""

    def test_evaluate_text_result_denied_invocation_never_calls_model(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = _create_node("EvaluateTextResult")
        node.set_parameter_value("criteria", "Is this correct?")
        node.set_parameter_value("input", "1 + 1")
        node.set_parameter_value("expected_output", "2")
        node.set_parameter_value("actual_output", "2")
        calls = ModelCalls()
        authorization_hook(_deny_hook(CheckpointAction.INVOKE_MODEL, "gtc_gpt_4_1"))

        thunk = _first_yielded_thunk(node)

        with override_model(calls.model), pytest.raises(RuntimeError, match="denied"):
            thunk()

        assert calls.count == 0


class TestInvokeModelDenialStopsHardcodedModelNodes:
    """Nodes with no user-facing model dropdown still gate the actual invocation."""

    def test_random_text_denied_invocation_never_calls_model(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = cast("RandomText", _create_node("RandomText"))
        calls = ModelCalls()
        authorization_hook(_deny_hook(CheckpointAction.INVOKE_MODEL, "gtc_gpt_4_1_nano"))

        with override_model(calls.model), pytest.raises(RuntimeError, match="denied"):
            node._generate_with_agent("sentence", seed=1)

        assert calls.count == 0

    def test_mcp_task_denied_invocation_never_calls_model(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = cast("MCPTaskNode", _create_node("MCPTaskNode"))
        calls = ModelCalls()
        authorization_hook(_deny_hook(CheckpointAction.INVOKE_MODEL, "gtc_claude_sonnet_5"))

        setup = node._setup_agent()
        assert setup is not None
        state, model_config = setup

        with override_model(calls.model), pytest.raises(RuntimeError, match="denied"):
            node._process_with_streaming(model_config, state, [], [], "hello")

        assert calls.count == 0

    def test_mcp_task_gates_the_connected_agents_model_not_the_dropdown(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = cast("MCPTaskNode", _create_node("MCPTaskNode"))
        node.set_parameter_value(
            "agent", AgentState(model=ModelConfig(provider=ModelProvider.GRIPTAPE_CLOUD, model="gpt-4.1")).to_wire()
        )
        calls = ModelCalls()
        authorization_hook(_deny_hook(CheckpointAction.INVOKE_MODEL, "gtc_gpt_4_1"))

        setup = node._setup_agent()
        assert setup is not None
        state, model_config = setup
        assert model_config.model == "gpt-4.1"

        with override_model(calls.model), pytest.raises(RuntimeError, match="denied"):
            node._process_with_streaming(model_config, state, [], [], "hello")

        assert calls.count == 0


class TestOfferModelDenialStopsSelection:
    """A model denied via `OFFER_MODEL` (the dropdown-populating checkpoint) must not
    run even before `INVOKE_MODEL` is ever declared -- `raise_if_denied()` at the top
    of `process()` re-queries live policy and stops the node first.
    """

    def test_askulator_denied_selection_raises_from_process_before_yielding(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        # Construct the node BEFORE the policy denies its default model, so the
        # ModelAccessComponent's constructor-time snapshot is clean and does not
        # relocate the stored value away from the (about-to-be-denied) default.
        node = _create_node("Askulator")
        node.set_parameter_value("instruction", "what is 2 + 2?")

        authorization_hook(_deny_hook(CheckpointAction.OFFER_MODEL, "gtc_gpt_4_1_mini"))

        with pytest.raises(RuntimeError, match="not permitted"):
            _first_yielded_thunk(node)

    def test_summarize_text_denied_selection_raises_from_process_before_yielding(
        self,
        authorization_hook: Callable[[AuthorizationHook], None],
    ) -> None:
        node = _create_node("SummarizeText")
        node.set_parameter_value("prompt", "hello")

        authorization_hook(_deny_hook(CheckpointAction.OFFER_MODEL, "gtc_gpt_4_1_nano"))

        with pytest.raises(RuntimeError, match="not permitted"):
            _first_yielded_thunk(node)

    def test_allowed_selection_does_not_raise_from_process(self) -> None:
        """Sanity check: with no denying hook registered, `raise_if_denied` is a no-op."""
        node = _create_node("Askulator")
        node.set_parameter_value("instruction", "what is 2 + 2?")

        # Must not raise; the generator proceeds to its first yield.
        _first_yielded_thunk(node)
