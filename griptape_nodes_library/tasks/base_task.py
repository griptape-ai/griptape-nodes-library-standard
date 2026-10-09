from collections.abc import Callable, Sequence
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from pydantic_ai.toolsets import AbstractToolset

from griptape_nodes_library.llm.model_config import cloud_model_config
from griptape_nodes_library.llm.runner import Prompt
from griptape_nodes_library.llm.task_support import TaskRunResult, run_task_agent
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

API_KEY_ENV_VAR = "GT_CLOUD_API_KEY"
SERVICE = "Griptape"

# The Griptape Cloud chat models every task node offers. Task nodes differ only in which
# of them they default to, so the list and its migration table belong to the family
# rather than to each node: a per-node copy has to be updated in lockstep on the next
# retirement, and each node's migration test only checks that node's own table.
MODEL_CHOICES = [
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "gpt-5",
]

# Migrates values saved before the dropdown stored the provider's own model id.
LEGACY_MODEL_VALUES = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 mini": "gpt-4.1-mini",
    "GPT-4.1 nano": "gpt-4.1-nano",
    "GPT-5": "gpt-5",
    "gtc_gpt_4_1": "gpt-4.1",
    "gtc_gpt_4_1_mini": "gpt-4.1-mini",
    "gtc_gpt_4_1_nano": "gpt-4.1-nano",
    "gtc_gpt_5": "gpt-5",
}


class BaseTask(ControlNode):
    def __init__(self, name: str, metadata: dict | None = None) -> None:
        super().__init__(name, metadata)
        # Installed by `_add_model_parameter`. Every task node calls it, but where the
        # dropdown sits in the node's layout is the subclass's to choose, so the base
        # cannot add it here.
        self._model_access: ModelAccessComponent | None = None

    def _add_model_parameter(self, *, default_model: str) -> None:
        """Add the license-filtered model dropdown at this point in the node's layout.

        Subclasses call this instead of declaring the parameter and constructing the
        component themselves: the component owns the `Options` + refresh `Button`
        traits, decorates each row with the caller's license entitlement, and
        migrates legacy stored values. `default_model` must be one of
        `MODEL_CHOICES`.
        """
        model_param = Parameter(
            name="model",
            type="str",
            default_value=default_model,
            tooltip="The model to use for the task.",
            ui_options={"hide": True},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=default_model,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

    def _require_permitted_model(self) -> str:
        """The selected model, refusing one the caller's license denies.

        The only way a task node reads its dropdown, so the gate cannot be left off a
        new node's `process`: there is no ungated read to reach for. Raises
        `RuntimeError` when the selection is denied.
        """
        if self._model_access is None:
            msg = (
                f"{type(self).__name__} has no model dropdown: call _add_model_parameter() in __init__ "
                "before reading the selected model."
            )
            raise RuntimeError(msg)
        self._model_access.raise_if_selection_denied()
        return self._model_access.selected_value or ""

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if self._model_access is not None:
            self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def _process(  # noqa: PLR0913
        self,
        prompt: Prompt | None,
        model: str,
        *,
        instructions: str | None = None,
        rulesets: Sequence[dict] = (),
        toolsets: Sequence[AbstractToolset[Any]] = (),
        output_type: Any = str,
        on_text: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str, str], None] | None = None,
        stream_output: bool = True,
    ) -> TaskRunResult:
        """Run `model` on `prompt`, streaming its text into `output` unless `on_text` or `stream_output=False` says otherwise."""
        # License-policy gate immediately before the model call. Shared by every subclass
        # that runs its agent through this method (a subclass that calls the model
        # directly must declare at its own site instead; see that subclass).
        require_model_invocation_sync(self, model)

        if on_text is None and stream_output:
            on_text = self._append_to_output

        return run_task_agent(
            cloud_model_config(model),
            prompt or None,
            instructions=instructions,
            rulesets=rulesets,
            toolsets=toolsets,
            output_type=output_type,
            on_text=on_text,
            on_tool_call=on_tool_call,
        )

    def _append_to_output(self, token: str) -> None:
        self.append_value_to_parameter("output", value=token)

    def _set_output(self, value: str) -> None:
        self.publish_update_to_parameter("output", value)

    def process(self) -> AsyncResult[str]:
        yield lambda: ""
