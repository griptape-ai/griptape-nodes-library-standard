from typing import Any

from griptape.artifacts import BaseArtifact
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.events import ActionChunkEvent, FinishStructureRunEvent, StartStructureRunEvent, TextChunkEvent
from griptape.structures import Agent, Structure
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult, ControlNode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent

from griptape_nodes_library.utils.cloud_driver_auth import cloud_driver_auth
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
    """Base task node for creating Griptape Tasks that can run on their own."""

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
        """Keep the model dropdown's denial badge in step with the selection."""
        if self._model_access is not None:
            self._model_access.on_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    def create_driver(self, model: str = "gpt-4.1") -> GriptapeCloudPromptDriver:
        return GriptapeCloudPromptDriver(
            model=model,
            stream=True,
            **cloud_driver_auth(),
        )

    def _process(self, agent: Agent, prompt: BaseArtifact | str, model: str) -> Structure:
        # License-policy gate immediately before the framework driver call. Shared by every
        # subclass that runs its agent through this method (a subclass that invokes its own
        # driver call directly -- e.g. bypassing this method -- must declare at its own site
        # instead; see that subclass for its own declaration).
        require_model_invocation_sync(self, model)

        args = [prompt] if prompt else []
        for event in agent.run_stream(
            *args, event_types=[StartStructureRunEvent, TextChunkEvent, ActionChunkEvent, FinishStructureRunEvent]
        ):
            if isinstance(event, TextChunkEvent):
                self.append_value_to_parameter("output", value=event.token)

        return agent

    def process(self) -> AsyncResult[Structure]:
        # Base implementation returns an empty Agent
        def _process() -> Structure:
            return Agent()

        yield _process
