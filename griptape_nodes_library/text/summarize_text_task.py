from typing import Any

from griptape.engines import PromptSummaryEngine
from griptape.structures import Agent, Structure
from griptape.tasks import TextSummaryTask
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import resolve_provider_model_id

from griptape_nodes_library.tasks.base_task import BaseTask

MODEL_CHOICES = [
    "gtc_gpt_4_1",
    "gtc_gpt_4_1_mini",
    "gtc_gpt_4_1_nano",
    "gtc_gpt_5",
]
DEFAULT_MODEL = "gtc_gpt_4_1_nano"

# Migrates values saved before the dropdown stored catalog keys.
LEGACY_MODEL_VALUES = {
    "GPT-4.1": "gtc_gpt_4_1",
    "GPT-4.1 mini": "gtc_gpt_4_1_mini",
    "GPT-4.1 nano": "gtc_gpt_4_1_nano",
    "GPT-5": "gtc_gpt_5",
    "gpt-4.1": "gtc_gpt_4_1",
    "gpt-4.1-mini": "gtc_gpt_4_1_mini",
    "gpt-4.1-nano": "gtc_gpt_4_1_nano",
    "gpt-5": "gtc_gpt_5",
}


class SummarizeText(BaseTask):
    """Base task node for creating Griptape Tasks that can run on their own.

    Attributes:
        prompt (BaseTool): A dictionary representation of the created tool.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="prompt",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                ui_options={"multiline": True, "placeholder_text": "Input text to process"},
            )
        )
        model_param = Parameter(
            name="model",
            type="str",
            default_value=DEFAULT_MODEL,
            tooltip="The model to use for the task.",
            ui_options={"hide": True},
        )
        self.add_parameter(model_param)
        self._model_access = ModelAccessComponent(
            node=self,
            parameter=model_param,
            model_choices=MODEL_CHOICES,
            default_model=DEFAULT_MODEL,
            deprecated_values=LEGACY_MODEL_VALUES,
        )

        self.add_parameter(
            Parameter(
                name="output",
                type="str",
                output_type="str",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The output of the task.",
                ui_options={"multiline": True, "placeholder_text": "Task output"},
            )
        )

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name == "model":
            self._model_access.on_value_changed(value)
        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult[Structure]:
        model = self.get_parameter_value("model")

        # License-policy runtime gate. Raises RuntimeError if the currently-selected
        # model is denied.
        self._model_access.raise_if_denied(model)

        # `model` is the catalog key the dropdown stores; the engine's driver needs the
        # upstream provider's own id instead.
        provider_model_id = resolve_provider_model_id(self, model) or ""
        engine = PromptSummaryEngine(prompt_driver=self.create_driver(model=provider_model_id))
        task = TextSummaryTask(summary_engine=engine)
        agent = Agent(tasks=[task])
        prompt = self.get_parameter_value("prompt")
        if prompt and not prompt.isspace():
            # Run the agent asynchronously
            yield lambda: self._process(agent, prompt, model)
