from typing import Any

from griptape.engines import PromptSummaryEngine
from griptape.structures import Agent, Structure
from griptape.tasks import TextSummaryTask
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent

from griptape_nodes_library.tasks.base_task import BaseTask

MODEL_CHOICES = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5"]
DEFAULT_MODEL = "gpt-4.1-nano"


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

        engine = PromptSummaryEngine(prompt_driver=self.create_driver(model=model))
        task = TextSummaryTask(summary_engine=engine)
        agent = Agent(tasks=[task])
        prompt = self.get_parameter_value("prompt")
        if prompt and not prompt.isspace():
            # Run the agent asynchronously
            yield lambda: self._process(agent, prompt, model)
