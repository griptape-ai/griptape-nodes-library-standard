from typing import Any

from griptape.artifacts import ListArtifact
from griptape.structures import Agent, Structure
from griptape.tasks import PromptTask
from griptape.tools import WebScraperTool as GtWebScraperTool
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent

from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import declare_model_invocation_sync

MODEL_CHOICES = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5"]
DEFAULT_MODEL = "gpt-4.1-mini"


class ScrapeWeb(BaseTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                default_value=None,
                tooltip="URL to scrape",
                ui_options={"placeholder_text": "Enter the URL to scrape."},
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
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                ui_options={"multiline": True, "placeholder_text": "Output from the web scraper."},
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
        prompt = self.get_parameter_value("prompt")
        model = self.get_parameter_value("model")

        # License-policy runtime gate. Raises RuntimeError if the currently-selected
        # model is denied.
        self._model_access.raise_if_denied(model)

        # Create the tool
        tool = GtWebScraperTool()
        scrape_task = PromptTask(
            tools=[tool],
            reflect_on_tool_use=False,
            prompt_driver=self.create_driver(model=model),
        )

        def _process() -> Structure:
            # License-policy gate immediately before the framework driver call. PromptTask.run
            # invokes the prompt driver directly rather than through BaseTask._process, so it
            # declares here rather than relying on the base implementation's declaration.
            declaration = declare_model_invocation_sync(self, model)
            if declaration.failed():
                details = str(declaration.result_details or f"{self.name}: model invocation was not permitted.")
                msg = f"Cannot run {type(self).__name__}: {details}"
                raise RuntimeError(msg)

            # Run the task
            output = ""
            response = scrape_task.run(f"Scrape the web for information about: {prompt}")
            if isinstance(response, ListArtifact):
                output += str(response[0].value[0].value)

            # Set the output
            self.parameter_output_values["output"] = output
            return Agent()  # Return a proper Structure instance

        yield _process
