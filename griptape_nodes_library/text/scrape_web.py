from typing import Any

from griptape.artifacts import ListArtifact
from griptape.structures import Agent, Structure
from griptape.tasks import PromptTask
from griptape.tools import WebScraperTool as GtWebScraperTool
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import resolve_provider_model_id

from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

MODEL_CHOICES = [
    "gtc_gpt_4_1",
    "gtc_gpt_4_1_mini",
    "gtc_gpt_4_1_nano",
    "gtc_gpt_5",
]
DEFAULT_MODEL = "gtc_gpt_4_1_mini"

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
            deprecated_values=LEGACY_MODEL_VALUES,
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
        # `model` is the catalog key the dropdown stores; the driver needs the upstream
        # provider's own id instead.
        provider_model_id = resolve_provider_model_id(self, model) or ""
        scrape_task = PromptTask(
            tools=[tool],
            reflect_on_tool_use=False,
            prompt_driver=self.create_driver(model=provider_model_id),
        )

        def _process() -> Structure:
            # License-policy gate immediately before the framework driver call. PromptTask.run
            # invokes the prompt driver directly rather than through BaseTask._process, so it
            # declares here rather than relying on the base implementation's declaration.
            require_model_invocation_sync(self, model)

            # Run the task
            output = ""
            response = scrape_task.run(f"Scrape the web for information about: {prompt}")
            if isinstance(response, ListArtifact):
                output += str(response[0].value[0].value)

            # Set the output
            self.parameter_output_values["output"] = output
            return Agent()  # Return a proper Structure instance

        yield _process
