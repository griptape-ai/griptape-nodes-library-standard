from griptape.artifacts import ListArtifact
from griptape.structures import Agent, Structure
from griptape.tasks import PromptTask
from griptape.tools import WebScraperTool as GtWebScraperTool
from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult

from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

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
        self._add_model_parameter(default_model=DEFAULT_MODEL)

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

    def process(self) -> AsyncResult[Structure]:
        prompt = self.get_parameter_value("prompt")
        model = self._require_permitted_model()

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
