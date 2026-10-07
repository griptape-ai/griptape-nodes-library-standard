from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.exe_types.node_types import AsyncResult

from griptape_nodes_library.llm.tools import web_scraper_output
from griptape_nodes_library.tasks.base_task import BaseTask

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

    def process(self) -> AsyncResult[str]:
        prompt = self.get_parameter_value("prompt")
        model = self._require_permitted_model()

        def _process() -> str:
            result = self._process(
                f"Scrape the web for information about: {prompt}",
                model,
                output_type=[web_scraper_output(), str],
                stream_output=False,
            )
            self._set_output(result.text)
            return result.text

        yield _process
