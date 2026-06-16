from griptape_nodes_library.tools.base_tool import BaseTool


class WebScraper(BaseTool):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.update_tool_info(
            value="The WebScraper tool can be given to an agent to help it perform web scraping operations.",
            title="WebScraper Tool",
        )
        self.hide_parameter_by_name("off_prompt")

    def process(self) -> None:
        off_prompt = self.get_parameter_value("off_prompt")
        self.parameter_output_values["tool"] = {"tool_type": "WebScraper", "off_prompt": off_prompt}
