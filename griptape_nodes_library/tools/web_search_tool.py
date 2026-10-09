from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.tools.base_tool import BaseTool
from griptape_nodes_library.utils.node_error_utils import missing_secret_error

SEARCH_ENGINE_MAP = {
    "Exa": {
        "api_keys": ["EXA_API_KEY"],
    },
    "DuckDuckGo": {
        "api_keys": None,
    },
    "Google": {
        "api_keys": ["GOOGLE_API_KEY", "GOOGLE_API_SEARCH_ID"],
    },
}
SEARCH_ENGINES = list(SEARCH_ENGINE_MAP.keys())


class WebSearch(BaseTool):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.update_tool_info(
            value="The WebSearch tool can be given to an agent to help search the web.\n\nIt uses Exa by default, but can be configured to use other search engines with their API keys.",
            title="WebSearch Tool",
        )
        self.add_parameter(
            Parameter(
                name="search_engine",
                type="str",
                tooltip="The search engine to use.",
                default_value=SEARCH_ENGINES[0],
                traits={Options(choices=SEARCH_ENGINES)},
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.move_element_to_position("tool", position="last")
        self.hide_parameter_by_name("off_prompt")

    def missing_api_keys(self) -> list[str]:
        search_engine = self.get_parameter_value("search_engine")
        api_keys = SEARCH_ENGINE_MAP[search_engine]["api_keys"] or []
        return [api_key for api_key in api_keys if not GriptapeNodes.SecretsManager().get_secret(api_key)]

    def check_api_keys(self) -> bool:
        return not self.missing_api_keys()

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "search_engine":
            if not self.check_api_keys():
                self.update_tool_info(
                    value=f"{value} requires the following API keys to be set: {SEARCH_ENGINE_MAP[value]['api_keys']}",
                    title="API Keys Required",
                    variant="warning",
                )
            else:
                self.update_tool_info(
                    value=f"Using {value} for web search.",
                    title="WebSearch Tool",
                )
        super().after_value_set(parameter, value)

    def validate_before_workflow_run(self) -> list[Exception] | None:
        search_engine = self.get_parameter_value("search_engine")
        exceptions: list[Exception] = [
            missing_secret_error(
                api_key,
                message=f"{api_key} is not set. {search_engine} search needs it. "
                "Add it in Settings → API Keys & Secrets, or pick another 'Search Engine'.",
            )
            for api_key in self.missing_api_keys()
        ]
        return exceptions or None

    def process(self) -> None:
        off_prompt = self.get_parameter_value("off_prompt")
        search_engine = self.get_parameter_value("search_engine")

        self.parameter_output_values["tool"] = {
            "tool_type": "WebSearch",
            "engine": search_engine,
            "off_prompt": off_prompt,
        }
