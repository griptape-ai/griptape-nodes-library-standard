from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.llm.tools import ToolType, build_toolsets, web_search_output
from griptape_nodes_library.tasks.base_task import BaseTask

SEARCH_ENGINE_MAP = {
    "DuckDuckGo": {
        "api_keys": None,
    },
    "Google": {
        "api_keys": ["GOOGLE_API_KEY", "GOOGLE_API_SEARCH_ID"],
    },
    "Exa": {
        "api_keys": ["EXA_API_KEY"],
    },
}
SEARCH_ENGINES = list(SEARCH_ENGINE_MAP.keys())
DEFAULT_MODEL = "gpt-4.1-mini"


class SearchWeb(BaseTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="prompt",
                type="str",
                default_value=None,
                tooltip="Search the web for information",
                ui_options={"placeholder_text": "Enter the search query."},
            )
        )
        self.add_parameter(
            Parameter(
                name="summarize",
                type="bool",
                default_value=False,
                tooltip="Summarize the results",
                ui_options={"hide": False},
            )
        )
        self._add_model_parameter(default_model=DEFAULT_MODEL)
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
        self.add_node_element(
            ParameterMessage(
                name="api_keys_message",
                value="Please ensure you have set appropriate API keys for the selected search engine.",
                variant="warning",
                title="API Keys",
                ui_options={"hide": True},
            )
        )

        self.add_parameter(
            Parameter(
                name="output",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                ui_options={"multiline": True, "placeholder_text": "Output from the web search."},
            )
        )

    def check_api_keys(self) -> bool:
        search_engine = self.get_parameter_value("search_engine")
        api_keys = SEARCH_ENGINE_MAP[search_engine]["api_keys"]
        if api_keys is None:
            return True
        for api_key in api_keys:
            if not GriptapeNodes.SecretsManager().get_secret(api_key):
                return False
        return True

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name == "search_engine":
            if value == "DuckDuckGo":
                self.hide_message_by_name("api_keys_message")
            else:
                api_key_message = self.get_message_by_name_or_element_id("api_keys_message")
                if api_key_message:
                    api_key_message.value = (
                        f"{value} requires the following API keys: {SEARCH_ENGINE_MAP[value]['api_keys']}"
                    )
                if not self.check_api_keys():
                    self.show_message_by_name("api_keys_message")
                else:
                    self.hide_message_by_name("api_keys_message")
        super().after_value_set(parameter, value)

    def validate_before_workflow_run(self) -> list[Exception] | None:
        if not self.check_api_keys():
            return [ValueError("Please ensure you have set appropriate API keys for the selected search engine.")]
        return None

    def process(self) -> AsyncResult[str]:
        prompt = self.get_parameter_value("prompt")
        search_engine = self.get_parameter_value("search_engine")
        summarize = self.get_parameter_value("summarize")
        model = self._require_permitted_model()

        if search_engine not in SEARCH_ENGINES:
            msg = f"Invalid search engine: {search_engine}"
            raise ValueError(msg)

        user_input = f"Search the web for {prompt}"
        if prompt and not prompt.isspace():

            def _process() -> str:
                if summarize:
                    toolsets = build_toolsets([{"tool_type": ToolType.WEB_SEARCH, "engine": search_engine}])
                    return self._process(user_input, model, toolsets=toolsets).text
                output_type = [web_search_output(search_engine), str]
                result = self._process(user_input, model, output_type=output_type, stream_output=False)
                self._set_output(result.text)
                return result.text

            yield _process
