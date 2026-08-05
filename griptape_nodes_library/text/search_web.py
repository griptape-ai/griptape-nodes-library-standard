from typing import Any

from griptape.drivers import DuckDuckGoWebSearchDriver, ExaWebSearchDriver, GoogleWebSearchDriver
from griptape.drivers.prompt.griptape_cloud import GriptapeCloudPromptDriver
from griptape.structures import Agent, Structure
from griptape.tasks import PromptTask
from griptape.tools import WebSearchTool
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import resolve_provider_model_id
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

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

    def _duck_duck_go_driver(self) -> DuckDuckGoWebSearchDriver:
        return DuckDuckGoWebSearchDriver()

    def _google_driver(self) -> GoogleWebSearchDriver:
        return GoogleWebSearchDriver(
            api_key=GriptapeNodes.SecretsManager().get_secret("GOOGLE_API_KEY"),
            search_id=GriptapeNodes.SecretsManager().get_secret("GOOGLE_API_SEARCH_ID"),
        )

    def _exa_driver(self) -> ExaWebSearchDriver:
        return ExaWebSearchDriver(
            api_key=GriptapeNodes.SecretsManager().get_secret("EXA_API_KEY"),
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
        if parameter.name == "model":
            self._model_access.on_value_changed(value)
        super().after_value_set(parameter, value)

    def validate_before_workflow_run(self) -> list[Exception] | None:
        if not self.check_api_keys():
            return [ValueError("Please ensure you have set appropriate API keys for the selected search engine.")]
        return None

    def process(self) -> AsyncResult[Structure]:
        prompt = self.get_parameter_value("prompt")
        search_engine = self.get_parameter_value("search_engine")
        model = self.get_parameter_value("model")

        # License-policy runtime gate. Raises RuntimeError if the currently-selected
        # model is denied.
        self._model_access.raise_if_denied(model)

        if search_engine == "DuckDuckGo":
            driver = self._duck_duck_go_driver()
        elif search_engine == "Google":
            driver = self._google_driver()
        elif search_engine == "Exa":
            driver = self._exa_driver()
        else:
            msg = f"Invalid search engine: {search_engine}"
            raise ValueError(msg)

        # Create the tool
        tool = WebSearchTool(web_search_driver=driver)
        # `model` is the catalog key the dropdown stores; the driver needs the upstream
        # provider's own id instead.
        provider_model_id = resolve_provider_model_id(self, model) or ""
        task = PromptTask(
            tools=[tool],
            reflect_on_tool_use=self.get_parameter_value("summarize"),
            prompt_driver=GriptapeCloudPromptDriver(model=provider_model_id, stream=True),
        )

        agent = Agent(tasks=[task])
        # Run the task
        user_input = f"Search the web for {prompt}"
        if prompt and not prompt.isspace():
            # Run the agent asynchronously
            yield lambda: self._process(agent, user_input, model)

        self.parameter_output_values["output"] = str(agent.output)
