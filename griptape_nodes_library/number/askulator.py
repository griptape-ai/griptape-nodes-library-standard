import json
from typing import Any

from griptape.artifacts import BaseArtifact
from griptape.events import ActionChunkEvent, TextChunkEvent
from griptape.rules import Rule
from griptape.structures import Agent, Structure
from griptape.tools import CalculatorTool as GtCalculatorTool
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from json_repair import repair_json  # json_repair
from pydantic import BaseModel

from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import declare_model_invocation_sync

MODEL_CHOICES = ["gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "gpt-5"]
DEFAULT_MODEL = "gpt-4.1-mini"


class Output(BaseModel):
    reasoning: str
    final_answer: str


class Askulator(BaseTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_parameter(
            Parameter(
                name="instruction",
                type="str",
                default_value=None,
                tooltip="URL to scrape",
                ui_options={"multiline": True, "placeholder_text": "Enter something to calculate."},
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
                name="result",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": False, "placeholder_text": "Output from the calculator."},
            )
        )
        self.add_parameter(
            Parameter(
                name="output",
                type="str",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="The reasoning for the answer.",
                ui_options={"multiline": True, "placeholder_text": "The reasoning for the answer."},
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

    def _process(self, agent: Agent, prompt: BaseArtifact | str, model: str) -> Structure:
        # License-policy gate immediately before the framework driver call. Askulator overrides
        # BaseTask._process wholesale (different streaming/parsing loop), so it declares here
        # rather than relying on the base implementation's declaration.
        declaration = declare_model_invocation_sync(self, model)
        if declaration.failed():
            details = str(declaration.result_details or f"{self.name}: model invocation was not permitted.")
            msg = f"Cannot run {type(self).__name__}: {details}"
            raise RuntimeError(msg)

        args = [prompt] if prompt else []
        full_result = ""
        last_reasoning = ""
        last_answer = ""

        for event in agent.run_stream(*args, event_types=[TextChunkEvent, ActionChunkEvent]):
            if isinstance(event, ActionChunkEvent) and event.name:
                self.append_value_to_parameter("output", value=(f"Using a {event.name}\n"))
            if isinstance(event, TextChunkEvent):
                full_result += event.token
                try:
                    result_json = json.loads(repair_json(full_result))  # pyright: ignore[reportArgumentType]
                    if "reasoning" in result_json:
                        new_reasoning = result_json["reasoning"]
                        if new_reasoning != last_reasoning:
                            self.append_value_to_parameter("output", value=new_reasoning[len(last_reasoning) :])
                            last_reasoning = new_reasoning
                    if "final_answer" in result_json:
                        new_answer = result_json["final_answer"]
                        if new_answer != last_answer:
                            self.append_value_to_parameter("result", value=new_answer[len(last_answer) :])
                            last_answer = new_answer
                except json.JSONDecodeError:
                    pass  # Ignore incomplete JSON

        return agent

    def process(self) -> Any:
        instruction = self.get_parameter_value("instruction")
        model = self.get_parameter_value("model")

        # License-policy runtime gate. Raises RuntimeError if the currently-selected
        # model is denied.
        self._model_access.raise_if_denied(model)

        # Create the tool
        tool = GtCalculatorTool()

        # Run the task
        agent = Agent(
            tools=[tool],
            rules=[
                Rule("You are a natural language calculator."),
                Rule(
                    "If given a prompt you don't have a number for, make something up that seems appropriate. Ex: Gajillion = 1,000,000,0000,0000"
                ),
                Rule(
                    "If there is insufficient information to answer the question, like a missing variable or something, use some likely number and explain why in your reasoning."
                ),
                Rule("You try your best to answer the question, your reasoning can be creative an interesting."),
                Rule("Feel free to use newlines in your reasoning to make it more readable."),
                Rule("Use the Calculate action with expression in the Calculator tool to do the math."),
                Rule("Your final answer should be concise. Only a number and unit if applicable."),
            ],
            prompt_driver=self.create_driver(model=model),
            output_schema=Output,
        )
        user_input = f"Give me the answer for: {instruction}\n."

        if instruction and not instruction.isspace():
            # Run the agent asynchronously
            yield lambda: self._process(agent, user_input, model)
