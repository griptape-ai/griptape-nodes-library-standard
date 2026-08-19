import json
from typing import Any

from griptape.artifacts import BaseArtifact
from griptape.events import ActionChunkEvent, TextChunkEvent
from griptape.rules import Rule
from griptape.structures import Agent, Structure
from griptape.tools import CalculatorTool as GtCalculatorTool
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from json_repair import repair_json  # json_repair
from pydantic import BaseModel

from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

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
        self._add_model_parameter(default_model=DEFAULT_MODEL)
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

    def _process(self, agent: Agent, prompt: BaseArtifact | str, model: str) -> Structure:
        # License-policy gate immediately before the framework driver call. Askulator overrides
        # BaseTask._process wholesale (different streaming/parsing loop), so it declares here
        # rather than relying on the base implementation's declaration.
        require_model_invocation_sync(self, model)

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
        model = self._require_permitted_model()

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
