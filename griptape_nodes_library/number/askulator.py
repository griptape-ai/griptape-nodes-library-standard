import json
from collections.abc import Callable

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from json_repair import repair_json  # json_repair
from pydantic import BaseModel
from pydantic_ai import PromptedOutput

from griptape_nodes_library.llm.tools import ToolType, build_toolsets
from griptape_nodes_library.tasks.base_task import BaseTask

DEFAULT_MODEL = "gpt-4.1-mini"

RULESETS = [
    {
        "name": "Default Ruleset",
        "rules": [
            "You are a natural language calculator.",
            "If given a prompt you don't have a number for, make something up that seems appropriate. Ex: Gajillion = 1,000,000,0000,0000",
            "If there is insufficient information to answer the question, like a missing variable or something, use some likely number and explain why in your reasoning.",
            "You try your best to answer the question, your reasoning can be creative an interesting.",
            "Feel free to use newlines in your reasoning to make it more readable.",
            "Use the Calculate action with expression in the Calculator tool to do the math.",
            "Your final answer should be concise. Only a number and unit if applicable.",
        ],
    }
]


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

    def _stream_answer(self, tokens: list[str]) -> Callable[[str], None]:
        """Stream the JSON answer's `reasoning` into `output` and `final_answer` into `result` as it grows."""
        last = {"reasoning": "", "final_answer": ""}
        destinations = {"reasoning": "output", "final_answer": "result"}

        def on_text(token: str) -> None:
            tokens.append(token)
            try:
                partial = json.loads(repair_json("".join(tokens)))  # pyright: ignore[reportArgumentType]
            except json.JSONDecodeError:
                return  # Incomplete JSON
            if not isinstance(partial, dict):
                return
            for key, destination in destinations.items():
                current = partial.get(key)
                if isinstance(current, str) and current != last[key]:
                    self.append_value_to_parameter(destination, value=current[len(last[key]) :])
                    last[key] = current

        return on_text

    def process(self) -> AsyncResult[str]:
        instruction = self.get_parameter_value("instruction")
        model = self._require_permitted_model()

        toolsets = build_toolsets([{"tool_type": ToolType.CALCULATOR}])
        user_input = f"Give me the answer for: {instruction}\n."

        if instruction and not instruction.isspace():
            tokens: list[str] = []

            def on_tool_call(tool_name: str, _args: str) -> None:
                self.append_value_to_parameter("output", value=f"Using a {tool_name}\n")

            def _process() -> str:
                # PromptedOutput makes the model answer in JSON text, which streams; a tool-call answer would not.
                result = self._process(
                    user_input,
                    model,
                    rulesets=RULESETS,
                    toolsets=toolsets,
                    output_type=PromptedOutput(Output),
                    on_text=self._stream_answer(tokens),
                    on_tool_call=on_tool_call,
                )
                # Streaming parses raw text, which retries or preamble can garble; the validated answer is authoritative.
                self.parameter_output_values["result"] = result.output.final_answer
                return result.text

            yield _process
