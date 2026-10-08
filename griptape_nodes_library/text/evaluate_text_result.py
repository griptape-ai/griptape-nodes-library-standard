from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.traits.options import Options
from pydantic import BaseModel

from griptape_nodes_library.llm.model_config import cloud_model_config
from griptape_nodes_library.llm.runner import prompt_model
from griptape_nodes_library.tasks.base_task import BaseTask
from griptape_nodes_library.utils.model_invocation import require_model_invocation_sync

EXAMPLES = [
    {
        "label": "Choose a preset..",
    },
    {
        "label": "Paraphrase",
        "criteria": "Does the output accurately paraphrase the input without losing meaning?",
        "input": "The quick brown fox jumps over the lazy dog.",
        "expected_output": "A swift brown fox leaps above a sleeping dog.",
        "actual_output": "A fast fox jumps over a dog that's not awake.",
    },
    {
        "label": "Factual",
        "criteria": "Is the output factually correct based on the input?",
        "input": "The capital of France is Paris.",
        "expected_output": "Paris is the capital city of France.",
        "actual_output": "France's capital is Paris.",
    },
    {
        "label": "Analogy",
        "criteria": "Does the output correctly complete the analogy?",
        "input": "A bird is to sky as a fish is to ______.",
        "expected_output": "water",
        "actual_output": "concrete",
    },
]

EXAMPLE_OPTIONS = [example["label"] for example in EXAMPLES]
DEFAULT_MODEL = "gpt-4.1"

# The prompts griptape's EvalEngine used: criteria -> evaluation steps -> score and reason.
STEPS_INSTRUCTIONS = """Given an evaluation criteria which outlines how you should judge the {evaluation_params}, generate 3-4 concise evaluation steps based on the criteria below.
You MUST make it clear how to evaluate {evaluation_params} in relation to one another.

Evaluation Criteria:
{criteria}"""
RESULTS_INSTRUCTIONS = """Given the evaluation steps, return a JSON with two keys:
1) a `score` key ranging from 0 - 10, with 10 being that it follows the criteria outlined in the steps and 0 being that it does not.
2) a `reason` key, a reason for the given score. Please mention specific information from {evaluation_params} in your reason, but be very concise with it!

Evaluation Steps:
{evaluation_steps}

{evaluation_text}"""
JSON_PROMPT = "JSON:"


class EvaluationSteps(BaseModel):
    steps: list[str]


class EvaluationResults(BaseModel):
    score: float
    reason: str


class EvaluateTextResult(BaseTask):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_node_element(
            ParameterMessage(
                name="Testing Inputs",
                variant="info",
                title="Testing Inputs",
                value="Run an evaluation by providing some criteria to test by, an input, expected output, and actual output.\nUnsure what to do? Try a preset example!",
            )
        )
        self.add_parameter(
            Parameter(
                name="Examples",
                type="str",
                default_value=EXAMPLE_OPTIONS[0],
                tooltip="Whether to automatically provide evaluation steps",
                traits={Options(choices=EXAMPLE_OPTIONS)},
                allowed_modes={ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="input",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": True, "placeholder_text": "Input text to process"},
            )
        )
        self.add_parameter(
            Parameter(
                name="expected_output",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="Expected output",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": False, "placeholder_text": "Expected output"},
            )
        )
        self.add_parameter(
            Parameter(
                name="actual_output",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="Actual output",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"multiline": False, "placeholder_text": "Actual output"},
            )
        )
        self.add_parameter(
            Parameter(
                name="criteria",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="Criteria for the evaluation",
                ui_options={"multiline": True, "placeholder_text": "Criteria for the evaluation"},
            )
        )

        self._add_model_parameter(default_model=DEFAULT_MODEL)
        self.add_node_element(
            ParameterMessage(
                name="Review Results",
                variant="success",
                title="Review Results",
                value="View the results of the evaluation, including a score and the reason for the result.",
            )
        )

        self.add_parameter(
            Parameter(
                name="score",
                type="float",
                default_value=0.0,
                tooltip="Score",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )
        self.add_parameter(
            Parameter(
                name="reason",
                type="str",
                default_value="",
                tooltip="Feedback",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"multiline": True, "placeholder_text": "Reason for result"},
            )
        )

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name in ["criteria", "input", "expected_output", "actual_output"]:
            self.set_parameter_value("Examples", EXAMPLE_OPTIONS[0])

        if parameter.name == "Examples" and value != EXAMPLE_OPTIONS[0]:
            self.set_parameter_value("criteria", EXAMPLES[EXAMPLE_OPTIONS.index(value)]["criteria"])
            self.set_parameter_value("input", EXAMPLES[EXAMPLE_OPTIONS.index(value)]["input"])
            self.set_parameter_value("expected_output", EXAMPLES[EXAMPLE_OPTIONS.index(value)]["expected_output"])
            self.set_parameter_value("actual_output", EXAMPLES[EXAMPLE_OPTIONS.index(value)]["actual_output"])

            self.parameter_output_values["criteria"] = EXAMPLES[EXAMPLE_OPTIONS.index(value)]["criteria"]
            self.parameter_output_values["input"] = EXAMPLES[EXAMPLE_OPTIONS.index(value)]["input"]
            self.parameter_output_values["expected_output"] = EXAMPLES[EXAMPLE_OPTIONS.index(value)]["expected_output"]
            self.parameter_output_values["actual_output"] = EXAMPLES[EXAMPLE_OPTIONS.index(value)]["actual_output"]

        return super().after_value_set(parameter, value)

    def _evaluate(self, model: str, criteria: str, evaluation_params: dict[str, str]) -> tuple[float, str]:
        """Generate evaluation steps from `criteria`, then score `evaluation_params` against them."""
        # License-policy gate immediately before the model call. The evaluation prompts the
        # model directly rather than through BaseTask._process, so it declares here.
        require_model_invocation_sync(self, model)

        config = cloud_model_config(model)
        param_names = ", ".join(evaluation_params)
        steps = prompt_model(
            config,
            JSON_PROMPT,
            instructions=STEPS_INSTRUCTIONS.format(evaluation_params=param_names, criteria=criteria),
            output_type=EvaluationSteps,
        ).steps
        results = prompt_model(
            config,
            JSON_PROMPT,
            instructions=RESULTS_INSTRUCTIONS.format(
                evaluation_params=param_names,
                evaluation_steps=steps,
                evaluation_text="\n\n".join(f"{key}: {value}" for key, value in evaluation_params.items()),
            ),
            output_type=EvaluationResults,
        )
        # The model scores 0-10 to avoid floating point ambiguity; the node reports 0-1.
        return results.score / 10, results.reason

    def process(self) -> AsyncResult[None]:
        criteria = self.get_parameter_value("criteria")
        model = self._require_permitted_model()
        if not criteria:
            msg = "criteria must not be empty"
            raise ValueError(msg)

        evaluation_params = {
            "Input": self.get_parameter_value("input"),
            "Actual Output": self.get_parameter_value("actual_output"),
            "Expected Output": self.get_parameter_value("expected_output"),
        }

        def _process() -> None:
            score, reason = self._evaluate(model, criteria, evaluation_params)
            self.parameter_output_values["score"] = score
            self.parameter_output_values["reason"] = reason

        yield _process
