from typing import Any

from griptape.engines import EvalEngine
from griptape.structures import Agent, Structure
from griptape_nodes.exe_types.core_types import Parameter, ParameterMessage, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult
from griptape_nodes.exe_types.param_components.model_access_component import ModelAccessComponent
from griptape_nodes.node_library.library_registry import resolve_provider_model_id
from griptape_nodes.traits.options import Options

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
MODEL_CHOICES = [
    "gtc_gpt_4_1",
    "gtc_gpt_4_1_mini",
    "gtc_gpt_4_1_nano",
    "gtc_gpt_5",
]
DEFAULT_MODEL = "gtc_gpt_4_1"

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

        if parameter.name == "model":
            self._model_access.on_value_changed(value)

        return super().after_value_set(parameter, value)

    def process(self) -> AsyncResult[Structure]:
        criteria = self.get_parameter_value("criteria")
        model = self.get_parameter_value("model")

        # License-policy runtime gate. Raises RuntimeError if the currently-selected
        # model is denied.
        self._model_access.raise_if_denied(model)

        # `model` is the catalog key the dropdown stores; the engine's driver needs the
        # upstream provider's own id instead.
        provider_model_id = resolve_provider_model_id(self, model) or ""
        engine = EvalEngine(criteria=criteria, prompt_driver=self.create_driver(model=provider_model_id))

        user_input = self.get_parameter_value("input")
        expected_output = self.get_parameter_value("expected_output")
        actual_output = self.get_parameter_value("actual_output")

        def _process() -> Structure:
            # License-policy gate immediately before the framework driver call. EvalEngine.evaluate
            # invokes the prompt driver directly rather than through BaseTask._process, so it
            # declares here rather than relying on the base implementation's declaration.
            require_model_invocation_sync(self, model)

            score, reason = engine.evaluate(
                input=user_input,
                expected_output=expected_output,
                actual_output=actual_output,
            )
            self.parameter_output_values["score"] = score
            self.parameter_output_values["reason"] = reason
            return Agent()

        yield _process
