import random
from enum import StrEnum
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_components.seed_parameter import SeedParameter
from griptape_nodes.traits.options import Options


class NumberMode(StrEnum):
    INTEGER = "integer"
    FLOAT = "float"


class RandomNumber(ControlNode):
    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.add_parameter(
            Parameter(
                name="mode",
                tooltip="Generate a whole integer or a decimal float.",
                type="str",
                default_value="integer",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(NumberMode))},
            )
        )
        self.add_parameter(
            Parameter(
                name="minimum",
                tooltip="Minimum value (inclusive).",
                input_types=["int", "float"],
                default_value=0,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )
        self.add_parameter(
            Parameter(
                name="maximum",
                tooltip="Maximum value (inclusive for integers, exclusive for floats).",
                input_types=["int", "float"],
                default_value=100,
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            )
        )

        self._seed_parameter = SeedParameter(self)
        self._seed_parameter.add_input_parameters()

        self.add_parameter(
            Parameter(
                name="result",
                tooltip="The generated random number.",
                output_type="float",
                allowed_modes={ParameterMode.OUTPUT},
                default_value=0,
            )
        )

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        self._seed_parameter.after_value_set(parameter, value)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        self._seed_parameter.preprocess()

        mode = self.get_parameter_value("mode") or NumberMode.INTEGER
        minimum = self.get_parameter_value("minimum") or 0
        maximum = self.get_parameter_value("maximum") or 100
        seed = self._seed_parameter.get_seed()

        rng = random.Random(seed)

        match mode:
            case NumberMode.INTEGER:
                self.parameter_output_values["result"] = rng.randint(int(minimum), int(maximum))
            case NumberMode.FLOAT:
                self.parameter_output_values["result"] = rng.uniform(float(minimum), float(maximum))
            case _:
                msg = f"Unknown mode: {mode!r}"
                raise ValueError(msg)
