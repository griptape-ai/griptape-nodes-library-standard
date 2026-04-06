"""RetryGroup Node - A group that retries its child nodes on failure."""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_groups.base_while_node_group import BaseWhileNodeGroup
from griptape_nodes.exe_types.node_groups.subflow_node_group import RIGHT_PARAMETERS_KEY

logger = logging.getLogger("griptape_nodes")


class RetryGroupNode(BaseWhileNodeGroup):
    """Retry Group Node that re-executes its child nodes on failure.

    Place nodes inside this group and connect their success/failure outputs
    to the group's Succeeded and Failed control inputs on the right side.
    If Failed is triggered and retry attempts remain, the entire group
    re-executes. When Succeeded is triggered or retries are exhausted,
    execution continues downstream.

    Parameters:
        max_iterations (input/property): Maximum number of retry attempts (default 3)
        iteration (output, left): Current attempt number (0-based) - connect to internal nodes
        done (control input, right): Connect success outputs here to stop retrying
        continue_loop (control input, right): Connect failure outputs here to trigger a retry
        was_successful (output, right): Whether the group ultimately succeeded
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # Override display names to use retry-specific vocabulary
        self.done.display_name = "Succeeded"
        self.continue_loop.display_name = "Failed"

        # Add retry-specific parameter
        self.was_successful = Parameter(
            name="was_successful",
            tooltip="Whether the group ultimately succeeded after all attempts",
            type=ParameterTypeBuiltin.BOOL.value,
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
            default_value=False,
        )
        self.add_parameter(self.was_successful)
        self.metadata[RIGHT_PARAMETERS_KEY].append("was_successful")

        self.raise_on_failure = Parameter(
            name="raise_on_failure",
            tooltip="Raise an error and halt the workflow if all retry attempts are exhausted",
            type=ParameterTypeBuiltin.BOOL.value,
            allowed_modes={ParameterMode.PROPERTY},
            default_value=False,
        )
        self.add_parameter(self.raise_on_failure)

        self.move_element_to_position("max_iterations", 0)
        self.move_element_to_position("raise_on_failure", 1)
        self.move_element_to_position("was_successful", 2)
        self.add_parameter_to_group_settings(self.max_iterations)
        self.add_parameter_to_group_settings(self.raise_on_failure)

    def _before_loop_iteration(self, iteration: int) -> None:
        """Called before each retry attempt.

        Args:
            iteration: The upcoming attempt number (0-based, so 1 means first retry)
        """
        logger.info(
            "Retry Group '%s': retrying (attempt %d of %d)",
            self.name,
            iteration + 1,
            self._max_iterations_value + 1,
        )

    def _on_complete(self, *, condition_met: bool, iterations: int) -> None:
        """Called after all retry attempts are finished, before the node resolves.

        Args:
            condition_met: Whether the group ultimately succeeded
            iterations: Total number of attempts that were executed
        """
        super()._on_complete(condition_met=condition_met, iterations=iterations)
        self.parameter_output_values["was_successful"] = condition_met

        if condition_met:
            return

        raise_on_failure = self.get_parameter_value("raise_on_failure")
        if not raise_on_failure:
            return

        msg = f"Retry Group '{self.name}' failed after {iterations} attempt(s)"
        raise RuntimeError(msg)
