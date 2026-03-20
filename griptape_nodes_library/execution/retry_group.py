"""RetryGroup Node - A group that retries its child nodes on failure."""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode, ParameterTypeBuiltin
from griptape_nodes.exe_types.node_groups import BaseRetryNodeGroup

logger = logging.getLogger("griptape_nodes")


class RetryGroupNode(BaseRetryNodeGroup):
    """Retry Group Node that re-executes its child nodes on failure.

    Place nodes inside this group and connect their success/failure outputs
    to the group's Succeeded and Failed control inputs on the right side.
    If Failed is triggered and retry attempts remain, the entire group
    re-executes. When Succeeded is triggered or retries are exhausted,
    execution continues downstream.

    Parameters:
        max_retries (input/property): Maximum number of retry attempts (default 3)
        attempt_number (output, left): Current attempt number (0-based) - connect to internal nodes
        succeeded (control input, right): Connect success outputs here to stop retrying
        failed (control input, right): Connect failure outputs here to trigger a retry
        was_successful (output, right): Whether the group ultimately succeeded
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.raise_on_failure = Parameter(
            name="raise_on_failure",
            tooltip="Raise an error and halt the workflow if all retry attempts are exhausted",
            type=ParameterTypeBuiltin.BOOL.value,
            allowed_modes={ParameterMode.PROPERTY},
            default_value=False,
        )
        self.add_parameter(self.raise_on_failure)

        self.move_element_to_position("max_retries", 0)
        self.move_element_to_position("raise_on_failure", 1)
        self.move_element_to_position("was_successful", 2)
        self.add_parameter_to_group_settings(self.max_retries)
        self.add_parameter_to_group_settings(self.raise_on_failure)

    def _on_retry(self, attempt: int) -> None:
        """Called before each retry attempt.

        Args:
            attempt: The upcoming attempt number (0-based, so 1 means first retry)
        """
        logger.info(
            "Retry Group '%s': retrying (attempt %d of %d)",
            self.name,
            attempt + 1,
            self._max_retries_value + 1,
        )

    def _on_complete(self, *, succeeded: bool, attempts: int) -> None:
        """Called after all retry attempts are finished, before the node resolves.

        Args:
            succeeded: Whether the group ultimately succeeded
            attempts: Total number of attempts that were executed
        """
        if succeeded:
            return

        raise_on_failure = self.get_parameter_value("raise_on_failure")
        if not raise_on_failure:
            return

        msg = f"Retry Group '{self.name}' failed after {attempts} attempt(s)"
        raise RuntimeError(msg)
