"""ForLoop Group Node - A single node that iterates over a numeric range with an encapsulated loop body."""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_groups import BaseIterativeNodeGroup
from griptape_nodes.exe_types.node_groups.subflow_node_group import LEFT_PARAMETERS_KEY
from griptape_nodes.traits.clamp import Clamp

logger = logging.getLogger("griptape_nodes")


class ForLoopGroupNode(BaseIterativeNodeGroup):
    """ForLoop Group Node that iterates over a numeric range.

    This node combines the functionality of ForLoopStartNode and ForLoopEndNode
    into a single group node. Child nodes added to the group become the loop body
    and are executed for each value in the numeric range.

    Parameters:
        start (input/property): Starting value for the loop (default 1)
        end (input/property): Ending value for the loop (default 10)
        end_inclusive (property): Include end value in the loop (default True)
        step (input/property): Step size for each iteration (default 1, min 1, max 1000)
        index (output, left): Current loop value (start, start+step, …) — connect to internal nodes
        new_item_to_add (input, right): Item to collect from each iteration
        skip_iteration (control input, right): Skip current iteration and continue to next
        break_loop (control input, right): Break out of loop immediately
        results (output, right): Collected results from all iterations

    Supports both sequential and parallel execution modes. skip_iteration and
    break_loop are only available in sequential mode.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        self.start_value = Parameter(
            name="start",
            tooltip="Starting value for the loop",
            type=ParameterTypeBuiltin.INT.value,
            input_types=["int", "float"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value=1,
        )
        self.add_parameter(self.start_value)

        self.end_value = Parameter(
            name="end",
            tooltip="Ending value for the loop",
            type=ParameterTypeBuiltin.INT.value,
            input_types=["int", "float"],
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value=10,
        )
        self.add_parameter(self.end_value)

        self.end_inclusive = Parameter(
            name="end_inclusive",
            tooltip="Include the end value in the loop (True) or exclude it (False)",
            type=ParameterTypeBuiltin.BOOL.value,
            allowed_modes={ParameterMode.PROPERTY},
            default_value=True,
            ui_options={"display_name": "Include end value"},
        )
        self.add_parameter(self.end_inclusive)

        self.step_value = Parameter(
            name="step",
            tooltip="Step size for each iteration (always positive)",
            type=ParameterTypeBuiltin.INT.value,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            default_value=1,
        )
        self.step_value.add_trait(Clamp(min_val=1, max_val=1000))
        self.add_parameter(self.step_value)

        # Register start/end/step on the left rail, after exec_in, before index
        left = self.metadata.setdefault(LEFT_PARAMETERS_KEY, [])
        insert_at = left.index("exec_in") + 1 if "exec_in" in left else 0
        for name in ("start", "end", "step"):
            left.insert(insert_at, name)
            insert_at += 1

        self.move_element_to_position("start", 0)
        self.move_element_to_position("end", 1)
        self.move_element_to_position("end_inclusive", 2)
        self.move_element_to_position("step", 3)
        self.move_element_to_position("results", 4)
        self.move_element_to_position("execution_mode", 5)
        self.add_parameter_to_group_settings(self.execution_mode)
        self.add_parameter_to_group_settings(self.end_inclusive)

    # ------------------------------------------------------------------
    # Iteration math (sibling implementation — does not import ForLoopStartNode)
    # ------------------------------------------------------------------

    def _get_total_iterations(self) -> int:
        start = self.get_parameter_value("start")
        end = self.get_parameter_value("end")
        step = self.get_parameter_value("step")
        end_inclusive = self.get_parameter_value("end_inclusive")

        if step <= 0:
            return 0

        if start <= end:  # ascending
            if end_inclusive:
                return int(max(0, (end - start) // step + 1))
            return int(max(0, (end - start + step - 1) // step))
        # descending
        if end_inclusive:
            return int(max(0, (start - end) // step + 1))
        return int(max(0, (start - end + step - 1) // step))

    def get_all_iteration_values(self) -> list[int]:
        """Return the actual loop values (e.g. [5, 7, 9] for start=5, end=10, step=2)."""
        start = self.get_parameter_value("start")
        end = self.get_parameter_value("end")
        step = self.get_parameter_value("step")
        total_iterations = self._get_total_iterations()

        if total_iterations == 0:
            return []

        ascending = start <= end

        values = []
        current_value = start
        for _ in range(total_iterations):
            values.append(current_value)
            if ascending:
                current_value += step
            else:
                current_value -= step

        return values

    def _get_iteration_items(self) -> list[Any]:
        """Return the numeric range values as the iteration items."""
        return self.get_all_iteration_values()

    def _get_current_item_value(self, iteration_index: int) -> Any:
        """Return the loop value at the given iteration index."""
        if self._items and iteration_index < len(self._items):
            return self._items[iteration_index]
        return None

    def _initialize_iteration_data(self) -> None:
        """Populate _items from the numeric range before iteration starts."""
        self._items = self.get_all_iteration_values()
        self._total_iterations = len(self._items)
        self._current_iteration_count = 0
        self._results_list = []

    def get_current_index(self) -> int:
        """Return the current loop value (start, start+step, ...) for the current iteration."""
        if self._items and self._current_iteration_count < len(self._items):
            return self._items[self._current_iteration_count]
        return self.get_parameter_value("start")

    def _validate_parameter_values(self) -> list[Exception]:
        exceptions: list[Exception] = []
        start = self.get_parameter_value("start")
        end = self.get_parameter_value("end")
        step = self.get_parameter_value("step")
        end_inclusive = self.get_parameter_value("end_inclusive")

        if step < 1:
            msg = f"{self.name}: Step value must be positive (>= 1), got {step}"
            exceptions.append(Exception(msg))

        if start == end:
            if end_inclusive:
                logger.info(
                    "%s: Loop will execute 1 iteration since start (%s) equals end (%s) and end_inclusive is True.",
                    self.name,
                    start,
                    end,
                )
            else:
                logger.info(
                    "%s: Loop will execute 0 iterations since start (%s) equals end (%s) and end_inclusive is False.",
                    self.name,
                    start,
                    end,
                )

        return exceptions

    def validate_before_workflow_run(self) -> list[Exception] | None:
        exceptions: list[Exception] = []

        if validation_exceptions := self._validate_parameter_values():
            exceptions.extend(validation_exceptions)

        parent_exceptions = super().validate_before_workflow_run()
        if parent_exceptions:
            exceptions.extend(parent_exceptions)

        return exceptions if exceptions else None
