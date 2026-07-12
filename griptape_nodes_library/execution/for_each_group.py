"""ForEach Group Node - A single node that iterates over items with an encapsulated loop body."""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
    ParameterTypeBuiltin,
)
from griptape_nodes.exe_types.node_groups import BaseIterativeNodeGroup
from griptape_nodes.exe_types.param_types.parameter_bool import ParameterBool
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt

logger = logging.getLogger("griptape_nodes")


class ForEachGroupNode(BaseIterativeNodeGroup):
    """ForEach Group Node that iterates over a list or dictionary.

    This node combines the functionality of ForEachStartNode and ForEachEndNode
    into a single group node. Child nodes added to the group become the loop body
    and are executed for each item in the input list or dictionary.

    Parameters:
        items (input): List or dictionary to iterate through
        current_item (output, left): Current item being processed - connect to internal nodes
        index (output, left): Current iteration index - connect to internal nodes
        new_item_to_add (input, right): Item to collect from each iteration
        skip_iteration (control input, right): Skip current item and continue to next iteration
        break_loop (control input, right): Break out of loop immediately
        results (output, right): Collected results from all iterations

    The node supports both sequential and parallel execution modes via the
    'run_in_order' property, and can execute locally, in a private subprocess,
    or via cloud services depending on the 'execution_environment' setting.

    Note: skip_iteration and break_loop are only available in sequential mode
    (Run Group Items One at a Time). They are hidden in parallel mode.
    """

    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
    ) -> None:
        super().__init__(name, metadata)

        # ForEach-specific: items input parameter (left side)
        self.items_param = Parameter(
            name="items",
            tooltip="List or dictionary to iterate through",
            input_types=["list", "dict"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items_param)

        # Add to left parameters for UI layout (insert at beginning)
        if "left_parameters" in self.metadata:
            self.metadata["left_parameters"].insert(0, "items")
        else:
            self.metadata["left_parameters"] = ["items"]

        # ForEach-specific: current_item output parameter (left side - feeds into group)
        self.current_item = Parameter(
            name="current_item",
            tooltip="Current item being processed",
            output_type=ParameterTypeBuiltin.ALL.value,
            allowed_modes={ParameterMode.OUTPUT},
            settable=False,
        )
        self.add_parameter(self.current_item)

        # Add current_item to left parameters (after items, before index)
        if "left_parameters" in self.metadata:
            # Insert after items but before index
            idx = self.metadata["left_parameters"].index("index")
            self.metadata["left_parameters"].insert(idx, "current_item")
        else:
            self.metadata["left_parameters"] = ["items", "current_item", "index"]

        # Testing mode — run the loop body for a single chosen item
        self.testing_mode = ParameterBool(
            name="testing_mode",
            tooltip="When enabled, run only one iteration using the item at the chosen index",
            allowed_modes={ParameterMode.PROPERTY},
            default_value=False,
            display_name="Testing Mode",
        )
        self.add_parameter(self.testing_mode)

        self.test_item_index = ParameterInt(
            name="test_item_index",
            tooltip="Index of the item to test (0-based). Clamped to the valid range at runtime.",
            allowed_modes={ParameterMode.PROPERTY},
            default_value=0,
            min_val=0,
            display_name="Test Item Index",
            hide=True,
        )
        self.add_parameter(self.test_item_index)

        self.move_element_to_position("items", 0)
        self.move_element_to_position("results", 1)
        self.move_element_to_position("execution_mode", 2)
        self.add_parameter_to_group_settings(self.execution_mode)
        self.add_parameter_to_group_settings(self.testing_mode)
        self.add_parameter_to_group_settings(self.test_item_index)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        super().after_value_set(parameter, value)
        if parameter == self.testing_mode:
            self.test_item_index.hide = not value
            self.execution_mode.hide = value
            if not value:
                # Restore skip/break visibility to match current execution_mode
                super().after_value_set(self.execution_mode, self.get_parameter_value("execution_mode"))

    def _get_iteration_items(self) -> list[Any]:
        """Get the list of items to iterate over.

        Accepts either a list or dict. If dict, converts to list of {"key": k, "value": v} dicts.
        """
        items = self.get_parameter_value("items")

        # Handle case where items parameter is not connected or has no value
        if items is None:
            logger.info("ForEach Group '%s': No items provided, skipping loop execution", self.name)
            return []

        if isinstance(items, dict):
            if len(items) == 0:
                logger.info("ForEach Group '%s': Empty dictionary provided, skipping loop execution", self.name)
                return []
            all_items = [{"key": k, "value": v} for k, v in items.items()]
        elif isinstance(items, list):
            if len(items) == 0:
                logger.info("ForEach Group '%s': Empty list provided, skipping loop execution", self.name)
            all_items = items
        else:
            error_msg = f"ForEach Group '{self.name}' expected a list or dict but got {type(items).__name__}: {items}"
            raise TypeError(error_msg)

        if self.get_parameter_value("testing_mode") and all_items:
            index = self.get_parameter_value("test_item_index")
            index = max(0, min(index, len(all_items) - 1))
            return [all_items[index]]

        return all_items

    def _get_current_item_value(self, iteration_index: int) -> Any:
        """Get the current item value for a specific iteration.

        Args:
            iteration_index: 0-based iteration index

        Returns:
            The item at the given index, or None if out of bounds
        """
        if self._items and iteration_index < len(self._items):
            return self._items[iteration_index]
        return None

    def get_current_index(self) -> int:
        """Return the current array position (0, 1, 2, ...)."""
        return self._current_iteration_count
