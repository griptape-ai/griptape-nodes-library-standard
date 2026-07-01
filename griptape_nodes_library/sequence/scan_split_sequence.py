"""ScanSplitSequenceNode: scan a path or pattern into one Sequence per contiguous run.

Multi-sequence scanner. Always uses `MissingItemPolicy.SPLIT` internally, so
items 1..5, 8..12, 15 on disk produce three Sequence objects covering each
contiguous run. There is no policy dropdown — picking this node IS the policy.

Output is `sequences: list[Sequence]`. Wire it into any list-aware node
(ForEach Group, Get From List, etc.) — pulling out a single sub-sequence
goes into Inspect Sequence or any other `type="Sequence"` consumer.

Disk I/O, fileseq parsing, and macro resolution all run inside
`ScanSequencesRequest` on the engine's event bus, so they happen on a
worker thread without blocking the event loop.
"""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.common.sequences import MissingItemPolicy
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultFailure,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.file_system_picker import FileSystemPicker

from griptape_nodes_library.sequence.advanced_sequence_component import AdvancedSequenceControls

logger = logging.getLogger("griptape_nodes")


class ScanSplitSequenceNode(SuccessFailureNode):
    """Scan a path or pattern into one Sequence per contiguous run of present items.

    Visible by default: `template`, `sequences`. The Item range controls live in
    the collapsed Advanced Sequence Control group.

    On total failure (bad path or pattern, missing directory, no matches),
    the node emits an empty list and routes through the Failure control-flow
    edge.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._path_param = ParameterString(
            name="template",
            default_value="",
            tooltip=(
                "A path to a numbered file sequence. Use a token like `####`, `%04d`, or `@@@` "
                "where the frame number goes — for example `render.####.png` matches "
                "`render.0001.png`, `render.0002.png`, … A plain path with no token works too; "
                "it's read as a sequence of one. Project macros like `{inputs}/render.####.png` "
                "are accepted and preserved end-to-end (the items you get back keep the "
                "`{inputs}` form), so workflows stay portable across machines."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={FileSystemPicker(allow_files=True, allow_directories=True, allow_sequences=True)},
            ui_options={"display_name": "Path or pattern"},
        )
        self.add_parameter(self._path_param)

        self._sequences_param = Parameter(
            name="sequences",
            tooltip=(
                "One sequence per contiguous run of items found on disk. Wire into any node "
                "that consumes sequences directly, or into a list-aware node "
                "(**ForEach Group**, **Get From List**, etc.)."
            ),
            type="list",
            output_type="list[Sequence]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Sequences"},
        )
        self.add_parameter(self._sequences_param)

        self._fail_on_empty_param = Parameter(
            name="fail_on_empty_result",
            input_types=["bool"],
            type="bool",
            output_type="bool",
            default_value=True,
            tooltip=(
                "When on, the node reports a failure if the scan finds nothing — usually a sign "
                "the path, range, or pattern is off. When off, the node succeeds with an empty "
                "list; useful for sweeps that may legitimately come up empty."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Fail when no items are found"},
        )
        self.add_parameter(self._fail_on_empty_param)

        self._advanced = AdvancedSequenceControls()
        self.add_node_element(self._advanced.group)

        self._create_status_parameters(
            result_details_tooltip="Details about the scan",
            result_details_placeholder="Scan results will appear here.",
        )

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Skip SuccessFailureNode's status reset.

        The parent class clears `was_successful` / `result_details` here, which
        causes downstream re-resolutions of this node to leave the status reading
        False / placeholder if the engine doesn't re-run `aprocess()` immediately
        afterward. We skip the reset and let `aprocess()` own the status entirely.
        """
        return BaseNode.validate_before_workflow_run(self)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Skip SuccessFailureNode's status reset (see `validate_before_workflow_run`)."""
        return BaseNode.validate_before_node_run(self)

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Forward item-range mode toggles to the shared helper."""
        self._advanced.handle_after_value_set(self, parameter, value)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        self._clear_execution_status()

        path = self.get_parameter_value(self._path_param.name).strip()
        if not path:
            self._emit_failure("No path or pattern provided.")
            return

        bounds_or_error = await self._advanced.resolve_bounds(self, path)
        if isinstance(bounds_or_error, str):
            self._emit_failure(bounds_or_error)
            return

        scan_result = await GriptapeNodes.ahandle_request(
            ScanSequencesRequest(
                path=path,
                policy=MissingItemPolicy.SPLIT,
                no_token_behavior=self._advanced.behavior_for_request(self),
                start_number=bounds_or_error.start_number,
                end_number=bounds_or_error.end_number,
            )
        )
        if isinstance(scan_result, ScanSequencesResultFailure):
            self._emit_failure(str(scan_result.result_details))
            return
        if not isinstance(scan_result, ScanSequencesResultSuccess):
            self._emit_failure(f"Unexpected scan result type: {type(scan_result).__name__}")
            return

        if not scan_result.has_entries:
            fail_on_empty = bool(self.get_parameter_value(self._fail_on_empty_param.name))
            details = self._build_empty_result_details(
                directory_had_matching_files=scan_result.directory_had_matching_files,
                discovered_first=scan_result.discovered_first,
                discovered_last=scan_result.discovered_last,
                active_first=bounds_or_error.start_number,
                active_last=bounds_or_error.end_number,
                path=path,
            )
            if fail_on_empty:
                self._emit_failure(details)
            else:
                self._emit_empty_success(f"{details} (*Fail when no items are found* is off.)")
            return

        self._emit_success(scan_result.sequences)

    def _emit_empty_success(self, details: str) -> None:
        """Emit empty outputs and mark the node successful (fail_on_empty_result=False path)."""
        self.parameter_output_values[self._sequences_param.name] = []
        self._set_status_results(was_successful=True, result_details=details)

    def _emit_failure(self, details: str) -> None:
        """Emit empty outputs and mark the node failed."""
        self.parameter_output_values[self._sequences_param.name] = []
        self._set_status_results(was_successful=False, result_details=details)

    def _emit_success(self, sequences: list[Any]) -> None:
        """Populate outputs and mark the node successful."""
        self.parameter_output_values[self._sequences_param.name] = sequences
        ranges = ", ".join(_format_run(s) for s in sequences)
        details = f"Found {len(sequences)} sub-sequence(s): {ranges}."
        if any(s.dropped_negative_number_count for s in sequences):
            total_dropped = sum(s.dropped_negative_number_count for s in sequences)
            details += f" Dropped {total_dropped} negative number(s)."
        self._set_status_results(was_successful=True, result_details=details)

    @staticmethod
    def _build_empty_result_details(
        *,
        directory_had_matching_files: bool,
        discovered_first: int | None,
        discovered_last: int | None,
        active_first: int | None,
        active_last: int | None,
        path: str,
    ) -> str:
        """Pick the most informative status string for an empty scan result.

        Three diagnostic cases (least to most informative):
        1. directory_had_matching_files=False → no files match basename/extension
        2. directory_had_matching_files=True, discovered_first/last=None → padding mismatch
        3. discovered_first/last set → subset clipped every present item
        """
        if not directory_had_matching_files:
            return f"No items matched `{path}` — nothing in that location matches that name and extension."
        if discovered_first is None or discovered_last is None:
            return (
                f"No items matched `{path}` — files exist with that name, but their numbering "
                "width doesn't match the token (e.g. 3-digit numbers against `####`)."
            )
        if active_first is not None and active_last is not None:
            return (
                "No items in the active range. "
                f"Items {discovered_first}–{discovered_last} exist on disk; "
                f"you asked for {active_first}–{active_last}."
            )
        return f"No items in the active range. Items {discovered_first}–{discovered_last} exist on disk."


def _format_run(sequence: Any) -> str:
    """Render a single sub-sequence's run for the status detail line.

    Singletons render as `item N` rather than `N–N` so the status reads
    naturally — e.g. *Found 3 sub-sequences: items 1–2, item 4, items 6–7.*
    """
    if sequence.first == sequence.last:
        return f"item {sequence.first}"
    return f"items {sequence.first}–{sequence.last}"
