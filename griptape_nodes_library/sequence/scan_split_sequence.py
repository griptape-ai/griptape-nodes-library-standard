"""ScanSplitSequenceNode: scan a fileseq template into one Sequence per contiguous run.

Multi-sequence scanner. Always uses `MissingItemPolicy.SPLIT` internally, so
a directory like `[1..5, 8..12, 15]` produces three Sequence objects covering each
contiguous run. There is no policy dropdown — picking this node IS the policy.

Output is `sequences: list[Sequence]`. The artist who wants paths-per-sub-sequence
iterates over `sequences` and pulls `seq.entries` themselves.

Disk I/O and fileseq parsing run via `ScanSequencesRequest` on the engine's
event bus, so they happen on a worker thread without blocking the event loop.
"""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.common.macro_parser import (
    MacroSyntaxError,
    ParsedMacro,
)
from griptape_nodes.common.sequences import MissingItemPolicy
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultFailure,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.sequence.advanced_sequence_component import (
    ItemRangeGroup,
    resolve_project_macro,
    split_resolved_path,
)

logger = logging.getLogger("griptape_nodes")


class ScanSplitSequenceNode(SuccessFailureNode):
    """Scan a fileseq template into one Sequence per contiguous run of present items.

    Visible by default: `template`, `sequences`. The Item range controls live in
    the collapsed Advanced Sequence Control group.

    On total failure (bad template, unresolvable project vars, missing directory,
    no matches), the node emits an empty list and routes through the Failure
    control-flow edge.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._template_param = ParameterString(
            name="template",
            default_value="",
            tooltip=(
                "Macro template. May contain a fileseq sequence token (`####` or `%04d`). "
                "Non-sequence templates are NOT supported here."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Sequence path"},
        )
        self.add_parameter(self._template_param)

        self._sequences_param = Parameter(
            name="sequences",
            tooltip=("List of Sequence objects, one per contiguous run of present items in the active range."),
            type="list",
            output_type="list[Sequence]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._sequences_param)

        self._fail_on_empty_param = Parameter(
            name="fail_on_empty_result",
            input_types=["bool"],
            type="bool",
            output_type="bool",
            default_value=True,
            tooltip=(
                "Fail the node when the scan returns no items. Most artists want this — an empty result almost "
                "always means a misconfigured template, range, or path. Turn off if your workflow legitimately "
                "tolerates empty scans (e.g. a sweep that may find nothing)."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Treat empty sequence as a failure"},
        )
        self.add_parameter(self._fail_on_empty_param)

        with ParameterGroup(name="Advanced Sequence Control", collapsed=True) as advanced_group:
            self._item_range = ItemRangeGroup()
        self.add_node_element(advanced_group)

        self._create_status_parameters(
            result_details_tooltip="Details about the split-sequence scan",
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
        self._item_range.handle_after_value_set(self, parameter, value)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        self._clear_execution_status()

        template = self.get_parameter_value(self._template_param.name).strip()
        if not template:
            self._emit_failure("No template provided.")
            return

        try:
            parsed = ParsedMacro(template)
        except MacroSyntaxError as e:
            self._emit_failure(f"Invalid template: {e}")
            return

        resolved_path = resolve_project_macro(parsed)
        if resolved_path is None:
            self._emit_failure(f"Could not resolve project variables for template: {template}")
            return

        split = split_resolved_path(resolved_path)
        if split.filename_pattern is None:
            self._emit_failure(f"Resolved template '{resolved_path}' has no filename component.")
            return

        bounds_or_error = await self._item_range.resolve_bounds(self, split.directory, split.filename_pattern)
        if isinstance(bounds_or_error, str):
            self._emit_failure(bounds_or_error)
            return

        scan_result = await GriptapeNodes.ahandle_request(
            ScanSequencesRequest(
                directory=split.directory,
                pattern=split.filename_pattern,
                policy=MissingItemPolicy.SPLIT,
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
                pattern=split.filename_pattern,
            )
            if fail_on_empty:
                self._emit_failure(details)
            else:
                self._emit_empty_success(f"{details} (fail_on_empty_result=False)")
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
        ranges = ", ".join(f"{s.first}-{s.last}" for s in sequences)
        details = f"Scanned {len(sequences)} sub-sequence(s): {ranges}."
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
        pattern: str,
    ) -> str:
        """Pick the most informative status string for an empty scan result.

        Three diagnostic cases (least to most informative):
        1. directory_had_matching_files=False → no files match basename/extension
        2. directory_had_matching_files=True, discovered_first/last=None → padding mismatch
        3. discovered_first/last set → subset clipped every present item
        """
        if not directory_had_matching_files:
            return f"Scan found no items: directory contains no files matching '{pattern}'."
        if discovered_first is None or discovered_last is None:
            return (
                f"Scan found no items: directory contains files matching '{pattern}', but their numbering "
                f"padding does not match the template's padding."
            )
        if active_first is not None and active_last is not None:
            return (
                f"Scan found no items: discovered range is {discovered_first}..{discovered_last}, "
                f"but the active subset is {active_first}..{active_last}."
            )
        return f"Scan found no items: discovered range is {discovered_first}..{discovered_last}."
