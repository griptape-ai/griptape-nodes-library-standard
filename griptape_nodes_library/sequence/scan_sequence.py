"""ScanSequenceNode: scan a fileseq template into a flat list of file paths.

The 80% artist-facing entry point for sequence work. Inputs:
- a macro / fileseq template (with `{inputs}` etc. resolvable through the project)
- a missing-item policy (default: SKIP)

Default output:
- `paths: list[str]` — the file paths in ascending number order, ready for
  a for-loop / Load Image / Video Player.

The full structured `Sequence` object is available under an "Advanced Sequence
Control" group (collapsed by default), alongside `Item range` controls for
sub-setting the discovered range. Multi-sequence (SPLIT) work lives in
ScanSplitSequenceNode.

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
from griptape_nodes.common.sequences import MissingItemPolicy, Sequence
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultFailure,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

from griptape_nodes_library.sequence.advanced_sequence_component import (
    ItemRangeGroup,
    resolve_project_macro,
    split_resolved_path,
)

logger = logging.getLogger("griptape_nodes")


# Friendly dropdown labels for the engine's `MissingItemPolicy`. SPLIT is
# intentionally omitted here — multi-sequence work is the job of
# ScanSplitSequenceNode. Order in this dict drives the dropdown order.
_POLICY_LABELS: dict[MissingItemPolicy, str] = {
    MissingItemPolicy.SKIP: "Skip over gaps",
    MissingItemPolicy.FILL_NEAREST: "Fill gaps with nearest",
    MissingItemPolicy.ABORT: "Abort on sequence gaps",
}
_LABEL_TO_POLICY: dict[str, MissingItemPolicy] = {label: policy for policy, label in _POLICY_LABELS.items()}


class ScanSequenceNode(SuccessFailureNode):
    """Scan a fileseq template into a flat list of file paths.

    Visible by default: `template`, `missing_item_policy`, `paths`. The full
    structured `Sequence` object lives in the collapsed Advanced Sequence
    Control group, along with the Item range controls.

    On total failure (bad template, unresolvable project vars, missing
    directory, no matches, ABORT-policy gap), the node emits empty outputs
    and routes through the Failure control-flow edge.
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

        self._policy_param = ParameterString(
            name="missing_item_policy",
            default_value=_POLICY_LABELS[MissingItemPolicy.SKIP],
            tooltip=(
                "How to handle gaps inside a sequence's range. "
                "`Skip over gaps` returns only present items. "
                "`Fill gaps with nearest` replaces each missing slot with the nearest neighbor's path. "
                "`Abort on sequence gaps` fails the node on the first gap, reporting the offending item number."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(_POLICY_LABELS.values()))},
            ui_options={"display_name": "How to handle missing sequence entries"},
        )
        self.add_parameter(self._policy_param)

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

        self._paths_param = Parameter(
            name="paths",
            tooltip=(
                "File paths in the active range, in ascending item-number order. "
                "Wires directly into a for-loop, Load Image, Video Player, etc."
            ),
            type="list",
            output_type="list[str]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._paths_param)

        with ParameterGroup(name="Advanced Sequence Control", collapsed=True) as advanced_group:
            self._item_range = ItemRangeGroup()

            self._sequence_param = Parameter(
                name="sequence",
                tooltip=(
                    "The full structured Sequence object (Pydantic model). "
                    'Use this for technical nodes that consume `type="Sequence"` directly.'
                ),
                type="Sequence",
                output_type="Sequence",
                default_value=None,
                allowed_modes={ParameterMode.OUTPUT},
            )
        self.add_node_element(advanced_group)

        self._create_status_parameters(
            result_details_tooltip="Details about the sequence scan",
            result_details_placeholder="Scan results will appear here.",
        )

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Skip SuccessFailureNode's status reset.

        The parent class clears `was_successful` / `result_details` here, which
        causes downstream re-resolutions of this node to leave the status reading
        False / placeholder if the engine doesn't re-run `process()` immediately
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

        policy = _LABEL_TO_POLICY[self.get_parameter_value(self._policy_param.name)]

        template = self.get_parameter_value(self._template_param.name).strip()
        if not template:
            self._emit_failure("No template provided.", policy=policy)
            return

        try:
            parsed = ParsedMacro(template)
        except MacroSyntaxError as e:
            self._emit_failure(f"Invalid template: {e}", policy=policy)
            return

        resolved_path = resolve_project_macro(parsed)
        if resolved_path is None:
            self._emit_failure(f"Could not resolve project variables for template: {template}", policy=policy)
            return

        split = split_resolved_path(resolved_path)
        if split.filename_pattern is None:
            self._emit_failure(
                f"Resolved template '{resolved_path}' has no filename component.",
                policy=policy,
                directory=split.directory,
            )
            return

        bounds_or_error = await self._item_range.resolve_bounds(self, split.directory, split.filename_pattern)
        if isinstance(bounds_or_error, str):
            self._emit_failure(
                bounds_or_error,
                policy=policy,
                directory=split.directory,
                pattern=split.filename_pattern,
            )
            return

        scan_result = await GriptapeNodes.ahandle_request(
            ScanSequencesRequest(
                directory=split.directory,
                pattern=split.filename_pattern,
                policy=policy,
                start_number=bounds_or_error.start_number,
                end_number=bounds_or_error.end_number,
            )
        )
        if isinstance(scan_result, ScanSequencesResultFailure):
            self._emit_failure(
                str(scan_result.result_details),
                policy=policy,
                directory=split.directory,
                pattern=split.filename_pattern,
            )
            return
        if not isinstance(scan_result, ScanSequencesResultSuccess):
            self._emit_failure(
                f"Unexpected scan result type: {type(scan_result).__name__}",
                policy=policy,
                directory=split.directory,
                pattern=split.filename_pattern,
            )
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
            empty_sequence = self._make_empty_sequence(
                policy=policy,
                directory=split.directory,
                pattern=split.filename_pattern,
            )
            if fail_on_empty:
                self._emit_outputs(empty_sequence, was_successful=False, details=details)
            else:
                self._emit_outputs(
                    empty_sequence,
                    was_successful=True,
                    details=f"{details} (fail_on_empty_result=False)",
                )
            return

        # Non-SPLIT policies always return exactly one Sequence; the engine
        # guarantees this in `_apply_single`.
        self._emit_success(scan_result.sequences[0])

    def _emit_success(self, sequence: Sequence) -> None:
        """Populate outputs and mark the node successful."""
        paths = [entry.path for entry in sequence.entries]
        details = (
            f"Scanned sequence: items {sequence.first}..{sequence.last} "
            f"({len(paths)} paths, policy={sequence.policy.value})."
        )
        if sequence.dropped_negative_number_count:
            details += f" Dropped {sequence.dropped_negative_number_count} negative number(s)."
        self._emit_outputs(sequence, was_successful=True, details=details)

    def _emit_failure(
        self,
        details: str,
        *,
        policy: MissingItemPolicy,
        directory: str = "",
        pattern: str = "",
    ) -> None:
        """Emit an empty Sequence and mark the node failed.

        Downstream consumers (Inspect Sequence, etc.) treat `None` as "no sequence
        connected" and short-circuit; emitting an empty Sequence lets them render
        their normal "0 entries" view instead, so the artist can still see the
        diagnostic flow downstream.
        """
        empty_sequence = self._make_empty_sequence(policy=policy, directory=directory, pattern=pattern)
        self._emit_outputs(empty_sequence, was_successful=False, details=details)

    def _emit_outputs(self, sequence: Sequence, *, was_successful: bool, details: str) -> None:
        """Single output-write path for both success and failure branches."""
        self.parameter_output_values[self._paths_param.name] = [entry.path for entry in sequence.entries]
        self.parameter_output_values[self._sequence_param.name] = sequence
        self._set_status_results(was_successful=was_successful, result_details=details)

    @staticmethod
    def _make_empty_sequence(*, policy: MissingItemPolicy, directory: str = "", pattern: str = "") -> Sequence:
        """Build an empty Sequence — `first > last` so the integer range is empty.

        Used as the output payload whenever the node produces no items (failures
        and `fail_on_empty_result=False` both qualify). Downstream nodes can
        introspect padding/pattern/directory to render their own context, but
        `entries`, `present_numbers`, and `missing_numbers` are all empty.
        """
        return Sequence(
            entries=[],
            first=0,
            last=-1,
            discovered_first=0,
            discovered_last=-1,
            padding=0,
            pattern=pattern,
            directory=directory,
            policy=policy,
        )

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
