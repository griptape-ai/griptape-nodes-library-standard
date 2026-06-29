"""ScanSequenceNode: scan a path or pattern into a list of file paths.

The 80% artist-facing entry point for sequence work. Inputs:
- a path or pattern (`{inputs}/render.####.png`, `/work/render.####.png`,
  or even a literal single file like `/work/photo.png`).
- a missing-item policy (default: SKIP).

Top-level outputs:
- `paths: list[str]` — file paths in item-number order, ready for a
  for-loop / Load Image / Video Player. Macro-form inputs round-trip with
  the macro head intact.
- `sequence: Sequence` — the structured Sequence object, ready for Inspect
  Sequence and any other `type="Sequence"` consumer.

`Item range` sub-setting controls live in the collapsed "Advanced Sequence
Control" group. Multi-sequence (SPLIT) work lives in ScanSplitSequenceNode.

Disk I/O, fileseq parsing, and macro resolution all run inside
`ScanSequencesRequest` on the engine's event bus, so they happen on a
worker thread without blocking the event loop.
"""

from __future__ import annotations

import logging
from typing import Any

from griptape_nodes.common.sequences import MissingItemPolicy, Sequence
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
from griptape_nodes.traits.options import Options

from griptape_nodes_library.sequence.advanced_sequence_component import AdvancedSequenceControls

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
    """Scan a path or pattern into a list of file paths.

    Top-level outputs: `paths` (convenience flat list) and `sequence` (the
    structured Sequence object). `Item range` controls live in the collapsed
    Advanced Sequence Control group.

    On total failure (bad path or pattern, missing directory, no matches,
    ABORT-policy gap), the node emits an empty Sequence and routes through
    the Failure control-flow edge.
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

        self._policy_param = ParameterString(
            name="missing_item_policy",
            default_value=_POLICY_LABELS[MissingItemPolicy.SKIP],
            tooltip=(
                "Choose how to handle missing items inside the range — for example, when "
                "items 1, 2, 4, 6, 7 exist on disk but 3 and 5 don't.\n"
                "- *Skip over gaps* keeps only what's on disk.\n"
                "- *Fill gaps with nearest* fills missing items by reusing the nearest existing "
                "item's file path.\n"
                "- *Abort on sequence gaps* fails the node at the first missing item.\n"
                "\n"
                "Need each contiguous run as its own sequence instead? Use the **Scan Split "
                "Sequence** node — it returns one sequence per run."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            traits={Options(choices=list(_POLICY_LABELS.values()))},
            ui_options={"display_name": "What to do at gaps"},
        )
        self._policy_param.set_badge(
            variant="help",
            title="What happens at gaps",
            message=(
                "You have items 1, 2, 4, 6, 7 on disk — items 3 and 5 are missing."
                "\n\n"
                "**Skip over gaps**\n"
                "Returns only what's on disk.\n"
                "→ `[1, 2, 4, 6, 7]`"
                "\n\n"
                "**Fill gaps with nearest**\n"
                "Fills the missing slots by reusing the nearest existing item's file path.\n"
                "→ `[1, 2, 2(=3), 4, 4(=5), 6, 7]` — slot 3 reuses item 2, slot 5 reuses item 4."
                "\n\n"
                "**Abort on sequence gaps**\n"
                "Fails the node at the first missing item; routes through the Failure edge.\n"
                '→ Status: *"sequence has a gap at item 3."*'
                "\n\n"
                "If you'd rather get one sequence per contiguous run (`[1, 2]`, `[4]`, `[6, 7]`) "
                "instead of a single sequence with a gap policy, use the **Scan Split Sequence** node."
            ),
        )
        self.add_parameter(self._policy_param)

        self._fail_on_empty_param = Parameter(
            name="fail_on_empty_result",
            input_types=["bool"],
            type="bool",
            output_type="bool",
            default_value=True,
            tooltip=(
                "When on, the node reports a failure if the scan finds nothing — usually a sign "
                "the path, range, or pattern is off. When off, the node succeeds with an empty "
                "sequence; useful for sweeps that may legitimately come up empty."
            ),
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
            ui_options={"display_name": "Fail when no items are found"},
        )
        self.add_parameter(self._fail_on_empty_param)

        self._paths_param = Parameter(
            name="paths",
            tooltip=(
                "The matched file paths in item-number order. If the input was a macro like "
                "`{inputs}/render.####.png`, these stay in macro form (`{inputs}/render.0001.png`, "
                "…). Wire into anything that takes a list of file paths."
            ),
            type="list",
            output_type="list[str]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "File paths"},
        )
        self.add_parameter(self._paths_param)

        self._sequence_param = Parameter(
            name="sequence",
            tooltip=(
                "The full sequence — items, range, and metadata in one object. Wire into any "
                "node that consumes a sequence directly, or into a list-aware node like "
                "**ForEach Group**. The `File paths` output above is a flat view of the same "
                "data for nodes that just want strings."
            ),
            type="Sequence",
            output_type="Sequence",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Sequence"},
        )
        self.add_parameter(self._sequence_param)

        self._advanced = AdvancedSequenceControls()
        self.add_node_element(self._advanced.group)

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
        self._advanced.handle_after_value_set(self, parameter, value)
        return super().after_value_set(parameter, value)

    async def aprocess(self) -> None:
        self._clear_execution_status()

        policy = _LABEL_TO_POLICY[self.get_parameter_value(self._policy_param.name)]

        path = self.get_parameter_value(self._path_param.name).strip()
        if not path:
            self._emit_failure("No path or pattern provided.", policy=policy)
            return

        bounds_or_error = await self._advanced.resolve_bounds(self, path)
        if isinstance(bounds_or_error, str):
            self._emit_failure(bounds_or_error, policy=policy)
            return

        scan_result = await GriptapeNodes.ahandle_request(
            ScanSequencesRequest(
                path=path,
                policy=policy,
                no_token_behavior=self._advanced.behavior_for_request(self),
                start_number=bounds_or_error.start_number,
                end_number=bounds_or_error.end_number,
            )
        )
        if isinstance(scan_result, ScanSequencesResultFailure):
            self._emit_failure(str(scan_result.result_details), policy=policy)
            return
        if not isinstance(scan_result, ScanSequencesResultSuccess):
            self._emit_failure(
                f"Unexpected scan result type: {type(scan_result).__name__}",
                policy=policy,
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
                path=path,
            )
            empty_sequence = self._make_empty_sequence(policy=policy)
            if fail_on_empty:
                self._emit_outputs(empty_sequence, was_successful=False, details=details)
            else:
                self._emit_outputs(
                    empty_sequence,
                    was_successful=True,
                    details=f"{details} (*Fail when no items are found* is off.)",
                )
            return

        # Non-SPLIT policies always return exactly one Sequence; the engine
        # guarantees this in `_apply_single`.
        self._emit_success(scan_result.sequences[0])

    def _emit_success(self, sequence: Sequence) -> None:
        """Populate outputs and mark the node successful."""
        paths = [entry.path for entry in sequence.entries]
        policy_label = _POLICY_LABELS[sequence.policy] if sequence.policy in _POLICY_LABELS else sequence.policy.value
        details = (
            f"Scanned sequence: items {sequence.first}–{sequence.last} ({len(paths)} items, {policy_label.lower()})."
        )
        if sequence.dropped_negative_number_count:
            details += f" Dropped {sequence.dropped_negative_number_count} negative number(s)."
        self._emit_outputs(sequence, was_successful=True, details=details)

    def _emit_failure(self, details: str, *, policy: MissingItemPolicy) -> None:
        """Emit an empty Sequence and mark the node failed.

        Downstream consumers (Inspect Sequence, etc.) treat `None` as "no sequence
        connected" and short-circuit; emitting an empty Sequence lets them render
        their normal "0 entries" view instead, so the artist can still see the
        diagnostic flow downstream.
        """
        empty_sequence = self._make_empty_sequence(policy=policy)
        self._emit_outputs(empty_sequence, was_successful=False, details=details)

    def _emit_outputs(self, sequence: Sequence, *, was_successful: bool, details: str) -> None:
        """Single output-write path for both success and failure branches."""
        self.parameter_output_values[self._paths_param.name] = [entry.path for entry in sequence.entries]
        self.parameter_output_values[self._sequence_param.name] = sequence
        self._set_status_results(was_successful=was_successful, result_details=details)

    @staticmethod
    def _make_empty_sequence(*, policy: MissingItemPolicy) -> Sequence:
        """Build an empty Sequence — `first > last` so the integer range is empty.

        Used as the output payload whenever the node produces no items (failures
        and `fail_on_empty_result=False` both qualify). Downstream nodes get
        `entries`, `present_numbers`, and `missing_numbers` all empty;
        `directory` and `pattern` are blank since neither is meaningful for an
        empty result.
        """
        return Sequence(
            entries=[],
            first=0,
            last=-1,
            discovered_first=0,
            discovered_last=-1,
            padding=0,
            pattern="",
            directory="",
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
