"""InspectSequenceNode: surface a Sequence's structure for expert/technical workflows.

Pure pass-through inspection node. The `sequence` parameter is both INPUT and
OUTPUT — the same Sequence flows through unchanged — plus a handful of
structural primitives that downstream expert workflows commonly need:

- `summary: str` (multiline) — human-readable description of the Sequence,
  suitable for a Display node or a Note. Surfaces range, policy, padding,
  pattern, directory, gap count, dropped-negatives count, and a peek at the
  first/last entry paths.
- `existing_entries: list[SequenceEntry]` — only the entries whose number is
  actually present on disk (filters out FILL_NEAREST's gap-fill duplicates).
- `gaps: list[int]` — sorted item numbers absent from the active range.
- `is_complete: bool` — True if there are zero gaps in the active range.
- `gap_count: int` — `len(seq.missing_numbers)`.
- `entry_count: int` — `len(seq.entries)`.

The Sequence pass-through means an artist can drop this node into the middle
of a chain without breaking it: input → InspectSequence → downstream Sequence
consumer.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.common.sequences import Sequence
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString


class InspectSequenceNode(SuccessFailureNode):
    """Inspect a Sequence and surface its structural primitives.

    On total failure (no input connected, or unexpected input shape), emits
    empty/default outputs and routes through the Failure control-flow edge.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self._sequence_param = Parameter(
            name="sequence",
            tooltip=(
                "The Sequence to inspect. Pass-through: the same Sequence flows through unchanged "
                "as the output, so this node can sit inline between any two Sequence-aware nodes."
            ),
            type="Sequence",
            input_types=["Sequence"],
            output_type="Sequence",
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
        )
        self.add_parameter(self._sequence_param)

        self._summary_param = ParameterString(
            name="summary",
            default_value="",
            tooltip=(
                "Human-readable description of the Sequence: range, policy, padding, "
                "pattern, directory, gap count, and a peek at the first/last entries."
            ),
            allowed_modes={ParameterMode.OUTPUT},
            multiline=True,
        )
        self.add_parameter(self._summary_param)

        self._existing_entries_param = Parameter(
            name="existing_entries",
            tooltip=(
                "List of SequenceEntry objects whose number is actually present on disk. "
                "Under FILL_NEAREST, this filters out gap entries (which carry the nearest "
                "neighbor's path); under SKIP / ABORT it equals `sequence.entries`."
            ),
            type="list",
            output_type="list[SequenceEntry]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._existing_entries_param)

        self._gaps_param = Parameter(
            name="gaps",
            tooltip="Sorted list of item numbers missing from the active range.",
            type="list",
            output_type="list[int]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._gaps_param)

        self._is_complete_param = Parameter(
            name="is_complete",
            tooltip="True when the Sequence has no gaps in its active range.",
            type="bool",
            output_type="bool",
            default_value=False,
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._is_complete_param)

        self._gap_count_param = ParameterInt(
            name="gap_count",
            default_value=0,
            tooltip="Number of items missing from the active range (`len(seq.missing_numbers)`).",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._gap_count_param)

        self._entry_count_param = ParameterInt(
            name="entry_count",
            default_value=0,
            tooltip="Number of entries in the Sequence (`len(seq.entries)`).",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self._entry_count_param)

        self._create_status_parameters(
            result_details_tooltip="Details about the Sequence inspection",
            result_details_placeholder="Inspection results will appear here.",
        )

    def validate_before_workflow_run(self) -> list[Exception] | None:
        """Skip SuccessFailureNode's status reset.

        The parent class clears `was_successful` / `result_details` here, which
        causes downstream re-resolutions of this node to leave the status reading
        False / placeholder if the engine doesn't re-run `process()` immediately
        afterward. We skip the reset and let `process()` own the status entirely.
        """
        return BaseNode.validate_before_workflow_run(self)

    def validate_before_node_run(self) -> list[Exception] | None:
        """Skip SuccessFailureNode's status reset (see `validate_before_workflow_run`)."""
        return BaseNode.validate_before_node_run(self)

    def process(self) -> None:
        self._clear_execution_status()

        raw = self.get_parameter_value(self._sequence_param.name)
        if raw is None:
            self._emit_failure("No sequence connected.")
            return

        # Accept either a Sequence instance (typical) or a dict (e.g. after a
        # save/load round-trip if the engine handed it back as JSON).
        sequence = Sequence.model_validate(raw)

        gaps = sorted(sequence.missing_numbers)
        gap_count = len(gaps)
        entry_count = len(sequence.entries)
        is_complete = gap_count == 0

        existing_entries = [entry for entry in sequence.entries if entry.number in sequence.present_numbers]

        self.parameter_output_values[self._sequence_param.name] = sequence
        self.parameter_output_values[self._summary_param.name] = _build_summary(sequence)
        self.parameter_output_values[self._existing_entries_param.name] = existing_entries
        self.parameter_output_values[self._gaps_param.name] = gaps
        self.parameter_output_values[self._is_complete_param.name] = is_complete
        self.parameter_output_values[self._gap_count_param.name] = gap_count
        self.parameter_output_values[self._entry_count_param.name] = entry_count

        details = (
            f"Inspected sequence: items {sequence.first}..{sequence.last} "
            f"({entry_count} entries, {gap_count} gap(s), policy={sequence.policy.value})."
        )
        self._set_status_results(was_successful=True, result_details=details)

    def _emit_failure(self, details: str) -> None:
        """Emit empty/default outputs and mark the node failed."""
        self.parameter_output_values[self._sequence_param.name] = None
        self.parameter_output_values[self._summary_param.name] = ""
        self.parameter_output_values[self._existing_entries_param.name] = []
        self.parameter_output_values[self._gaps_param.name] = []
        self.parameter_output_values[self._is_complete_param.name] = False
        self.parameter_output_values[self._gap_count_param.name] = 0
        self.parameter_output_values[self._entry_count_param.name] = 0
        self._set_status_results(was_successful=False, result_details=details)


def _build_summary(sequence: Sequence) -> str:
    """Render a Sequence as a multi-line summary suitable for the `summary` output."""
    lines: list[str] = []
    lines.append(f"Range:           {sequence.first}..{sequence.last}")
    lines.append(f"Discovered:      {sequence.discovered_first}..{sequence.discovered_last}")
    lines.append(f"Policy:          {sequence.policy.value}")
    lines.append(f"Padding:         {sequence.padding}")
    lines.append(f"Pattern:         {sequence.pattern}")
    lines.append(f"Directory:       {sequence.directory}")
    lines.append(f"Entries:         {len(sequence.entries)}")
    lines.append(f"Present:         {len(sequence.present_numbers)}")

    missing = sorted(sequence.missing_numbers)
    if missing:
        if len(missing) <= 10:
            missing_repr = ", ".join(str(n) for n in missing)
        else:
            head = ", ".join(str(n) for n in missing[:5])
            tail = ", ".join(str(n) for n in missing[-5:])
            missing_repr = f"{head}, ..., {tail}"
        lines.append(f"Missing ({len(missing)}):  {missing_repr}")
    else:
        lines.append("Missing:         (none)")

    if sequence.dropped_negative_number_count:
        lines.append(f"Dropped negatives: {sequence.dropped_negative_number_count}")

    if sequence.entries:
        first_entry = sequence.entries[0]
        last_entry = sequence.entries[-1]
        lines.append("")
        lines.append(f"First entry:     #{first_entry.number} ({first_entry.padded_number}) → {first_entry.path}")
        if last_entry is not first_entry:
            lines.append(f"Last entry:      #{last_entry.number} ({last_entry.padded_number}) → {last_entry.path}")

    return "\n".join(lines)
