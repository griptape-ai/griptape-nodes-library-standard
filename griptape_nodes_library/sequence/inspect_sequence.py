"""InspectSequenceNode: surface a Sequence's structure for expert/technical workflows.

Pure pass-through inspection node. The `sequence` parameter is both INPUT and
OUTPUT — the same Sequence flows through unchanged — plus a handful of
structural outputs that downstream workflows commonly need:

- `summary: str` (multiline) — human-readable description of the Sequence,
  suitable for a Display node or a Note. Surfaces range, gap policy, token
  width, pattern, directory, gap count, dropped-negatives count, and a peek
  at the first/last entry paths.
- `existing_entries: list[SequenceEntry]` — only the entries whose number is
  actually present on disk (filters out the gap-fill duplicates emitted by
  the *Fill gaps with nearest* policy).
- `gaps: list[int]` — sorted item numbers absent from the active range.
- `is_complete: bool` — True if there are zero gaps in the active range.
- `gap_count: int` — how many items are missing from the active range.
- `entry_count: int` — how many items the sequence carries.

The Sequence pass-through means an artist can drop this node into the middle
of a chain without breaking it: input → InspectSequence → downstream Sequence
consumer.
"""

from __future__ import annotations

from typing import Any

from griptape_nodes.common.sequences import MissingItemPolicy, Sequence
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, SuccessFailureNode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

# Friendly labels for the engine's `MissingItemPolicy`. Mirrors the dropdown
# labels in `scan_sequence.py` so the summary's "Gap policy" line reads the
# same way the artist set it.
_POLICY_LABELS: dict[MissingItemPolicy, str] = {
    MissingItemPolicy.SKIP: "Skip over gaps",
    MissingItemPolicy.FILL_NEAREST: "Fill gaps with nearest",
    MissingItemPolicy.ABORT: "Abort on sequence gaps",
    MissingItemPolicy.SPLIT: "Split into contiguous runs",
}


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
                "The sequence to inspect. Whatever you wire in flows through to this node's own "
                "`Sequence` output unchanged, so this node can sit inline between any two nodes "
                "that pass sequences around."
            ),
            type="Sequence",
            input_types=["Sequence"],
            output_type="Sequence",
            default_value=None,
            allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY, ParameterMode.OUTPUT},
            ui_options={"display_name": "Sequence"},
        )
        self.add_parameter(self._sequence_param)

        self._summary_param = ParameterString(
            name="summary",
            default_value="",
            tooltip=(
                "A multi-line description of the sequence: range, gap count, padding, pattern, "
                "directory, and a peek at the first and last items. Wire into a Note or Display node."
            ),
            allowed_modes={ParameterMode.OUTPUT},
            multiline=True,
            ui_options={"display_name": "Summary"},
        )
        self.add_parameter(self._summary_param)

        self._existing_entries_param = Parameter(
            name="existing_entries",
            tooltip=(
                "Just the items that actually exist on disk — gap-fill items (where the policy "
                "reused a neighbor's path) are filtered out."
            ),
            type="list",
            output_type="list[SequenceEntry]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Items present on disk"},
        )
        self.add_parameter(self._existing_entries_param)

        self._gaps_param = Parameter(
            name="gaps",
            tooltip="The item numbers missing from the active range, sorted ascending.",
            type="list",
            output_type="list[int]",
            default_value=[],
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Missing item numbers"},
        )
        self.add_parameter(self._gaps_param)

        self._is_complete_param = Parameter(
            name="is_complete",
            tooltip="True when every item in the active range exists on disk.",
            type="bool",
            output_type="bool",
            default_value=False,
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "No gaps"},
        )
        self.add_parameter(self._is_complete_param)

        self._gap_count_param = ParameterInt(
            name="gap_count",
            default_value=0,
            tooltip="How many items are missing from the active range.",
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Missing item count"},
        )
        self.add_parameter(self._gap_count_param)

        self._entry_count_param = ParameterInt(
            name="entry_count",
            default_value=0,
            tooltip="How many items the sequence carries.",
            allowed_modes={ParameterMode.OUTPUT},
            ui_options={"display_name": "Item count"},
        )
        self.add_parameter(self._entry_count_param)

        self._create_status_parameters(
            result_details_tooltip="Details about the inspection",
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

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Recompute outputs whenever the input Sequence is reassigned in place."""
        if parameter.name == self._sequence_param.name:
            self._recompute()
        return super().after_value_set(parameter, value)

    def after_incoming_connection(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Recompute outputs as soon as a Sequence is wired in (no node run required)."""
        if target_parameter.name == self._sequence_param.name:
            self._recompute()
        return super().after_incoming_connection(source_node, source_parameter, target_parameter)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        """Reset outputs when the upstream Sequence is disconnected."""
        if target_parameter.name == self._sequence_param.name:
            self._recompute()
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    def process(self) -> None:
        self._clear_execution_status()
        self._recompute()

    def _recompute(self) -> None:
        """Compute every output from the current `sequence` input value.

        Called from `process()` (node-run path) and from the connection /
        value-change hooks (reactive path). Both paths must produce identical
        outputs, so they share this single helper. The status flag is set the
        same way in both paths — the node looks "live" in the editor without
        needing a manual run.
        """
        raw = self.get_parameter_value(self._sequence_param.name)
        if raw is None:
            self._publish_outputs(
                sequence=None,
                summary="",
                existing_entries=[],
                gaps=[],
                is_complete=False,
                gap_count=0,
                entry_count=0,
            )
            self._set_status_results(was_successful=False, result_details="No sequence connected.")
            return

        # Accept either a Sequence instance (typical) or a dict (e.g. after a
        # save/load round-trip if the engine handed it back as JSON).
        sequence = Sequence.model_validate(raw)

        gaps = sorted(sequence.missing_numbers)
        gap_count = len(gaps)
        entry_count = len(sequence.entries)
        existing_entries = [entry for entry in sequence.entries if entry.number in sequence.present_numbers]

        self._publish_outputs(
            sequence=sequence,
            summary=_build_summary(sequence),
            existing_entries=existing_entries,
            gaps=gaps,
            is_complete=gap_count == 0,
            gap_count=gap_count,
            entry_count=entry_count,
        )

        policy_label = _POLICY_LABELS[sequence.policy] if sequence.policy in _POLICY_LABELS else sequence.policy.value
        details = (
            f"Inspected sequence: items {sequence.first}–{sequence.last} "
            f"({entry_count} items, {gap_count} gap(s), {policy_label.lower()})."
        )
        self._set_status_results(was_successful=True, result_details=details)

    def _publish_outputs(
        self,
        *,
        sequence: Sequence | None,
        summary: str,
        existing_entries: list,
        gaps: list[int],
        is_complete: bool,
        gap_count: int,
        entry_count: int,
    ) -> None:
        """Write every output value AND publish each one to the editor.

        `parameter_output_values[...]` alone is sufficient for the engine's
        run-time value flow; `publish_update_to_parameter(...)` is what makes
        the editor see the change without a node run, which is the whole point
        of the reactive path.
        """
        outputs: dict[str, Any] = {
            self._sequence_param.name: sequence,
            self._summary_param.name: summary,
            self._existing_entries_param.name: existing_entries,
            self._gaps_param.name: gaps,
            self._is_complete_param.name: is_complete,
            self._gap_count_param.name: gap_count,
            self._entry_count_param.name: entry_count,
        }
        for name, value in outputs.items():
            self.parameter_output_values[name] = value
            self.publish_update_to_parameter(name, value)


def _build_summary(sequence: Sequence) -> str:
    """Render a Sequence as a multi-line summary suitable for the `summary` output."""
    policy_label = _POLICY_LABELS[sequence.policy] if sequence.policy in _POLICY_LABELS else sequence.policy.value
    if sequence.padding == 0:
        token_width = "no padding"
    elif sequence.padding == 1:
        token_width = "1 digit"
    else:
        token_width = f"{sequence.padding} digits"

    lines: list[str] = []
    lines.append(f"Range:             {sequence.first}–{sequence.last}")
    lines.append(f"Found on disk:     {sequence.discovered_first}–{sequence.discovered_last}")
    lines.append(f"Gap policy:        {policy_label}")
    lines.append(f"Token width:       {token_width}")
    lines.append(f"Pattern:           {sequence.pattern}")
    lines.append(f"Directory:         {sequence.directory}")
    lines.append(f"Items:             {len(sequence.entries)}")
    lines.append(f"Present on disk:   {len(sequence.present_numbers)}")

    missing = sorted(sequence.missing_numbers)
    if missing:
        if len(missing) <= 10:
            missing_repr = ", ".join(str(n) for n in missing)
        else:
            head = ", ".join(str(n) for n in missing[:5])
            tail = ", ".join(str(n) for n in missing[-5:])
            missing_repr = f"{head}, ..., {tail}"
        lines.append(f"Missing ({len(missing)}):    {missing_repr}")
    else:
        lines.append("Missing:           (none)")

    if sequence.dropped_negative_number_count:
        lines.append(f"Negatives dropped: {sequence.dropped_negative_number_count}")

    if sequence.entries:
        first_entry = sequence.entries[0]
        last_entry = sequence.entries[-1]
        lines.append("")
        lines.append(f"First item:        #{first_entry.number} ({first_entry.padded_number}) → {first_entry.path}")
        if last_entry is not first_entry:
            lines.append(f"Last item:         #{last_entry.number} ({last_entry.padded_number}) → {last_entry.path}")

    return "\n".join(lines)
