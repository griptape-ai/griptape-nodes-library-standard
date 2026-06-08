"""Reusable component for nodes that scan sequences.

This is the shared "Advanced Sequence Control" component — any node that
consumes a sequence path or pattern can drop `AdvancedSequenceControls` into
its own `__init__`, add the resulting `ParameterGroup` to the node, and
forward `after_value_set` to it. The component owns the entire collapsed
"Advanced Sequence Control" group:

- Item-range start/end dropdowns and value sub-parameters (visibility-toggled
  in `handle_after_value_set`).
- A `When there's no sequence marker (e.g., ###)` dropdown that maps onto
  the engine's `NoTokenBehavior` enum (single-file, explore, reject).

`resolve_bounds` walks the dropdowns, optionally probes the path for the
first/last item on disk, and produces the final `start_number`/`end_number`
to feed into `ScanSequencesRequest`. `behavior_for_request` returns the
selected `NoTokenBehavior` value to pass on the same request.

Discovery probes (used when `start_mode` or `end_mode` is `RELATIVE`) go
through the engine's `ScanSequencesRequest` event so disk I/O and fileseq
parsing happen on a worker thread.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

from griptape_nodes.common.sequences import MissingItemPolicy, NoTokenBehavior
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultFailure,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

if TYPE_CHECKING:
    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")


class StartMode(StrEnum):
    """How the active range's start is determined."""

    FIRST_IN_SEQUENCE = "First item in sequence"
    SPECIFY = "Specify start"
    RELATIVE = "Relative to sequence start"


class EndMode(StrEnum):
    """How the active range's end is determined."""

    LAST_IN_SEQUENCE = "Last item in sequence"
    SPECIFY = "Specify end"
    RELATIVE = "Relative to sequence end"


# Friendly labels for the engine's `NoTokenBehavior` enum. The dictionary
# order is the order rendered in the dropdown.
_NO_TOKEN_LABELS: dict[NoTokenBehavior, str] = {
    NoTokenBehavior.SINGLE_FILE: "Treat as a single file",
    NoTokenBehavior.EXPLORE_SEQUENCE: "Treat as part of a sequence",
    NoTokenBehavior.REJECT: "Fail unless a token is present",
}
_LABEL_TO_NO_TOKEN: dict[str, NoTokenBehavior] = {label: behavior for behavior, label in _NO_TOKEN_LABELS.items()}


class ResolvedBounds(NamedTuple):
    """Item bounds ready to feed into `ScanSequencesRequest`."""

    start_number: int | None
    end_number: int | None


class AdvancedSequenceControls:
    """Reusable advanced-controls group for sequence-scanning nodes.

    Owns the parameters under the collapsed "Advanced Sequence Control" group
    and the `ParameterGroup` they live in. Instantiate inside a node's
    `__init__`, add `self.group` via `node.add_node_element(...)`, and forward
    `after_value_set` calls to `handle_after_value_set`. At run time, call
    `resolve_bounds(node, path)` to get the (start_number, end_number) pair
    and `behavior_for_request(node)` to get the `NoTokenBehavior` to send.
    """

    def __init__(self) -> None:
        self.group = ParameterGroup(name="Advanced Sequence Control", collapsed=True)
        with self.group:
            self.no_token_behavior = ParameterString(
                name="no_token_behavior",
                default_value=_NO_TOKEN_LABELS[NoTokenBehavior.SINGLE_FILE],
                tooltip=(
                    "What to do when the path has no sequence token (`####`, `%04d`, etc.) — "
                    "for example, when the artist types `render.0002.png`.\n"
                    "- *Treat as a single file* (default) — the path names exactly one file. "
                    "Other items in the same directory are ignored. Returns a one-item sequence "
                    "if the file exists.\n"
                    "- *Treat as part of a sequence* — read the digits in the filename as a "
                    "sequence number and walk the surrounding sibling files. Useful when a "
                    "downstream tool gave you one filename but you want the whole take.\n"
                    "- *Fail unless a token is present* — strict mode. The node fails with an "
                    "error telling the artist to add a token. Use this in pipelines that should "
                    "never silently widen the artist's intent."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=list(_NO_TOKEN_LABELS.values()))},
                ui_options={"display_name": "When there's no sequence marker (e.g., ###)"},
            )
            self.no_token_behavior.set_badge(
                variant="help",
                title="What happens when there's no sequence marker",
                message=(
                    "You typed `render.0002.png`. The directory holds `render.0001.png` "
                    "through `render.0005.png`."
                    "\n\n"
                    "**Treat as a single file** *(default)*\n"
                    "Returns just the one file you named.\n"
                    "→ `[render.0002.png]` (1 item)"
                    "\n\n"
                    "**Treat as part of a sequence**\n"
                    "Reads the `0002` as a frame number and walks the surrounding siblings.\n"
                    "→ `[render.0001.png, render.0002.png, …, render.0005.png]` (5 items)"
                    "\n\n"
                    "**Fail unless a token is present**\n"
                    "Refuses the path; routes through the Failure edge.\n"
                    '→ Status: *"add a token like `####` to scan the surrounding sequence"*.'
                ),
            )
            self.start_mode = ParameterString(
                name="start_mode",
                default_value=StartMode.FIRST_IN_SEQUENCE.value,
                tooltip=(
                    "Where the active range begins.\n"
                    "- `First item in sequence` — start at the first item that exists on disk.\n"
                    "- `Specify start` — start at a specific item number you provide.\n"
                    "- `Relative to sequence start` — start a few items in from the first one found "
                    "(e.g. skip the first 5)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[m.value for m in StartMode])},
                ui_options={"display_name": "Start at"},
            )
            self.start_mode.set_badge(
                variant="help",
                title="Where the active range begins",
                message=(
                    "The directory holds items 1 through 10."
                    "\n\n"
                    "**First item in sequence** *(default)*\n"
                    "Begin at whatever's first on disk.\n"
                    "→ active range starts at item `1`"
                    "\n\n"
                    "**Specify start**\n"
                    "Begin at the item number you type into *Start at item*.\n"
                    "→ with *Start at item* = `3`, active range starts at item `3`"
                    "\n\n"
                    "**Relative to sequence start**\n"
                    "Skip the first N items, where N is *Skip items from start*.\n"
                    "→ with *Skip items from start* = `2`, active range starts at item `3` "
                    "(first item `1` + offset `2`)"
                ),
            )
            self.start_number = ParameterInt(
                name="start_number",
                default_value=0,
                tooltip=(
                    'The item number to start at — for example, `10` means "ignore everything '
                    'before item 10." Active when **Start at** is set to *Specify start*.'
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True, "display_name": "Start at item"},
            )
            self.start_offset = ParameterInt(
                name="start_offset",
                default_value=0,
                tooltip=(
                    "How many items to skip past the first one found — `0` keeps everything, "
                    "`5` skips the first 5. Active when **Start at** is set to "
                    "*Relative to sequence start*."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True, "display_name": "Skip items from start"},
            )
            self.end_mode = ParameterString(
                name="end_mode",
                default_value=EndMode.LAST_IN_SEQUENCE.value,
                tooltip=(
                    "Where the active range ends.\n"
                    "- `Last item in sequence` — end at the last item that exists on disk.\n"
                    "- `Specify end` — end at a specific item number you provide.\n"
                    "- `Relative to sequence end` — end a few items short of the last one found "
                    "(e.g. drop the final 2)."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[m.value for m in EndMode])},
                ui_options={"display_name": "End at"},
            )
            self.end_mode.set_badge(
                variant="help",
                title="Where the active range ends",
                message=(
                    "The directory holds items 1 through 10."
                    "\n\n"
                    "**Last item in sequence** *(default)*\n"
                    "End at whatever's last on disk.\n"
                    "→ active range ends at item `10`"
                    "\n\n"
                    "**Specify end**\n"
                    "End at the item number you type into *End at item*.\n"
                    "→ with *End at item* = `7`, active range ends at item `7`"
                    "\n\n"
                    "**Relative to sequence end**\n"
                    "Drop the last N items, where N is *Skip items from end*.\n"
                    "→ with *Skip items from end* = `2`, active range ends at item `8` "
                    "(last item `10` − offset `2`)"
                ),
            )
            self.end_number = ParameterInt(
                name="end_number",
                default_value=0,
                tooltip=(
                    'The item number to end at — for example, `100` means "ignore everything '
                    'after item 100." Active when **End at** is set to *Specify end*.'
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True, "display_name": "End at item"},
            )
            self.end_offset = ParameterInt(
                name="end_offset",
                default_value=0,
                tooltip=(
                    "How many items to drop off the end — `0` keeps everything, `2` drops the "
                    "final 2. Active when **End at** is set to *Relative to sequence end*."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True, "display_name": "Skip items from end"},
            )

    def handle_after_value_set(self, node: BaseNode, parameter: Parameter, value: Any) -> bool:
        """Toggle the visibility of value sub-parameters when a mode dropdown changes.

        Returns True if the parameter was one of ours and we handled it; False
        otherwise. The hosting node uses the return value to decide whether to
        keep dispatching to its own logic.
        """
        if parameter.name == self.start_mode.name:
            if value == StartMode.SPECIFY.value:
                node.show_parameter_by_name(self.start_number.name)
                node.hide_parameter_by_name(self.start_offset.name)
            elif value == StartMode.RELATIVE.value:
                node.hide_parameter_by_name(self.start_number.name)
                node.show_parameter_by_name(self.start_offset.name)
            else:
                node.hide_parameter_by_name(self.start_number.name)
                node.hide_parameter_by_name(self.start_offset.name)
            return True
        if parameter.name == self.end_mode.name:
            if value == EndMode.SPECIFY.value:
                node.show_parameter_by_name(self.end_number.name)
                node.hide_parameter_by_name(self.end_offset.name)
            elif value == EndMode.RELATIVE.value:
                node.hide_parameter_by_name(self.end_number.name)
                node.show_parameter_by_name(self.end_offset.name)
            else:
                node.hide_parameter_by_name(self.end_number.name)
                node.hide_parameter_by_name(self.end_offset.name)
            return True
        return False

    def behavior_for_request(self, node: BaseNode) -> NoTokenBehavior:
        """Translate the user-facing dropdown label into the engine's enum value.

        Returns the `NoTokenBehavior` the host node should pass to
        `ScanSequencesRequest.no_token_behavior`. Falls back to
        `SINGLE_FILE` if the parameter is set to a value not in the
        dropdown (shouldn't happen via the UI, but keep the host robust).
        """
        label = node.get_parameter_value(self.no_token_behavior.name)
        return _LABEL_TO_NO_TOKEN.get(label, NoTokenBehavior.SINGLE_FILE)

    async def resolve_bounds(self, node: BaseNode, path: str) -> ResolvedBounds | str:
        """Compute (start_number, end_number) for the scan based on the dropdown modes.

        Returns either a `ResolvedBounds` ready to feed into the request, or a
        human-readable failure message string. Dispatches a discovery
        `ScanSequencesRequest` only when at least one side is set to `RELATIVE`.
        The discovery probe sends the same `path` value the host node will use,
        so macro resolution stays the engine's job. The probe also sends the
        same `no_token_behavior` the host has selected, so a literal-file
        path doesn't get explored just because we're sniffing for offsets.
        """
        start_mode = StartMode(node.get_parameter_value(self.start_mode.name))
        end_mode = EndMode(node.get_parameter_value(self.end_mode.name))
        start_number = int(node.get_parameter_value(self.start_number.name))
        start_offset = int(node.get_parameter_value(self.start_offset.name))
        end_number = int(node.get_parameter_value(self.end_number.name))
        end_offset = int(node.get_parameter_value(self.end_offset.name))

        if start_mode is StartMode.SPECIFY and start_number < 0:
            return f"*Start at item* must be 0 or greater (you entered {start_number})."
        if start_mode is StartMode.RELATIVE and start_offset < 0:
            return f"*Skip items from start* must be 0 or greater (you entered {start_offset})."
        if end_mode is EndMode.SPECIFY and end_number < 0:
            return f"*End at item* must be 0 or greater (you entered {end_number})."
        if end_mode is EndMode.RELATIVE and end_offset < 0:
            return f"*Skip items from end* must be 0 or greater (you entered {end_offset})."

        needs_discovery = start_mode is StartMode.RELATIVE or end_mode is EndMode.RELATIVE
        discovered_first: int | None = None
        discovered_last: int | None = None
        if needs_discovery:
            probe = await GriptapeNodes.ahandle_request(
                ScanSequencesRequest(
                    path=path,
                    policy=MissingItemPolicy.SPLIT,
                    no_token_behavior=self.behavior_for_request(node),
                )
            )
            if isinstance(probe, ScanSequencesResultFailure):
                return f"Couldn't read the sequence to compute the offsets: {probe.result_details}"
            if not isinstance(probe, ScanSequencesResultSuccess) or not probe.has_entries:
                return f"Couldn't find any items at `{path}` to anchor the offsets."
            discovered_first = min(seq.discovered_first for seq in probe.sequences)
            discovered_last = max(seq.discovered_last for seq in probe.sequences)

        resolved_start: int | None
        if start_mode is StartMode.FIRST_IN_SEQUENCE:
            resolved_start = None
        elif start_mode is StartMode.SPECIFY:
            resolved_start = start_number
        elif discovered_first is None:
            return f"Couldn't read the first item at `{path}`."
        else:
            resolved_start = discovered_first + start_offset

        resolved_end: int | None
        if end_mode is EndMode.LAST_IN_SEQUENCE:
            resolved_end = None
        elif end_mode is EndMode.SPECIFY:
            resolved_end = end_number
        elif discovered_last is None:
            return f"Couldn't read the last item at `{path}`."
        else:
            resolved_end = discovered_last - end_offset

        return ResolvedBounds(start_number=resolved_start, end_number=resolved_end)
