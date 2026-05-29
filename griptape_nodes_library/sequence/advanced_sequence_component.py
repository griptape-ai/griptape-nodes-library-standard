"""Reusable component for nodes that scan sequences and need item-range controls.

This is the shared "Advanced Sequence Control" component — any node that
consumes a sequence template can drop the `ItemRangeGroup` into its own
ParameterGroup (typically called "Advanced Sequence Control") and forward
`after_value_set` to it. The component owns:

- The `Item range` ParameterGroup (start/end mode dropdowns + value sub-parameters).
- Visibility-toggling logic in `handle_after_value_set` for the value sub-parameters.
- Project-macro / template-resolution helpers (`resolve_project_macro`, `split_resolved_path`).
- `resolve_bounds`, which walks the dropdowns, optionally probes the directory
  for relative offsets, and produces absolute `start_number`/`end_number` for
  `ScanSequencesRequest`.

The discovery probe (used when `start_mode` or `end_mode` is `RELATIVE`) is
dispatched via the engine's `ScanSequencesRequest` event so disk I/O and
fileseq parsing happen on a worker thread.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING, Any, NamedTuple

from griptape_nodes.common.sequences import MissingItemPolicy
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.retained_mode.events.os_events import (
    ScanSequencesRequest,
    ScanSequencesResultFailure,
    ScanSequencesResultSuccess,
)
from griptape_nodes.retained_mode.events.project_events import (
    GetPathForMacroRequest,
    GetPathForMacroResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.options import Options

if TYPE_CHECKING:
    from griptape_nodes.common.macro_parser import ParsedMacro
    from griptape_nodes.exe_types.node_types import BaseNode

logger = logging.getLogger("griptape_nodes")


class StartMode(StrEnum):
    """How the active subset's start is determined."""

    FIRST_IN_SEQUENCE = "First item in sequence"
    SPECIFY = "Specify start"
    RELATIVE = "Relative to sequence start"


class EndMode(StrEnum):
    """How the active subset's end is determined."""

    LAST_IN_SEQUENCE = "Last item in sequence"
    SPECIFY = "Specify end"
    RELATIVE = "Relative to sequence end"


class ResolvedBounds(NamedTuple):
    """Absolute item bounds ready to feed into `ScanSequencesRequest`."""

    start_number: int | None
    end_number: int | None


class ResolvedPath(NamedTuple):
    """An absolute path split into its directory and filename-pattern halves."""

    directory: str
    filename_pattern: str | None


class ItemRangeGroup:
    """Reusable `Item range` parameter group with start/end dropdowns and value sub-parameters.

    Owns six parameters and the `ParameterGroup` they live in. Instantiate inside
    a node's `__init__` (typically inside an outer `with ParameterGroup(...) as ...`
    context if the host wants the Item range nested under another group). The
    hosting node forwards `after_value_set` calls to `handle_after_value_set` and
    resolves bounds at run time via `resolve_bounds`.
    """

    def __init__(self) -> None:
        self.group = ParameterGroup(name="Item range")
        with self.group:
            self.start_mode = ParameterString(
                name="start_mode",
                default_value=StartMode.FIRST_IN_SEQUENCE.value,
                tooltip=(
                    "How to choose the lower bound of the active subset. The bound is global across all "
                    "sub-sequences (it's compared to the smallest item number on disk). "
                    "`Specify start` reads `start_number` as an absolute item number. "
                    "`Relative to sequence start` reads `start_offset` as a non-negative offset "
                    "added to the discovered first item across all sub-sequences."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[m.value for m in StartMode])},
            )
            self.start_number = ParameterInt(
                name="start_number",
                default_value=0,
                tooltip=(
                    "Absolute lower-bound item number. Items below this are dropped from output. "
                    "Used only when `start_mode` is `Specify start`."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
            self.start_offset = ParameterInt(
                name="start_offset",
                default_value=0,
                tooltip=(
                    "Non-negative offset added to the discovered first item (across all sub-sequences) "
                    "to compute the lower bound. `0` keeps the first item; `5` skips 5 items in. "
                    "Used only when `start_mode` is `Relative to sequence start`."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
            self.end_mode = ParameterString(
                name="end_mode",
                default_value=EndMode.LAST_IN_SEQUENCE.value,
                tooltip=(
                    "How to choose the upper bound of the active subset. The bound is global across all "
                    "sub-sequences (it's compared to the largest item number on disk). "
                    "`Specify end` reads `end_number` as an absolute item number. "
                    "`Relative to sequence end` reads `end_offset` as a non-negative offset "
                    "subtracted from the discovered last item across all sub-sequences."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                traits={Options(choices=[m.value for m in EndMode])},
            )
            self.end_number = ParameterInt(
                name="end_number",
                default_value=0,
                tooltip=(
                    "Absolute upper-bound item number. Items above this are dropped from output. "
                    "Used only when `end_mode` is `Specify end`."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True},
            )
            self.end_offset = ParameterInt(
                name="end_offset",
                default_value=0,
                tooltip=(
                    "Non-negative offset subtracted from the discovered last item (across all sub-sequences) "
                    "to compute the upper bound. `0` keeps the last item; `2` trims 2 items off the end. "
                    "Used only when `end_mode` is `Relative to sequence end`."
                ),
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                ui_options={"hide": True},
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

    async def resolve_bounds(self, node: BaseNode, directory: str, filename_pattern: str) -> ResolvedBounds | str:
        """Compute absolute (start_number, end_number) for `ScanSequencesRequest` based on the dropdown modes.

        Returns either a `ResolvedBounds` ready to feed into the request, or a
        human-readable failure message string. Dispatches a discovery
        `ScanSequencesRequest` only when at least one side is set to `RELATIVE`.
        """
        start_mode = StartMode(node.get_parameter_value(self.start_mode.name))
        end_mode = EndMode(node.get_parameter_value(self.end_mode.name))
        start_number = int(node.get_parameter_value(self.start_number.name))
        start_offset = int(node.get_parameter_value(self.start_offset.name))
        end_number = int(node.get_parameter_value(self.end_number.name))
        end_offset = int(node.get_parameter_value(self.end_offset.name))

        if start_mode is StartMode.SPECIFY and start_number < 0:
            return f"Start number must be >= 0; got {start_number}."
        if start_mode is StartMode.RELATIVE and start_offset < 0:
            return f"Start offset must be >= 0; got {start_offset}."
        if end_mode is EndMode.SPECIFY and end_number < 0:
            return f"End number must be >= 0; got {end_number}."
        if end_mode is EndMode.RELATIVE and end_offset < 0:
            return f"End offset must be >= 0; got {end_offset}."

        needs_discovery = start_mode is StartMode.RELATIVE or end_mode is EndMode.RELATIVE
        discovered_first: int | None = None
        discovered_last: int | None = None
        if needs_discovery:
            probe = await GriptapeNodes.ahandle_request(
                ScanSequencesRequest(
                    directory=directory,
                    pattern=filename_pattern,
                    policy=MissingItemPolicy.SPLIT,
                )
            )
            if isinstance(probe, ScanSequencesResultFailure):
                return f"Could not probe sequences for relative offsets: {probe.result_details}"
            if not isinstance(probe, ScanSequencesResultSuccess) or not probe.has_entries:
                return f"No sequences found in '{directory}' to anchor relative offsets."
            discovered_first = min(seq.discovered_first for seq in probe.sequences)
            discovered_last = max(seq.discovered_last for seq in probe.sequences)

        resolved_start: int | None
        if start_mode is StartMode.FIRST_IN_SEQUENCE:
            resolved_start = None
        elif start_mode is StartMode.SPECIFY:
            resolved_start = start_number
        elif discovered_first is None:
            return f"Could not determine the discovered first item for relative start in '{directory}'."
        else:
            resolved_start = discovered_first + start_offset

        resolved_end: int | None
        if end_mode is EndMode.LAST_IN_SEQUENCE:
            resolved_end = None
        elif end_mode is EndMode.SPECIFY:
            resolved_end = end_number
        elif discovered_last is None:
            return f"Could not determine the discovered last item for relative end in '{directory}'."
        else:
            resolved_end = discovered_last - end_offset

        return ResolvedBounds(start_number=resolved_start, end_number=resolved_end)


def resolve_project_macro(parsed: ParsedMacro) -> str | None:
    """Resolve a ParsedMacro through the project's variable bag.

    Returns the absolute filesystem path with any sequence tokens preserved
    as literal text. Suppresses client toasts on resolution failure.
    """
    result = GriptapeNodes.handle_request(
        GetPathForMacroRequest(
            parsed_macro=parsed,
            variables={},
            failure_log_level=logging.DEBUG,
            broadcast_result=False,
        )
    )
    if not isinstance(result, GetPathForMacroResultSuccess):
        return None
    return str(result.absolute_path)


def split_resolved_path(resolved: str) -> ResolvedPath:
    """Split an absolute path into directory + filename pattern.

    `filename_pattern` is None when the path has no filename portion (e.g. a
    bare directory). Sequence tokens may appear in the filename portion only;
    tokens in directory components aren't supported by these nodes.
    """
    sep_index = max(resolved.rfind("/"), resolved.rfind("\\"))
    if sep_index < 0:
        return ResolvedPath(directory="", filename_pattern=resolved)
    directory = resolved[:sep_index]
    filename = resolved[sep_index + 1 :]
    if not filename:
        return ResolvedPath(directory=directory, filename_pattern=None)
    return ResolvedPath(directory=directory, filename_pattern=filename)
