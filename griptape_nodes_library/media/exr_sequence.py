"""Helpers for nodes that return a ZIP of EXR frames as a project file sequence.

``add_exr_sequence_outputs`` adds the output parameters; ``write_exr_zip_to_sequence`` unpacks a
ZIP into a ``FileSequenceDestination`` and returns the resulting ``Sequence``.

Known rough edge: the ``ProjectFileSequenceParameter`` cog button targets a ``FileSequenceSettings``
node that the engine does not ship yet, so clicking it does nothing useful.

The sequence is exposed via ``FileSequence.scan()`` rather than ``FileSequence.location``. The
latter returns the unresolved macro template (griptape-ai/griptape-nodes-engine#6183), and the
save parameter's own output type is ``FileSequence``, which no consumer accepts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
import logging
import pathlib
import re
import shutil
import zipfile
import zlib
from typing import TYPE_CHECKING

from griptape_nodes.common.sequences import MissingItemPolicy, Sequence
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.param_components.project_file_sequence_parameter import ProjectFileSequenceParameter
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString

if TYPE_CHECKING:
    import os

    from griptape_nodes.exe_types.node_types import BaseNode
    from griptape_nodes.files.file_sequence import FileSequenceDestination

logger = logging.getLogger("griptape_nodes")

DEFAULT_MAX_MEMBERS = 4000
# Real LTX HDR output peaks around 18 GB unpacked (273 frames of 1080p, or 65 of 4K, half-float).
DEFAULT_MAX_TOTAL_BYTES = 32 * 1024**3
# Ceiling on one decompressed frame. A 4K half-float RGBA EXR is about 66 MB uncompressed, so this is
# generous; it stops one member that lies about its size from being inflated in memory up to the
# whole-archive budget before the size check can fire.
DEFAULT_MAX_MEMBER_BYTES = 512 * 1024**2

# LTX `output_colorspace` value -> OCIO colour space name, for OCIOColorParameters.source_colorspace.
#
# These names follow the ACES studio config. Another OCIO config may name the same spaces
# differently, and then the name sent to OCIOColorParameters will not match a space in that
# config. This table does not read the config's roles or mappings, so a studio that names them
# differently has to map them itself. `ocio_colour_space_name` passes an unknown value through
# unchanged, so a value that is already a name from the studio's own config is used as written.
OCIO_COLOUR_SPACE_NAMES: dict[str, str] = {
    "acescg": "ACEScg",
    "acescct": "ACEScct",
    "srgb_linear": "Linear Rec.709 (sRGB)",
    "aces2065-1": "ACES2065-1",
}
DEFAULT_OCIO_COLOUR_SPACE_NAME = "ACEScg"

_NUMBER_RE = re.compile(r"(\d+)")


def ocio_colour_space_name(value: str | None) -> str:
    """Return the OCIO colour space name for an LTX `output_colorspace` value (ACEScg when None).

    Unknown values are passed through unchanged, so a value that is already an OCIO name survives.
    """
    if value is None:
        return DEFAULT_OCIO_COLOUR_SPACE_NAME
    return OCIO_COLOUR_SPACE_NAMES.get(value, value)


@dataclasses.dataclass(frozen=True)
class ExrSequenceOutputs:
    """Handles for the parameters added by ``add_exr_sequence_outputs``."""

    file_sequence: ProjectFileSequenceParameter
    sequence_param_name: str
    colour_space_param_name: str


def add_exr_sequence_outputs(
    node: BaseNode,
    *,
    sequence_param_name: str = "exr_sequence",
    file_param_name: str = "exr_frames",
    default_filename: str = "ltx.exr",
    colour_space_param_name: str = "colour_space",
) -> ExrSequenceOutputs:
    """Add the EXR save target, a `Sequence` output and a colour space output to ``node``."""
    file_sequence = ProjectFileSequenceParameter(node, file_param_name, default_filename=default_filename)

    # The save target is added at the node's top level: ProjectFileSequenceParameter.add_parameter()
    # attaches to the node directly rather than to an enclosing group.
    file_sequence.add_parameter()
    with ParameterGroup(name="EXR Output") as group:
        Parameter(
            name=sequence_param_name,
            type="Sequence",
            output_type="Sequence",
            default_value=None,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip=("The saved EXR frames as a sequence. Wire into an EXR sequence loader or preview node."),
            ui_options={"display_name": "EXR Sequence"},
        )
        ParameterString(
            name=colour_space_param_name,
            allowed_modes={ParameterMode.OUTPUT},
            tooltip="OCIO colour space name of the EXR frames, for an OpenColorIO source colour space.",
            ui_options={"display_name": "Colour Space"},
        )
    node.add_node_element(group)

    return ExrSequenceOutputs(
        file_sequence=file_sequence,
        sequence_param_name=sequence_param_name,
        colour_space_param_name=colour_space_param_name,
    )


def _natural_key(name: str) -> list[tuple[int, int, str]]:
    # Digit runs sort numerically; text sorts case-insensitively. Tuples keep types comparable.
    return [(0, int(part), "") if part.isdigit() else (1, 0, part.lower()) for part in _NUMBER_RE.split(name) if part]


def _is_exr_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir() or not info.filename.lower().endswith(".exr"):
        return False
    parts = info.filename.split("/")
    if parts[0] == "__MACOSX":
        return False
    return not any(part.startswith(".") and part not in (".", "..") for part in parts)


def _select_members(
    zf: zipfile.ZipFile, *, max_members: int, max_total_bytes: int, max_member_bytes: int
) -> list[zipfile.ZipInfo]:
    members = sorted((i for i in zf.infolist() if _is_exr_member(i)), key=lambda i: _natural_key(i.filename))
    if not members:
        msg = "The ZIP contains no .exr files. Check that the generation returned EXR output."
        raise NodeError(msg)
    if len(members) > max_members:
        msg = f"The ZIP contains {len(members)} EXR frames, which is over the limit of {max_members} frames."
        raise NodeError(msg)
    # Checked before anything is written, so an oversized frame never leaves a partial version folder.
    for member in members:
        if member.file_size > max_member_bytes:
            msg = (
                f"The ZIP's EXR frame '{member.filename}' is {member.file_size} bytes when unpacked, "
                f"which is over the per-frame limit of {max_member_bytes} bytes."
            )
            raise NodeError(msg)
    declared_total = sum(m.file_size for m in members)
    if declared_total > max_total_bytes:
        msg = (
            f"The ZIP's EXR frames total {declared_total} bytes when unpacked, "
            f"which is over the limit of {max_total_bytes} bytes."
        )
        raise NodeError(msg)
    return members


def _check_free_disk_space(dest: FileSequenceDestination, needed_bytes: int) -> None:
    """Refuse to start unpacking when the destination volume cannot hold the declared size.

    Checked before anything is written, so a full disk does not leave a partial sequence behind once
    the generation has already been billed. The destination folder may not exist yet, so the nearest
    existing parent is the one measured.
    """
    directory = pathlib.Path(dest.entry(1).resolve()).parent
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    free_bytes = shutil.disk_usage(directory).free
    if needed_bytes > free_bytes:
        msg = (
            f"The ZIP's EXR frames need {needed_bytes} bytes when unpacked, "
            f"but only {free_bytes} bytes are free where the project saves them."
        )
        raise NodeError(msg)


def _read_member(
    zf: zipfile.ZipFile, info: zipfile.ZipInfo, remaining_bytes: int, max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES
) -> bytes:
    """Read one member, refusing to decompress past the budget (``file_size`` can lie).

    The read is bounded by the smaller of the archive's remaining budget and the per-frame cap, so
    peak memory stays at one frame.
    """
    limit = min(remaining_bytes, max_member_bytes)
    with zf.open(info) as member:
        data = member.read(limit + 1)
    if len(data) > limit:
        if max_member_bytes <= remaining_bytes:
            msg = (
                f"The ZIP's EXR frame '{info.filename}' is over the per-frame limit of "
                f"{max_member_bytes} bytes when unpacked."
            )
        else:
            msg = (
                f"The ZIP's EXR frames exceed the limit of the unpacked size while reading '{info.filename}'. "
                "The archive's declared sizes do not match its contents."
            )
        raise NodeError(msg)
    return data


async def write_exr_zip_to_sequence(
    zip_bytes_or_path: bytes | bytearray | memoryview | str | os.PathLike[str],
    dest: FileSequenceDestination,
    *,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_total_bytes: int = DEFAULT_MAX_TOTAL_BYTES,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
) -> Sequence | None:
    """Write the `.exr` members of a ZIP into ``dest`` as frames 1..N and return the scanned Sequence.

    Members are written by position through ``dest.entry(i)``, never by member name, so a hostile
    archive cannot write outside the destination. Members are read one at a time. Returns None
    when the scan finds no sequence.

    ``dest`` should come from ``ProjectFileSequenceParameter.build_sequence()``, whose default
    situation writes each run into a fresh versioned folder. A destination whose policy is FAIL on
    a non-versioned path fails on the second run, and that failure is reported as a NodeError.

    Raises:
        NodeError: The input is not a valid ZIP, has no EXR members, or exceeds a limit, or the
        destination volume has too little free space for the unpacked frames.
    """
    in_memory = isinstance(zip_bytes_or_path, (bytes, bytearray, memoryview))
    source = io.BytesIO(bytes(zip_bytes_or_path)) if in_memory else zip_bytes_or_path
    try:
        zf = await asyncio.to_thread(zipfile.ZipFile, source)
    except zipfile.BadZipFile as e:
        msg = "The downloaded file is not a valid ZIP archive."
        raise NodeError(msg) from e
    except OSError as e:
        msg = f"Could not open the downloaded ZIP archive: {e}"
        raise NodeError(msg) from e

    with zf:
        members = _select_members(
            zf, max_members=max_members, max_total_bytes=max_total_bytes, max_member_bytes=max_member_bytes
        )
        await asyncio.to_thread(_check_free_disk_space, dest, sum(m.file_size for m in members))
        remaining = max_total_bytes
        for index, info in enumerate(members, start=1):
            try:
                data = await asyncio.to_thread(_read_member, zf, info, remaining, max_member_bytes)
            except (zipfile.BadZipFile, RuntimeError, NotImplementedError, zlib.error, EOFError, OSError) as e:
                msg = f"Could not read '{info.filename}' from the ZIP. The archive may be corrupt or encrypted."
                raise NodeError(msg) from e
            remaining -= len(data)
            try:
                await dest.entry(index).awrite_bytes(data)
            except NodeError:
                raise
            except Exception as e:
                msg = f"Could not save EXR frame {index} to the project: {e}"
                raise NodeError(msg) from e

    file_sequence = dest.file_sequence
    if file_sequence is None:
        logger.warning("No EXR frames were written, so there is no sequence to return.")
        return None
    # Bound the scan to the frames just written, so leftovers from an earlier, longer run in the same
    # directory (possible under an OVERWRITE policy) are not reported as part of this sequence.
    sequences = await asyncio.to_thread(file_sequence.scan, policy=MissingItemPolicy.SKIP, start=1, end=len(members))
    if not sequences:
        logger.warning("Scanning the EXR output directory found no sequence.")
        return None
    return sequences[0]
