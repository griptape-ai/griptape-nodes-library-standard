"""Tests for the EXR ZIP -> file sequence helper."""

from __future__ import annotations

import io
import shutil
import tempfile
import zipfile
import zlib
from pathlib import Path
from typing import Any

import pytest
from griptape_nodes.common.sequences import MissingItemPolicy
from griptape_nodes.exe_types.core_types import NodeError
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.media import exr_sequence


def _make_zip(members: dict[str, bytes], *, compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


class _FakeEntry:
    def __init__(self, dest: _FakeDest, number: int) -> None:
        self._dest = dest
        self._number = number

    def resolve(self) -> str:
        return str(self._dest.directory / f"frame.{self._number:04d}.exr")

    async def awrite_bytes(self, data: bytes) -> None:
        self._dest.calls.append(("write", self._number, data))
        self._dest.file_sequence = _FakeFileSequence(self._dest)

    def write_bytes(self, data: bytes) -> None:  # pragma: no cover - the helper uses the async API
        msg = "Blocking write used"
        raise AssertionError(msg)


class _FakeFileSequence:
    def __init__(self, dest: _FakeDest) -> None:
        self._dest = dest

    def scan(self, *, policy: MissingItemPolicy, start: int | None = None, end: int | None = None) -> list[Any]:
        self._dest.calls.append(("scan", policy, start, end))
        return self._dest.scan_result


class _FakeDest:
    def __init__(self, scan_result: list[Any] | None = None) -> None:
        self.calls: list[tuple] = []
        # Deliberately absent, so the free-space check has to fall back to an existing parent.
        self.directory = Path(tempfile.gettempdir()) / "exr-sequence-test-not-created" / "v1"
        self.file_sequence: _FakeFileSequence | None = None
        self.scan_result = ["SEQ"] if scan_result is None else scan_result

    def entry(self, number: int) -> _FakeEntry:
        return _FakeEntry(self, number)

    @property
    def writes(self) -> list[tuple[int, bytes]]:
        return [(c[1], c[2]) for c in self.calls if c[0] == "write"]


@pytest.mark.asyncio
async def test_natural_sort_and_one_based_numbering_then_scan() -> None:
    data = _make_zip({"f_10.exr": b"ten", "f_2.exr": b"two", "f_1.exr": b"one", "F_9.EXR": b"nine"})
    dest = _FakeDest()

    result = await exr_sequence.write_exr_zip_to_sequence(data, dest)  # type: ignore[arg-type]

    assert result == "SEQ"
    assert dest.writes == [(1, b"one"), (2, b"two"), (3, b"nine"), (4, b"ten")]
    # The scan is bounded to the frames just written.
    assert dest.calls[-1][:2] == ("scan", MissingItemPolicy.SKIP)
    assert dest.calls[-1][2] == 1


@pytest.mark.asyncio
async def test_ignores_non_exr_directories_macosx_and_dotfiles() -> None:
    data = _make_zip(
        {
            "a.exr": b"a",
            "readme.txt": b"x",
            "sub/": b"",
            "__MACOSX/a.exr": b"junk",
            ".hidden.exr": b"junk",
            "d/.hidden.exr": b"junk",
            "./a2.exr": b"dot",
            "b.exr.bak": b"junk",
            "b.exr": b"b",
        }
    )
    dest = _FakeDest()

    await exr_sequence.write_exr_zip_to_sequence(data, dest)  # type: ignore[arg-type]

    assert dest.writes == [(1, b"dot"), (2, b"a"), (3, b"b")]


@pytest.mark.asyncio
async def test_member_names_never_reach_the_destination() -> None:
    dest = _FakeDest()

    await exr_sequence.write_exr_zip_to_sequence(_make_zip({"../../evil.exr": b"x"}), dest)  # type: ignore[arg-type]

    assert dest.writes == [(1, b"x")]


@pytest.mark.asyncio
async def test_no_exr_members_raises() -> None:
    with pytest.raises(NodeError, match="no .exr files"):
        await exr_sequence.write_exr_zip_to_sequence(_make_zip({"a.txt": b"x"}), _FakeDest())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_invalid_zip_raises() -> None:
    with pytest.raises(NodeError, match="not a valid ZIP"):
        await exr_sequence.write_exr_zip_to_sequence(b"definitely not a zip", _FakeDest())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_member_count_cap_raises_before_writing() -> None:
    dest = _FakeDest()
    data = _make_zip({f"f_{i}.exr": b"x" for i in range(3)})

    with pytest.raises(NodeError, match="limit of 2 frames"):
        await exr_sequence.write_exr_zip_to_sequence(data, dest, max_members=2)  # type: ignore[arg-type]

    assert dest.calls == []


@pytest.mark.asyncio
async def test_declared_size_cap_raises_before_writing() -> None:
    dest = _FakeDest()
    data = _make_zip({"a.exr": b"x" * 600, "b.exr": b"x" * 600})

    with pytest.raises(NodeError, match="limit of 1000 bytes"):
        await exr_sequence.write_exr_zip_to_sequence(data, dest, max_total_bytes=1000)  # type: ignore[arg-type]

    assert dest.calls == []


def test_read_guard_rejects_content_larger_than_remaining_budget() -> None:
    # zipfile stops reading at the declared size, so a lying header cannot be crafted through
    # its public API; exercise the guard directly with a budget below the real content.
    zf = zipfile.ZipFile(io.BytesIO(_make_zip({"a.exr": b"x" * 100})))
    info = zf.infolist()[0]

    with pytest.raises(NodeError, match="declared sizes do not match"):
        exr_sequence._read_member(zf, info, 50)

    assert exr_sequence._read_member(zf, info, 100) == b"x" * 100


@pytest.mark.asyncio
async def test_budget_is_shared_across_members(monkeypatch: pytest.MonkeyPatch) -> None:
    # The running budget is defence in depth behind the upfront declared-size check (a header that
    # understates a member's size is caught by zipfile's own CRC check). Bypass the upfront check so
    # the running guard across members is what has to stop the second frame.
    data = _make_zip({"a.exr": b"x" * 100, "b.exr": b"x" * 100})
    dest = _FakeDest()
    real_select = exr_sequence._select_members

    def lenient_select(zf: zipfile.ZipFile, **kwargs: Any) -> list[zipfile.ZipInfo]:
        return real_select(
            zf, max_members=kwargs["max_members"], max_total_bytes=10**9, max_member_bytes=kwargs["max_member_bytes"]
        )

    monkeypatch.setattr(exr_sequence, "_select_members", lenient_select)

    with pytest.raises(NodeError, match="exceed the limit"):
        await exr_sequence.write_exr_zip_to_sequence(
            data,
            dest,  # type: ignore[arg-type]
            max_total_bytes=150,
        )
    # The first frame fit the budget and was written; the second overflowed it and was not.
    assert [call[0] for call in dest.calls] == ["write"]


@pytest.mark.asyncio
async def test_accepts_path_input(tmp_path: Path) -> None:
    zip_path = tmp_path / "frames.zip"
    zip_path.write_bytes(_make_zip({"a.exr": b"a", "b.exr": b"b"}))
    dest = _FakeDest()

    await exr_sequence.write_exr_zip_to_sequence(zip_path, dest)  # type: ignore[arg-type]
    await exr_sequence.write_exr_zip_to_sequence(str(zip_path), dest)  # type: ignore[arg-type]

    assert dest.writes == [(1, b"a"), (2, b"b")] * 2


@pytest.mark.asyncio
async def test_returns_none_when_scan_finds_nothing() -> None:
    dest = _FakeDest(scan_result=[])

    assert await exr_sequence.write_exr_zip_to_sequence(_make_zip({"a.exr": b"a"}), dest) is None  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("acescg", "ACEScg"),
        ("acescct", "ACEScct"),
        ("srgb_linear", "Linear Rec.709 (sRGB)"),
        ("aces2065-1", "ACES2065-1"),
        (None, "ACEScg"),
        ("Already An OCIO Name", "Already An OCIO Name"),
    ],
)
def test_ocio_colour_space_name(value: str | None, expected: str) -> None:
    assert exr_sequence.ocio_colour_space_name(value) == expected


def test_add_exr_sequence_outputs_adds_parameters() -> None:
    node = DataNode(name="n")

    outputs = exr_sequence.add_exr_sequence_outputs(node)

    seq = node.get_parameter_by_name("exr_sequence")
    assert seq is not None
    assert seq.output_type == "Sequence"
    assert node.get_parameter_by_name("exr_frames") is not None
    assert node.get_parameter_by_name("colour_space") is not None
    assert outputs.sequence_param_name == "exr_sequence"


@pytest.mark.asyncio
async def test_real_file_sequence_destination_round_trip(tmp_path: Path) -> None:
    config = GriptapeNodes.ConfigManager()
    previous_workspace = config.get_config_value("workspace_directory")
    config.set_config_value("workspace_directory", str(tmp_path))
    try:
        await _round_trip_into(tmp_path)
    finally:
        config.set_config_value("workspace_directory", previous_workspace)


async def _round_trip_into(tmp_path: Path) -> None:
    node = DataNode(name="n")
    outputs = exr_sequence.add_exr_sequence_outputs(node)
    dest = outputs.file_sequence.build_sequence()
    data = _make_zip({"f_10.exr": b"ten", "f_2.exr": b"two"})

    result = await exr_sequence.write_exr_zip_to_sequence(data, dest)

    assert result is not None
    assert (result.first, result.last) == (1, 2)
    assert result.pattern.endswith(".exr")
    assert "{" not in result.directory
    frames = sorted(Path(result.directory).glob("*.exr"))
    assert [f.read_bytes() for f in frames] == [b"two", b"ten"]
    assert str(tmp_path) in result.directory


def test_read_guard_caps_a_single_frame_below_the_archive_budget() -> None:
    """One frame can't be inflated up to the whole-archive budget before the size check fires."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("frame.0001.exr", b"x" * 100)
    with zipfile.ZipFile(buffer) as zf:
        info = zf.getinfo("frame.0001.exr")
        with pytest.raises(NodeError):
            exr_sequence._read_member(zf, info, 10_000, max_member_bytes=50)
        assert exr_sequence._read_member(zf, info, 10_000, max_member_bytes=100) == b"x" * 100


@pytest.mark.asyncio
async def test_oversized_frame_is_rejected_before_anything_is_written() -> None:
    dest = _FakeDest()
    data = _make_zip({"a.0001.exr": b"x" * 10, "a.0002.exr": b"y" * 100})

    with pytest.raises(NodeError, match=r"a\.0002\.exr.*per-frame limit of 50 bytes"):
        await exr_sequence.write_exr_zip_to_sequence(data, dest, max_member_bytes=50)  # type: ignore[arg-type]

    assert dest.writes == []


def test_read_guard_message_names_the_per_frame_cap() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("frame.0001.exr", b"x" * 100)
    with zipfile.ZipFile(buffer) as zf:
        with pytest.raises(NodeError, match="per-frame limit") as excinfo:
            exr_sequence._read_member(zf, zf.getinfo("frame.0001.exr"), 10_000, max_member_bytes=50)
        assert "declared sizes" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_scan_is_bounded_to_the_frames_written() -> None:
    dest = _FakeDest()
    await exr_sequence.write_exr_zip_to_sequence(
        _make_zip({"a.0001.exr": b"1", "a.0002.exr": b"2", "a.0003.exr": b"3"}),
        dest,  # type: ignore[arg-type]
    )
    assert dest.calls[-1] == ("scan", MissingItemPolicy.SKIP, 1, 3)


@pytest.mark.asyncio
async def test_corrupt_member_is_reported_as_node_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_read(*_args: Any, **_kwargs: Any) -> bytes:
        raise zlib.error("invalid stored block lengths")

    monkeypatch.setattr(exr_sequence, "_read_member", broken_read)
    with pytest.raises(NodeError, match="corrupt or encrypted"):
        await exr_sequence.write_exr_zip_to_sequence(
            _make_zip({"a.exr": b"x"}),
            _FakeDest(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_write_failure_is_reported_as_node_error() -> None:
    class _FailingEntry:
        def resolve(self) -> str:
            return str(Path(tempfile.gettempdir()) / "frame.0001.exr")

        async def awrite_bytes(self, _data: bytes) -> None:
            msg = "file exists"
            raise FileExistsError(msg)

    class _FailingDest:
        file_sequence = None

        def entry(self, _index: int) -> _FailingEntry:
            return _FailingEntry()

    with pytest.raises(NodeError, match="Could not save EXR frame 1"):
        await exr_sequence.write_exr_zip_to_sequence(
            _make_zip({"a.exr": b"x"}),
            _FailingDest(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_accepts_bytearray_input() -> None:
    dest = _FakeDest()
    await exr_sequence.write_exr_zip_to_sequence(
        bytearray(_make_zip({"a.exr": b"x"})),
        dest,  # type: ignore[arg-type]
    )
    assert dest.calls[0][0] == "write"


@pytest.mark.asyncio
async def test_missing_path_is_reported_as_node_error(tmp_path: Path) -> None:
    with pytest.raises(NodeError, match="Could not open"):
        await exr_sequence.write_exr_zip_to_sequence(
            tmp_path / "missing.zip",
            _FakeDest(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_insufficient_free_disk_space_is_rejected_before_writing(monkeypatch: pytest.MonkeyPatch) -> None:
    dest = _FakeDest()
    data = _make_zip({"a.exr": b"x" * 600, "b.exr": b"x" * 600})
    checked: list[Path] = []

    def fake_disk_usage(path: Path) -> shutil._ntuple_diskusage:
        checked.append(Path(path))
        return shutil._ntuple_diskusage(total=10_000, used=8_801, free=1199)

    monkeypatch.setattr(exr_sequence.shutil, "disk_usage", fake_disk_usage)

    with pytest.raises(NodeError, match=r"need 1200 bytes.*only 1199 bytes are free"):
        await exr_sequence.write_exr_zip_to_sequence(data, dest)  # type: ignore[arg-type]

    assert dest.calls == []
    # The folder does not exist yet, so the nearest existing parent is measured.
    assert checked
    assert checked[0].exists()
    assert not dest.directory.exists()


@pytest.mark.asyncio
async def test_enough_free_disk_space_still_unpacks(monkeypatch: pytest.MonkeyPatch) -> None:
    dest = _FakeDest()
    data = _make_zip({"a.exr": b"x" * 600, "b.exr": b"x" * 600})
    monkeypatch.setattr(
        exr_sequence.shutil, "disk_usage", lambda _path: shutil._ntuple_diskusage(total=10_000, used=8_800, free=1200)
    )

    await exr_sequence.write_exr_zip_to_sequence(data, dest)  # type: ignore[arg-type]

    assert [number for number, _data in dest.writes] == [1, 2]
