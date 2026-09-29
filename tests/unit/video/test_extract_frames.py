"""Unit tests for ExtractFrames' Sequence output.

Frame files are empty placeholders in ``tmp_path``. The scan only looks at filenames,
so no ffmpeg or real video is needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from griptape_nodes_library.video.extract_frames import ExtractFrames

if TYPE_CHECKING:
    from pathlib import Path


def _write_frames(directory: Path, numbers: list[int], *, prefix: str = "frames", padding: int = 4) -> None:
    for n in numbers:
        (directory / f"{prefix}.{str(n).zfill(padding)}.png").touch()


@pytest.fixture
def node() -> ExtractFrames:
    return ExtractFrames(name="test_extract_frames")


class TestScanOutputSequence:
    def test_sparse_selection_keeps_gaps(self, node: ExtractFrames, tmp_path: Path) -> None:
        frames = [1, 4, 5, 6, 7, 8, 9, 11]
        _write_frames(tmp_path, frames)

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", frames)

        assert sequence is not None
        assert sequence.pattern == "frames.####.png"
        assert sequence.padding == 4
        assert sorted(sequence.present_numbers) == frames
        assert [entry.number for entry in sequence.entries] == frames
        assert sequence.entries[0].path == str(tmp_path / "frames.0001.png")

    def test_single_frame_is_a_one_item_sequence(self, node: ExtractFrames, tmp_path: Path) -> None:
        _write_frames(tmp_path, [7])

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [7])

        assert sequence is not None
        assert [entry.number for entry in sequence.entries] == [7]

    def test_frames_outside_requested_range_are_excluded(self, node: ExtractFrames, tmp_path: Path) -> None:
        # Frames 1 and 20 are left over from an earlier run into the same directory.
        _write_frames(tmp_path, [1, 5, 6, 7, 20])

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [5, 6, 7])

        assert sequence is not None
        assert sorted(sequence.present_numbers) == [5, 6, 7]

    def test_other_prefixes_are_ignored(self, node: ExtractFrames, tmp_path: Path) -> None:
        _write_frames(tmp_path, [1, 2, 3])
        _write_frames(tmp_path, [1, 2, 3], prefix="other")

        sequence = node._scan_output_sequence(tmp_path, "frames", 4, "png", [1, 2, 3])

        assert sequence is not None
        assert all(entry.path.startswith(str(tmp_path / "frames.")) for entry in sequence.entries)

    def test_empty_directory_returns_none(self, node: ExtractFrames, tmp_path: Path) -> None:
        assert node._scan_output_sequence(tmp_path, "frames", 4, "png", [1, 2, 3]) is None


class TestSequenceOutputWiring:
    def test_perform_extraction_sets_sequence_output(
        self, node: ExtractFrames, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        frames = [2, 4, 6]

        def fake_extract_frames(**_kwargs: object) -> list[Path]:
            _write_frames(tmp_path, frames)
            return [tmp_path / f"frames.{str(n).zfill(4)}.png" for n in frames]

        monkeypatch.setattr(node, "_resolve_output_dir", lambda: tmp_path)
        monkeypatch.setattr(node, "_extract_frames", fake_extract_frames)

        node._perform_extraction("unused.mp4", frames)

        sequence = node.parameter_output_values["sequence"]
        assert sequence is not None
        assert [entry.number for entry in sequence.entries] == frames
        assert node.parameter_output_values["output_frames"] == [entry.path for entry in sequence.entries]

    def test_safe_defaults_clear_sequence_output(self, node: ExtractFrames) -> None:
        node.parameter_output_values["sequence"] = object()

        node._set_safe_defaults()

        assert node.parameter_output_values["sequence"] is None
