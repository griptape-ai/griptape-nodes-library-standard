"""Fixtures for the GLB to USD tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ._glb_builder import GlbSpec, build_glb

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@pytest.fixture
def make_glb(tmp_path: Path) -> Callable[..., Path]:
    def _make(name: str = "model.glb", spec: GlbSpec | None = None) -> Path:
        return build_glb(tmp_path / name, spec)

    return _make
