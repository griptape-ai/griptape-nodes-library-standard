"""Tests for how SeedreamImageGeneration maps the proxy's hosted artifact list

onto its dynamically named image_url / image_url_N output parameters.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.image.seedream_image_generation import SeedreamImageGeneration
from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifact

LIBRARY_NAME = "Griptape Nodes Library"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> SeedreamImageGeneration:  # noqa: ARG001
    """Construct through the library, matching how the node is actually built.

    `_parse_result` reaches `self._output_file`, which `ProjectFileParameter`
    only sets up through the real node constructor.
    """
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast(
        "SeedreamImageGeneration",
        library.create_node(node_type="SeedreamImageGeneration", name="Seedream Image Generation"),
    )


class _FakeDestination:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def awrite_bytes(self, data: bytes) -> SimpleNamespace:
        self._path.write_bytes(data)
        return SimpleNamespace(location=str(self._path), name=self._path.name)


def _stub_output_file(node: SeedreamImageGeneration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        node._output_file,
        "build_file",
        lambda _index=0, **_kwargs: _FakeDestination(tmp_path / f"seedream_{_index}.png"),
    )


@pytest.mark.asyncio
async def test_parse_result_saves_multiple_images_on_dynamically_named_params(
    node: SeedreamImageGeneration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = [
        HostedArtifact(index=0, kind="image", url="https://example/0.png"),
        HostedArtifact(index=1, kind="image", url="https://example/1.png"),
        HostedArtifact(index=2, kind="image", url="https://example/2.png"),
    ]
    image_bytes = {0: b"image-one", 1: b"image-two", 2: b"image-three"}

    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return artifacts

    async def fake_download_artifact(artifact: HostedArtifact) -> bytes:
        return image_bytes[artifact.index]

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)
    _stub_output_file(node, monkeypatch, tmp_path)

    await node._parse_result({}, "gen-1")

    # image_url keeps the pre-multi-image name for backwards compatibility; the rest
    # are numbered from 2, not 1 -- a change to that scheme must fail here.
    assert Path(node.parameter_output_values["image_url"].value).read_bytes() == b"image-one"
    assert Path(node.parameter_output_values["image_url_2"].value).read_bytes() == b"image-two"
    assert Path(node.parameter_output_values["image_url_3"].value).read_bytes() == b"image-three"
    assert node.parameter_output_values.get("image_url_4") is None
    assert node.parameter_output_values["was_successful"] is True


@pytest.mark.asyncio
async def test_parse_result_saves_a_single_image_as_image_url(
    node: SeedreamImageGeneration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return [HostedArtifact(index=0, kind="image", url="https://example/0.png")]

    async def fake_download_artifact(_artifact: HostedArtifact) -> bytes:
        return b"only-image"

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)
    _stub_output_file(node, monkeypatch, tmp_path)

    await node._parse_result({}, "gen-1")

    assert Path(node.parameter_output_values["image_url"].value).read_bytes() == b"only-image"
    assert node.parameter_output_values.get("image_url_2") is None
    assert node.parameter_output_values["was_successful"] is True


@pytest.mark.asyncio
async def test_parse_result_reports_failure_and_clears_images_when_retrieval_fails(
    node: SeedreamImageGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Seeded as if a prior run succeeded, so the test can tell a real clear from an
    # already-empty parameter.
    node.parameter_output_values["image_url"] = "stale-artifact"

    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return [HostedArtifact(index=0, kind="image", url="https://example/0.png")]

    async def fake_download_artifact(_artifact: HostedArtifact) -> bytes:
        msg = "network error"
        raise RuntimeError(msg)

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)

    await node._parse_result({}, "gen-1")

    assert node.parameter_output_values["image_url"] is None
    assert node.parameter_output_values["was_successful"] is False
