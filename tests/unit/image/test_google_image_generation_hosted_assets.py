"""Tests for how GoogleImageGeneration reports a short or empty hosted image set.

A list that could not be read, a generation that hosted nothing, and bytes that could
not be downloaded are three different causes and must not share one status message.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from griptape_nodes.node_library.library_registry import LibraryRegistry
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.image.google_image_generation import GoogleImageGeneration
from griptape_nodes_library.proxy.hosted_artifacts import HostedArtifact, HostedArtifactError

LIBRARY_NAME = "Griptape Nodes Library"


@pytest.fixture
def node(griptape_nodes: GriptapeNodes) -> GoogleImageGeneration:  # noqa: ARG001
    library = LibraryRegistry.get_library(name=LIBRARY_NAME)
    return cast(
        "GoogleImageGeneration",
        library.create_node(node_type="GoogleImageGeneration", name="Google Image Generation"),
    )


class _FakeDestination:
    def __init__(self, path: Path) -> None:
        self._path = path

    async def awrite_bytes(self, data: bytes) -> SimpleNamespace:
        self._path.write_bytes(data)
        return SimpleNamespace(location=str(self._path), name=self._path.name)


def _capture_status(node: GoogleImageGeneration, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    status_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(node, "_set_status_results", lambda **kwargs: status_calls.append(kwargs))
    return status_calls


@pytest.mark.asyncio
async def test_a_listing_failure_names_the_listing_as_the_cause(
    node: GoogleImageGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        raise HostedArtifactError("HTTP 401")

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    status_calls = _capture_status(node, monkeypatch)

    await node._handle_response({"candidates": [{"content": {"parts": []}}]}, "gen-1")

    assert status_calls[0]["was_successful"] is False
    assert "could not be listed: HTTP 401" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_a_generation_that_hosted_nothing_says_so(
    node: GoogleImageGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return []

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    status_calls = _capture_status(node, monkeypatch)

    await node._handle_response({"candidates": [{"content": {"parts": []}}]}, "gen-1")

    assert status_calls[0]["was_successful"] is False
    assert "no images were hosted" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_every_download_failing_is_a_retrieval_failure(
    node: GoogleImageGeneration, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return [HostedArtifact(index=0, kind="image", url="https://example/0.png")]

    async def fake_download_artifact(_artifact: HostedArtifact) -> bytes:
        msg = "artifact gone"
        raise RuntimeError(msg)

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)
    status_calls = _capture_status(node, monkeypatch)

    await node._handle_response({"candidates": [{"content": {"parts": []}}]}, "gen-1")

    assert status_calls[0]["was_successful"] is False
    assert "could not be retrieved" in status_calls[0]["result_details"]


@pytest.mark.asyncio
async def test_a_partial_set_succeeds_and_counts_what_is_missing(
    node: GoogleImageGeneration, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifacts = [
        HostedArtifact(index=0, kind="image", url="https://example/0.png"),
        HostedArtifact(index=1, kind="image", url="https://example/1.png"),
    ]

    async def fake_hosted_artifacts(_generation_id: str) -> list[HostedArtifact]:
        return artifacts

    async def fake_download_artifact(artifact: HostedArtifact) -> bytes:
        if artifact.index == 1:
            msg = "artifact gone"
            raise RuntimeError(msg)
        return b"image-one"

    monkeypatch.setattr(node, "_hosted_artifacts", fake_hosted_artifacts)
    monkeypatch.setattr(node, "_download_artifact", fake_download_artifact)
    monkeypatch.setattr(node._output_file, "build_file", lambda **_kwargs: _FakeDestination(tmp_path / "google_0.png"))
    status_calls = _capture_status(node, monkeypatch)

    await node._handle_response({"candidates": [{"content": {"parts": []}}]}, "gen-1")

    assert status_calls[0]["was_successful"] is True
    assert "1 image(s) could not be retrieved" in status_calls[0]["result_details"]
    assert len(node.parameter_output_values["all_images"]) == 1
