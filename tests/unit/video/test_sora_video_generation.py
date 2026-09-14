from __future__ import annotations

import pytest

from griptape_nodes_library.proxy import ArtifactKind
from griptape_nodes_library.video.sora_video_generation import SoraVideoGeneration


def _node(name: str = "SoraVideoGeneration") -> SoraVideoGeneration:
    return SoraVideoGeneration(name=name)


@pytest.mark.asyncio
async def test_parse_result_reports_the_providers_own_error_on_an_internal_failure_status() -> None:
    # The outer polling wrapper already reported COMPLETED, so only this in-payload
    # status catches a provider-reported failure riding inside a completed generation.
    node = _node()

    await node._parse_result(
        {"status": "failed", "error": {"message": "Content violates usage policies"}},
        "gen-1",
    )

    assert node.parameter_output_values["was_successful"] is False
    assert "Content violates usage policies" in node.parameter_output_values["result_details"]
    assert node.parameter_output_values["video_url"] is None


@pytest.mark.asyncio
async def test_parse_result_reports_the_providers_error_case_insensitively() -> None:
    node = _node()

    await node._parse_result({"status": "Error", "error": {"message": "Rejected"}}, "gen-1")

    assert node.parameter_output_values["was_successful"] is False
    assert "Rejected" in node.parameter_output_values["result_details"]


@pytest.mark.asyncio
async def test_parse_result_saves_the_hosted_video_on_a_non_failing_status(monkeypatch: pytest.MonkeyPatch) -> None:
    # A payload that carries no failed/error status must still reach the hosted-media
    # save, not just skip the new branch and stop.
    node = _node()
    captured: dict = {}

    async def fake_save_generated_media(self, generation_id, output_param, _factory, **kwargs) -> bool:  # noqa: ARG001
        captured["generation_id"] = generation_id
        captured["output_param"] = output_param
        captured["kind"] = kwargs.get("kind")
        return True

    monkeypatch.setattr(SoraVideoGeneration, "_save_generated_media", fake_save_generated_media)

    await node._parse_result({"status": "completed"}, "gen-1")

    assert captured == {"generation_id": "gen-1", "output_param": "video_url", "kind": ArtifactKind.VIDEO}


@pytest.mark.asyncio
async def test_parse_result_saves_the_hosted_video_when_no_status_is_present(monkeypatch: pytest.MonkeyPatch) -> None:
    node = _node()
    captured: dict = {}

    async def fake_save_generated_media(self, generation_id, output_param, _factory, **kwargs) -> bool:  # noqa: ARG001
        captured["generation_id"] = generation_id
        return True

    monkeypatch.setattr(SoraVideoGeneration, "_save_generated_media", fake_save_generated_media)

    await node._parse_result({}, "gen-1")

    assert captured == {"generation_id": "gen-1"}
