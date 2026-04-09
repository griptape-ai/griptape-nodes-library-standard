"""Tests for CartwheelCharacterGeneration."""

import pytest
from griptape.artifacts.image_url_artifact import ImageUrlArtifact
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.cartwheel_character_generation import (
    REFERENCE_IMAGE_MODE,
    CartwheelCharacterGeneration,
)


class TestCartwheelCharacterGeneration:
    @pytest.fixture
    def node(
        self, griptape_nodes: GriptapeNodes  # noqa: ARG002
    ) -> CartwheelCharacterGeneration:
        return CartwheelCharacterGeneration(name="test_cartwheel_character_generation")

    @pytest.mark.asyncio
    async def test_build_payload_for_text_mode(
        self, node: CartwheelCharacterGeneration
    ) -> None:
        node.set_parameter_value("prompt", "  blue robot hero  ")
        node.set_parameter_value("character_name", "  Robot Hero  ")

        payload = await node._build_payload()

        assert payload == {
            "prompt": "blue robot hero",
            "characterName": "Robot Hero",
        }

    @pytest.mark.asyncio
    async def test_build_payload_for_reference_image_mode(
        self, node: CartwheelCharacterGeneration
    ) -> None:
        node.set_parameter_value("mode", REFERENCE_IMAGE_MODE)
        node.set_parameter_value("reference_media_id", " media-123 ")
        node.set_parameter_value("character_name", " Character From Ref ")

        payload = await node._build_payload()

        assert payload == {
            "mediaID": "media-123",
            "characterName": "Character From Ref",
        }

    def test_validate_before_node_run_requires_reference_media_id_in_reference_mode(
        self, node: CartwheelCharacterGeneration
    ) -> None:
        node.set_parameter_value("mode", REFERENCE_IMAGE_MODE)

        exceptions = node.validate_before_node_run()

        assert exceptions is not None
        assert any(
            "reference image media ID" in str(exception) for exception in exceptions
        )

    @pytest.mark.asyncio
    async def test_parse_result_falls_back_to_provider_thumbnail_url(
        self, node: CartwheelCharacterGeneration, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def no_download(_url: str) -> bytes | None:
            return None

        monkeypatch.setattr(
            CartwheelCharacterGeneration,
            "_download_bytes_from_url",
            staticmethod(no_download),
        )

        await node._parse_result(
            {
                "character": {
                    "characterID": "char-123",
                    "uploadStatus": "COMPLETE",
                    "generatedStatus": "3D_CONVERT_COMPLETE",
                    "estimatedSecondsWaitTime": 480,
                    "thumbnailURL": "https://example.com/thumb.png",
                }
            },
            "gen-123",
        )

        thumbnail = node.parameter_output_values["thumbnail_image"]
        assert isinstance(thumbnail, ImageUrlArtifact)
        assert thumbnail.value == "https://example.com/thumb.png"
        assert node.parameter_output_values["character_id"] == "char-123"
        assert node.parameter_output_values["upload_status"] == "COMPLETE"
        assert node.parameter_output_values["generated_status"] == "3D_CONVERT_COMPLETE"
