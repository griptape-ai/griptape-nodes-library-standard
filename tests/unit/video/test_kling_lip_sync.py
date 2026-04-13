from unittest.mock import patch

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.kling_lip_sync import KlingLipSync


class TestKlingLipSync:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> KlingLipSync:  # noqa: ARG002
        return KlingLipSync(name="kling_lip_sync")

    @pytest.mark.asyncio
    async def test_build_payload_for_text_mode(self, node: KlingLipSync) -> None:
        node.parameter_values.update(
            {
                "model_name": "kling-v2-1",
                "video_input_type": "video_id",
                "video_id": "video-123",
                "mode": "text2video",
                "text": "Hello world",
                "voice_id": "oversea_male1",
                "voice_language": "en",
                "voice_speed": 1.2,
            }
        )

        payload = await node._build_payload()

        assert payload == {
            "input": {
                "model_name": "kling-v2-1",
                "mode": "text2video",
                "video_id": "video-123",
                "text": "Hello world",
                "voice_id": "oversea_male1",
                "voice_language": "en",
                "voice_speed": 1.2,
            }
        }

    @pytest.mark.asyncio
    async def test_build_payload_for_audio_mode(self, node: KlingLipSync) -> None:
        node.parameter_values.update(
            {
                "video_input_type": "video_url",
                "mode": "audio2video",
            }
        )

        with (
            patch.object(
                node._public_video_url_parameter,
                "get_public_url_for_parameter",
                return_value="https://example.com/video.mp4",
            ),
            patch.object(
                node._public_audio_url_parameter,
                "get_public_url_for_parameter",
                return_value="https://example.com/audio.mp3",
            ),
        ):
            payload = await node._build_payload()

        assert payload == {
            "input": {
                "model_name": "kling-v2-1",
                "mode": "audio2video",
                "video_url": "https://example.com/video.mp4",
                "audio_type": "url",
                "audio_url": "https://example.com/audio.mp3",
            }
        }

    def test_validate_rejects_long_text(self, node: KlingLipSync) -> None:
        node.parameter_values.update(
            {
                "video_input_type": "video_id",
                "video_id": "video-123",
                "mode": "text2video",
                "text": "x" * 121,
                "voice_id": "voice",
            }
        )

        errors = node.validate_before_node_run()

        assert errors is not None
        assert any("exceeds 120 characters" in str(error) for error in errors)
