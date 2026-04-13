from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.kling_advanced_lip_sync import KlingAdvancedLipSync


class _FakeFileWriter:
    async def awrite_bytes(self, data: bytes) -> SimpleNamespace:  # noqa: ARG002
        return SimpleNamespace(location="project://kling_advanced_lip_sync.mp4", name="kling_advanced_lip_sync.mp4")


class TestKlingAdvancedLipSync:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> KlingAdvancedLipSync:  # noqa: ARG002
        return KlingAdvancedLipSync(name="kling_advanced_lip_sync")

    @pytest.mark.asyncio
    async def test_build_payload_for_sound_file(self, node: KlingAdvancedLipSync) -> None:
        node.parameter_values.update(
            {
                "session_id": "session-123",
                "selected_face": {"face_id": "2"},
                "audio_input_type": "sound_file",
                "sound_start_time": 0,
                "sound_end_time": 3000,
                "sound_insert_time": 1000,
                "sound_volume": 1.5,
                "original_audio_volume": 0.5,
                "watermark": True,
            }
        )

        with patch.object(
            node._public_audio_url_parameter,
            "get_public_url_for_parameter",
            return_value="https://example.com/audio.mp3",
        ):
            payload = await node._build_payload()

        assert payload == {
            "session_id": "session-123",
            "face_choose": [
                {
                    "face_id": "2",
                    "sound_start_time": 0,
                    "sound_end_time": 3000,
                    "sound_insert_time": 1000,
                    "sound_volume": 1.5,
                    "original_audio_volume": 0.5,
                    "sound_file": "https://example.com/audio.mp3",
                }
            ],
            "watermark_info": {"enabled": True},
        }

    def test_validate_requires_face_overlap(self, node: KlingAdvancedLipSync) -> None:
        node.parameter_values.update(
            {
                "session_id": "session-123",
                "selected_face": {"face_id": "2", "start_time": 0, "end_time": 2500},
                "audio_input_type": "audio_id",
                "audio_id": "audio-123",
                "sound_start_time": 0,
                "sound_end_time": 3000,
                "sound_insert_time": 5000,
            }
        )

        errors = node.validate_before_node_run()

        assert errors is not None
        assert any("overlap" in str(error) for error in errors)

    @pytest.mark.asyncio
    async def test_parse_result_saves_video(self, node: KlingAdvancedLipSync) -> None:
        with patch.object(node, "_download_bytes_from_url", AsyncMock(return_value=b"video-bytes")):
            node._output_file.build_file = Mock(return_value=_FakeFileWriter())
            await node._parse_result(
                {
                    "data": {
                        "task_result": {
                            "videos": [
                                {
                                    "id": "video-123",
                                    "url": "https://example.com/output.mp4",
                                }
                            ]
                        }
                    }
                },
                "generation-123",
            )

        assert node.parameter_output_values["kling_video_id"] == "video-123"
        assert node.parameter_output_values["video_url"].value == "project://kling_advanced_lip_sync.mp4"
