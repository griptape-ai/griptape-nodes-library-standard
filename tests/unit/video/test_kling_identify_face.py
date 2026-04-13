from unittest.mock import patch

import pytest
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

from griptape_nodes_library.video.kling_identify_face import KlingIdentifyFace


class TestKlingIdentifyFace:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> KlingIdentifyFace:  # noqa: ARG002
        return KlingIdentifyFace(name="kling_identify_face")

    @pytest.mark.asyncio
    async def test_build_payload_for_video_id(self, node: KlingIdentifyFace) -> None:
        node.parameter_values["video_input_type"] = "video_id"
        node.parameter_values["video_id"] = "video-123"

        payload = await node._build_payload()

        assert payload == {"video_id": "video-123"}

    @pytest.mark.asyncio
    async def test_build_payload_for_video_url(self, node: KlingIdentifyFace) -> None:
        node.parameter_values["video_input_type"] = "video_url"

        with patch.object(
            node._public_video_url_parameter,
            "get_public_url_for_parameter",
            return_value="https://example.com/video.mp4",
        ):
            payload = await node._build_payload()

        assert payload == {"video_url": "https://example.com/video.mp4"}

    @pytest.mark.asyncio
    async def test_parse_result_updates_selected_face(self, node: KlingIdentifyFace) -> None:
        node.parameter_values["selected_face_index"] = 1

        await node._parse_result(
            {
                "data": {
                    "session_id": "session-123",
                    "face_data": [
                        {"face_id": "0", "start_time": 0, "end_time": 4000},
                        {"face_id": "1", "start_time": 5000, "end_time": 9000},
                    ],
                }
            },
            "generation-123",
        )

        assert node.parameter_output_values["session_id"] == "session-123"
        assert node.parameter_output_values["selected_face"] == {
            "face_id": "1",
            "start_time": 5000,
            "end_time": 9000,
        }
        assert node.parameter_output_values["selected_face_id"] == "1"

    def test_selected_face_updates_when_index_changes(self, node: KlingIdentifyFace) -> None:
        node.parameter_output_values["face_data"] = [
            {"face_id": "0"},
            {"face_id": "1"},
        ]

        node.after_value_set(
            next(p for p in node.parameters if p.name == "selected_face_index"),
            1,
        )
        node.parameter_values["selected_face_index"] = 1
        node._update_selected_face_output()

        assert node.parameter_output_values["selected_face_id"] == "1"
