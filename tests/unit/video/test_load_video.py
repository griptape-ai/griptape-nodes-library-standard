"""Unit tests for LoadVideo.process()."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files.file import FileLoadError
from griptape_nodes.retained_mode.events.os_events import FileIOFailureReason
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

import griptape_nodes_library.video.load_video as load_video_module
from griptape_nodes_library.utils.macro_path_utils import MacroPathResult
from griptape_nodes_library.video.load_video import LoadVideo

_MACRO_PATH = "{inputs}/video.mp4"


class TestLoadVideoProcess:
    @pytest.fixture
    def node(self, griptape_nodes: GriptapeNodes) -> LoadVideo:  # noqa: ARG002
        return LoadVideo(name="test_load_video")

    @pytest.fixture
    def mock_update_external_file_controls(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        mock_update_external_file_controls = Mock(spec=load_video_module.update_external_file_controls)
        monkeypatch.setattr(load_video_module, "update_external_file_controls", mock_update_external_file_controls)
        return mock_update_external_file_controls

    @pytest.fixture
    def mock_resolve_to_macro_path(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        mock_resolve_to_macro_path = Mock(spec=load_video_module.resolve_to_macro_path)
        mock_resolve_to_macro_path.return_value = MacroPathResult(resolved_path=_MACRO_PATH, is_external=False)
        monkeypatch.setattr(load_video_module, "resolve_to_macro_path", mock_resolve_to_macro_path)
        return mock_resolve_to_macro_path

    @pytest.fixture
    def mock_file_cls(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        mock_file_cls = Mock(spec=load_video_module.File)
        monkeypatch.setattr(load_video_module, "File", mock_file_cls)
        return mock_file_cls

    @pytest.fixture
    def mock_extract_video_player_metadata(self, monkeypatch: pytest.MonkeyPatch) -> Mock:
        mock_extract_video_player_metadata = Mock(spec=load_video_module.ffmpeg_utils.extract_video_player_metadata)
        monkeypatch.setattr(
            load_video_module.ffmpeg_utils, "extract_video_player_metadata", mock_extract_video_player_metadata
        )
        mock_extract_video_player_metadata.return_value = {"duration": 1.0}
        return mock_extract_video_player_metadata

    def test_relative_video_url_artifact(
        self,
        node: LoadVideo,
        mock_update_external_file_controls: Mock,
        mock_resolve_to_macro_path: Mock,
        mock_file_cls: Mock,
        mock_extract_video_player_metadata: Mock,
    ) -> None:
        mock_file = mock_file_cls.return_value

        node.parameter_values["video"] = VideoUrlArtifact(_MACRO_PATH)

        node.process()

        mock_update_external_file_controls.assert_called_once_with(
            mock_resolve_to_macro_path.return_value, node._external_warning, node._copy_button, node.name, "video"
        )
        mock_resolve_to_macro_path.assert_called_once_with(_MACRO_PATH)
        mock_file_cls.assert_called_once_with(_MACRO_PATH)
        mock_extract_video_player_metadata.assert_called_once_with(mock_file.resolve.return_value)

        assert node.parameter_output_values["video"].value == _MACRO_PATH
        assert node.parameter_output_values["video"].meta == {"duration": 1.0}
        assert node.parameter_output_values["path"] == _MACRO_PATH

    def test_relative_video_url_artifact_fails_to_resolve(
        self,
        node: LoadVideo,
        monkeypatch: pytest.MonkeyPatch,
        mock_update_external_file_controls: Mock,
        mock_resolve_to_macro_path: Mock,
        mock_file_cls: Mock,
        mock_extract_video_player_metadata: Mock,
    ) -> None:
        mock_file = mock_file_cls.return_value
        mock_file.resolve.side_effect = FileLoadError(
            failure_reason=FileIOFailureReason.MISSING_MACRO_VARIABLES,
            result_details="Boom!",
        )

        mock_logger = Mock(spec=load_video_module.logger)
        monkeypatch.setattr(load_video_module, "logger", mock_logger)

        node.parameter_values["path"] = "some_path"
        node.parameter_values["video"] = VideoUrlArtifact(_MACRO_PATH)

        node.process()

        mock_update_external_file_controls.assert_called_once_with(
            mock_resolve_to_macro_path.return_value, node._external_warning, node._copy_button, node.name, "video"
        )
        mock_resolve_to_macro_path.assert_called_once_with(_MACRO_PATH)
        mock_file_cls.assert_called_once_with(_MACRO_PATH)
        mock_extract_video_player_metadata.assert_not_called()
        mock_logger.warning.assert_called_once_with(
            f"LoadVideo '{node.name}': Attempted to resolve '{_MACRO_PATH}' to extract video "
            f"metadata. Failed due to Boom!"
        )

        assert node.parameter_output_values["video"].value == _MACRO_PATH
        assert node.parameter_output_values["video"].meta == {}
        assert node.parameter_output_values["path"] == _MACRO_PATH

    def test_external_video_url_artifact(
        self,
        node: LoadVideo,
        mock_update_external_file_controls: Mock,
        mock_resolve_to_macro_path: Mock,
        mock_file_cls: Mock,
        mock_extract_video_player_metadata: Mock,
    ) -> None:
        mock_resolve_to_macro_path.return_value = MacroPathResult(resolved_path=_MACRO_PATH, is_external=True)

        node.parameter_values["path"] = "some_path"
        node.parameter_values["video"] = VideoUrlArtifact(_MACRO_PATH)

        node.process()

        mock_update_external_file_controls.assert_called_once_with(
            mock_resolve_to_macro_path.return_value, node._external_warning, node._copy_button, node.name, "video"
        )
        mock_resolve_to_macro_path.assert_called_once_with(_MACRO_PATH)
        mock_file_cls.assert_not_called()
        mock_extract_video_player_metadata.assert_not_called()

        assert node.parameter_output_values["video"].value == _MACRO_PATH
        assert node.parameter_output_values["video"].meta == {}
        assert node.parameter_output_values["path"] == "some_path"

    def test_not_a_video_url_artifact(
        self,
        node: LoadVideo,
        mock_update_external_file_controls: Mock,
        mock_resolve_to_macro_path: Mock,
        mock_file_cls: Mock,
        mock_extract_video_player_metadata: Mock,
    ) -> None:
        node.parameter_values["path"] = "some_path"
        node.parameter_values["video"] = "something"

        node.process()

        mock_update_external_file_controls.assert_not_called()
        mock_resolve_to_macro_path.assert_not_called()
        mock_file_cls.assert_not_called()
        mock_extract_video_player_metadata.assert_not_called()

        assert node.parameter_output_values["video"] == "something"
        assert node.parameter_output_values["path"] == "some_path"
