from pathlib import Path

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import FileDestination
from griptape_nodes.retained_mode.events.project_events import (
    AttemptMapAbsolutePathToProjectRequest,
    AttemptMapAbsolutePathToProjectResultSuccess,
)
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget


class VideoCapture(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterDict(
                name="recording",
                default_value={"state": "idle"},
                tooltip="Webcam video capture widget. Record video, then click Accept.",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="VideoCapture", library="Griptape Nodes Library")},
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="output_video",
                tooltip="The captured video.",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"pulse_on_run": True},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="webcam_capture.mp4",
        )
        self._output_file.add_parameter()

    def after_value_set(self, parameter, value):
        if parameter.name == "recording" and isinstance(value, dict):
            match value.get("state"):
                case "requesting_upload_url":
                    server_url = GriptapeNodes.StaticFilesManager().static_server_base_url
                    dest = self._output_file.build_file()
                    # resolve() fails for CREATE_NEW situations because _index is only
                    # computed at write time. Call the base-class write_bytes directly to
                    # allocate the indexed path without the ProjectFileDestination macro
                    # mapping — that gives us the raw absolute path we need for the URL.
                    placeholder = FileDestination.write_bytes(dest, b"")
                    abs_path = Path(placeholder.location)
                    workspace = GriptapeNodes.ConfigManager().workspace_path
                    rel_path = str(abs_path.relative_to(workspace))
                    upload_url = f"{server_url}/static-uploads/{rel_path}"
                    map_result = GriptapeNodes.handle_request(
                        AttemptMapAbsolutePathToProjectRequest(absolute_path=abs_path)
                    )
                    artifact_url = (
                        map_result.mapped_path
                        if isinstance(map_result, AttemptMapAbsolutePathToProjectResultSuccess)
                        and map_result.mapped_path
                        else str(abs_path)
                    )
                    ready = {
                        "state": "upload_ready",
                        "_uploadUrl": upload_url,
                        "_artifactUrl": artifact_url,
                        "_emitSeq": value.get("_emitSeq", 0),
                    }
                    self.set_parameter_value("recording", ready)
                    return

                case "accepted":
                    if value.get("url"):
                        artifact = VideoUrlArtifact(value["url"])
                        self.parameter_output_values["output_video"] = artifact
                        self.publish_update_to_parameter("output_video", artifact)
                        processed = {"state": "processed", "url": artifact.value, "_emitSeq": value.get("_emitSeq", 0)}
                        self.set_parameter_value("recording", processed)
                        return

        return super().after_value_set(parameter, value)

    def process(self) -> None:
        recording = self.get_parameter_value("recording")
        if recording and recording.get("state") == "processed" and recording.get("url"):
            self.parameter_output_values["output_video"] = VideoUrlArtifact(recording["url"])
            return
        msg = "No video recorded. Use the widget to capture and accept a recording before running."
        raise ValueError(msg)
