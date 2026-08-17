import base64

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
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
            default_filename="webcam_capture.webm",
        )
        self._output_file.add_parameter()

    def after_value_set(self, parameter, value):
        if parameter.name == "recording" and isinstance(value, dict):
            if value.get("state") == "accepted" and value.get("value"):
                artifact = self._save_recording(value)
                self.parameter_output_values["output_video"] = artifact
                self.publish_update_to_parameter("output_video", artifact)
                # Replace stored recording with a slim reference so the workflow
                # never serialises the base64 blob, and signal the JS overlay to hide.
                processed = {"state": "processed", "url": artifact.value, "_emitSeq": value.get("_emitSeq", 0)}
                self.set_parameter_value("recording", processed)
                self.publish_update_to_parameter("recording", processed)
        return super().after_value_set(parameter, value)

    def _save_recording(self, recording: dict) -> VideoUrlArtifact:
        raw = recording["value"]
        if "base64," in raw:
            raw = raw.split("base64,")[1]
        video_bytes = base64.b64decode(raw)
        dest = self._output_file.build_file()
        saved = dest.write_bytes(video_bytes)
        return VideoUrlArtifact(saved.location)

    def process(self) -> None:
        recording = self.get_parameter_value("recording")
        if recording and recording.get("state") == "processed" and recording.get("url"):
            self.parameter_output_values["output_video"] = VideoUrlArtifact(recording["url"])
            return
        msg = "No video recorded. Use the widget to capture and accept a recording before running."
        raise ValueError(msg)
