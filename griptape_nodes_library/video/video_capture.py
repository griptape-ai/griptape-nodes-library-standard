from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.traits.widget import Widget

from griptape_nodes_library.utils.video_utils import dict_to_video_url_artifact


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

    def after_value_set(self, parameter, value):
        if parameter.name == "recording" and isinstance(value, dict):
            if value.get("state") == "accepted" and value.get("value"):
                artifact = dict_to_video_url_artifact(value)
                self.parameter_output_values["output_video"] = artifact
                self.publish_update_to_parameter("output_video", artifact)
                # Replace stored recording with a slim reference — no blob, just the saved path.
                # This prevents the base64 payload from being serialised into the workflow file.
                processed = {"state": "processed", "url": artifact.value, "_emitSeq": value.get("_emitSeq", 0)}
                self.set_parameter_value("recording", processed)
                self.publish_update_to_parameter("recording", processed)
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        recording = self.get_parameter_value("recording")
        if recording and recording.get("state") == "processed" and recording.get("url"):
            self.parameter_output_values["output_video"] = VideoUrlArtifact(recording["url"])
            return
        msg = "No video recorded. Use the widget to capture and accept a recording before running."
        raise ValueError(msg)
