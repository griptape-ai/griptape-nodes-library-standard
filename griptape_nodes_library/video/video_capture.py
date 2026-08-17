import re
from pathlib import Path

from griptape.artifacts import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes
from griptape_nodes.traits.widget import Widget


class VideoCapture(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # Absolute path to the temp upload file; set in requesting_upload_url, consumed in accepted.
        self._pending_upload_path: Path | None = None
        # MIME type of the recorded blob (e.g. "video/webm;codecs=vp9,opus").
        self._pending_mime: str = "video/mp4"

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
                    workspace = GriptapeNodes.ConfigManager().workspace_path
                    # Predictable name: no orphan files accumulate on re-record.
                    # Static server creates temp/ and writes the file on PUT.
                    mime = value.get("_mime", "video/mp4")
                    self._pending_mime = mime
                    ext = ".webm" if mime.startswith("video/webm") else ".mp4"
                    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", self.name)
                    rel_path = f"temp/_vc_{safe_name}{ext}"
                    self._pending_upload_path = workspace / rel_path
                    ready = {
                        "state": "upload_ready",
                        "_uploadUrl": f"{server_url}/static-uploads/{rel_path}",
                        "_emitSeq": value.get("_emitSeq", 0),
                    }
                    self.set_parameter_value("recording", ready)
                    return

                case "accepted":
                    pending = self._pending_upload_path
                    if pending is None or not pending.exists():
                        error = {
                            "state": "error",
                            "message": "Upload not found. Please re-record.",
                            "_emitSeq": value.get("_emitSeq", 0),
                        }
                        self.set_parameter_value("recording", error)
                        return
                    self._pending_upload_path = None
                    data = pending.read_bytes()
                    pending.unlink(missing_ok=True)
                    # Coerce the output filename extension to match the recorded container.
                    # Firefox records webm even when mp4 is the default; updating the
                    # parameter here also corrects what the user sees in the UI.
                    correct_ext = ".webm" if self._pending_mime.startswith("video/webm") else ".mp4"
                    output_val = self.get_parameter_value("output_file") or ""
                    if output_val and Path(output_val).suffix.lower() != correct_ext:
                        self.set_parameter_value("output_file", str(Path(output_val).with_suffix(correct_ext)))
                    saved = self._output_file.build_file().write_bytes(data)
                    artifact = VideoUrlArtifact(saved.location)
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
