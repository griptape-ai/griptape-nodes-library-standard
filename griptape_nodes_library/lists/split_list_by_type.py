from typing import Any

from griptape.artifacts import AudioArtifact, ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import (
    Parameter,
    ParameterMode,
)
from griptape_nodes.exe_types.node_types import ControlNode

from griptape_nodes_library.utils.audio_utils import is_audio_url_artifact
from griptape_nodes_library.utils.video_utils import is_video_url_artifact


class SplitListByType(ControlNode):
    """SplitListByType Node that routes a mixed list into one output list per media type.

    Multi-modal generation nodes take each media type on its own parameter (images on one,
    audio on another), but a list assembled at runtime can hold a mix of them. This node sits
    between the two: give it one mixed list and wire each typed output where it belongs.

    Items that match none of the typed outputs go to `other`, so nothing is silently dropped.
    """

    def __init__(self, name: str, metadata: dict[Any, Any] | None = None) -> None:
        super().__init__(name, metadata)

        self.items = Parameter(
            name="items",
            tooltip="Mixed list of items to route by type",
            input_types=["list"],
            allowed_modes={ParameterMode.INPUT},
        )
        self.add_parameter(self.items)

        self.images = Parameter(
            name="images",
            tooltip="Items that are images",
            output_type="list[ImageUrlArtifact]",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.images)

        self.audio = Parameter(
            name="audio",
            tooltip="Items that are audio",
            output_type="list[AudioUrlArtifact]",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.audio)

        self.video = Parameter(
            name="video",
            tooltip="Items that are video",
            output_type="list[VideoUrlArtifact]",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.video)

        self.text = Parameter(
            name="text",
            tooltip="Items that are text",
            output_type="list[str]",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.text)

        self.other = Parameter(
            name="other",
            tooltip="Items that did not match any of the typed outputs",
            output_type="list",
            allowed_modes={ParameterMode.OUTPUT},
        )
        self.add_parameter(self.other)

    def process(self) -> None:
        items = self.get_parameter_value("items")
        if not isinstance(items, list):
            items = []

        images = []
        audio = []
        video = []
        text = []
        other = []

        for item in items:
            # Checked in order of likelihood. Audio and video need predicate helpers rather than
            # isinstance, because those artifacts are defined per-library and are not a single type.
            match item:
                case ImageUrlArtifact() | ImageArtifact():
                    images.append(item)
                case _ if is_video_url_artifact(item):
                    video.append(item)
                case AudioArtifact():
                    audio.append(item)
                case _ if is_audio_url_artifact(item):
                    audio.append(item)
                case str():
                    text.append(item)
                case _:
                    other.append(item)

        self.parameter_output_values["images"] = images
        self.parameter_output_values["audio"] = audio
        self.parameter_output_values["video"] = video
        self.parameter_output_values["text"] = text
        self.parameter_output_values["other"] = other
