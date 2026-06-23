from typing import Any

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterGroup, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File

from griptape_nodes_library.utils import ffmpeg_utils


class GetVideoMetadata(DataNode):
    def __init__(
        self,
        name: str,
        metadata: dict[Any, Any] | None = None,
        value: Any = None,
    ) -> None:
        super().__init__(name, metadata)

        # Input parameter
        self.add_parameter(
            ParameterVideo(
                name="video",
                default_value=value,
                tooltip="The video to analyse for metadata",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        # File Details Parameter Group
        with ParameterGroup(name="File Details") as file_group:
            ParameterString(
                name="codec_name",
                default_value=None,
                tooltip="Video codec name (e.g., h264, vp8)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterString(
                name="codec_type",
                default_value=None,
                tooltip="Stream type (always 'video' for video streams)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterFloat(
                name="optional_duration",
                default_value=None,
                tooltip="Video duration in seconds (may not be available)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterInt(
                name="optional_bit_rate",
                default_value=None,
                tooltip="Video bitrate (may not be available)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterString(
                name="optional_codec_long_name",
                default_value=None,
                tooltip="Full codec name (may not be available)",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="Will be filled if available",
            )
        self.add_node_element(file_group)

        # Dimensions Parameter Group
        with ParameterGroup(name="Dimensions") as dimensions_group:
            ParameterInt(
                name="width",
                default_value=None,
                tooltip="Video width in pixels",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterInt(
                name="height",
                default_value=None,
                tooltip="Video height in pixels",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterFloat(
                name="aspect_ratio_decimal",
                default_value=None,
                tooltip="Aspect ratio as decimal (width/height)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterString(
                name="aspect_ratio_string",
                default_value=None,
                tooltip="Aspect ratio as string (e.g., '16:9')",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="Will be filled if available",
            )
            ParameterInt(
                name="optional_coded_width",
                default_value=None,
                tooltip="Coded width (may differ from display width)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterInt(
                name="optional_coded_height",
                default_value=None,
                tooltip="Coded height (may differ from display height)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        self.add_node_element(dimensions_group)

        # Color Details Parameter Group
        with ParameterGroup(name="Color Details", ui_options={"collapsed": True}) as color_group:
            ParameterString(
                name="optional_color_space",
                default_value=None,
                tooltip="Color space (e.g., bt709, bt2020)",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="Will be filled if available",
            )
            ParameterString(
                name="optional_color_transfer",
                default_value=None,
                tooltip="Color transfer characteristic",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="Will be filled if available",
            )
            ParameterString(
                name="optional_color_primaries",
                default_value=None,
                tooltip="Color primaries",
                allowed_modes={ParameterMode.OUTPUT},
                placeholder_text="Will be filled if available",
            )
        self.add_node_element(color_group)

        # Frame Details Parameter Group
        with ParameterGroup(name="Frame Details", ui_options={"collapsed": True}) as frame_group:
            ParameterFloat(
                name="frame_rate",
                default_value=None,
                tooltip="Playback frame rate as decimal (e.g., 29.97). Selected from avg_frame_rate when valid, else r_frame_rate, else nb_frames/duration.",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterString(
                name="r_frame_rate",
                default_value=None,
                tooltip="Raw ffprobe r_frame_rate fraction (e.g., '60/1'). Lowest timing-grid rate; not necessarily playback rate.",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterString(
                name="avg_frame_rate",
                default_value=None,
                tooltip="Raw ffprobe avg_frame_rate fraction (e.g., '30000/1001'). Average playback rate; '0/0' on live or fragmented streams.",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterInt(
                name="optional_nb_frames",
                default_value=None,
                tooltip="Total number of frames (may not be available)",
                allowed_modes={ParameterMode.OUTPUT},
            )
            ParameterFloat(
                name="optional_start_time",
                default_value=None,
                tooltip="Start time offset in seconds",
                allowed_modes={ParameterMode.OUTPUT},
            )
        self.add_node_element(frame_group)

    def _get_video_url(self, video_input: Any) -> str:
        """Extract video URL from VideoUrlArtifact.

        The ParameterVideo converter should have already normalised string inputs
        to VideoUrlArtifact before this method is called.
        """
        if isinstance(video_input, VideoUrlArtifact):
            return File(video_input.value).resolve()

        if hasattr(video_input, "value") and not isinstance(video_input, str):
            return File(video_input.value).resolve()

        msg = f"Unsupported video input type: {type(video_input)}. Expected VideoUrlArtifact (ParameterVideo should convert strings automatically)."
        raise ValueError(msg)

    def _set_metadata_output_values(self, metadata: ffmpeg_utils.VideoMetadata) -> None:
        """Set output values from extracted metadata."""
        # File Details
        self.parameter_output_values["codec_name"] = metadata.file_details.codec_name
        self.parameter_output_values["codec_type"] = metadata.file_details.codec_type
        self.parameter_output_values["optional_duration"] = metadata.file_details.optional_duration
        self.parameter_output_values["optional_bit_rate"] = metadata.file_details.optional_bit_rate
        self.parameter_output_values["optional_codec_long_name"] = metadata.file_details.optional_codec_long_name

        # Dimensions
        self.parameter_output_values["width"] = metadata.dimensions.width
        self.parameter_output_values["height"] = metadata.dimensions.height
        self.parameter_output_values["aspect_ratio_decimal"] = metadata.dimensions.aspect_ratio_decimal
        self.parameter_output_values["aspect_ratio_string"] = metadata.dimensions.aspect_ratio_string
        self.parameter_output_values["optional_coded_width"] = metadata.dimensions.optional_coded_width
        self.parameter_output_values["optional_coded_height"] = metadata.dimensions.optional_coded_height

        # Color Details
        self.parameter_output_values["optional_color_space"] = metadata.color_details.optional_color_space
        self.parameter_output_values["optional_color_transfer"] = metadata.color_details.optional_color_transfer
        self.parameter_output_values["optional_color_primaries"] = metadata.color_details.optional_color_primaries

        # Frame Details
        self.parameter_output_values["frame_rate"] = metadata.frame_details.frame_rate
        self.parameter_output_values["r_frame_rate"] = metadata.frame_details.r_frame_rate
        self.parameter_output_values["avg_frame_rate"] = metadata.frame_details.avg_frame_rate
        self.parameter_output_values["optional_nb_frames"] = metadata.frame_details.optional_nb_frames
        self.parameter_output_values["optional_start_time"] = metadata.frame_details.optional_start_time

    def _set_default_output_values(self) -> None:
        """Set all output values to their defaults when no valid video input."""
        self.parameter_output_values["codec_name"] = "unknown"
        self.parameter_output_values["codec_type"] = "video"
        self.parameter_output_values["optional_duration"] = None
        self.parameter_output_values["optional_bit_rate"] = None
        self.parameter_output_values["optional_codec_long_name"] = None

        self.parameter_output_values["width"] = 0
        self.parameter_output_values["height"] = 0
        self.parameter_output_values["aspect_ratio_decimal"] = 0.0
        self.parameter_output_values["aspect_ratio_string"] = "0:0"
        self.parameter_output_values["optional_coded_width"] = None
        self.parameter_output_values["optional_coded_height"] = None

        self.parameter_output_values["optional_color_space"] = None
        self.parameter_output_values["optional_color_transfer"] = None
        self.parameter_output_values["optional_color_primaries"] = None

        self.parameter_output_values["frame_rate"] = 0.0
        self.parameter_output_values["r_frame_rate"] = "0/0"
        self.parameter_output_values["avg_frame_rate"] = "0/0"
        self.parameter_output_values["optional_nb_frames"] = None
        self.parameter_output_values["optional_start_time"] = None

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        """Extract metadata when video parameter is set."""
        if parameter.name == "video":
            if not value:
                self._set_default_output_values()
            else:
                video_url = self._get_video_url(value)
                metadata = ffmpeg_utils.extract_video_metadata_structured(video_url)
                self._set_metadata_output_values(metadata)

    def process(self) -> None:
        """Ensures metadata extraction runs regardless of when parameter was set."""
        video_input = self.get_parameter_value("video")

        if not video_input:
            self._set_default_output_values()
        else:
            video_url = self._get_video_url(video_input)
            metadata = ffmpeg_utils.extract_video_metadata_structured(video_url)
            self._set_metadata_output_values(metadata)
