from typing import Any

from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import ControlNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo

from griptape_nodes_library.traits.compare_videos import CompareVideosTrait


class CompareVideos(ControlNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterVideo(
                name="Video_1",
                tooltip="Video 1",
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterVideo(
                name="Video_2",
                tooltip="Video 2",
                default_value=None,
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="Compare",
                tooltip="Compare two videos",
                default_value={"input_video_1": None, "input_video_2": None},
                allowed_modes={ParameterMode.PROPERTY},
                traits={CompareVideosTrait()},
                ui_options={"video_compare": True},
            )
        )

    def _update_compare(self) -> None:
        """Update the Compare parameter with current videos."""
        video_1 = self.get_parameter_value("Video_1")
        video_2 = self.get_parameter_value("Video_2")

        result_dict = {"input_video_1": video_1, "input_video_2": video_2}

        self.set_parameter_value("Compare", result_dict)
        self.parameter_output_values["Compare"] = result_dict

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name not in {"Video_1", "Video_2"}:
            return super().after_value_set(parameter, value)

        self._update_compare()
        return super().after_value_set(parameter, value)

    def process(self) -> None:
        """Process the node during execution."""
        self._update_compare()
