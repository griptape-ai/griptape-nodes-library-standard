from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_types.parameter_three_d import Parameter3D


class LoadGLTF(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Need to define the category
        self.category = "3D"
        self.description = "Load a GLTF file"
        gltf_parameter = Parameter3D(
            name="gltf",
            default_value=None,
            ui_options={
                "clickable_file_browser": True,
                "expander": True,
                "show_snapshot_button": True,
            },
            tooltip="The GLTF file that has been loaded.",
        )
        self.add_parameter(gltf_parameter)

        image_parameter = ParameterImage(
            name="image",
            default_value=None,
            tooltip="The image of the GLTF file.",
            allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
        )
        self.add_parameter(image_parameter)

    def after_value_set(
        self,
        parameter: Parameter,
        value: Any,
    ) -> None:
        if parameter.name == "gltf":
            image_url = value.get("metadata", {}).get("imageUrl")
            if image_url:
                image_artifact = ImageUrlArtifact(value=image_url)
                self.set_parameter_value("image", image_artifact)
                self.parameter_output_values["image"] = image_artifact

        return super().after_value_set(parameter, value)

    def process(self) -> None:
        from griptape_nodes_library.utils.gltf_utils import dict_to_gltf_url_artifact

        gltf = self.get_parameter_value("gltf")
        image = self.get_parameter_value("image")

        if isinstance(gltf, dict):
            gltf_artifact = dict_to_gltf_url_artifact(gltf)
        else:
            gltf_artifact = gltf

        self.parameter_output_values["image"] = image
        self.parameter_output_values["gltf"] = gltf_artifact
