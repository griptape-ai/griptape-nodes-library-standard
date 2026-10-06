"""Base class for image generation configuration nodes."""

from typing import Any

from griptape_nodes.exe_types.core_types import Parameter
from griptape_nodes.traits.options import Options

from griptape_nodes_library.config.base_driver import BaseDriver


class BaseImageDriver(BaseDriver):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        driver_parameter = self.get_parameter_by_name("driver")
        if driver_parameter is not None:
            driver_parameter.name = "image_model_config"
            driver_parameter.output_type = "Image Generation Driver"
            driver_parameter._ui_options = {"display_name": "image model config"}

        self.add_parameter(
            Parameter(
                name="message",
                type="str",
                default_value="⚠️ This node requires an API key to function.",
                tooltip="",
                allowed_modes={},  # type: ignore  # noqa: PGH003
                ui_options={"is_full_width": True, "multiline": True, "hide": True},
            )
        )
        # Parameter for model selection. Subclasses call `_install_model_access`
        # to offer a license-filtered dropdown of their models.
        self.add_parameter(
            Parameter(
                name="model",
                input_types=["str"],
                type="str",
                output_type="str",
                default_value="",
                tooltip="Select the model you want to use from the available options.",
            )
        )
        self.add_parameter(
            Parameter(
                name="image_size",
                type="str",
                default_value="",
                tooltip="Select the size of the generated image.",
                traits={Options(choices=[])},
            )
        )

    def _get_common_driver_args(self, params: dict[str, Any]) -> dict[str, Any]:
        driver_args = {"model": params.get("model"), "image_size": params.get("image_size")}

        return driver_args
