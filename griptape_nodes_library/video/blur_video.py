from typing import Any

from griptape_nodes.exe_types.core_types import ParameterGroup
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor


class BlurVideo(BaseVideoProcessor):
    """Blur video using fast ffmpeg blur filters (boxblur, gblur, avgblur)."""

    # Blur radius / sigma constants
    MIN_RADIUS = 1
    MAX_RADIUS = 50
    DEFAULT_RADIUS = 5

    MIN_SIGMA = 0.1
    MAX_SIGMA = 50.0
    DEFAULT_SIGMA = 5.0

    # boxblur power (number of times to apply the box filter)
    MIN_POWER = 1
    MAX_POWER = 5
    DEFAULT_POWER = 1

    BLUR_TYPES = ("box", "gaussian", "average")
    DEFAULT_BLUR_TYPE = "box"

    def _setup_custom_parameters(self) -> None:
        """Setup blur-specific parameters."""
        with ParameterGroup(name="blur_settings", ui_options={"collapsed": False}) as blur_group:
            blur_type_parameter = ParameterString(
                name="blur_type",
                default_value=self.DEFAULT_BLUR_TYPE,
                tooltip=(
                    "Blur algorithm: 'box' (boxblur, fastest), 'gaussian' (gblur, smoother),"
                    " or 'average' (avgblur, simple box average)"
                ),
            )
            self.add_parameter(blur_type_parameter)
            blur_type_parameter.add_trait(Options(choices=list(self.BLUR_TYPES)))

            radius_parameter = ParameterInt(
                name="radius",
                default_value=self.DEFAULT_RADIUS,
                tooltip=(
                    f"Blur radius in pixels for box/average blur ({self.MIN_RADIUS}-{self.MAX_RADIUS})."
                    " Ignored when blur_type is 'gaussian'."
                ),
            )
            self.add_parameter(radius_parameter)
            radius_parameter.add_trait(Slider(min_val=self.MIN_RADIUS, max_val=self.MAX_RADIUS))

            sigma_parameter = ParameterFloat(
                name="sigma",
                default_value=self.DEFAULT_SIGMA,
                tooltip=(
                    f"Gaussian blur sigma ({self.MIN_SIGMA}-{self.MAX_SIGMA}). Only used when blur_type is 'gaussian'."
                ),
                hide=True,  # Hidden by default, shown when blur_type is 'gaussian'
            )
            self.add_parameter(sigma_parameter)
            sigma_parameter.add_trait(Slider(min_val=self.MIN_SIGMA, max_val=self.MAX_SIGMA))

            power_parameter = ParameterInt(
                name="power",
                default_value=self.DEFAULT_POWER,
                tooltip=(
                    f"Number of times to apply the box blur filter ({self.MIN_POWER}-{self.MAX_POWER})."
                    " Higher values produce a stronger blur. Only used when blur_type is 'box'."
                ),
            )
            self.add_parameter(power_parameter)
            power_parameter.add_trait(Slider(min_val=self.MIN_POWER, max_val=self.MAX_POWER))

        self.add_node_element(blur_group)

    def after_value_set(self, parameter, value):
        if parameter.name in {"blur_type"}:
            if value == "box":
                self.show_parameter_by_name("radius")
                self.show_parameter_by_name("power")
                self.hide_parameter_by_name("sigma")
            elif value == "gaussian":
                self.hide_parameter_by_name("radius")
                self.hide_parameter_by_name("power")
                self.show_parameter_by_name("sigma")
            elif value == "average":
                self.show_parameter_by_name("radius")
                self.hide_parameter_by_name("power")
                self.hide_parameter_by_name("sigma")
        return super().after_value_set(parameter, value)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "video blur"

    def _build_blur_filter(self, blur_type: str, radius: int, sigma: float, power: int) -> str:
        """Build the ffmpeg blur filter expression for the chosen blur type."""
        if blur_type == "gaussian":
            return f"gblur=sigma={sigma}"
        if blur_type == "average":
            return f"avgblur={radius}"
        # box
        return f"boxblur=luma_radius={radius}:luma_power={power}"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:
        """Build FFmpeg command for video blur."""
        blur_type = kwargs.get("blur_type", self.DEFAULT_BLUR_TYPE)
        radius = kwargs.get("radius", self.DEFAULT_RADIUS)
        sigma = kwargs.get("sigma", self.DEFAULT_SIGMA)
        power = kwargs.get("power", self.DEFAULT_POWER)

        custom_filter = self._build_blur_filter(blur_type, radius, sigma, power)

        # Combine with frame rate filter if needed
        filter_complex = self._combine_video_filters(custom_filter, input_frame_rate)

        # Get processing speed settings
        preset, pix_fmt, crf = self._get_processing_speed_settings()

        # Get ffmpeg executable path
        ffmpeg_path, _ = self._get_ffmpeg_paths()

        return [
            ffmpeg_path,
            "-i",
            input_url,
            "-vf",
            filter_complex,
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            pix_fmt,
            "-movflags",
            "+faststart",
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate blur parameters."""
        exceptions = []

        blur_type = self.get_parameter_value("blur_type")
        if blur_type is not None and blur_type not in self.BLUR_TYPES:
            msg = f"{self.name} - blur_type must be one of {list(self.BLUR_TYPES)}, got {blur_type}"
            exceptions.append(ValueError(msg))

        radius = self.get_parameter_value("radius")
        if radius is not None and (radius < self.MIN_RADIUS or radius > self.MAX_RADIUS):
            msg = f"{self.name} - radius must be between {self.MIN_RADIUS} and {self.MAX_RADIUS}, got {radius}"
            exceptions.append(ValueError(msg))

        sigma = self.get_parameter_value("sigma")
        if sigma is not None and (sigma < self.MIN_SIGMA or sigma > self.MAX_SIGMA):
            msg = f"{self.name} - sigma must be between {self.MIN_SIGMA} and {self.MAX_SIGMA}, got {sigma}"
            exceptions.append(ValueError(msg))

        power = self.get_parameter_value("power")
        if power is not None and (power < self.MIN_POWER or power > self.MAX_POWER):
            msg = f"{self.name} - power must be between {self.MIN_POWER} and {self.MAX_POWER}, got {power}"
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get blur parameters."""
        return {
            "blur_type": self.get_parameter_value("blur_type"),
            "radius": self.get_parameter_value("radius"),
            "sigma": self.get_parameter_value("sigma"),
            "power": self.get_parameter_value("power"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        blur_type = kwargs.get("blur_type", self.DEFAULT_BLUR_TYPE)
        if blur_type == "gaussian":
            sigma = kwargs.get("sigma", self.DEFAULT_SIGMA)
            return f"_blurred_gaussian_s{sigma:.2f}"
        if blur_type == "average":
            radius = kwargs.get("radius", self.DEFAULT_RADIUS)
            return f"_blurred_average_r{radius}"
        radius = kwargs.get("radius", self.DEFAULT_RADIUS)
        power = kwargs.get("power", self.DEFAULT_POWER)
        return f"_blurred_box_r{radius}_p{power}"
