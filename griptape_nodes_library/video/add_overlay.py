from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any, ClassVar
from urllib.parse import urlparse

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import ParameterMode
from griptape_nodes.exe_types.param_types.parameter_float import ParameterFloat
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_types.parameter_string import ParameterString
from griptape_nodes.exe_types.param_types.parameter_video import ParameterVideo
from griptape_nodes.files.file import File
from griptape_nodes.traits.options import Options
from griptape_nodes.traits.slider import Slider

from griptape_nodes_library.media import resolve_media_location
from griptape_nodes_library.video.base_video_processor import BaseVideoProcessor


class AlphaConvention(StrEnum):
    """Whether the overlay's colour channels have been scaled by its own alpha."""

    STRAIGHT = "straight"
    PREMULTIPLIED = "premultiplied"


class AddOverlay(BaseVideoProcessor):
    """Add an overlay video or image on top of a base video using FFmpeg's blend modes.

    This node supports various blend modes that match industry-standard compositing software.
    It can overlay videos, images (PNG with transparency), and supports different channels
    and positioning options.

    Key Features:
    - Multiple blend modes (overlay, screen, grainmerge, etc.)
    - Overlay transparency is respected by every blend mode
    - Optional matte input that supplies the overlay's alpha
    - Straight or premultiplied alpha for overlays that carry their own alpha
    - Choke and edge softness to trim a contaminated soft edge
    - Either input can be a still image: the other input's video sets the length of the result
    - Sizing options (scale to cover/fit)

    Common Use Cases:
    - Adding film grain/noise effects
    - Overlaying logos or watermarks
    - Adding dust/scratch effects
    - Compositing multiple video layers
    """

    # Blend mode constants - matching Nuke's standard blend modes
    BLEND_MODES: ClassVar[list[str]] = [
        "overlay",  # Original overlay with alpha blending
        "screen",  # Brighten (good for dust/scratches)
        "lighten",  # Only brighten where overlay is brighter
        "softlight",  # Natural blending for film grain
        "grainmerge",  # Merge grain/noise (recommended for noise effects)
        "grainextract",  # Extract grain/noise
        "glow",  # Glow effect
        "hardlight",  # Hard light blending
    ]

    # Amount constants
    MIN_AMOUNT = 0.0
    MAX_AMOUNT = 1.0
    DEFAULT_AMOUNT = 1.0

    # Locations with these extensions are still images, not videos
    STILL_IMAGE_EXTENSIONS: ClassVar[frozenset[str]] = frozenset(
        {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    )

    # Resolved location of the base input when it arrived as an image artifact. An image URL need not
    # have an image extension, so the artifact type is the reliable signal and the extension is the fallback.
    _still_base_url: str | None = None

    # Edge cleanup limits
    MIN_CHOKE = 0
    MAX_CHOKE = 10
    MIN_EDGE_SOFTNESS = 0.0
    MAX_EDGE_SOFTNESS = 10.0

    # Sizing options
    SIZING_OPTIONS: ClassVar[list[str]] = [
        "Scale to cover",  # Scale to cover (maintains aspect ratio)
        "Scale to fit",  # Scale to fit (doesn't maintain aspect ratio)
    ]

    def _setup_custom_parameters(self) -> None:
        """Setup overlay parameters."""
        base_param = self.get_parameter_by_name("video")
        if base_param is not None:
            base_param.tooltip = (
                "The base video, or a still image to composite a video overlay onto. "
                "With an image, the overlay video sets the length and frame rate of the result."
            )
        overlay_param = ParameterVideo(
            name="overlay_video",
            input_types=["VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip="The video or image to overlay on top of the base video (PNG with transparency supported)",
            hide_property=True,
        )
        self.add_parameter(overlay_param)

        matte_param = ParameterVideo(
            name="matte_video",
            input_types=["VideoArtifact", "VideoUrlArtifact", "ImageArtifact", "ImageUrlArtifact"],
            type="VideoUrlArtifact",
            tooltip=(
                "Optional matte. Its brightness (white = visible, black = transparent) replaces the overlay's "
                "alpha. It is stretched to the overlay's size."
            ),
            hide_property=True,
        )
        self.add_parameter(matte_param)

        alpha_param = ParameterString(
            name="alpha_convention",
            default_value=AlphaConvention.STRAIGHT,
            tooltip=(
                "Alpha convention of the overlay's own alpha channel. Most generators emit straight alpha; "
                "renderers and keyers often emit premultiplied. Choosing the wrong one fringes soft edges: "
                "dark if premultiplied is treated as straight, bright the other way. A matte replaces the "
                "overlay's alpha, so this only matters for the alpha the overlay carries."
            ),
            allowed_modes={ParameterMode.PROPERTY},
            traits={Options(choices=list(AlphaConvention))},
        )
        self.add_parameter(alpha_param)

        blend_mode_param = ParameterString(
            name="blend_mode",
            default_value="overlay",
            tooltip="Blend mode for combining the overlay with the base video",
        )
        blend_mode_param.add_trait(Options(choices=self.BLEND_MODES))
        self.add_parameter(blend_mode_param)

        amount_param = ParameterFloat(
            name="amount",
            default_value=self.DEFAULT_AMOUNT,
            tooltip=f"Strength of the overlay effect ({self.MIN_AMOUNT}-{self.MAX_AMOUNT})",
        )
        amount_param.add_trait(Slider(min_val=self.MIN_AMOUNT, max_val=self.MAX_AMOUNT))
        self.add_parameter(amount_param)

        choke_param = ParameterInt(
            name="choke",
            default_value=self.MIN_CHOKE,
            tooltip=(
                "Shrinks the overlay's alpha inward by this many pixels. Soft edges often carry color from "
                "the footage's original background, which shows as a halo on a new background. Choking "
                "removes the most contaminated edge pixels. 0 leaves the edge alone."
            ),
        )
        choke_param.add_trait(Slider(min_val=self.MIN_CHOKE, max_val=self.MAX_CHOKE))
        self.add_parameter(choke_param)

        edge_softness_param = ParameterFloat(
            name="edge_softness",
            default_value=self.MIN_EDGE_SOFTNESS,
            tooltip="Blurs the overlay's alpha by this many pixels after choking, so a choked edge does not look hard.",
        )
        edge_softness_param.add_trait(Slider(min_val=self.MIN_EDGE_SOFTNESS, max_val=self.MAX_EDGE_SOFTNESS))
        self.add_parameter(edge_softness_param)

        sizing_param = ParameterString(
            name="sizing",
            default_value="Scale to cover",
            tooltip="How to size the overlay: Scale to cover (maintains aspect), Scale to fit (stretches)",
        )
        sizing_param.add_trait(Options(choices=self.SIZING_OPTIONS))
        self.add_parameter(sizing_param)

    def _get_processing_description(self) -> str:
        """Get description of what this processor does."""
        return "video overlay"

    def _build_ffmpeg_command(self, input_url: str, output_path: str, input_frame_rate: float, **kwargs) -> list[str]:
        """Build FFmpeg command for video overlay."""
        overlay_video = kwargs.get("overlay_video")
        blend_mode = kwargs.get("blend_mode", "overlay")
        amount = kwargs.get("amount", self.DEFAULT_AMOUNT)

        if not overlay_video:
            # Get processing speed settings
            preset, pix_fmt, crf = self._get_processing_speed_settings()

            # Get ffmpeg executable path
            ffmpeg_path, _ = self._get_ffmpeg_paths()

            # No overlay video provided, just copy the input
            return [
                ffmpeg_path,
                "-i",
                input_url,
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
                "aac",
                "-b:a",
                "192k",
                "-y",
                output_path,
            ]

        overlay_url = self._resolve_overlay_location(overlay_video)

        matte_video = kwargs.get("matte_video")
        matte_url = self._resolve_overlay_location(matte_video, parameter_name="matte_video") if matte_video else None
        alpha_convention = kwargs.get("alpha_convention", AlphaConvention.STRAIGHT)

        # Get base video duration for proper trimming
        ffmpeg_path, ffprobe_path = self._get_ffmpeg_paths()
        _, (base_width, base_height), base_duration = self._detect_video_properties(input_url, ffprobe_path)

        base_input_options: list[str] = []
        if input_url == self._still_base_url or self._is_still_image(input_url):
            if self._is_still_input(overlay_video, overlay_url):
                msg = "At least one of the base and the overlay must be a video, so the result has a length."
                raise ValueError(msg)
            # A still base has no length of its own, so it takes the overlay video's length and frame rate.
            overlay_frame_rate, _, base_duration = self._detect_video_properties(overlay_url, ffprobe_path)
            if base_duration <= 0:
                msg = "Could not read the overlay video's length, which the still image base needs."
                raise ValueError(msg)
            input_frame_rate = overlay_frame_rate
            base_input_options = ["-loop", "1", "-framerate", str(overlay_frame_rate), "-t", str(base_duration)]

        # The overlay filter accepts any overlay size, so overlay mode always centers the overlay unscaled
        overlay_position = "(W-w)/2:(H-h)/2"

        # Inputs: [0] = overlay, [1] = base video, [2] = matte (only when connected).
        # Every blend mode composites from one straight-RGBA overlay stream, so all of them honour the
        # overlay's transparency. That stream is built in this order: straight RGBA, size (blend modes
        # only), matte alpha if there is one, loop and trim to the base duration, then alpha scaled by amount.
        graph = ["[0]format=rgba[src]"]
        source = "src"

        match alpha_convention:
            case AlphaConvention.STRAIGHT:
                pass
            case AlphaConvention.PREMULTIPLIED:
                # FFmpeg's unpremultiply filter leaves the colour untouched on ProRes input, so divide
                # the colour planes by the alpha plane with the blend filter and put the alpha back.
                graph.append(
                    "[src]split=3[pm_color][pm_alpha_divisor][pm_alpha_keep];"
                    "[pm_color]format=gbrp[pm_rgb];"
                    "[pm_alpha_divisor]alphaextract,format=gbrp[pm_divisor];"
                    "[pm_alpha_keep]alphaextract[pm_alpha];"
                    "[pm_rgb][pm_divisor]blend=all_mode=divide[pm_straight];"
                    "[pm_straight][pm_alpha]alphamerge,format=rgba[unpremultiplied]"
                )
                source = "unpremultiplied"
            case _:
                msg = f"Unknown alpha convention: {alpha_convention!r}"
                raise ValueError(msg)

        if blend_mode == "overlay":
            # The overlay filter accepts any overlay size, so the overlay stays at its own size.
            match_size_filter = ""
            if matte_url:
                _, (overlay_width, overlay_height), _ = self._detect_video_properties(overlay_url, ffprobe_path)
                match_size_filter = f"scale={overlay_width}:{overlay_height},"
        else:
            # The blend filter requires both inputs to share size and pixel format, so scale the overlay
            # (and its matte, the same way) to the base video.
            sizing_filter = self._get_sizing_filter(kwargs.get("sizing", "Scale to cover"), base_width, base_height)
            graph.append(f"[{source}]{sizing_filter},setsar=1[sized]")
            source = "sized"
            match_size_filter = f"{sizing_filter},setsar=1,"

        if matte_url:
            # scale2ref would size the matte to the overlay without probing, but FFmpeg 7.0 aborts on it
            # when an input is a still image, so size it explicitly.
            graph.append(f"[2]{match_size_filter}format=gray[matte];[{source}][matte]alphamerge,format=rgba[matted]")
            source = "matted"

        choke = int(kwargs.get("choke") or 0)
        edge_softness = float(kwargs.get("edge_softness") or 0.0)
        if choke > 0 or edge_softness > 0:
            # Rebuild the alpha from itself: erode it (each pass is a 3x3 minimum, so one pixel), then blur
            # it, and put it back on the unchanged colour.
            alpha_steps = ",".join(
                ["alphaextract", *["erosion"] * choke, *([f"gblur=sigma={edge_softness}"] if edge_softness > 0 else [])]
            )
            graph.append(
                f"[{source}]split[edge_color][edge_alpha_src];"
                f"[edge_alpha_src]{alpha_steps}[edge_alpha];"
                "[edge_color][edge_alpha]alphamerge,format=rgba[edged]"
            )
            source = "edged"

        graph.append(
            f"[{source}]loop=loop=-1,trim=duration={base_duration},format=rgba,colorchannelmixer=aa={amount}[fg]"
        )

        if blend_mode == "overlay":
            graph.append(f"[1][fg]overlay={overlay_position}[out]")
        else:
            # Blend the colour, then mix the blended result back over the base using the overlay alpha
            graph.append(
                "[fg]split[fg_color_src][fg_alpha_src];"
                "[fg_color_src]format=gbrp[fg_color];"
                "[fg_alpha_src]alphaextract[fg_alpha];"
                "[1]setsar=1,format=gbrp,split[bg_blend][bg_keep];"
                f"[bg_blend][fg_color]blend=all_mode={blend_mode}:all_opacity=1[blended];"
                "[bg_keep][blended][fg_alpha]maskedmerge[out]"
            )

        custom_filter = ";".join(graph)

        # Add frame rate filter if needed (after the overlay effect)
        frame_rate_filter = self._get_frame_rate_filter(input_frame_rate)
        if frame_rate_filter:
            # For complex filters, we need to add the frame rate filter to the output
            custom_filter = custom_filter.replace("[out]", "[out_temp]")
            custom_filter = f"{custom_filter};[out_temp]{frame_rate_filter}[out]"

        filter_complex = custom_filter

        # Get processing speed settings
        preset, pix_fmt, crf = self._get_processing_speed_settings()

        # Get ffmpeg executable path
        ffmpeg_path, _ = self._get_ffmpeg_paths()

        matte_inputs = ["-i", matte_url] if matte_url else []  # Matte - [2]

        return [
            ffmpeg_path,
            "-i",
            overlay_url,  # Overlay video (film grain) - [0]
            *base_input_options,
            "-i",
            input_url,  # Base video (main content) - [1]
            *matte_inputs,
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-shortest",  # End when shortest stream ends
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
            "aac",
            "-b:a",
            "192k",
            "-y",
            output_path,
        ]

    def _is_still_image(self, location: str) -> bool:
        """Whether a resolved path or URL points at a still image rather than a video."""
        if location.startswith("data:"):
            return location.startswith("data:image/")
        return PurePosixPath(urlparse(location).path).suffix.lower() in self.STILL_IMAGE_EXTENSIONS

    @staticmethod
    def _holds_still_image(value: Any) -> bool:
        """Whether an input value is an image artifact, or the serialized dict of one."""
        if isinstance(value, ImageUrlArtifact | ImageArtifact):
            return True
        return isinstance(value, dict) and value.get("type") in {"ImageUrlArtifact", "ImageArtifact"}

    def _is_still_input(self, value: Any, location: str) -> bool:
        """Whether an input is a still image: by its artifact type, else by its location's extension."""
        return self._holds_still_image(value) or self._is_still_image(location)

    def _validate_video_input(self) -> list[Exception] | None:
        """Accept an image as well as a video for the base input."""
        base = self.parameter_values.get("video")
        if isinstance(base, ImageUrlArtifact | ImageArtifact) and isinstance(base.value, str) and base.value:
            return None
        return super()._validate_video_input()

    def _get_video_input_data(self) -> tuple[str, str]:
        """Resolve the base input; a still image still produces an mp4."""
        base = self.parameter_values.get("video")
        if isinstance(base, ImageUrlArtifact | ImageArtifact) and isinstance(base.value, str):
            self._still_base_url = File(base.value).resolve()
            return self._still_base_url, "mp4"
        self._still_base_url = None
        input_url, detected_format = super()._get_video_input_data()
        if self._is_still_image(input_url):
            detected_format = "mp4"
        return input_url, detected_format

    def _check_read_codec_permission(self, input_url: str) -> None:
        """Skip the video codec permission check for a still image."""
        if input_url == self._still_base_url or self._is_still_image(input_url):
            return
        super()._check_read_codec_permission(input_url)

    def _check_extra_input_permissions(self, **kwargs: Any) -> None:
        """Ask the engine whether reading the overlay and matte videos is permitted.

        The base class only checks the base input, but FFmpeg also decodes these two.
        """
        for parameter_name in ("overlay_video", "matte_video"):
            value = kwargs.get(parameter_name)
            if not value:
                continue
            location = self._resolve_overlay_location(value, parameter_name=parameter_name)
            if not self._is_still_input(value, location):
                self._check_read_codec_permission(location)

    def _process_video(self, input_url: str, output_path: str, **kwargs) -> None:
        """Check the overlay and matte reads before the base class checks the base and runs FFmpeg."""
        try:
            self._check_extra_input_permissions(**kwargs)
        except Exception as e:
            self.append_value_to_parameter("logs", f"ERROR: Error during video processing: {e!s}\n")
            raise
        super()._process_video(input_url, output_path, **kwargs)

    def _get_sizing_filter(self, sizing: str, width: int, height: int) -> str:
        """Return an FFmpeg filter that sizes the overlay to exactly width x height."""
        match sizing:
            case "Scale to cover":
                return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"
            case "Scale to fit":
                return f"scale={width}:{height}"
            case _:
                msg = f"Unknown sizing option: {sizing!r}"
                raise ValueError(msg)

    def _resolve_overlay_location(self, overlay_video: Any, parameter_name: str = "overlay_video") -> str:
        """Return a path or URL FFmpeg can open for the overlay input.

        Project-saved artifacts carry macro paths like ``{outputs}/images/foo.png``,
        which FFmpeg can't open until they are resolved.

        Raises:
            ValueError: If the input has no string location (e.g. a raw byte artifact).
            FileLoadError: If macro resolution fails.
        """
        if isinstance(overlay_video, dict):
            location = overlay_video.get("value")
        else:
            location = getattr(overlay_video, "value", None)
        if not isinstance(location, str) or not location:
            msg = f"'{parameter_name}' must reference a file or URL, got {type(overlay_video).__name__}."
            raise ValueError(msg)  # noqa: TRY004
        return resolve_media_location(location)

    def _validate_custom_parameters(self) -> list[Exception] | None:
        """Validate overlay parameters."""
        exceptions = []

        blend_mode = self.get_parameter_value("blend_mode")
        if blend_mode is not None and blend_mode not in self.BLEND_MODES:
            msg = f"Blend mode must be one of {self.BLEND_MODES}, got {blend_mode}."
            exceptions.append(ValueError(msg))

        amount = self.get_parameter_value("amount")
        if amount is not None and (amount < self.MIN_AMOUNT or amount > self.MAX_AMOUNT):
            msg = f"Amount must be between {self.MIN_AMOUNT} and {self.MAX_AMOUNT}, got {amount}."
            exceptions.append(ValueError(msg))

        sizing = self.get_parameter_value("sizing")
        if sizing is not None and sizing not in self.SIZING_OPTIONS:
            msg = f"Sizing must be one of {self.SIZING_OPTIONS}, got {sizing}."
            exceptions.append(ValueError(msg))

        choke = self.get_parameter_value("choke")
        if choke is not None and (choke < self.MIN_CHOKE or choke > self.MAX_CHOKE):
            msg = f"Choke must be between {self.MIN_CHOKE} and {self.MAX_CHOKE}, got {choke}."
            exceptions.append(ValueError(msg))

        edge_softness = self.get_parameter_value("edge_softness")
        if edge_softness is not None and (
            edge_softness < self.MIN_EDGE_SOFTNESS or edge_softness > self.MAX_EDGE_SOFTNESS
        ):
            msg = f"Edge softness must be between {self.MIN_EDGE_SOFTNESS} and {self.MAX_EDGE_SOFTNESS}, got {edge_softness}."
            exceptions.append(ValueError(msg))

        alpha_convention = self.get_parameter_value("alpha_convention")
        if alpha_convention is not None and alpha_convention not in list(AlphaConvention):
            msg = f"Alpha convention must be one of {[str(c) for c in AlphaConvention]}, got {alpha_convention}."
            exceptions.append(ValueError(msg))

        return exceptions if exceptions else None

    def _get_custom_parameters(self) -> dict[str, Any]:
        """Get overlay parameters."""
        return {
            "overlay_video": self.get_parameter_value("overlay_video"),
            "matte_video": self.get_parameter_value("matte_video"),
            "alpha_convention": self.get_parameter_value("alpha_convention"),
            "choke": self.get_parameter_value("choke"),
            "edge_softness": self.get_parameter_value("edge_softness"),
            "blend_mode": self.get_parameter_value("blend_mode"),
            "amount": self.get_parameter_value("amount"),
            "sizing": self.get_parameter_value("sizing"),
        }

    def _get_output_suffix(self, **kwargs) -> str:
        """Get output filename suffix."""
        blend_mode = kwargs.get("blend_mode", "overlay")
        amount = kwargs.get("amount", self.DEFAULT_AMOUNT)
        sizing = kwargs.get("sizing", "Scale to cover")

        return f"_overlay_{sizing.replace(' ', '_')}_{blend_mode}_amount{amount:.2f}"
