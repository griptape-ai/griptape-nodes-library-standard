import math
from io import BytesIO
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.widget import Widget
from PIL import Image, ImageDraw, ImageFont

from griptape_nodes_library.utils.color_utils import parse_color_to_rgba


def _default_annotation_data() -> dict:
    return {
        "image_url": "",
        "raw_url": "",
        "canvas_width": 0,
        "canvas_height": 0,
        "annotations": [],
        "active_tool": "select",
        "tool_settings": {
            "paint": {"color": "#ff0000", "size": 8},
            "text": {"color": "#ffffff", "font_size": 48},
            "arrow": {"color": "#ff0000", "width": 3},
        },
        "selected_id": None,
    }


class AnnotateImage(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            Parameter(
                name="image",
                input_types=["ImageUrlArtifact", "ImageArtifact"],
                default_value=None,
                tooltip="Input image to annotate",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="annotation_data",
                default_value=_default_annotation_data(),
                tooltip="Canvas annotations (paint, text, arrows)",
                display_name="Canvas",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="AnnotateImageSimple", library="Griptape Nodes Library")},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageUrlArtifact",
                tooltip="Image with annotations composited",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self, name="output_file", default_filename="annotated.png"
        )
        self._output_file.add_parameter()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _resolve_url(self, artifact: Any) -> tuple[str, str]:
        """Return (raw_path, browser_url) for an image artifact."""
        from griptape_nodes.retained_mode.events.static_file_events import (
            CreateStaticFileDownloadUrlFromPathRequest,
            CreateStaticFileDownloadUrlFromPathResultSuccess,
        )

        raw = getattr(artifact, "value", "") or ""
        if not raw:
            return "", ""
        if raw.startswith(("http://", "https://", "data:")):
            return raw, raw
        try:
            resolved = File(raw).resolve()
        except Exception:
            resolved = str(raw)
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return resolved, result.url
        except Exception:
            pass
        return resolved, raw

    def _get_dimensions(self, raw_path: str) -> tuple[int, int]:
        try:
            data = File(raw_path).read_bytes()
            with Image.open(BytesIO(data)) as img:
                return img.size  # (width, height)
        except Exception:
            return 0, 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if parameter.name == "image" and value:
            raw, browser_url = self._resolve_url(value)
            if not browser_url:
                return super().after_value_set(parameter, value)
            w, h = self._get_dimensions(raw)
            data = self.get_parameter_value("annotation_data") or _default_annotation_data()
            if not isinstance(data, dict):
                data = _default_annotation_data()
            # Preserve all annotations; only refresh image/canvas fields.
            new_data = {
                **data,
                "image_url": browser_url,
                "raw_url": raw,
                "canvas_width": w or data.get("canvas_width", 0),
                "canvas_height": h or data.get("canvas_height", 0),
            }
            self.set_parameter_value("annotation_data", new_data)
            self.publish_update_to_parameter("annotation_data", new_data)
        return super().after_value_set(parameter, value)

    # ── compositing ───────────────────────────────────────────────────────────

    def _parse_color(self, color_str: str, opacity: float = 1.0) -> tuple[int, int, int, int]:
        try:
            r, g, b, _ = parse_color_to_rgba(color_str)
        except Exception:
            r, g, b = 255, 0, 0
        return (r, g, b, int(255 * opacity))

    def _draw_paint(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        for stroke in ann.get("strokes", []):
            points = stroke.get("points", [])
            if not points:
                continue
            color = self._parse_color(stroke.get("color", "#ff0000"))
            base_size = max(1, int(stroke.get("size", 8)))
            for pt in points:
                px, py = pt[0], pt[1]
                sz = max(1, int(pt[2])) if len(pt) > 2 else base_size
                draw.ellipse([px - sz / 2, py - sz / 2, px + sz / 2, py + sz / 2], fill=color)
            for i in range(len(points) - 1):
                x0, y0 = points[i][0], points[i][1]
                x1, y1 = points[i + 1][0], points[i + 1][1]
                sz = max(1, int((points[i][2] + points[i + 1][2]) / 2)) if len(points[i]) > 2 else base_size
                draw.line([x0, y0, x1, y1], fill=color, width=sz)

    def _draw_text(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        text = ann.get("text", "")
        if not text:
            return
        x = float(ann.get("x", 0))
        y = float(ann.get("y", 0))
        font_size = max(8, int(ann.get("font_size", 48)))
        color = self._parse_color(ann.get("color", "#ffffff"))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((x, y), text, font=font, fill=color)

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        x1, y1 = float(ann.get("x1", 0)), float(ann.get("y1", 0))
        x2, y2 = float(ann.get("x2", 0)), float(ann.get("y2", 0))
        color = self._parse_color(ann.get("color", "#ff0000"))
        width = max(1, int(ann.get("width", 3)))
        draw.line([x1, y1, x2, y2], fill=color, width=width)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = max(15, width * 4)
        tip = (x2, y2)
        left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
        right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
        draw.polygon([tip, left, right], fill=color)

    def process(self) -> None:
        image_artifact = self.get_parameter_value("image")
        if not image_artifact:
            msg = f"{self.name}: No input image provided"
            raise ValueError(msg)

        annotation_data = self.get_parameter_value("annotation_data") or _default_annotation_data()
        if not isinstance(annotation_data, dict):
            annotation_data = _default_annotation_data()

        raw_url = annotation_data.get("raw_url") or getattr(image_artifact, "value", "")
        try:
            img_data = File(raw_url).read_bytes()
            bg = Image.open(BytesIO(img_data)).convert("RGBA")
        except Exception as e:
            msg = f"{self.name}: Could not load image: {e}"
            raise ValueError(msg) from e

        overlay = Image.new("RGBA", bg.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for ann in annotation_data.get("annotations", []):
            ann_type = ann.get("type")
            if ann_type == "paint":
                self._draw_paint(draw, ann)
            elif ann_type == "text":
                self._draw_text(draw, ann)
            elif ann_type == "arrow":
                self._draw_arrow(draw, ann)

        canvas = Image.alpha_composite(bg, overlay)

        dest = self._output_file.build_file()
        buf = BytesIO()
        canvas.convert("RGB").save(buf, format="PNG")
        saved = dest.write_bytes(buf.getvalue())

        artifact = ImageUrlArtifact(value=saved.location)
        self.set_parameter_value("output_image", artifact)
        self.parameter_output_values["output_image"] = artifact
        self.publish_update_to_parameter("output_image", artifact)
        logger.debug(f"{self.name}: Output saved to {artifact.value}")
