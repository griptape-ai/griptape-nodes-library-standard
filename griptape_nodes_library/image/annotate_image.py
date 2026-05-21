import math
from io import BytesIO
from typing import Any

from griptape.artifacts import ImageUrlArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import DataNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.widget import Widget
from PIL import Image, ImageDraw, ImageFont

from griptape_nodes_library.utils.color_utils import parse_color_to_rgba

DEFAULT_CANVAS_WIDTH = 1920
DEFAULT_CANVAS_HEIGHT = 1080


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
            ParameterImage(
                name="image",
                default_value=None,
                tooltip="Input image to annotate",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True
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
            ParameterImage(
                name="output_image",
                tooltip="Image with annotations composited",
                allowed_modes={ParameterMode.OUTPUT},
                ui_options={"expander": True, "pulse_on_run": True},
                hide_property=True
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
        import math as _math

        cx = ann.get("cx", 0) or 0
        cy = ann.get("cy", 0) or 0
        tx = ann.get("x", 0) or 0
        ty = ann.get("y", 0) or 0
        sx = ann.get("scaleX", 1) or 1
        sy = ann.get("scaleY", 1) or 1
        rot = ann.get("rotation", 0) or 0
        cos_r, sin_r = _math.cos(rot), _math.sin(rot)

        def xform(nx: float, ny: float) -> tuple[float, float]:
            lx, ly = (nx - cx) * sx, (ny - cy) * sy
            return cx + tx + lx * cos_r - ly * sin_r, cy + ty + lx * sin_r + ly * cos_r

        size_scale = float(ann.get("sizeScale", 1.0) or 1.0)
        for stroke in ann.get("strokes", []):
            points = stroke.get("points", [])
            if not points:
                continue
            color = self._parse_color(stroke.get("color", "#ff0000"))
            base_size = max(1, int(stroke.get("size", 8)))
            for i, pt in enumerate(points):
                px, py = xform(pt[0], pt[1])
                raw_sz = pt[2] if len(pt) > 2 and pt[2] is not None else base_size
                sz = max(1, int(raw_sz * size_scale))
                draw.ellipse([px - sz / 2, py - sz / 2, px + sz / 2, py + sz / 2], fill=color)
                if i > 0:
                    ppx, ppy = xform(points[i - 1][0], points[i - 1][1])
                    prev = points[i - 1]
                    raw_sz2 = prev[2] if len(prev) > 2 and prev[2] is not None else base_size
                    sz2 = max(1, int(raw_sz2 * size_scale))
                    w = max(1, int((sz + sz2) / 2))
                    draw.line([ppx, ppy, px, py], fill=color, width=w)

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
        draw.text((x, y), text, font=font, fill=color, spacing=int(font_size * 0.2))

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        x1, y1 = float(ann.get("x1", 0)), float(ann.get("y1", 0))
        x2, y2 = float(ann.get("x2", 0)), float(ann.get("y2", 0))
        is_bezier = bool(ann.get("is_bezier", False))
        cp1x = float(ann.get("cp1x", x1 + (x2 - x1) / 3)) if is_bezier else x1 + (x2 - x1) / 3
        cp1y = float(ann.get("cp1y", y1 + (y2 - y1) / 3)) if is_bezier else y1 + (y2 - y1) / 3
        cp2x = float(ann.get("cp2x", x1 + (x2 - x1) * 2 / 3)) if is_bezier else x1 + (x2 - x1) * 2 / 3
        cp2y = float(ann.get("cp2y", y1 + (y2 - y1) * 2 / 3)) if is_bezier else y1 + (y2 - y1) * 2 / 3
        color = self._parse_color(ann.get("color", "#ff0000"))
        width = max(1, int(ann.get("width", 3)))
        has_end_arrow = ann.get("has_end_arrow", True)
        has_start_arrow = bool(ann.get("has_start_arrow", False))
        n = 30
        pts = []
        for i in range(n + 1):
            t = i / n
            mt = 1 - t
            bx = mt**3 * x1 + 3 * mt**2 * t * cp1x + 3 * mt * t**2 * cp2x + t**3 * x2
            by = mt**3 * y1 + 3 * mt**2 * t * cp1y + 3 * mt * t**2 * cp2y + t**3 * y2
            pts.append((bx, by))
        for i in range(len(pts) - 1):
            draw.line([pts[i], pts[i + 1]], fill=color, width=width)
        head = max(15, width * 4)
        setback = head * math.cos(math.pi / 6)  # pull line back so it doesn't poke through arrowhead
        if has_end_arrow:
            dx, dy = x2 - cp2x, y2 - cp2y
            angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.1 else math.atan2(y2 - y1, x2 - x1)
            # Shorten the last segment(s) to the arrowhead base
            base_x = x2 - setback * math.cos(angle)
            base_y = y2 - setback * math.sin(angle)
            if pts:
                pts[-1] = (base_x, base_y)
            tip = (x2, y2)
            left = (x2 - head * math.cos(angle - math.pi / 6), y2 - head * math.sin(angle - math.pi / 6))
            right = (x2 - head * math.cos(angle + math.pi / 6), y2 - head * math.sin(angle + math.pi / 6))
            draw.polygon([tip, left, right], fill=color)
        if has_start_arrow:
            dx, dy = x1 - cp1x, y1 - cp1y
            angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.1 else math.atan2(y1 - y2, x1 - x2)
            base_x = x1 - setback * math.cos(angle)
            base_y = y1 - setback * math.sin(angle)
            if pts:
                pts[0] = (base_x, base_y)
            tip = (x1, y1)
            left = (x1 - head * math.cos(angle - math.pi / 6), y1 - head * math.sin(angle - math.pi / 6))
            right = (x1 - head * math.cos(angle + math.pi / 6), y1 - head * math.sin(angle + math.pi / 6))
            draw.polygon([tip, left, right], fill=color)

    def _draw_rect(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        x = float(ann.get("x", 0))
        y = float(ann.get("y", 0))
        w = float(ann.get("w", 100))
        h = float(ann.get("h", 100))
        rotation = float(ann.get("rotation", 0))
        color = self._parse_color(ann.get("color", "#ff0000"))
        width = max(1, int(ann.get("width", 2)))
        fill_color_str = ann.get("fill_color", "") or ""
        fill = self._parse_color(fill_color_str) if fill_color_str else None
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        hw, hh = w / 2, h / 2
        corners = [
            (x + lx * cos_r - ly * sin_r, y + lx * sin_r + ly * cos_r)
            for lx, ly in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        ]
        draw.polygon(corners, fill=fill, outline=color, width=width)

    def _draw_ellipse(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        x = float(ann.get("x", 0))
        y = float(ann.get("y", 0))
        w = float(ann.get("w", 100))
        h = float(ann.get("h", 100))
        fill_color_str = ann.get("fill_color", "") or ""
        fill = self._parse_color(fill_color_str) if fill_color_str else None
        color = self._parse_color(ann.get("color", "#ff0000"))
        width = max(1, int(ann.get("width", 2)))
        bbox = [x - w / 2, y - h / 2, x + w / 2, y + h / 2]
        draw.ellipse(bbox, fill=fill, outline=color, width=width)

    def process(self) -> None:
        image_artifact = self.get_parameter_value("image")

        annotation_data = self.get_parameter_value("annotation_data") or _default_annotation_data()
        if not isinstance(annotation_data, dict):
            annotation_data = _default_annotation_data()

        bg = None
        if image_artifact:
            raw_url = annotation_data.get("raw_url") or getattr(image_artifact, "value", "")
            try:
                img_data = File(raw_url).read_bytes()
                bg = Image.open(BytesIO(img_data)).convert("RGBA")
            except Exception:
                bg = None

        if bg is None:
            w = annotation_data.get("canvas_width") or DEFAULT_CANVAS_WIDTH
            h = annotation_data.get("canvas_height") or DEFAULT_CANVAS_HEIGHT
            bg = Image.new("RGBA", (int(w), int(h)), (0, 0, 0, 0))

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
            elif ann_type == "rect":
                self._draw_rect(draw, ann)
            elif ann_type == "ellipse":
                self._draw_ellipse(draw, ann)

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
