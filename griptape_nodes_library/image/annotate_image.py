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
            "text": {"color": "#ff0000", "font_size": 48},
            "arrow": {"color": "#ff0000", "width": 8, "has_start_arrow": False, "has_end_arrow": True, "is_bezier": False, "taper": False},
            "rect": {"color": "#ff0000", "width": 8, "fill_color": ""},
            "ellipse": {"color": "#ff0000", "width": 8, "fill_color": ""},
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
                allowed_modes={ParameterMode.INPUT, ParameterMode.OUTPUT},
                hide_property=True
            )
        )

        self.add_parameter(
            ParameterDict(
                name="import_annotations",
                default_value=None,
                tooltip="Annotation data to import from another node (overrides can be applied in canvas)",
                allowed_modes={ParameterMode.INPUT},
                hide_property=True,
            )
        )

        self.add_parameter(
            ParameterDict(
                name="annotation_data",
                default_value=_default_annotation_data(),
                tooltip="Canvas annotations (paint, text, arrows)",
                display_name="Canvas",
                allowed_modes={ParameterMode.PROPERTY, ParameterMode.OUTPUT},
                traits={Widget(name="AnnotateImageSimple", library="Griptape Nodes Library")},
            )
        )

        self.add_parameter(
            ParameterImage(
                name="output_image",
                tooltip="Image with annotations composited",
                allowed_modes={ParameterMode.OUTPUT},
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
            new_data = {
                **data,
                "image_url": browser_url,
                "raw_url": raw,
                "canvas_width": w or data.get("canvas_width", 0),
                "canvas_height": h or data.get("canvas_height", 0),
            }
            self.set_parameter_value("annotation_data", new_data)
            self.publish_update_to_parameter("annotation_data", new_data)

        if parameter.name == "import_annotations" and isinstance(value, dict):
            # Accept full annotation_data dict from an upstream node's annotation_data output.
            # Compute the effective (merged) annotations so overrides and deletions are resolved.
            imported = self._effective_annotations(value)
            data = self.get_parameter_value("annotation_data") or _default_annotation_data()
            if not isinstance(data, dict):
                data = _default_annotation_data()
            new_data = {**data, "imported_annotations": imported}
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

    def _paint_natural_center(self, ann: dict) -> tuple[float, float]:
        if ann.get("cx") is not None and ann.get("cy") is not None:
            return float(ann["cx"]), float(ann["cy"])
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        for stroke in ann.get("strokes", []):
            for pt in stroke.get("points", []):
                min_x = min(min_x, pt[0]); min_y = min(min_y, pt[1])
                max_x = max(max_x, pt[0]); max_y = max(max_y, pt[1])
        if math.isinf(min_x):
            return 0.0, 0.0
        return (min_x + max_x) / 2, (min_y + max_y) / 2

    def _draw_paint(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        cx, cy = self._paint_natural_center(ann)
        tx = ann.get("x", 0) or 0
        ty = ann.get("y", 0) or 0
        sx = ann.get("scaleX", 1) or 1
        sy = ann.get("scaleY", 1) or 1
        rot = ann.get("rotation", 0) or 0
        cos_r, sin_r = math.cos(rot), math.sin(rot)

        def xform(nx: float, ny: float) -> tuple[float, float]:
            lx, ly = (nx - cx) * sx, (ny - cy) * sy
            return cx + tx + lx * cos_r - ly * sin_r, cy + ty + lx * sin_r + ly * cos_r

        size_scale = float(ann.get("sizeScale", 1.0) or 1.0)
        # sx/sy scale both point positions (via xform) and brush radius
        transform_scale = math.sqrt(abs(sx * sy))
        effective_scale = size_scale * transform_scale
        for stroke in ann.get("strokes", []):
            points = stroke.get("points", [])
            if not points:
                continue
            color = self._parse_color(stroke.get("color", "#ff0000"))
            base_size = max(1, float(stroke.get("size", 8)))
            for i, pt in enumerate(points):
                px, py = xform(pt[0], pt[1])
                raw_sz = pt[2] if len(pt) > 2 and pt[2] is not None else base_size
                sz = max(1, raw_sz * effective_scale)
                r = sz / 2
                draw.ellipse([px - r, py - r, px + r, py + r], fill=color)
                if i > 0:
                    ppx, ppy = xform(points[i - 1][0], points[i - 1][1])
                    prev = points[i - 1]
                    raw_sz2 = prev[2] if len(prev) > 2 and prev[2] is not None else base_size
                    sz2 = max(1, raw_sz2 * effective_scale)
                    w = max(1, int((sz + sz2) / 2))
                    draw.line([ppx, ppy, px, py], fill=color, width=w)

    def _draw_text(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        text = ann.get("text", "")
        if not text:
            return
        x = float(ann.get("x", 0))
        y = float(ann.get("y", 0))
        font_size = max(8, int(ann.get("font_size", 48)))
        color = self._parse_color(ann.get("color", "#ff0000"))
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
        draw.text((x, y), text, font=font, fill=color, spacing=int(font_size * 0.2))

    def _draw_arrow(self, draw: ImageDraw.ImageDraw, ann: dict) -> None:
        x1, y1 = float(ann.get("x1", 0)), float(ann.get("y1", 0))
        x2, y2 = float(ann.get("x2", 0)), float(ann.get("y2", 0))
        cp1x = float(ann.get("cp1x", x1 + (x2 - x1) / 3))
        cp1y = float(ann.get("cp1y", y1 + (y2 - y1) / 3))
        cp2x = float(ann.get("cp2x", x1 + (x2 - x1) * 2 / 3))
        cp2y = float(ann.get("cp2y", y1 + (y2 - y1) * 2 / 3))
        color = self._parse_color(ann.get("color", "#ff0000"))
        w = max(1.0, float(ann.get("width", 8)))
        has_end_arrow = bool(ann.get("has_end_arrow", True))
        has_start_arrow = bool(ann.get("has_start_arrow", False))
        taper = bool(ann.get("taper", False))

        head = max(15.0, w * 4)
        setback = head * math.cos(math.pi / 6)

        # Arrowhead angles from tangent at endpoints
        end_angle = start_angle = 0.0
        if has_end_arrow:
            dx, dy = x2 - cp2x, y2 - cp2y
            end_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.1 else math.atan2(y2 - y1, x2 - x1)
        if has_start_arrow:
            dx, dy = x1 - cp1x, y1 - cp1y
            start_angle = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.1 else math.atan2(y1 - y2, x1 - x2)

        # Pull endpoints back to arrowhead base
        lx2 = x2 - setback * math.cos(end_angle)   if has_end_arrow   else x2
        ly2 = y2 - setback * math.sin(end_angle)   if has_end_arrow   else y2
        lx1 = x1 - setback * math.cos(start_angle) if has_start_arrow else x1
        ly1 = y1 - setback * math.sin(start_angle) if has_start_arrow else y1

        # Sample bezier and compute parametric speed (first derivative magnitude)
        n = 48
        pts, speeds, tangents = [], [], []
        for i in range(n + 1):
            t = i / n
            mt = 1 - t
            bx = mt**3*lx1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*lx2
            by = mt**3*ly1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ly2
            dvx = 3*(mt**2*(cp1x-lx1) + 2*mt*t*(cp2x-cp1x) + t**2*(lx2-cp2x))
            dvy = 3*(mt**2*(cp1y-ly1) + 2*mt*t*(cp2y-cp1y) + t**2*(ly2-cp2y))
            spd = math.hypot(dvx, dvy)
            pts.append((bx, by))
            speeds.append(max(spd, 0.001))
            tangents.append((dvx, dvy, max(spd, 0.001)))

        min_spd = min(speeds)
        is_straight = (max(speeds) - min_spd) < 0.001

        if not taper or is_straight:
            # Uniform width — round caps via overlapping circles + connecting lines
            for i in range(n + 1):
                bx, by = pts[i]
                r = w / 2
                draw.ellipse([bx - r, by - r, bx + r, by + r], fill=color)
            for i in range(n):
                draw.line([pts[i], pts[i + 1]], fill=color, width=int(w))
        else:
            # Velocity taper — thick in curves (slow), thin on straights (fast).
            # Mirrors the JS formula: hw = (minSpd / spd) * w / 2
            left_pts, right_pts = [], []
            for i in range(n + 1):
                bx, by = pts[i]
                dvx, dvy, spd = tangents[i]
                hw = (min_spd / spd) * w / 2
                px, py = (-dvy / spd * hw, dvx / spd * hw)
                left_pts.append((bx + px, by + py))
                right_pts.append((bx - px, by - py))
            polygon = left_pts + list(reversed(right_pts))
            draw.polygon([(int(x), int(y)) for x, y in polygon], fill=color)

        # Arrowheads
        if has_end_arrow:
            tip = (x2, y2)
            left = (x2 - head * math.cos(end_angle - math.pi/6), y2 - head * math.sin(end_angle - math.pi/6))
            right = (x2 - head * math.cos(end_angle + math.pi/6), y2 - head * math.sin(end_angle + math.pi/6))
            draw.polygon([tip, left, right], fill=color)
        if has_start_arrow:
            tip = (x1, y1)
            left = (x1 - head * math.cos(start_angle - math.pi/6), y1 - head * math.sin(start_angle - math.pi/6))
            right = (x1 - head * math.cos(start_angle + math.pi/6), y1 - head * math.sin(start_angle + math.pi/6))
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

    def _effective_annotations(self, annotation_data: dict) -> list:
        """Return imported (with overrides applied, deleted ones skipped) + local annotations."""
        imported = annotation_data.get("imported_annotations", []) or []
        overrides = annotation_data.get("overrides", {}) or {}
        local = annotation_data.get("annotations", []) or []

        merged_imported = []
        for ann in imported:
            ov = overrides.get(ann.get("id", ""), {})
            if ov.get("deleted"):
                continue
            merged_imported.append({**ann, **{k: v for k, v in ov.items() if k != "deleted"}})

        return merged_imported + local

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

        all_annotations = self._effective_annotations(annotation_data)
        for ann in all_annotations:
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

        if image_artifact:
            self.parameter_output_values["image"] = image_artifact
            self.publish_update_to_parameter("image", image_artifact)

        self.parameter_output_values["annotation_data"] = annotation_data
        self.publish_update_to_parameter("annotation_data", annotation_data)

        logger.debug(f"{self.name}: Output saved to {artifact.value}")
