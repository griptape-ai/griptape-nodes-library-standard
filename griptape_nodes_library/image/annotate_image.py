import math
from io import BytesIO
from typing import Any
from urllib.parse import unquote, urlparse

from griptape.artifacts import ImageUrlArtifact, JsonArtifact
from griptape_nodes.exe_types.core_types import Parameter, ParameterList, ParameterMode
from griptape_nodes.exe_types.node_types import BaseNode, DataNode
from griptape_nodes.exe_types.param_types.parameter_dict import ParameterDict
from griptape_nodes.exe_types.param_types.parameter_int import ParameterInt
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.files.file import File
from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes, logger
from griptape_nodes.traits.widget import Widget
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

from griptape_nodes_library.utils.color_utils import parse_color_to_rgba
from griptape_nodes_library.utils.image_utils import load_pil_from_url


def _default_annotation_data() -> dict:
    return {
        "canvas_width": 1920,
        "canvas_height": 1080,
        "layers": [],
        "active_tool": "select",
        "tool_settings": {
            "paint": {"color": "#ff0000", "size": 10},
            "text": {"color": "#000000", "font_size": 24, "font": "Arial"},
            "arrow": {"color": "#ff0000", "width": 3},
        },
        "selected_layer_id": None,
        "layers_panel_open": True,
    }


class AnnotateImage(DataNode):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_parameter(
            ParameterList(
                name="input_images",
                default_value=None,
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageArtifact",
                tooltip="Images to use as layers in the canvas editor",
                allowed_modes={ParameterMode.INPUT},
            )
        )

        self.add_parameter(
            ParameterDict(
                name="annotation_data",
                default_value=_default_annotation_data(),
                tooltip="Canvas editor with layers, paint, text, and arrow annotations",
                display_name="Canvas Editor",
                allowed_modes={ParameterMode.PROPERTY},
                traits={Widget(name="AnnotateImage", library="Griptape Nodes Library")},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="canvas_width",
                default_value=1920,
                tooltip="Canvas width (auto-set from first connected image)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            ParameterInt(
                name="canvas_height",
                default_value=1080,
                tooltip="Canvas height (auto-set from first connected image)",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_image",
                input_types=["ImageArtifact", "ImageUrlArtifact"],
                type="ImageUrlArtifact",
                tooltip="Flattened composite of all layers and annotations",
                ui_options={"expander": True},
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self.add_parameter(
            Parameter(
                name="output_annotations",
                type="JsonArtifact",
                tooltip="Raw annotation data (layers, strokes, text, arrows) as JSON",
                allowed_modes={ParameterMode.OUTPUT},
            )
        )

        self._output_file = ProjectFileParameter(
            node=self,
            name="output_file",
            default_filename="annotate_image.png",
        )
        self._output_file.add_parameter()

    # ─── helpers ──────────────────────────────────────────────────────────────

    def _get_raw_value(self, artifact: Any) -> str:
        """Extract the raw path/URL string from an artifact or dict."""
        if isinstance(artifact, dict):
            return artifact.get("value", "")
        return getattr(artifact, "value", "") or ""

    def _resolve_image_url(self, raw_value: str) -> str:
        """Convert a raw file path / macro path to a browser-accessible HTTP URL."""
        from griptape_nodes.retained_mode.events.static_file_events import (
            CreateStaticFileDownloadUrlFromPathRequest,
            CreateStaticFileDownloadUrlFromPathResultSuccess,
        )

        if not raw_value:
            return ""
        if raw_value.startswith(("http://", "https://", "data:")):
            return raw_value
        try:
            resolved_path = File(raw_value).resolve()
        except Exception:
            resolved_path = str(raw_value)
        try:
            result = GriptapeNodes.handle_request(
                CreateStaticFileDownloadUrlFromPathRequest(file_path=resolved_path)
            )
            if isinstance(result, CreateStaticFileDownloadUrlFromPathResultSuccess):
                return result.url
        except Exception:
            pass
        return raw_value

    def _get_image_dimensions(self, raw_path: str) -> tuple[int, int]:
        """Get image dimensions using the raw file path (not the HTTP URL)."""
        try:
            image_bytes = File(raw_path).read_bytes()
            with Image.open(BytesIO(image_bytes)) as img:
                return img.size
        except Exception:
            return (400, 300)

    def _get_image_name(self, url: str, index: int) -> str:
        try:
            parsed = urlparse(url)
            filename = unquote(parsed.path.split("/")[-1])
            if filename and "." in filename:
                return filename.split(".")[0].split("?")[0]
        except Exception:
            pass
        return f"Image {index + 1}"

    def _find_existing_image_layer(self, layers: list[dict], url: str) -> dict | None:
        for layer in layers:
            if layer.get("type") == "image" and layer.get("url") == url:
                return layer
        return None

    # ─── layer sync ───────────────────────────────────────────────────────────

    def _sync_image_layers(self, images: list | None) -> None:
        annotation_data = self.get_parameter_value("annotation_data") or _default_annotation_data()
        if not isinstance(annotation_data, dict):
            annotation_data = _default_annotation_data()

        existing_layers = annotation_data.get("layers", [])
        canvas_width = annotation_data.get("canvas_width", 0)
        canvas_height = annotation_data.get("canvas_height", 0)

        # Build (raw_path, browser_url) pairs — no PIL reads, JS handles dimensions
        new_pairs: list[tuple[str, str]] = []
        for img in (images or []):
            raw = self._get_raw_value(img)
            if raw:
                browser_url = self._resolve_image_url(raw)
                if browser_url:
                    new_pairs.append((raw, browser_url))

        # Keep non-image layers (paint, text, arrow) unchanged
        non_image_layers = [l for l in existing_layers if l.get("type") != "image"]

        # Build new image layers, preserving existing data for known URLs
        new_image_layers: list[dict] = []
        existing_image_layers = [l for l in existing_layers if l.get("type") == "image"]

        for i, (raw, browser_url) in enumerate(new_pairs):
            existing = self._find_existing_image_layer(existing_image_layers, browser_url)
            if existing:
                new_image_layers.append(existing)
            else:
                # New image — store URL only; JS widget fills in width/height/scale on first render
                new_image_layers.append({
                    "id": f"img-{i + 1}",
                    "type": "image",
                    "name": self._get_image_name(browser_url, i),
                    "url": browser_url,   # browser-accessible for JS widget
                    "raw_url": raw,       # file path for Python PIL compositing
                    "visible": True,
                    "opacity": 1.0,
                    "x": None,   # JS will center after it knows image dimensions
                    "y": None,
                    "width": None,
                    "height": None,
                    "scaleX": None,
                    "scaleY": None,
                    "rotation": 0,
                    "order": i,
                })

        # Rebuild layer list: image layers first, then non-image layers with updated order
        all_layers = new_image_layers[:]
        for j, layer in enumerate(non_image_layers):
            all_layers.append({**layer, "order": len(new_image_layers) + j})

        annotation_data = {
            **annotation_data,
            "canvas_width": canvas_width,
            "canvas_height": canvas_height,
            "layers": all_layers,
        }

        self.set_parameter_value("annotation_data", annotation_data)
        self.publish_update_to_parameter("annotation_data", annotation_data)
        self.publish_update_to_parameter("canvas_width", canvas_width)
        self.publish_update_to_parameter("canvas_height", canvas_height)

    def _handle_images_removed(self) -> None:
        annotation_data = self.get_parameter_value("annotation_data") or _default_annotation_data()
        if not isinstance(annotation_data, dict):
            annotation_data = _default_annotation_data()

        non_image_layers = [l for l in annotation_data.get("layers", []) if l.get("type") != "image"]
        annotation_data = {**annotation_data, "layers": non_image_layers}

        self.set_parameter_value("annotation_data", annotation_data)
        self.publish_update_to_parameter("annotation_data", annotation_data)

    # ─── lifecycle hooks ──────────────────────────────────────────────────────

    def after_value_set(self, parameter: Parameter, value: Any) -> None:
        if "input_images" in parameter.name:
            # Always read the full list — after_value_set may receive a single item
            current = self.get_parameter_value("input_images") or []
            if len(current) == 0:
                self._handle_images_removed()
            else:
                self._sync_image_layers(current)
        elif parameter.name == "annotation_data" and isinstance(value, dict):
            cw = value.get("canvas_width")
            ch = value.get("canvas_height")
            if cw:
                self.publish_update_to_parameter("canvas_width", cw)
            if ch:
                self.publish_update_to_parameter("canvas_height", ch)
        return super().after_value_set(parameter, value)

    def after_incoming_connection_removed(
        self,
        source_node: BaseNode,
        source_parameter: Parameter,
        target_parameter: Parameter,
    ) -> None:
        if "input_images" in target_parameter.name:
            current = self.get_parameter_value("input_images") or []
            if len(current) == 0:
                self._handle_images_removed()
            else:
                self._sync_image_layers(current)
        return super().after_incoming_connection_removed(source_node, source_parameter, target_parameter)

    # ─── compositing ─────────────────────────────────────────────────────────

    def _draw_paint_layer(self, draw: ImageDraw.ImageDraw, layer: dict) -> None:
        for stroke in layer.get("strokes", []):
            points = stroke.get("points", [])
            if not points:
                continue
            color_str = stroke.get("color", "#ff0000")
            try:
                r, g, b, _ = parse_color_to_rgba(color_str)
            except Exception:
                r, g, b = 255, 0, 0
            size = max(1, int(stroke.get("size", 10)))
            opacity = layer.get("opacity", 1.0)
            a = int(255 * opacity)
            color = (r, g, b, a)

            # Draw circles at each point for round caps
            for px, py in points:
                draw.ellipse(
                    [px - size / 2, py - size / 2, px + size / 2, py + size / 2],
                    fill=color,
                )
            # Connect points with lines
            for i in range(len(points) - 1):
                x0, y0 = points[i]
                x1, y1 = points[i + 1]
                draw.line([x0, y0, x1, y1], fill=color, width=size)

    def _draw_text_layer(self, draw: ImageDraw.ImageDraw, layer: dict) -> None:
        text = layer.get("text", "")
        if not text:
            return
        x = float(layer.get("x", 0))
        y = float(layer.get("y", 0))
        font_size = max(8, int(layer.get("font_size", 24)))
        color_str = layer.get("color", "#000000")
        try:
            r, g, b, _ = parse_color_to_rgba(color_str)
        except Exception:
            r, g, b = 0, 0, 0
        opacity = layer.get("opacity", 1.0)
        a = int(255 * opacity)
        color = (r, g, b, a)

        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

        draw.text((x, y), text, font=font, fill=color)

    def _draw_arrow_layer(self, draw: ImageDraw.ImageDraw, layer: dict) -> None:
        x1 = float(layer.get("x1", 0))
        y1 = float(layer.get("y1", 0))
        x2 = float(layer.get("x2", 0))
        y2 = float(layer.get("y2", 0))
        color_str = layer.get("color", "#ff0000")
        try:
            r, g, b, _ = parse_color_to_rgba(color_str)
        except Exception:
            r, g, b = 255, 0, 0
        opacity = layer.get("opacity", 1.0)
        a = int(255 * opacity)
        color = (r, g, b, a)
        width = max(1, int(layer.get("width", 3)))

        draw.line([x1, y1, x2, y2], fill=color, width=width)

        # Arrowhead
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = max(15, width * 4)
        tip = (x2, y2)
        left = (
            x2 - head_len * math.cos(angle - math.pi / 6),
            y2 - head_len * math.sin(angle - math.pi / 6),
        )
        right = (
            x2 - head_len * math.cos(angle + math.pi / 6),
            y2 - head_len * math.sin(angle + math.pi / 6),
        )
        draw.polygon([tip, left, right], fill=color)

    def _composite_image_layer(self, canvas: Image.Image, layer: dict) -> None:
        # Prefer raw_url (file path) for PIL loading; fall back to browser url
        load_url = layer.get("raw_url") or layer.get("url")
        if not load_url:
            return
        try:
            img = load_pil_from_url(load_url)
        except Exception as e:
            logger.warning(f"{self.name}: Could not load image layer '{layer.get('name')}': {e}")
            return

        if img.mode != "RGBA":
            img = img.convert("RGBA")

        base_w = layer.get("width", img.width) or img.width
        base_h = layer.get("height", img.height) or img.height
        scale_x = float(layer.get("scaleX", 1.0) or 1.0)
        scale_y = float(layer.get("scaleY", 1.0) or 1.0)
        target_w = max(1, round(base_w * scale_x))
        target_h = max(1, round(base_h * scale_y))

        if img.size != (target_w, target_h):
            img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

        rotation = float(layer.get("rotation", 0) or 0)
        if rotation != 0:
            img = img.rotate(-rotation, expand=True, resample=Image.Resampling.BICUBIC)

        opacity = float(layer.get("opacity", 1.0) or 1.0)
        if opacity < 1.0:
            alpha = img.getchannel("A")
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            img.putalpha(alpha)

        cx = float(layer.get("x") or canvas.width / 2)
        cy = float(layer.get("y") or canvas.height / 2)
        paste_x = round(cx - img.width / 2)
        paste_y = round(cy - img.height / 2)
        canvas.paste(img, (paste_x, paste_y), img)

    def process(self) -> None:
        annotation_data = self.get_parameter_value("annotation_data") or _default_annotation_data()
        if not isinstance(annotation_data, dict):
            annotation_data = _default_annotation_data()

        canvas_width = int(annotation_data.get("canvas_width") or 1920)
        canvas_height = int(annotation_data.get("canvas_height") or 1080)

        canvas = Image.new("RGBA", (canvas_width, canvas_height), (255, 255, 255, 255))

        layers = sorted(annotation_data.get("layers", []), key=lambda l: l.get("order", 0))

        for layer in layers:
            if not layer.get("visible", True):
                continue

            layer_type = layer.get("type", "image")

            if layer_type == "image":
                self._composite_image_layer(canvas, layer)

            elif layer_type in ("paint", "text", "arrow"):
                # Render annotation onto a transparent overlay, then composite
                overlay = Image.new("RGBA", (canvas_width, canvas_height), (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                if layer_type == "paint":
                    self._draw_paint_layer(draw, layer)
                elif layer_type == "text":
                    self._draw_text_layer(draw, layer)
                elif layer_type == "arrow":
                    self._draw_arrow_layer(draw, layer)

                canvas = Image.alpha_composite(canvas, overlay)

        dest = self._output_file.build_file()
        img_bytes = self._pil_to_bytes(canvas.convert("RGB"), "PNG")
        saved = dest.write_bytes(img_bytes)

        output_artifact = ImageUrlArtifact(value=saved.location)
        self.set_parameter_value("output_image", output_artifact)
        self.parameter_output_values["output_image"] = output_artifact
        self.publish_update_to_parameter("output_image", output_artifact)

        annotations_artifact = JsonArtifact(annotation_data)
        self.set_parameter_value("output_annotations", annotations_artifact)
        self.parameter_output_values["output_annotations"] = annotations_artifact
        self.publish_update_to_parameter("output_annotations", annotations_artifact)

        logger.debug(f"{self.name}: Output image saved to {output_artifact.value}")

    def _pil_to_bytes(self, img: Image.Image, img_format: str) -> bytes:
        import io

        with io.BytesIO() as buf:
            img.save(buf, format=img_format)
            buf.seek(0)
            data = buf.getvalue()

        if not data:
            msg = f"{self.name}: Failed to convert image to bytes"
            raise ValueError(msg)
        return data
