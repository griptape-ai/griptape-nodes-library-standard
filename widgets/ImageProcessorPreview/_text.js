// Create Text Image and Add Text to Existing Image. The node hands the widget the font Pillow draws with
// (font_url), and the layout follows Pillow's multiline text: the "la" anchor, lines spaced by the bottom
// of "A" plus the spacing, and each line shifted within the widest one. Glyphs rasterise slightly
// differently in the browser, so edges can differ by a pixel.

export const PREVIEW_FONT = "GriptapePillowDefault";

const num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const fontSpec = (px) => `${px}px "${PREVIEW_FONT}", sans-serif`;

// ── Colours (utils/color_utils.py parse_color_to_rgba) ───────────────────────

const NAMED_COLORS = {
  transparent: [0, 0, 0, 0], black: [0, 0, 0, 255], white: [255, 255, 255, 255], red: [255, 0, 0, 255],
  green: [0, 128, 0, 255], blue: [0, 0, 255, 255], yellow: [255, 255, 0, 255], cyan: [0, 255, 255, 255],
  magenta: [255, 0, 255, 255], gray: [128, 128, 128, 255], grey: [128, 128, 128, 255],
  orange: [255, 165, 0, 255], purple: [128, 0, 128, 255], pink: [255, 192, 203, 255],
  brown: [165, 42, 42, 255], lime: [0, 255, 0, 255], navy: [0, 0, 128, 255], teal: [0, 128, 128, 255],
  olive: [128, 128, 0, 255], maroon: [128, 0, 0, 255], silver: [192, 192, 192, 255], gold: [255, 215, 0, 255],
};

// colorsys.hls_to_rgb
function hlsToRgb(h, l, s) {
  if (s === 0) return [l, l, l];
  const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s;
  const m1 = 2 * l - m2;
  const v = (hue) => {
    hue = ((hue % 1) + 1) % 1;
    if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6;
    if (hue < 0.5) return m2;
    if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6;
    return m1;
  };
  return [v(h + 1 / 3), v(h), v(h - 1 / 3)];
}

// Returns [r, g, b, a], or null where the node would raise.
export function parseColor(input) {
  const s = String(input ?? "").trim().toLowerCase();
  if (s.startsWith("#")) {
    const hex = s.slice(1);
    if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/.test(hex)) return null;
    const bytes = hex.match(/../g).map((b) => parseInt(b, 16));
    return bytes.length === 3 ? [...bytes, 255] : bytes;
  }
  const byte = (x) => (x >= 0 && x <= 255 ? x : NaN);
  const unit = (x) => (x >= 0 && x <= 1 ? Math.trunc(x * 255) : NaN);
  let m;
  if ((m = /^rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(s))) {
    const c = [byte(+m[1]), byte(+m[2]), byte(+m[3]), 255];
    return c.some(Number.isNaN) ? null : c;
  }
  if ((m = /^rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/.exec(s))) {
    const c = [byte(+m[1]), byte(+m[2]), byte(+m[3]), unit(+m[4])];
    return c.some(Number.isNaN) ? null : c;
  }
  if ((m = /^hsla?\((\d+),\s*(\d+)%,\s*(\d+)%(?:,\s*([\d.]+))?\)/.exec(s))) {
    const [h, sat, l] = [+m[1], +m[2], +m[3]];
    if (h > 360 || sat > 100 || l > 100) return null;
    const alpha = m[4] === undefined ? 255 : unit(+m[4]);
    if (Number.isNaN(alpha) || (s.startsWith("hsla") && m[4] === undefined)) return null;
    return [...hlsToRgb(h / 360, l / 100, sat / 100).map((c) => Math.trunc(c * 255)), alpha];
  }
  return NAMED_COLORS[s] ?? null;
}

const rgbCss = ([r, g, b]) => `rgb(${r},${g},${b})`;

// ── Layout ───────────────────────────────────────────────────────────────────

// Lines, each with its offset from the text origin, and their combined bounding box, rounded outward
// the way Pillow's hinted integer boxes are.
function layoutText(ctx, text, size, spacing, align) {
  ctx.font = fontSpec(size);
  const a = ctx.measureText("A");
  const ascent = Math.round(a.fontBoundingBoxAscent);
  const lineSpacing = Math.round(ascent + a.actualBoundingBoxDescent) + spacing;
  const lines = text.split("\n");
  const widths = lines.map((line) => ctx.measureText(line).width);
  const widest = Math.max(0, ...widths);
  const shift = align === "center" ? 0.5 : align === "right" ? 1 : 0;
  let left = Infinity, top = Infinity, right = -Infinity, bottom = -Infinity;
  const placed = lines.map((line, i) => {
    const x = (widest - widths[i]) * shift;
    const y = i * lineSpacing;
    if (line) {
      const m = ctx.measureText(line);
      left = Math.min(left, x - m.actualBoundingBoxLeft);
      top = Math.min(top, y + ascent - m.actualBoundingBoxAscent);
      right = Math.max(right, x + m.actualBoundingBoxRight);
      bottom = Math.max(bottom, y + ascent + m.actualBoundingBoxDescent);
    }
    return { line, x, y };
  });
  if (!Number.isFinite(left)) left = top = right = bottom = 0;
  return { lines: placed, ascent, box: [Math.floor(left), Math.floor(top), Math.ceil(right), Math.ceil(bottom)] };
}

function drawLayout(ctx, layout, size, originX, originY) {
  ctx.font = fontSpec(size);
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  for (const { line, x, y } of layout.lines) ctx.fillText(line, originX + x, originY + y + layout.ascent);
}

// ── Create Text Image ────────────────────────────────────────────────────────

export const textImage = {
  generate: true,
  usesFont: true,
  size(v) {
    const width = Math.trunc(num(v.width, 512));
    const height = Math.trunc(num(v.height, 512));
    if (width <= 0 || height <= 0) return { error: "Width and height must be greater than 0" };
    return { width, height };
  },
  render(ctx, v, { width, height }) {
    const background = parseColor(v.background_color ?? "#000080");
    const ink = parseColor(v.text_color ?? "#00FFFF");
    if (!background || !ink) throw new Error("A colour isn't one the node can read");
    ctx.fillStyle = rgbCss(background);
    ctx.fillRect(0, 0, width, height);
    const text = v.text ?? "Hello, world!";
    const size = Math.trunc(num(v.font_size, 36));
    if (size <= 0 || !text.trim()) return;
    const layout = layoutText(ctx, text, size, 4, "left");
    const [l, t, r, b] = layout.box;
    // The node centres the text's size but draws from its origin, not its box, as here.
    const x = Math.floor((width - (r - l)) / 2);
    const y = Math.floor((height - (b - t)) / 2);
    ctx.fillStyle = rgbCss(ink);
    drawLayout(ctx, layout, size, x, y);
  },
};

// ── Add Text to Existing Image ───────────────────────────────────────────────

// Python's str() for template values.
function pyStr(value) {
  if (value === true) return "True";
  if (value === false) return "False";
  if (value === null || value === undefined) return "None";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function expandTemplate(template, values) {
  const dict = values && typeof values === "object" && !Array.isArray(values) ? values : {};
  let text = template;
  for (const match of template.matchAll(/\{([\p{L}\p{N}_]+)\}/gu)) {
    if (Object.hasOwn(dict, match[1])) text = text.split(`{${match[1]}}`).join(pyStr(dict[match[1]]));
  }
  return text;
}

const blend = (m, a, b) => Math.round((a * (255 - m) + b * m) / 255);

export const textOverlay = {
  usesFont: true,
  draw(ctx, img, v, view) {
    const { width: W, height: H, scale } = view;
    ctx.drawImage(img, 0, 0, W, H);
    const text = expandTemplate(String(v.text ?? ""), v.template_values);
    const ink = parseColor(v.text_color || "#ffffffff");
    const background = parseColor(v.text_background || "#000000ff");
    const size = num(v.font_size, 36) * scale;
    if (!text.trim() || !ink || !background || size <= 0) return;
    const margin = Math.max(0, num(v.margin, 10)) * scale;
    const pad = 4 * scale;
    const align = v.text_horizontal_alignment || "left";
    const valign = v.text_vertical_alignment || "top";

    const scratch = document.createElement("canvas");
    scratch.width = W; scratch.height = H;
    const sctx = scratch.getContext("2d", { willReadFrequently: true });
    const layout = layoutText(sctx, text, size, 4 * scale, align);
    const [l, t, r, b] = layout.box;
    const textW = Math.trunc(r - l), textH = Math.trunc(b - t);

    // _compute_aligned_text_bbox_origin: inside the margin, then clamped onto the image.
    const insetRight = Math.max(margin, W - margin), insetBottom = Math.max(margin, H - margin);
    const insetW = Math.max(0, insetRight - margin), insetH = Math.max(0, insetBottom - margin);
    let x = margin, y = margin;
    if (insetW > textW) {
      if (align === "center") x = margin + Math.floor((insetW - textW) / 2);
      else if (align === "right") x = insetRight - textW;
    }
    if (insetH > textH) {
      if (valign === "center") y = margin + Math.floor((insetH - textH) / 2);
      else if (valign === "bottom") y = insetBottom - textH;
    }
    x = Math.max(0, Math.min(x, Math.max(0, W - textW)));
    y = Math.max(0, Math.min(y, Math.max(0, H - textH)));

    // Glyph coverage, drawn white so its alpha is the coverage.
    sctx.fillStyle = "#fff";
    drawLayout(sctx, layout, size, x - Math.trunc(l), y - Math.trunc(t));
    const coverage = sctx.getImageData(0, 0, W, H).data;

    // The overlay: the background box written as-is, then the text blended the way Pillow fills an
    // RGBA layer, where text alpha replaces the box's rather than stacking on it.
    const overlay = new ImageData(W, H);
    const o = overlay.data;
    const x0 = Math.max(0, Math.round(x - pad)), y0 = Math.max(0, Math.round(y - pad));
    const x1 = Math.min(W - 1, Math.round(x + textW + pad)), y1 = Math.min(H - 1, Math.round(y + textH + pad));
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) o.set(background, (py * W + px) * 4);
    }
    for (let i = 0; i < o.length; i += 4) {
      const m = coverage[i + 3];
      if (!m) continue;
      const cm = o[i + 3] === 255 ? m : 255;
      for (let c = 0; c < 3; c++) o[i + c] = blend(cm, o[i + c], ink[c]);
      o[i + 3] = blend(m, o[i + 3], ink[3]);
    }
    sctx.clearRect(0, 0, W, H);
    sctx.putImageData(overlay, 0, 0);
    ctx.drawImage(scratch, 0, 0);
  },
};
