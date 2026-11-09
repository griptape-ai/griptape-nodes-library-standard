// How each image processor's result looks, keyed by the node's PREVIEW_OPERATION. Each mirrors the
// node's _process_image for the settings in `v`, which arrive from the node's own parameters.
//
//   size(v, w, h)            output size in source pixels for a w × h input: { width, height } or
//                            { error }. Defaults to the input size.
//   label(v)                 text shown before the size in the status bar (optional).
//   draw(ctx, img, v, view)  draws the result into ctx, which is view.width × view.height.
//   pixels(data, v, view)    edits the scaled input's RGBA pixels in place, for colour operations.
//
// view: { width, height, scale, sourceWidth, sourceHeight, outputWidth, outputHeight }. width and height
// are the preview's pixels, scale is preview pixels per source pixel, and the output size is in source pixels.

import { applyLut, blendByte, gaussianBlur, luma } from "./_pixels.js";
import { VIDEO_OPERATIONS } from "./_video_operations.js";
import { colorBars } from "./_color_bars.js";
import { textImage, textOverlay } from "./_text.js";
import {
  colorToTransparent,
  combineMasks,
  displayChannel,
  edgeFade,
  gridSplit,
  invert,
  invertMask,
} from "./_pixel_nodes.js";

const num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

// ── Rotate Image ─────────────────────────────────────────────────────────────

const QUARTER_TURNS = {
  none: 0,
  "90° clockwise": 90,
  "180°": 180,
  "90° counter-clockwise": -90,
};

function resolveRotation(v, w, h) {
  const turn = QUARTER_TURNS[v.rotation ?? "90° clockwise"];
  if (turn !== undefined) {
    const swap = Math.abs(turn) === 90;
    return { degrees: turn, width: swap ? h : w, height: swap ? w : h, custom: false };
  }
  const degrees = num(v.angle, 0);
  if (v.expand_canvas === false) return { degrees, width: w, height: h, custom: true };
  const rad = (degrees * Math.PI) / 180;
  const cos = Math.abs(Math.cos(rad));
  const sin = Math.abs(Math.sin(rad));
  return {
    degrees,
    width: Math.max(1, Math.round(w * cos + h * sin)),
    height: Math.max(1, Math.round(w * sin + h * cos)),
    custom: true,
  };
}

function fmtDegrees(d) {
  const r = Math.round(d * 100) / 100;
  return (r > 0 ? "+" : "") + r + "°";
}

const rotate = {
  size: (v, w, h) => resolveRotation(v, w, h),
  label: (v) => (QUARTER_TURNS[v.rotation ?? "90° clockwise"] !== undefined ? v.rotation : fmtDegrees(num(v.angle, 0))),
  draw(ctx, img, v, view) {
    const { degrees, custom } = resolveRotation(v, view.sourceWidth, view.sourceHeight);
    if (custom) {
      ctx.fillStyle = v.background_color || "#00000000";
      ctx.fillRect(0, 0, view.width, view.height);
    }
    const w = view.sourceWidth * view.scale;
    const h = view.sourceHeight * view.scale;
    ctx.save();
    ctx.translate(view.width / 2, view.height / 2);
    ctx.rotate((degrees * Math.PI) / 180);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();
  },
};

// ── Flip Image ───────────────────────────────────────────────────────────────

const flip = {
  label: (v) => v.direction || "horizontal",
  draw(ctx, img, v, view) {
    const direction = v.direction || "horizontal";
    const mirror = direction === "horizontal" || direction === "both";
    const upsideDown = direction === "vertical" || direction === "both";
    ctx.save();
    ctx.translate(mirror ? view.width : 0, upsideDown ? view.height : 0);
    ctx.scale(mirror ? -1 : 1, upsideDown ? -1 : 1);
    ctx.drawImage(img, 0, 0, view.width, view.height);
    ctx.restore();
  },
};

// ── Grayscale Image ──────────────────────────────────────────────────────────

const grayscale = {
  pixels(data) {
    for (let i = 0; i < data.length; i += 4) {
      const l = luma(data[i], data[i + 1], data[i + 2]);
      data[i] = l;
      data[i + 1] = l;
      data[i + 2] = l;
    }
  },
};

// ── Adjust Image Levels ──────────────────────────────────────────────────────

function levelsTable(v) {
  const highlights = Math.trunc(num(v.highlights, 255));
  let shadows = Math.trunc(num(v.shadows, 0));
  const midtones = num(v.midtones, 1);
  const outShadows = Math.trunc(num(v.output_shadows, 0));
  const outHighlights = Math.trunc(num(v.output_highlights, 255));
  if (shadows >= highlights) shadows = Math.min(shadows, highlights - 1);

  const table = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    let iv;
    if (i <= shadows) iv = 0;
    else if (i >= highlights) iv = 255;
    else iv = Math.trunc((255 * (i - shadows)) / (highlights - shadows));

    let gv;
    if (iv === 0) gv = 0;
    else if (iv === 255) gv = 255;
    else gv = Math.trunc(255 * Math.pow(iv / 255, 1 / midtones));

    let ov;
    if (gv === 0) ov = outShadows;
    else if (gv === 255) ov = outHighlights;
    else ov = Math.trunc(outShadows + (outHighlights - outShadows) * (gv / 255));
    table[i] = Math.max(0, Math.min(255, ov));
  }
  return table;
}

const levels = {
  pixels(data, v) {
    applyLut(data, levelsTable(v));
  },
};

// ── Adjust Image EQ ──────────────────────────────────────────────────────────

const eq = {
  pixels(data, v) {
    const brightness = num(v.brightness, 1);
    const contrast = num(v.contrast, 1);
    const saturation = num(v.saturation, 1);
    const gamma = num(v.gamma, 1);

    if (brightness !== 1) {
      for (let i = 0; i < data.length; i += 4) {
        for (let c = 0; c < 3; c++) data[i + c] = blendByte(0, data[i + c], brightness);
      }
    }
    if (contrast !== 1) {
      // The mean of L over every pixel, transparent ones included, as ImageStat does.
      let total = 0;
      for (let i = 0; i < data.length; i += 4) total += luma(data[i], data[i + 1], data[i + 2]);
      const mean = Math.trunc(total / (data.length / 4) + 0.5);
      for (let i = 0; i < data.length; i += 4) {
        for (let c = 0; c < 3; c++) data[i + c] = blendByte(mean, data[i + c], contrast);
      }
    }
    if (saturation !== 1) {
      for (let i = 0; i < data.length; i += 4) {
        const l = luma(data[i], data[i + 1], data[i + 2]);
        for (let c = 0; c < 3; c++) data[i + c] = blendByte(l, data[i + c], saturation);
      }
    }
    if (gamma !== 1) {
      const table = new Uint8Array(256);
      for (let x = 0; x < 256; x++) table[x] = Math.min(255, Math.trunc(Math.pow(x / 255, 1 / gamma) * 255));
      applyLut(data, table);
    }
  },
};

// ── Gaussian Blur Image ──────────────────────────────────────────────────────

const gaussianBlurOperation = {
  label: (v) => `radius ${num(v.radius, 5)}`,
  pixels(data, v, view) {
    gaussianBlur(data, view.width, view.height, num(v.radius, 5) * view.scale);
  },
};

// ── Bloom Effect ─────────────────────────────────────────────────────────────

const bloom = {
  pixels(data, v, view) {
    const amount = num(v.bloom_amount, 0.5);
    if (amount <= 0) return;
    // The node converts to RGB, so the result is opaque.
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
    const original = Uint8ClampedArray.from(data);
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) data[i + c] = blendByte(0, data[i + c], 1 + amount);
    }
    gaussianBlur(data, view.width, view.height, num(v.bloom_radius, 5) * view.scale);
    const t = Math.min(amount, 1);
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) data[i + c] = blendByte(original[i + c], data[i + c], t);
      data[i + 3] = 255;
    }
  },
};

// ── Resize Image ─────────────────────────────────────────────────────────────

// Python's round(): halves go to the even neighbour.
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

function parseAspectRatio(text) {
  const raw = String(text ?? "").trim();
  if (raw.includes(":")) {
    const parts = raw.split(":");
    if (parts.length !== 2) return null;
    const ratio = Number(parts[0].trim()) / Number(parts[1].trim());
    return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
  }
  const ratio = Number(raw);
  return raw !== "" && Number.isFinite(ratio) && ratio > 0 ? ratio : null;
}

// Output size, and the target box for "width and height" and "aspect ratio", which use fit_mode.
function resizeLayout(v, w, h) {
  const mode = v.resize_mode || "percentage";
  if (mode === "width") {
    const size = Math.trunc(num(v.target_size, 1000));
    return { width: size, height: Math.trunc(h * (size / w)) };
  }
  if (mode === "height") {
    const size = Math.trunc(num(v.target_size, 1000));
    return { width: Math.trunc(w * (size / h)), height: size };
  }
  if (mode === "percentage") {
    const s = num(v.percentage_scale, 100) / 100;
    return { width: Math.trunc(w * s), height: Math.trunc(h * s) };
  }
  let targetW;
  let targetH;
  if (mode === "width and height") {
    targetW = Math.trunc(num(v.target_width, 1000));
    targetH = Math.trunc(num(v.target_height, 1000));
  } else if (mode === "aspect ratio") {
    const ratio = parseAspectRatio(v.aspect_ratio);
    if (ratio === null) return { error: `"${v.aspect_ratio}" isn't an aspect ratio` };
    if (ratio >= 1) {
      targetW = w;
      targetH = Math.max(1, roundHalfEven(w / ratio));
    } else {
      targetH = h;
      targetW = Math.max(1, roundHalfEven(h * ratio));
    }
  } else {
    return { error: `Unknown resize mode "${mode}"` };
  }
  return { width: Math.max(1, targetW), height: Math.max(1, targetH), fit: v.fit_mode || "fit" };
}

// The node's _create_canvas: 3- or 6-digit hex, anything else white.
function fitBackground(color) {
  const text = String(color ?? "");
  if (text === "transparent") return "rgba(0,0,0,0)";
  const hex = text.startsWith("#") ? text.slice(1) : "";
  if (/^[0-9a-fA-F]{3}$/.test(hex) || /^[0-9a-fA-F]{6}$/.test(hex)) return `#${hex}`;
  return "#ffffff";
}

const resize = {
  size(v, w, h) {
    const layout = resizeLayout(v, w, h);
    if (layout.error) return layout;
    if (layout.width < 1 || layout.height < 1) return { error: "The result would be 0 pixels wide or tall" };
    return layout;
  },
  label: (v) => v.resize_mode || "percentage",
  draw(ctx, img, v, view) {
    const { fit } = resizeLayout(v, view.sourceWidth, view.sourceHeight);
    ctx.imageSmoothingEnabled = (v.resample_filter || "lanczos") !== "nearest";
    ctx.imageSmoothingQuality = "high";
    if (fit === undefined || fit === "stretch") {
      ctx.drawImage(img, 0, 0, view.width, view.height);
      return;
    }
    // Scale the input to cover ("fill") or fit inside ("fit") the target box, in source pixels.
    const targetW = view.outputWidth;
    const targetH = view.outputHeight;
    const sx = targetW / view.sourceWidth;
    const sy = targetH / view.sourceHeight;
    const s = fit === "fill" ? Math.max(sx, sy) : Math.min(sx, sy);
    const nw = Math.max(1, Math.trunc(view.sourceWidth * s));
    const nh = Math.max(1, Math.trunc(view.sourceHeight * s));
    const x = fit === "fill" ? -Math.trunc((nw - targetW) / 2) : Math.trunc((targetW - nw) / 2);
    const y = fit === "fill" ? -Math.trunc((nh - targetH) / 2) : Math.trunc((targetH - nh) / 2);
    if (fit === "fit") {
      ctx.fillStyle = fitBackground(v.background_color);
      ctx.fillRect(0, 0, view.width, view.height);
    }
    ctx.drawImage(img, x * view.scale, y * view.scale, nw * view.scale, nh * view.scale);
  },
};

// ── Extend Canvas ────────────────────────────────────────────────────────────

// Where the original image sits for each position, as the node's _calculate_position.
const EXTEND_POSITIONS = {
  center: (dx, dy) => [Math.floor(dx / 2), Math.floor(dy / 2)],
  top_left: () => [0, 0],
  top_right: (dx) => [dx, 0],
  bottom_left: (dx, dy) => [0, dy],
  bottom_right: (dx, dy) => [dx, dy],
  top: (dx) => [Math.floor(dx / 2), 0],
  bottom: (dx, dy) => [Math.floor(dx / 2), dy],
  left: (dx, dy) => [0, Math.floor(dy / 2)],
  right: (dx, dy) => [dx, Math.floor(dy / 2)],
};

// Canvas size and where the original image goes. Presets are "W:H label"; anything else extends by
// the pixel margins, which only "custom" also uses as the image's offset.
function extendLayout(v, w, h) {
  const preset = v.aspect_ratio_preset ?? "1:1 square";
  const ratio = preset === "custom" ? null : /^(\d+):(\d+) /.exec(preset);
  const margin = (name) => Math.max(0, Math.trunc(num(v[name], 0)));
  let targetW;
  let targetH;
  if (ratio) {
    const rw = Number(ratio[1]);
    const rh = Number(ratio[2]);
    if (w / h > rw / rh) {
      targetW = w;
      targetH = Math.trunc((w * rh) / rw);
    } else {
      targetH = h;
      targetW = Math.trunc((h * rw) / rh);
    }
  } else {
    targetW = w + margin("left") + margin("right");
    targetH = h + margin("top") + margin("bottom");
  }
  // The upscale factor grows the canvas; the image keeps its size.
  const upscale = num(v.upscale_factor, 1);
  if (upscale !== 1) {
    targetW = Math.trunc(targetW * upscale);
    targetH = Math.trunc(targetH * upscale);
  }
  const width = Math.max(targetW, w);
  const height = Math.max(targetH, h);
  let x;
  let y;
  if (preset === "custom") {
    x = margin("left");
    y = margin("top");
  } else {
    const place = EXTEND_POSITIONS[v.position || "center"] ?? EXTEND_POSITIONS.center;
    [x, y] = place(width - w, height - h);
  }
  return { width, height, x, y };
}

const extendCanvas = {
  size: (v, w, h) => extendLayout(v, w, h),
  label: (v) => v.aspect_ratio_preset ?? "1:1 square",
  draw(ctx, img, v, view) {
    const { x, y } = extendLayout(v, view.sourceWidth, view.sourceHeight);
    const background = v.background_color || "black";
    if (background !== "transparent") {
      ctx.fillStyle = background;
      ctx.fillRect(0, 0, view.width, view.height);
    }
    ctx.drawImage(img, x * view.scale, y * view.scale, view.sourceWidth * view.scale, view.sourceHeight * view.scale);
  },
};

export const OPERATIONS = {
  rotate,
  flip,
  grayscale,
  levels,
  eq,
  gaussian_blur: gaussianBlurOperation,
  bloom,
  resize,
  extend_canvas: extendCanvas,
  video_eq: VIDEO_OPERATIONS.video_eq,
  video_blur: VIDEO_OPERATIONS.video_blur,
  vignette: VIDEO_OPERATIONS.vignette,
  rgb_shift: VIDEO_OPERATIONS.rgb_shift,
  curves: VIDEO_OPERATIONS.curves,
  film_grain: VIDEO_OPERATIONS.film_grain,
  color_bars: colorBars,
  text_image: textImage,
  text_overlay: textOverlay,
  invert,
  invert_mask: invertMask,
  display_channel: displayChannel,
  color_to_transparent: colorToTransparent,
  grid_split: gridSplit,
  edge_fade: edgeFade,
  combine_masks: combineMasks,
  resize_video: VIDEO_OPERATIONS.resize_video,
};
