// Invert Image, Invert Mask, Display Channel/Mask, Set Color to Transparent and Image Grid Splitter.
// The browser decodes every image to RGBA, so the node sends its Pillow mode (image_mode) and these
// follow the node's branch for that mode.

import { gaussianBlur, luma } from "./_pixels.js";

const num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

// Python's round(): halves go to the even neighbour.
function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

// Without a mode, an image with any transparency is treated as RGBA.
function modeOf(v, data) {
  if (v.image_mode) return v.image_mode;
  for (let i = 3; i < data.length; i += 4) if (data[i] < 255) return "RGBA";
  return "RGB";
}

function writeGray(data, i, g) {
  data[i] = g;
  data[i + 1] = g;
  data[i + 2] = g;
  data[i + 3] = 255;
}

// ── Invert Image ─────────────────────────────────────────────────────────────

export const invert = {
  pixels(data, v) {
    // RGBA keeps its alpha; every other mode becomes opaque RGB.
    const keepAlpha = modeOf(v, data) === "RGBA";
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
      if (!keepAlpha) data[i + 3] = 255;
    }
  },
};

// ── Invert Mask ──────────────────────────────────────────────────────────────

export const invertMask = {
  pixels(data, v) {
    // An RGBA mask is read from its alpha, an L mask from its value, anything else by luma.
    const mode = modeOf(v, data);
    for (let i = 0; i < data.length; i += 4) {
      let source;
      if (mode === "RGBA") source = data[i + 3];
      else if (mode === "L") source = data[i];
      else source = luma(data[i], data[i + 1], data[i + 2]);
      writeGray(data, i, 255 - source);
    }
  },
};

// ── Display Channel / Display Mask ───────────────────────────────────────────

const BANDS = { red: 0, green: 1, blue: 2, alpha: 3 };

export const displayChannel = {
  label: (v) => v.channel || "",
  pixels(data, v) {
    const mode = modeOf(v, data);
    const channel = v.channel || "red";
    if (!["L", "LA", "RGB", "RGBA"].includes(mode)) throw new Error(`The node can't read ${mode} images`);
    for (let i = 0; i < data.length; i += 4) {
      let g;
      if (mode === "L") g = data[i];
      else if (mode === "LA") g = channel === "alpha" ? data[i + 3] : data[i];
      else if (mode === "RGB" && channel === "alpha") g = 255;
      else g = data[i + (BANDS[channel] ?? 0)];
      writeGray(data, i, g);
    }
  },
};

// ── Set Color to Transparent ─────────────────────────────────────────────────

// MinFilter/MaxFilter(3) repeated n times: the min or max over a (2n+1)² window, edges clamped.
function rankFilter(mask, w, h, n, pick) {
  const out = new Uint8ClampedArray(mask.length);
  const pass = (src, dst, horizontal) => {
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        let best = src[y * w + x];
        for (let k = -n; k <= n; k++) {
          const xx = horizontal ? Math.max(0, Math.min(w - 1, x + k)) : x;
          const yy = horizontal ? y : Math.max(0, Math.min(h - 1, y + k));
          best = pick(best, src[yy * w + xx]);
        }
        dst[y * w + x] = best;
      }
    }
  };
  pass(mask, out, true);
  const result = new Uint8ClampedArray(mask.length);
  pass(out, result, false);
  return result;
}

export const colorToTransparent = {
  pixels(data, v, view) {
    // parse_hex_color: three hex pairs after an optional "#".
    const hex = String(v.color || "#00ff00").replace(/^#/, "");
    const target = [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16));
    if (target.some(Number.isNaN)) throw new Error(`"${v.color}" isn't a hex colour`);
    const tolerance = v.tolerance === null || v.tolerance === undefined ? 10 : num(v.tolerance, 10);
    const { width: w, height: h, scale } = view;

    let mask = new Uint8ClampedArray(w * h);
    for (let p = 0; p < w * h; p++) {
      const i = p * 4;
      const within = [0, 1, 2].every((c) => Math.abs(data[i + c] - target[c]) <= tolerance);
      mask[p] = within ? 0 : 255;
    }
    // Positive grow_shrink erodes the opaque area, negative dilates it.
    const growShrink = num(v.grow_shrink, 0);
    const iterations = Math.round(Math.abs(growShrink) * scale);
    if (iterations > 0) mask = rankFilter(mask, w, h, iterations, growShrink > 0 ? Math.min : Math.max);
    const blur = num(v.blur_edges, 0);
    if (blur > 0) {
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let p = 0; p < w * h; p++) rgba[p * 4] = mask[p];
      gaussianBlur(rgba, w, h, blur * scale);
      for (let p = 0; p < w * h; p++) mask[p] = rgba[p * 4];
    }
    for (let p = 0; p < w * h; p++) data[p * 4 + 3] = Math.min(data[p * 4 + 3], mask[p]);
  },
};

// ── Image Grid Splitter ──────────────────────────────────────────────────────

const clampGrid = (value) => Math.max(1, Math.min(12, Math.trunc(num(value, 2))));

function gridSize(v) {
  if (v.grid_detection_mode === "manual") return [clampGrid(v.rows), clampGrid(v.cols)];
  return [Math.max(1, num(v.detected_rows, 1)), Math.max(1, num(v.detected_cols, 1))];
}

export const gridSplit = {
  label(v) {
    const [rows, cols] = gridSize(v);
    return `${v.grid_detection_mode === "manual" ? "" : "auto: "}${rows} × ${cols}`;
  },
  draw(ctx, img, v, view) {
    ctx.drawImage(img, 0, 0, view.width, view.height);
    const [rows, cols] = gridSize(v);
    const w = view.sourceWidth, h = view.sourceHeight;
    // The node's dotted lines, 2 px wide in 10 px dashes with 6 px gaps, drawn in source pixels and
    // never thinner than a screen pixel.
    const thick = Math.max(2, 1 / view.scale);
    ctx.save();
    ctx.scale(view.scale, view.scale);
    ctx.fillStyle = "rgb(255,255,0)";
    for (let c = 1; c < cols; c++) {
      const x = pyRound((c * w) / cols);
      for (let y = 0; y < h - 1; ) {
        const end = Math.min(y + 10, h - 1);
        ctx.fillRect(x, y, thick, end - y + 1);
        y = end + 6;
      }
    }
    for (let r = 1; r < rows; r++) {
      const y = pyRound((r * h) / rows);
      for (let x = 0; x < w - 1; ) {
        const end = Math.min(x + 10, w - 1);
        ctx.fillRect(x, y, end - x + 1, thick);
        x = end + 6;
      }
    }
    ctx.restore();
  },
};

// ── Gaussian Edge Fade ───────────────────────────────────────────────────────

// The fade mask at source-pixel coordinates, as _create_square_mask / _create_rounded_mask.
function edgeFadeMask(v, view) {
  const { width: w, height: h, scale, sourceWidth: W, sourceHeight: H } = view;
  const distance = num(v.fade_distance, 5);
  const fp = Math.max(0, (v.fade_mode || "percentage") === "percentage"
    ? Math.trunc((Math.min(W, H) * distance) / 100)
    : Math.trunc(distance));
  const curve = num(v.fade_curve, 2);
  const on = (name) => v[name] !== false;
  const mask = new Uint8ClampedArray(w * h).fill(255);
  if (fp <= 0) return mask;
  const rounded = v.edge_shape === "rounded";
  const ft = on("apply_top") ? fp : 0, fb = on("apply_bottom") ? fp : 0;
  const fl = on("apply_left") ? fp : 0, fr = on("apply_right") ? fp : 0;
  for (let py = 0; py < h; py++) {
    const y = Math.floor(py / scale);
    for (let px = 0; px < w; px++) {
      const x = Math.floor(px / scale);
      let a = 255;
      if (rounded) {
        const inLeft = x < fl, inRight = x >= W - fr, inTop = y < ft, inBottom = y >= H - fb;
        const dxl = inLeft ? x - fl : 0, dxr = inRight ? x - (W - fr - 1) : 0;
        const dyt = inTop ? y - ft : 0, dyb = inBottom ? y - (H - fb - 1) : 0;
        let d = 0;
        if (inTop && inLeft) d = Math.hypot(dxl, dyt);
        else if (inTop && inRight) d = Math.hypot(dxr, dyt);
        else if (inBottom && inLeft) d = Math.hypot(dxl, dyb);
        else if (inBottom && inRight) d = Math.hypot(dxr, dyb);
        else if (inTop) d = Math.abs(dyt);
        else if (inBottom) d = Math.abs(dyb);
        else if (inLeft) d = Math.abs(dxl);
        else if (inRight) d = Math.abs(dxr);
        a = d <= 0 ? 255 : d >= fp ? 0 : Math.trunc(255 * Math.pow(1 - d / fp, curve));
      } else {
        // Square mode skips an edge whose fade would pass the middle of the image.
        const ramp = (t) => Math.trunc(255 * Math.pow(t / fp, curve));
        if (fp <= Math.floor(H / 2)) {
          if (ft && y < Math.min(fp, H)) a = Math.min(a, ramp(y));
          if (fb && y >= Math.max(0, H - fp)) a = Math.min(a, ramp(H - y));
        }
        if (fp <= Math.floor(W / 2)) {
          if (fl && x < Math.min(fp, W)) a = Math.min(a, ramp(x));
          if (fr && x >= Math.max(0, W - fp)) a = Math.min(a, ramp(W - x));
        }
      }
      mask[py * w + px] = a;
    }
  }
  return mask;
}

export const edgeFade = {
  pixels(data, v, view) {
    const { width: w, height: h, scale } = view;
    const mask = edgeFadeMask(v, view);
    const blur = num(v.blur_radius, 10);
    if (blur > 0) {
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let p = 0; p < w * h; p++) rgba[p * 4] = mask[p];
      gaussianBlur(rgba, w, h, blur * scale);
      for (let p = 0; p < w * h; p++) mask[p] = rgba[p * 4];
    }
    // Only an RGBA image keeps its own transparency, and only when replace_mask is off.
    const keepOwn = !v.replace_mask && modeOf(v, data) === "RGBA";
    for (let p = 0; p < w * h; p++) {
      const i = p * 4 + 3;
      data[i] = keepOwn ? Math.trunc((data[i] / 255) * (mask[p] / 255) * 255) : mask[p];
    }
  },
};

// ── Combine Masks ────────────────────────────────────────────────────────────

// _mask_to_l: alpha for RGBA/LA unless it is almost uniformly opaque, then the colour; luma for RGB.
function maskToL(data, mode) {
  const n = data.length / 4;
  const out = new Uint8ClampedArray(n);
  let useAlpha = false;
  if (mode === "RGBA" || mode === "LA") {
    let min = 255, max = 0;
    for (let p = 0; p < n; p++) {
      min = Math.min(min, data[p * 4 + 3]);
      max = Math.max(max, data[p * 4 + 3]);
    }
    useAlpha = !(min >= 200 && max - min <= 32);
  }
  for (let p = 0; p < n; p++) {
    const i = p * 4;
    if (useAlpha) out[p] = data[i + 3];
    else if (mode === "RGBA" || mode === "LA" || mode === "L") out[p] = data[i];
    else out[p] = luma(data[i], data[i + 1], data[i + 2]);
  }
  return out;
}

export const combineMasks = {
  multi: true,
  label: (v) => `${(v.images || []).length} masks`,
  // modes[k] is the mode of datas[k]; the shell keeps them paired.
  combine(datas, modes) {
    const combined = maskToL(datas[0], modes[0] || modeOf({}, datas[0]));
    for (let k = 1; k < datas.length; k++) {
      const mask = maskToL(datas[k], modes[k] || modeOf({}, datas[k]));
      for (let p = 0; p < combined.length; p++) combined[p] = Math.max(combined[p], mask[p]);
    }
    const out = new Uint8ClampedArray(combined.length * 4);
    for (let p = 0; p < combined.length; p++) writeGray(out, p * 4, combined[p]);
    return out;
  },
};
