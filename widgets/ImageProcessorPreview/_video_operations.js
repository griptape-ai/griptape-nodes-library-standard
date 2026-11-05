// Previews for the FFmpeg-based video processors, drawn on the current frame. Each mirrors the FFmpeg
// filter the node builds. FFmpeg works on limited-range BT.601 YUV for most of these, so the colour
// operations convert each pixel to YUV, apply the filter's math, and convert back. Film grain and the
// blurs are approximations; the rest follow FFmpeg's integer math.

const num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};

const clamp255 = (v) => (v < 0 ? 0 : v > 255 ? 255 : v);

// ── BT.601 limited range, as FFmpeg converts untagged video ───────────────────

function toYuv(r, g, b) {
  return [
    Math.round(16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255),
    Math.round(128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255),
    Math.round(128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255),
  ];
}

function writeRgb(data, i, y, u, v) {
  const yy = 1.164383 * (y - 16);
  data[i] = yy + 1.596027 * (v - 128);
  data[i + 1] = yy - 0.391762 * (u - 128) - 0.812968 * (v - 128);
  data[i + 2] = yy + 2.017232 * (u - 128);
}

// ── Adjust Video EQ (vf_eq) ──────────────────────────────────────────────────

// One plane's lookup table: the integer path when gamma is 1 and contrast is small, the LUT otherwise.
function eqLut(contrast, brightness, gamma) {
  const lut = new Uint8Array(256);
  if (contrast === 1 && brightness === 0 && gamma === 1) {
    for (let i = 0; i < 256; i++) lut[i] = i;
    return { lut, identity: true };
  }
  if (gamma === 1 && Math.abs(contrast) < 7.9) {
    const ci = Math.trunc(contrast * 4096);
    const bi = Math.trunc((Math.trunc(100 * brightness + 100) * 511) / 200) - 128 - Math.trunc(ci / 32);
    for (let i = 0; i < 256; i++) lut[i] = clamp255(((i * ci) >> 12) + bi);
    return { lut, identity: false };
  }
  for (let i = 0; i < 256; i++) {
    const v = contrast * (i / 255 - 0.5) + 0.5 + brightness;
    if (v <= 0) lut[i] = 0;
    else {
      const p = Math.pow(v, 1 / gamma);
      lut[i] = p >= 1 ? 255 : Math.trunc(256 * p);
    }
  }
  return { lut, identity: false };
}

const videoEq = {
  pixels(data, v) {
    // Brightness, contrast and gamma touch Y; saturation scales U and V.
    const y = eqLut(num(v.contrast, 1), num(v.brightness, 0), num(v.gamma, 1));
    const c = eqLut(num(v.saturation, 1), 0, 1);
    if (y.identity && c.identity) return;
    for (let i = 0; i < data.length; i += 4) {
      const [yy, u, vv] = toYuv(data[i], data[i + 1], data[i + 2]);
      writeRgb(data, i, y.lut[yy], c.lut[u], c.lut[vv]);
    }
  },
};

// ── Blur Video (gblur, boxblur, avgblur) ─────────────────────────────────────

// Runs fn over each colour channel as a Float32Array plane, then writes it back rounded.
function eachChannel(data, w, h, fn) {
  const plane = new Float32Array(w * h);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < w * h; i++) plane[i] = data[i * 4 + c];
    fn(plane);
    for (let i = 0; i < w * h; i++) data[i * 4 + c] = plane[i];
  }
}

// gblur with steps=1: a recursive filter run forwards and backwards along each row, then each column.
function gblur(data, w, h, sigma) {
  if (sigma <= 0) return;
  const lambda = (sigma * sigma) / 2;
  const nu = (1 + 2 * lambda - Math.sqrt(1 + 4 * lambda)) / (2 * lambda);
  const post = nu / lambda;
  const boundary = 1 / (1 - nu);
  const run = (plane, start, step, length) => {
    plane[start] *= boundary;
    for (let x = 1; x < length; x++) plane[start + x * step] += nu * plane[start + (x - 1) * step];
    plane[start + (length - 1) * step] *= boundary;
    for (let x = length - 1; x > 0; x--) plane[start + (x - 1) * step] += nu * plane[start + x * step];
  };
  eachChannel(data, w, h, (plane) => {
    for (let y = 0; y < h; y++) run(plane, y * w, 1, w);
    for (let x = 0; x < w; x++) run(plane, x, w, h);
    for (let i = 0; i < plane.length; i++) plane[i] = Math.round(plane[i] * post * post);
  });
}

// A running mean of 2r+1 pixels along rows (step 1) or columns (step w). index(k) maps a position
// outside the line back in: mirrored for boxblur, clamped for avgblur.
function meanPass(plane, w, h, radius, horizontal, index, round) {
  const lines = horizontal ? h : w;
  const length = horizontal ? w : h;
  const step = horizontal ? 1 : w;
  const out = new Float32Array(length);
  const size = 2 * radius + 1;
  for (let line = 0; line < lines; line++) {
    const start = horizontal ? line * w : line;
    const at = (k) => plane[start + index(k, length) * step];
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += at(k);
    for (let x = 0; x < length; x++) {
      out[x] = round(sum / size);
      sum += at(x + radius + 1) - at(x - radius);
    }
    for (let x = 0; x < length; x++) plane[start + x * step] = out[x];
  }
}

const mirror = (k, length) => (k < 0 ? Math.min(-k - 1, length - 1) : k >= length ? Math.max(2 * length - k - 1, 0) : k);
const clampIndex = (k, length) => (k < 0 ? 0 : k >= length ? length - 1 : k);

const videoBlur = {
  label: (v) => v.blur_type || "box",
  pixels(data, v, view) {
    const type = v.blur_type || "box";
    const { width: w, height: h, scale } = view;
    if (type === "gaussian") {
      gblur(data, w, h, num(v.sigma, 5) * scale);
      return;
    }
    const radius = Math.round(num(v.radius, 5) * scale);
    if (radius < 1) return;
    if (type === "average") {
      eachChannel(data, w, h, (plane) => {
        meanPass(plane, w, h, radius, true, clampIndex, (x) => x);
        meanPass(plane, w, h, radius, false, clampIndex, Math.floor);
      });
      return;
    }
    const power = Math.max(1, Math.trunc(num(v.power, 1)));
    eachChannel(data, w, h, (plane) => {
      for (let p = 0; p < power; p++) meanPass(plane, w, h, radius, true, mirror, Math.round);
      for (let p = 0; p < power; p++) meanPass(plane, w, h, radius, false, mirror, Math.round);
    });
  },
};

// ── Add Vignette (vf_vignette, square pixels, no dither) ─────────────────────

const vignette = {
  label: (v) => v.mode || "forward",
  pixels(data, v, view) {
    const W = view.sourceWidth;
    const H = view.sourceHeight;
    const angle = Math.max(0, Math.min(Math.PI / 2, num(v.angle, Math.PI / 5)));
    const x0 = W / 2 + (num(v.center_x, 0) * W) / 2;
    const y0 = H / 2 + (num(v.center_y, 0) * H) / 2;
    // FFmpeg keeps the scale as a 32-bit float, which decides which pixels land inside the circle.
    const yscale = Math.fround(num(v.aspect, 1));
    const backward = v.mode === "backward";
    const dmax = Math.hypot(W / 2, H / 2);
    for (let py = 0; py < view.height; py++) {
      const yy = Math.trunc((py / view.scale - y0) * yscale);
      for (let px = 0; px < view.width; px++) {
        const xx = Math.trunc(px / view.scale - x0);
        const d = Math.hypot(xx, yy) / dmax;
        let f = d > 1 ? 0 : Math.pow(Math.cos(angle * d), 4);
        if (backward) f = 1 / f;
        const i = (py * view.width + px) * 4;
        const [y, u, vv] = toYuv(data[i], data[i + 1], data[i + 2]);
        const scaled = (value, pivot) => {
          const out = f * (value - pivot) + pivot;
          return Number.isNaN(out) ? 0 : clamp255(Math.trunc(out));
        };
        writeRgb(data, i, scaled(y, 0), scaled(u, 127), scaled(vv, 127));
      }
    }
  },
};

// ── Add RGB Shift (rgbashift, edge smear) ────────────────────────────────────

// Shifts each channel within rows [top, bottom), repeating edge pixels. Positive moves right and down.
function shiftBand(data, source, w, top, bottom, shifts) {
  for (let y = top; y < bottom; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      for (let c = 0; c < 3; c++) {
        const [dx, dy] = shifts[c];
        const sx = clampIndex(x - dx, w);
        const sy = Math.max(top, Math.min(bottom - 1, y - dy));
        data[i + c] = source[(sy * w + sx) * 4 + c];
      }
    }
  }
}

const rgbShift = {
  pixels(data, v, view) {
    const intensity = num(v.intensity, 1);
    const shift = (name, fallback) => {
      const value = Math.max(-255, Math.min(255, Math.trunc(num(v[name], fallback) * intensity)));
      return Math.round(value * view.scale);
    };
    const shifts = [
      [shift("red_horizontal", -6), shift("red_vertical", 0)],
      [shift("green_horizontal", 6), shift("green_vertical", 0)],
      [shift("blue_horizontal", 0), shift("blue_vertical", 0)],
    ];
    const source = Uint8ClampedArray.from(data);
    const { width: w, height: h } = view;
    if (!v.tear_enabled) {
      shiftBand(data, source, w, 0, h, shifts);
      return;
    }
    // The tear: the band below tear_position is shifted on its own, with extra horizontal offset.
    const tearY = Math.round(Math.floor((view.sourceHeight * num(v.tear_position, 0.5)) / 2) * 2 * view.scale);
    const offset = Math.round(Math.trunc(num(v.tear_offset, 10)) * view.scale);
    shiftBand(data, source, w, 0, tearY, shifts);
    shiftBand(data, source, w, tearY, h, shifts.map(([dx, dy]) => [dx + offset, dy]));
  },
};

// ── Add Color Curves (vf_curves presets, natural cubic spline) ────────────────

const CURVE_PRESETS = {
  color_negative: {
    r: "0.129/1 0.466/0.498 0.725/0",
    g: "0.109/1 0.301/0.498 0.517/0",
    b: "0.098/1 0.235/0.498 0.423/0",
  },
  cross_process: {
    r: "0/0 0.25/0.156 0.501/0.501 0.686/0.745 1/1",
    g: "0/0 0.25/0.188 0.38/0.501 0.745/0.815 1/0.815",
    b: "0/0 0.231/0.094 0.709/0.874 1/1",
  },
  darker: { master: "0/0 0.5/0.4 1/1" },
  increase_contrast: { master: "0/0 0.149/0.066 0.831/0.905 0.905/0.98 1/1" },
  lighter: { master: "0/0 0.4/0.5 1/1" },
  linear_contrast: { master: "0/0 0.305/0.286 0.694/0.713 1/1" },
  medium_contrast: { master: "0/0 0.286/0.219 0.639/0.643 1/1" },
  negative: { master: "0/1 1/0" },
  strong_contrast: { master: "0/0 0.301/0.196 0.592/0.6 0.686/0.737 1/1" },
  vintage: {
    r: "0/0.11 0.42/0.51 1/0.95",
    g: "0/0 0.50/0.48 1/1",
    b: "0/0.22 0.49/0.44 1/0.8",
  },
};

function curveLut(spec) {
  const lut = new Uint8Array(256);
  const points = spec ? spec.split(" ").map((p) => p.split("/").map(Number)) : [];
  const n = points.length;
  const toByte = (y) => clamp255(Math.trunc(y * 255));
  if (n === 0) {
    for (let i = 0; i < 256; i++) lut[i] = i;
    return lut;
  }
  if (n === 1) {
    lut.fill(toByte(points[0][1]));
    return lut;
  }
  const x = points.map((p) => p[0]);
  const y = points.map((p) => p[1]);
  const hs = [];
  for (let i = 0; i < n - 1; i++) hs.push(x[i + 1] - x[i]);

  // Second derivatives M with M[0] = M[n-1] = 0, by a tridiagonal (Thomas) solve.
  const m = new Array(n).fill(0);
  if (n > 2) {
    const size = n - 2;
    const a = new Array(size), b = new Array(size), c = new Array(size), r = new Array(size);
    for (let k = 0; k < size; k++) {
      const i = k + 1;
      a[k] = hs[i - 1];
      b[k] = 2 * (hs[i - 1] + hs[i]);
      c[k] = hs[i];
      r[k] = 6 * ((y[i + 1] - y[i]) / hs[i] - (y[i] - y[i - 1]) / hs[i - 1]);
    }
    for (let k = 1; k < size; k++) {
      const w = a[k] / b[k - 1];
      b[k] -= w * c[k - 1];
      r[k] -= w * r[k - 1];
    }
    m[size] = r[size - 1] / b[size - 1];
    for (let k = size - 2; k >= 0; k--) m[k + 1] = (r[k] - c[k] * m[k + 2]) / b[k];
  }

  const first = Math.trunc(x[0] * 255);
  for (let X = 0; X < first; X++) lut[X] = toByte(y[0]);
  for (let i = 0; i < n - 1; i++) {
    const h = hs[i];
    const a = y[i];
    const b = (y[i + 1] - y[i]) / h - (h * m[i]) / 2 - (h * (m[i + 1] - m[i])) / 6;
    const c = m[i] / 2;
    const d = (m[i + 1] - m[i]) / (6 * h);
    const start = Math.trunc(x[i] * 255);
    const end = Math.trunc(x[i + 1] * 255);
    for (let X = start; X <= end; X++) {
      const t = (X - start) / 255;
      lut[X] = toByte(a + b * t + c * t * t + d * t * t * t);
    }
  }
  for (let X = Math.trunc(x[n - 1] * 255); X < 256; X++) lut[X] = toByte(y[n - 1]);
  return lut;
}

const curves = {
  label: (v) => v.curve_preset || "none",
  pixels(data, v) {
    const preset = CURVE_PRESETS[v.curve_preset];
    if (!preset) return;
    const channels = [curveLut(preset.r), curveLut(preset.g), curveLut(preset.b)];
    if (preset.master) {
      const master = curveLut(preset.master);
      for (const lut of channels) for (let j = 0; j < 256; j++) lut[j] = master[lut[j]];
    }
    for (let i = 0; i < data.length; i += 4) {
      data[i] = channels[0][data[i]];
      data[i + 1] = channels[1][data[i + 1]];
      data[i + 2] = channels[2][data[i + 2]];
    }
  },
};

// ── Add Film Grain (the node's noise and luminance-mask graph) ───────────────

// Seeded so the grain holds still while settings are dragged. FFmpeg's noise changes every frame.
function mulberry32(seed) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// The 3×3 neighbourhood helpers deflate and dilation share: rows clamp, columns mirror.
function neighbours(src, w, h, x, y, fn) {
  for (let dy = -1; dy <= 1; dy++) {
    const yy = Math.max(0, Math.min(h - 1, y + dy));
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      let xx = x + dx;
      if (xx < 0) xx = Math.min(1, w - 1);
      else if (xx >= w) xx = Math.max(w - 2, 0);
      fn(src[yy * w + xx]);
    }
  }
}

function grainNoise(w, h) {
  const random = mulberry32(0x9e3779b9);
  let noise = new Uint8Array(w * h);
  for (let i = 0; i < noise.length; i++) noise[i] = Math.min(255, Math.trunc(random() * 256));
  // deflate=threshold0=15
  let next = new Uint8Array(noise.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = noise[y * w + x];
      let sum = 0;
      neighbours(noise, w, h, x, y, (n) => { sum += n; });
      next[y * w + x] = Math.max(Math.min(Math.floor(sum / 8), p), Math.max(p - 15, 0));
    }
  }
  noise = next;
  // dilation=threshold0=10
  next = new Uint8Array(noise.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = noise[y * w + x];
      let max = p;
      neighbours(noise, w, h, x, y, (n) => { if (n > max) max = n; });
      next[y * w + x] = Math.min(max, Math.min(p + 10, 255));
    }
  }
  // eq=contrast=3
  for (let i = 0; i < next.length; i++) next[i] = clamp255(3 * next[i] - 257);
  return next;
}

// Bilinear upscale of the grain to the preview size (FFmpeg uses bicubic).
function sampleGrain(noise, nw, nh, x, y) {
  const fx = Math.max(0, Math.min(nw - 1, x - 0.5));
  const fy = Math.max(0, Math.min(nh - 1, y - 0.5));
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(nw - 1, x0 + 1), y1 = Math.min(nh - 1, y0 + 1);
  const tx = fx - x0, ty = fy - y0;
  const top = noise[y0 * nw + x0] * (1 - tx) + noise[y0 * nw + x1] * tx;
  const bottom = noise[y1 * nw + x0] * (1 - tx) + noise[y1 * nw + x1] * tx;
  return top * (1 - ty) + bottom * ty;
}

let grainCache = null;

const filmGrain = {
  pixels(data, v, view) {
    const grainScale = Math.max(1, num(v.grain_scale, 2));
    const intensity = num(v.grain_intensity, 0.15);
    const threshold = Math.trunc(num(v.luminance_threshold, 75));
    const { width: w, height: h } = view;
    // The node makes its grain at the frame's size divided by grain_scale, then scales it up.
    const nw = Math.max(1, Math.trunc(w / grainScale));
    const nh = Math.max(1, Math.trunc(h / grainScale));
    if (!grainCache || grainCache.w !== nw || grainCache.h !== nh) {
      grainCache = { w: nw, h: nh, noise: grainNoise(nw, nh) };
    }
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4;
        const n = sampleGrain(grainCache.noise, nw, nh, (px + 0.5) / grainScale, (py + 0.5) / grainScale);
        const luma = Math.round(16 + (65.481 * data[i] + 128.553 * data[i + 1] + 24.966 * data[i + 2]) / 255);
        const mask = clamp255(Math.trunc(intensity * (182 - Math.abs(threshold - luma))));
        const dark = Math.floor((n * mask) / 255);
        const alpha = clamp255(Math.round(((255 - dark - 16) * 255) / 219)) / 255;
        data[i] *= alpha;
        data[i + 1] *= alpha;
        data[i + 2] *= alpha;
      }
    }
  },
};


// ── Resize Video (scale / pad / crop as the node builds them) ────────────────

const makeEven = (v) => Math.max(2, Math.floor(Math.max(1, v) / 2) * 2);
// pad and crop centre the picture, then round the offset down to even for 4:2:0 chroma.
const evenOffset = (space) => Math.floor(Math.trunc(space / 2) / 2) * 2;

function videoResizeLayout(v, w, h) {
  const mode = v.resize_mode || "percentage";
  if (mode === "width") {
    const width = makeEven(Math.trunc(num(v.target_size, 1000)));
    return { width, height: Math.round((width * h) / (w * 2)) * 2 };
  }
  if (mode === "height") {
    const height = makeEven(Math.trunc(num(v.target_size, 1000)));
    return { width: Math.round((height * w) / (h * 2)) * 2, height };
  }
  if (mode === "percentage") {
    const f = num(v.percentage, 100) / 100;
    const width = Math.trunc((w * f) / 2) * 2, height = Math.trunc((h * f) / 2) * 2;
    if (width < 1 || height < 1) return { error: "The result would be 0 pixels wide or tall" };
    return { width, height };
  }
  if (mode !== "width and height") return { error: `Unknown resize mode "${mode}"` };
  const ew = makeEven(Math.trunc(num(v.target_width, 1000)));
  const eh = makeEven(Math.trunc(num(v.target_height, 1000)));
  const fit = v.fit_mode || "fit";
  if (fit === "stretch") return { width: ew, height: eh, fit };
  // force_original_aspect_ratio: decrease for fit, increase for fill.
  const tmpW = Math.round((eh * w) / h), tmpH = Math.round((ew * h) / w);
  const pick = fit === "fill" ? Math.max : Math.min;
  const sw = pick(tmpW, ew), sh = pick(tmpH, eh);
  const x = fit === "fill" ? -evenOffset(sw - ew) : evenOffset(ew - sw);
  const y = fit === "fill" ? -evenOffset(sh - eh) : evenOffset(eh - sh);
  return { width: ew, height: eh, fit, sw, sh, x, y };
}

// _format_ffmpeg_color: 3- or 6-digit hex, anything else white.
function padColor(color) {
  let hex = String(color || "").replace(/^#/, "");
  if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
  return /^[0-9a-fA-F]{6}$/.test(hex) ? `#${hex}` : "#ffffff";
}

const resizeVideo = {
  size: (v, w, h) => videoResizeLayout(v, w, h),
  label: (v) => v.resize_mode || "percentage",
  draw(ctx, img, v, view) {
    const layout = videoResizeLayout(v, view.sourceWidth, view.sourceHeight);
    ctx.imageSmoothingEnabled = v.scaling_algorithm !== "neighbor";
    ctx.imageSmoothingQuality = "high";
    if (layout.sw === undefined) {
      ctx.drawImage(img, 0, 0, view.width, view.height);
      return;
    }
    if (layout.fit === "fit") {
      ctx.fillStyle = padColor(v.background_color);
      ctx.fillRect(0, 0, view.width, view.height);
    }
    const s = view.scale;
    // In 4:2:0 a padded picture with an odd size loses its last row or column; the offset still comes
    // from the odd size.
    const even = (n) => (layout.fit === "fit" ? Math.floor(n / 2) * 2 : n);
    ctx.drawImage(img, layout.x * s, layout.y * s, even(layout.sw) * s, even(layout.sh) * s);
  },
};

export const VIDEO_OPERATIONS = {
  video_eq: videoEq,
  video_blur: videoBlur,
  vignette,
  rgb_shift: rgbShift,
  curves,
  film_grain: filmGrain,
  resize_video: resizeVideo,
};
