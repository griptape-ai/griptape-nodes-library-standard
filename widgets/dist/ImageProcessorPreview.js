// src/ImageProcessorPreview/_pixels.ts
function luma(r, g, b) {
  return r * 19595 + g * 38470 + b * 7471 + 32768 >> 16;
}
function blendByte(a, b, t) {
  const value = a + t * (b - a);
  if (t >= 0 && t <= 1) return Math.trunc(value);
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.trunc(value);
}
function applyLut(data, table) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = table[data[i]];
    data[i + 1] = table[data[i + 1]];
    data[i + 2] = table[data[i + 2]];
  }
}
function boxRadius(radius, passes) {
  const sigma2 = radius * radius / passes;
  const l = Math.floor((Math.sqrt(12 * sigma2 + 1) - 1) / 2);
  const a = (2 * l + 1) * (l * (l + 1) - 3 * sigma2) / (6 * (sigma2 - (l + 1) * (l + 1)));
  return l + a;
}
function boxPass(src, dst, width, height, radius, horizontal) {
  const whole = Math.floor(radius);
  const frac = radius - whole;
  const norm = 1 / (2 * radius + 1);
  const lines = horizontal ? height : width;
  const length = horizontal ? width : height;
  const step = horizontal ? 4 : 4 * width;
  const clamp = (k) => k < 0 ? 0 : k >= length ? length - 1 : k;
  for (let line = 0; line < lines; line++) {
    const start = horizontal ? line * width * 4 : line * 4;
    for (let c = 0; c < 4; c++) {
      const at = (k) => src[start + clamp(k) * step + c];
      let sum = 0;
      for (let k = -whole; k <= whole; k++) sum += at(k);
      for (let x = 0; x < length; x++) {
        dst[start + x * step + c] = (sum + frac * (at(x - whole - 1) + at(x + whole + 1))) * norm;
        sum += at(x + whole + 1) - at(x - whole);
      }
    }
  }
}
function gaussianBlur(data, width, height, radius) {
  if (radius <= 0) return;
  const box = boxRadius(radius, 3);
  let a = Float32Array.from(data);
  let b = new Float32Array(data.length);
  for (let pass = 0; pass < 3; pass++) {
    boxPass(a, b, width, height, box, true);
    [a, b] = [b, a];
  }
  for (let pass = 0; pass < 3; pass++) {
    boxPass(a, b, width, height, box, false);
    [a, b] = [b, a];
  }
  for (let i = 0; i < data.length; i++) data[i] = Math.round(a[i]);
}

// src/ImageProcessorPreview/_video_operations.ts
var num = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
var clamp255 = (v) => v < 0 ? 0 : v > 255 ? 255 : v;
function toYuv(r, g, b) {
  return [
    Math.round(16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255),
    Math.round(128 + (-37.797 * r - 74.203 * g + 112 * b) / 255),
    Math.round(128 + (112 * r - 93.786 * g - 18.214 * b) / 255)
  ];
}
function writeRgb(data, i, y, u, v) {
  const yy = 1.164383 * (y - 16);
  data[i] = yy + 1.596027 * (v - 128);
  data[i + 1] = yy - 0.391762 * (u - 128) - 0.812968 * (v - 128);
  data[i + 2] = yy + 2.017232 * (u - 128);
}
function eqLut(contrast, brightness, gamma) {
  const lut = new Uint8Array(256);
  if (contrast === 1 && brightness === 0 && gamma === 1) {
    for (let i = 0; i < 256; i++) lut[i] = i;
    return { lut, identity: true };
  }
  if (gamma === 1 && Math.abs(contrast) < 7.9) {
    const ci = Math.trunc(contrast * 4096);
    const bi = Math.trunc(Math.trunc(100 * brightness + 100) * 511 / 200) - 128 - Math.trunc(ci / 32);
    for (let i = 0; i < 256; i++) lut[i] = clamp255((i * ci >> 12) + bi);
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
var videoEq = {
  pixels(data, v) {
    const y = eqLut(num(v.contrast, 1), num(v.brightness, 0), num(v.gamma, 1));
    const c = eqLut(num(v.saturation, 1), 0, 1);
    if (y.identity && c.identity) return;
    for (let i = 0; i < data.length; i += 4) {
      const [yy, u, vv] = toYuv(data[i], data[i + 1], data[i + 2]);
      writeRgb(data, i, y.lut[yy], c.lut[u], c.lut[vv]);
    }
  }
};
function eachChannel(data, w, h, fn) {
  const plane = new Float32Array(w * h);
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < w * h; i++) plane[i] = data[i * 4 + c];
    fn(plane);
    for (let i = 0; i < w * h; i++) data[i * 4 + c] = plane[i];
  }
}
function gblur(data, w, h, sigma) {
  if (sigma <= 0) return;
  const lambda = sigma * sigma / 2;
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
var mirror = (k, length) => k < 0 ? Math.min(-k - 1, length - 1) : k >= length ? Math.max(2 * length - k - 1, 0) : k;
var clampIndex = (k, length) => k < 0 ? 0 : k >= length ? length - 1 : k;
var videoBlur = {
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
  }
};
var vignette = {
  label: (v) => v.mode || "forward",
  pixels(data, v, view) {
    const W = view.sourceWidth;
    const H = view.sourceHeight;
    const angle = Math.max(0, Math.min(Math.PI / 2, num(v.angle, Math.PI / 5)));
    const x0 = W / 2 + num(v.center_x, 0) * W / 2;
    const y0 = H / 2 + num(v.center_y, 0) * H / 2;
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
  }
};
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
var rgbShift = {
  pixels(data, v, view) {
    const intensity = num(v.intensity, 1);
    const shift = (name, fallback) => {
      const value = Math.max(-255, Math.min(255, Math.trunc(num(v[name], fallback) * intensity)));
      return Math.round(value * view.scale);
    };
    const shifts = [
      [shift("red_horizontal", -6), shift("red_vertical", 0)],
      [shift("green_horizontal", 6), shift("green_vertical", 0)],
      [shift("blue_horizontal", 0), shift("blue_vertical", 0)]
    ];
    const source = Uint8ClampedArray.from(data);
    const { width: w, height: h } = view;
    if (!v.tear_enabled) {
      shiftBand(data, source, w, 0, h, shifts);
      return;
    }
    const tearY = Math.round(Math.floor(view.sourceHeight * num(v.tear_position, 0.5) / 2) * 2 * view.scale);
    const offset = Math.round(Math.trunc(num(v.tear_offset, 10)) * view.scale);
    shiftBand(data, source, w, 0, tearY, shifts);
    shiftBand(data, source, w, tearY, h, shifts.map(([dx, dy]) => [dx + offset, dy]));
  }
};
var CURVE_PRESETS = {
  color_negative: {
    r: "0.129/1 0.466/0.498 0.725/0",
    g: "0.109/1 0.301/0.498 0.517/0",
    b: "0.098/1 0.235/0.498 0.423/0"
  },
  cross_process: {
    r: "0/0 0.25/0.156 0.501/0.501 0.686/0.745 1/1",
    g: "0/0 0.25/0.188 0.38/0.501 0.745/0.815 1/0.815",
    b: "0/0 0.231/0.094 0.709/0.874 1/1"
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
    b: "0/0.22 0.49/0.44 1/0.8"
  }
};
function curveLut(spec) {
  const lut = new Uint8Array(256);
  const points = spec ? spec.split(" ").map((p) => p.split("/").map(Number)) : [];
  const n = points.length;
  const toByte = (y2) => clamp255(Math.trunc(y2 * 255));
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
    const b = (y[i + 1] - y[i]) / h - h * m[i] / 2 - h * (m[i + 1] - m[i]) / 6;
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
var curves = {
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
  }
};
function mulberry32(seed) {
  let a = seed;
  return () => {
    a = a + 1831565813 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}
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
  const random = mulberry32(2654435769);
  let noise = new Uint8Array(w * h);
  for (let i = 0; i < noise.length; i++) noise[i] = Math.min(255, Math.trunc(random() * 256));
  let next = new Uint8Array(noise.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = noise[y * w + x];
      let sum = 0;
      neighbours(noise, w, h, x, y, (n) => {
        sum += n;
      });
      next[y * w + x] = Math.max(Math.min(Math.floor(sum / 8), p), Math.max(p - 15, 0));
    }
  }
  noise = next;
  next = new Uint8Array(noise.length);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const p = noise[y * w + x];
      let max = p;
      neighbours(noise, w, h, x, y, (n) => {
        if (n > max) max = n;
      });
      next[y * w + x] = Math.min(max, Math.min(p + 10, 255));
    }
  }
  for (let i = 0; i < next.length; i++) next[i] = clamp255(3 * next[i] - 257);
  return next;
}
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
var grainCache = null;
var filmGrain = {
  pixels(data, v, view) {
    const grainScale = Math.max(1, num(v.grain_scale, 2));
    const intensity = num(v.grain_intensity, 0.15);
    const threshold = Math.trunc(num(v.luminance_threshold, 75));
    const { width: w, height: h } = view;
    const nw = Math.max(1, Math.trunc(w / grainScale));
    const nh = Math.max(1, Math.trunc(h / grainScale));
    if (!grainCache || grainCache.w !== nw || grainCache.h !== nh) {
      grainCache = { w: nw, h: nh, noise: grainNoise(nw, nh) };
    }
    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const i = (py * w + px) * 4;
        const n = sampleGrain(grainCache.noise, nw, nh, (px + 0.5) / grainScale, (py + 0.5) / grainScale);
        const luma2 = Math.round(16 + (65.481 * data[i] + 128.553 * data[i + 1] + 24.966 * data[i + 2]) / 255);
        const mask = clamp255(Math.trunc(intensity * (182 - Math.abs(threshold - luma2))));
        const dark = Math.floor(n * mask / 255);
        const alpha = clamp255(Math.round((255 - dark - 16) * 255 / 219)) / 255;
        data[i] *= alpha;
        data[i + 1] *= alpha;
        data[i + 2] *= alpha;
      }
    }
  }
};
var makeEven = (v) => Math.max(2, Math.floor(Math.max(1, v) / 2) * 2);
var evenOffset = (space) => Math.floor(Math.trunc(space / 2) / 2) * 2;
function videoResizeLayout(v, w, h) {
  const mode = v.resize_mode || "percentage";
  if (mode === "width") {
    const width = makeEven(Math.trunc(num(v.target_size, 1e3)));
    return { width, height: Math.round(width * h / (w * 2)) * 2 };
  }
  if (mode === "height") {
    const height = makeEven(Math.trunc(num(v.target_size, 1e3)));
    return { width: Math.round(height * w / (h * 2)) * 2, height };
  }
  if (mode === "percentage") {
    const f = num(v.percentage, 100) / 100;
    const width = Math.trunc(w * f / 2) * 2, height = Math.trunc(h * f / 2) * 2;
    if (width < 1 || height < 1) return { error: "The result would be 0 pixels wide or tall" };
    return { width, height };
  }
  if (mode !== "width and height") return { error: `Unknown resize mode "${mode}"` };
  const ew = makeEven(Math.trunc(num(v.target_width, 1e3)));
  const eh = makeEven(Math.trunc(num(v.target_height, 1e3)));
  const fit = v.fit_mode || "fit";
  if (fit === "stretch") return { width: ew, height: eh, fit };
  const tmpW = Math.round(eh * w / h), tmpH = Math.round(ew * h / w);
  const pick = fit === "fill" ? Math.max : Math.min;
  const sw = pick(tmpW, ew), sh = pick(tmpH, eh);
  const x = fit === "fill" ? -evenOffset(sw - ew) : evenOffset(ew - sw);
  const y = fit === "fill" ? -evenOffset(sh - eh) : evenOffset(eh - sh);
  return { width: ew, height: eh, fit, sw, sh, x, y };
}
function padColor(color) {
  let hex = String(color || "").replace(/^#/, "");
  if (hex.length === 3) hex = hex.split("").map((c) => c + c).join("");
  return /^[0-9a-fA-F]{6}$/.test(hex) ? `#${hex}` : "#ffffff";
}
var resizeVideo = {
  size: (v, w, h) => videoResizeLayout(v, w, h),
  label: (v) => v.resize_mode || "percentage",
  draw(ctx, img, v, view) {
    const layout = videoResizeLayout(v, view.sourceWidth, view.sourceHeight);
    ctx.imageSmoothingEnabled = v.scaling_algorithm !== "neighbor";
    ctx.imageSmoothingQuality = "high";
    if (layout.sw === void 0) {
      ctx.drawImage(img, 0, 0, view.width, view.height);
      return;
    }
    if (layout.fit === "fit") {
      ctx.fillStyle = padColor(v.background_color);
      ctx.fillRect(0, 0, view.width, view.height);
    }
    const s = view.scale;
    const even = (n) => layout.fit === "fit" ? Math.floor(n / 2) * 2 : n;
    ctx.drawImage(img, layout.x * s, layout.y * s, even(layout.sw) * s, even(layout.sh) * s);
  }
};
var VIDEO_OPERATIONS = {
  video_eq: videoEq,
  video_blur: videoBlur,
  vignette,
  rgb_shift: rgbShift,
  curves,
  film_grain: filmGrain,
  resize_video: resizeVideo
};

// src/ImageProcessorPreview/_color_bars.ts
var num2 = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
var fl = Math.floor;
var tr = Math.trunc;
var pymod = (a, m) => {
  const r = a % m;
  return r !== 0 && r < 0 !== m < 0 ? r + m : r;
};
var BLACK = [0, 0, 0];
var WHITE = [255, 255, 255];
var GRAY = [128, 128, 128];
var C100 = [[255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0], [255, 0, 255], [255, 0, 0], [0, 0, 255]];
var C75 = C100.map((c) => c.map((v) => v === 255 ? 191 : v));
var SMALL75 = [[0, 0, 191], [191, 0, 191], [0, 191, 191], [191, 191, 191], [191, 191, 0], [191, 0, 0], [0, 191, 0]];
var SMALL100 = SMALL75.map((c) => c.map((v) => v === 191 ? 255 : v));
var FULL_FIELD = {
  "Full Field White": [255, 255, 255],
  "Full Field Blue": [0, 0, 255],
  "Full Field Cyan": [0, 255, 255],
  "Full Field Green": [0, 255, 0],
  "Full Field Magenta": [255, 0, 255],
  "Full Field Red": [255, 0, 0],
  "Full Field Yellow": [255, 255, 0]
};
var Canvas = class {
  constructor(w, h, fill = BLACK) {
    this.w = w;
    this.h = h;
    this.data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) this.set(i, fill);
  }
  set(i, [r, g, b]) {
    this.data[i * 4] = r;
    this.data[i * 4 + 1] = g;
    this.data[i * 4 + 2] = b;
    this.data[i * 4 + 3] = 255;
  }
  // PIL draw.rectangle: both corners inclusive, clipped to the image.
  rect(x0, y0, x1, y1, color) {
    const xa = Math.max(0, x0), ya = Math.max(0, y0);
    const xb = Math.min(this.w - 1, x1), yb = Math.min(this.h - 1, y1);
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) this.set(y * this.w + x, color);
  }
  pixel(x, y, color) {
    this.set(y * this.w + x, color);
  }
  // Vertical bars, the last one running to the right edge.
  bars(colors, bottom) {
    const bw = fl(this.w / colors.length);
    colors.forEach((c, i) => this.rect(i * bw, 0, i < colors.length - 1 ? (i + 1) * bw : this.w, bottom, c));
  }
};
var barBottom = (h) => tr(h * 0.67);
function smpteBars(cv) {
  const { w, h } = cv;
  const by = barBottom(h);
  cv.bars(C75, by);
  const lsw = fl(w / 4);
  const sbw = fl((w - lsw) / 7);
  cv.rect(0, by, fl(lsw / 3), h, BLACK);
  cv.rect(fl(lsw / 3), by, fl(lsw * 2 / 3), h, GRAY);
  cv.rect(fl(lsw * 2 / 3), by, lsw, h, [191, 191, 191]);
  SMALL75.forEach((c, i) => cv.rect(lsw + i * sbw, by, i < 6 ? lsw + (i + 1) * sbw : w, h, c));
}
function checker(cv, light) {
  const by = barBottom(cv.h);
  const cs = Math.min(fl(cv.w / 20), fl((cv.h - by) / 4));
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      cv.rect(col * cs, by + row * cs, col * cs + cs, by + row * cs + cs, (row + col) % 2 === 0 ? light : BLACK);
    }
  }
}
function fullFieldBars(cv, colors, white, small) {
  const { w, h } = cv;
  const by = barBottom(h);
  cv.bars(colors, by);
  const s = fl(w / 11);
  cv.rect(0, by, 3 * s, h, BLACK);
  cv.rect(s, by, 2 * s, h, white);
  small.forEach((c, i) => cv.rect(3 * s + i * s, by, i < 6 ? 3 * s + (i + 1) * s : w, h, c));
}
function stairs(cv, steps, vertical) {
  const length = vertical ? cv.h : cv.w;
  const sw = fl(length / steps);
  for (let i = 0; i < steps; i++) {
    const g = tr(255 * i / (steps - 1));
    const end = i < steps - 1 ? (i + 1) * sw : length;
    if (vertical) cv.rect(0, i * sw, cv.w, end, [g, g, g]);
    else cv.rect(i * sw, 0, end, cv.h, [g, g, g]);
  }
}
function eachColumn(cv, fn) {
  for (let x = 0; x < cv.w; x++) {
    const c = fn(x);
    for (let y = 0; y < cv.h; y++) cv.pixel(x, y, c);
  }
}
function hsvFull(h) {
  const i = tr(h * 6);
  const f = h * 6 - i;
  const q = 1 - f;
  const rgb = [[1, f, 0], [q, 1, 0], [0, 1, f], [0, q, 1], [f, 0, 1], [1, 0, q]][pymod(i, 6)];
  return rgb.map((c) => tr(c * 255));
}
function pluge(cv, v) {
  const { w, h } = cv;
  const setup = v.pluge_ire_setup || "NTSC 7.5 IRE";
  const ire = setup === "PAL 0 IRE" ? [-2, 0, 2] : setup === "RGB Full Range" ? [-1, 0, 1.5] : [3.5, 7.5, 11.5];
  const [superBlack, black, above] = ire.map((x) => Math.max(0, tr(x / 100 * 255)));
  const count = Math.max(2, Math.min(5, tr(num2(v.pluge_bar_count, 3))));
  const levels2 = count >= 3 ? [superBlack, black, above] : [black, above];
  for (let i = 0; i < count - 3; i++) levels2.push(tr(black + (above - black) * ((i + 1) / (count - 2))));
  if ((v.pluge_orientation || "vertical") === "horizontal") {
    const bh = fl(h / (count + 2)), bw = fl(w / 2), cx = fl(w / 2);
    const sy = fl((h - bh * count) / 2);
    levels2.forEach((g, i) => cv.rect(cx - fl(bw / 2), sy + i * bh, cx + fl(bw / 2), sy + i * bh + bh, [g, g, g]));
  } else {
    const bw = fl(w / (count + 3)), bh = fl(h / 2), by = fl((h - bh) / 2);
    const sx = fl(w / 2) - fl(bw * levels2.length / 2);
    levels2.forEach((g, i) => cv.rect(sx + i * bw, by, sx + i * bw + bw, by + bh, [g, g, g]));
  }
}
var PATTERNS = {
  "SMPTE 219-100 Bars": (cv) => cv.bars(C100, cv.h),
  "SMPTE 75% Bars": (cv) => cv.bars(C75, cv.h),
  "SMPTE Bars": smpteBars,
  "SMPTE 219+i Bars": (cv) => {
    smpteBars(cv);
    checker(cv, WHITE);
  },
  "100% Full Field Bars": (cv) => fullFieldBars(cv, C100, WHITE, SMALL100),
  "75% Full Field Bars": (cv) => fullFieldBars(cv, C75, [191, 191, 191], SMALL75),
  "75% Bars Over Red": (cv) => {
    fullFieldBars(cv, C75, [191, 191, 191], SMALL75);
    const by = barBottom(cv.h);
    const b = fl(cv.w / 3);
    cv.rect(0, by, cv.w, cv.h, [191, 0, 0]);
    cv.rect(0, by, b, cv.h, [191, 191, 191]);
    cv.rect(2 * b, by, cv.w, cv.h, BLACK);
  },
  "EBU Bars": (cv) => cv.bars([...C100, BLACK], cv.h),
  "ARIB 28-100": (cv) => {
    smpteBars(cv);
    checker(cv, WHITE);
  },
  "ARIB 28-75": (cv) => {
    const { w, h } = cv;
    const by = barBottom(h);
    cv.bars(C75, by);
    const lsw = fl(w / 4);
    cv.rect(0, by, fl(lsw / 3), h, BLACK);
    cv.rect(fl(lsw / 3), by, fl(lsw * 2 / 3), h, GRAY);
    cv.rect(fl(lsw * 2 / 3), by, lsw, h, [191, 191, 191]);
    checker(cv, [191, 191, 191]);
    const sbw = fl((w - lsw) / 7);
    SMALL75.forEach((c, i) => cv.rect(lsw + i * sbw, by, i < 6 ? lsw + (i + 1) * sbw : w, h, c));
  },
  "ARIB 28+i": (cv) => {
    smpteBars(cv);
    checker(cv, WHITE);
  },
  "HD Color Bars": (cv) => {
    const by = barBottom(cv.h);
    cv.bars(C100, by);
    const rw = fl(cv.w / 8);
    for (let i = 0; i < 8; i++) {
      const g = tr(255 * (i / 7));
      cv.rect(i * rw, by, i < 7 ? (i + 1) * rw : cv.w, cv.h, [g, g, g]);
    }
  },
  "Zone Plate": (cv) => {
    const { w, h } = cv;
    const cx = fl(w / 2), cy = fl(h / 2), r = fl(Math.min(w, h) / 2);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const dx = x - cx, dy = y - cy;
        const d = Math.sqrt(dx * dx + dy * dy);
        const ring = d < r ? tr(pymod(d / r * 10, 2) * 255) : 128;
        const radial = tr(pymod(Math.atan2(dy, dx) / Math.PI * 10, 2) * 255);
        const g = tr((ring + radial) / 2);
        cv.pixel(x, y, [g, g, g]);
      }
    }
  },
  "Tartan Bars": (cv) => {
    const colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], WHITE, BLACK];
    const t = fl(Math.min(cv.w, cv.h) / 16);
    if (t <= 0) throw new Error("Tartan Bars needs an image at least 16 pixels on each side");
    for (let y = 0; y < cv.h; y += t) {
      for (let x = 0; x < cv.w; x += t) {
        cv.rect(x, y, Math.min(x + t, cv.w), Math.min(y + t, cv.h), colors[(fl(x / t) + fl(y / t)) % 8]);
      }
    }
  },
  "Stair 5 Step": (cv) => stairs(cv, 5, false),
  "Stair 5 Step Vert": (cv) => stairs(cv, 5, true),
  "Stair 10 Step": (cv) => stairs(cv, 10, false),
  "Stair 10 Step Vert": (cv) => stairs(cv, 10, true),
  "Y Ramp Up": (cv) => {
    if (cv.w < 2) throw new Error("Y Ramp Up needs an image at least 2 pixels wide");
    eachColumn(cv, (x) => {
      const g = tr(255 * (x / (cv.w - 1)));
      return [g, g, g];
    });
  },
  "Y Ramp Down": (cv) => {
    if (cv.w < 2) throw new Error("Y Ramp Down needs an image at least 2 pixels wide");
    eachColumn(cv, (x) => {
      const g = tr(255 * (1 - x / (cv.w - 1)));
      return [g, g, g];
    });
  },
  "Vertical Ramp": (cv) => {
    if (cv.h < 2) throw new Error("Vertical Ramp needs an image at least 2 pixels tall");
    for (let y = 0; y < cv.h; y++) {
      const g = tr(255 * (y / (cv.h - 1)));
      for (let x = 0; x < cv.w; x++) cv.pixel(x, y, [g, g, g]);
    }
  },
  "Legal Chroma Ramp": (cv) => {
    if (cv.w < 2) throw new Error("Legal Chroma Ramp needs an image at least 2 pixels wide");
    eachColumn(cv, (x) => {
      const t = x / (cv.w - 1);
      if (t < 0.33) return [tr(255 * t * 3), 0, 0];
      if (t < 0.66) return [255, tr(255 * (t - 0.33) * 3), 0];
      return [tr(255 * (1 - (t - 0.66) * 3)), 255, tr(255 * (t - 0.66) * 3)];
    });
  },
  "Full Chroma Ramp": (cv) => eachColumn(cv, (x) => hsvFull(x / cv.w)),
  "Chroma Ramp": (cv) => eachColumn(cv, (x) => hsvFull(x / cv.w)),
  "Multi Burst": (cv) => {
    cv.rect(0, 0, cv.w, cv.h, GRAY);
    let cur = 0;
    for (const sw of [fl(cv.w / 4), fl(cv.w / 8), fl(cv.w / 16), fl(cv.w / 32)]) {
      if (sw <= 0) throw new Error("Multi Burst needs an image at least 32 pixels wide");
      const sec = fl(cv.w / 4);
      for (let x = cur; x < cur + sec; x += sw * 2) cv.rect(x, 0, Math.min(x + sw, cur + sec), cv.h, BLACK);
      cur += sec;
    }
  },
  "Pluge": pluge,
  "Bowtie": (cv) => {
    cv.rect(0, 0, cv.w, cv.h, GRAY);
    const cx = fl(cv.w / 2), sw = fl(cv.w / 20);
    if (cx === 0) throw new Error("Bowtie needs an image at least 2 pixels wide");
    for (let x = 0; x < cv.w; x++) {
      const f = Math.max(1, tr(sw * (1 + Math.abs(x - cx) / cx)));
      if (fl(x / f) % 2 === 0) cv.rect(x, 0, x + 1, cv.h, BLACK);
    }
  },
  "Pathological EG": (cv) => cv.rect(0, 0, cv.w, cv.h, [128, 0, 128]),
  "Pathological PLL": (cv) => cv.rect(0, 0, cv.w, cv.h, [64, 64, 64]),
  "Pathological EG/PLL": (cv) => {
    cv.rect(0, 0, cv.w, fl(cv.h / 2), [128, 0, 128]);
    cv.rect(0, fl(cv.h / 2), cv.w, cv.h, [64, 64, 64]);
  },
  "AV Delay Pattern 1": (cv) => {
    smpteBars(cv);
    checker(cv, WHITE);
    const by = barBottom(cv.h);
    const rs = fl(cv.w / 10), rx = fl((cv.w - rs) / 2), ry = by + fl((cv.h - by - rs) / 2);
    cv.rect(rx, ry, rx + rs, ry + rs, [255, 0, 0]);
  },
  "AV Delay Pattern 2": (cv) => {
    cv.rect(0, 0, cv.w, cv.h, GRAY);
    const bh = fl(cv.h / 20), byy = fl((cv.h - bh) / 2);
    cv.rect(0, byy, cv.w, byy + bh, WHITE);
    const sq = Math.min(fl(cv.w / 10), fl(cv.h / 10));
    const sx = cv.w - sq - fl(cv.w / 20), sy = byy - fl((sq - bh) / 2);
    cv.rect(sx, sy, sx + sq, sy + sq, BLACK);
  },
  "Bouncing Box": (cv) => {
    smpteBars(cv);
    const by = barBottom(cv.h);
    const bs = Math.min(fl(cv.w / 8), fl((cv.h - by) / 2));
    const bx = fl((cv.w - bs) / 2), bY = by + fl((cv.h - by - bs) / 2);
    cv.rect(bx, bY, bx + bs, bY + bs, [255, 0, 0]);
  }
};
for (const [name, color] of Object.entries(FULL_FIELD)) PATTERNS[name] = (cv) => cv.rect(0, 0, cv.w, cv.h, color);
var MAX_PIXELS = 16e6;
var colorBars = {
  generate: true,
  label: (v) => v.bar_type || "SMPTE 219-100 Bars",
  size(v) {
    const w = num2(v.width, 1920);
    const h = num2(v.height, 1080);
    if (w <= 0 || h <= 0) return { error: "Width and height must be greater than 0" };
    const width = Math.max(1, tr(w));
    const height = Math.max(1, tr(h));
    if (width * height > MAX_PIXELS) return { error: "Too large to preview" };
    return { width, height };
  },
  pixels(v, width, height) {
    const pattern = PATTERNS[v.bar_type] ?? PATTERNS["SMPTE 219-100 Bars"];
    const cv = new Canvas(width, height);
    pattern(cv, v);
    return cv.data;
  }
};

// src/ImageProcessorPreview/_text.ts
var PREVIEW_FONT = "GriptapePillowDefault";
var num3 = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
var fontSpec = (px) => `${px}px "${PREVIEW_FONT}", sans-serif`;
var NAMED_COLORS = {
  transparent: [0, 0, 0, 0],
  black: [0, 0, 0, 255],
  white: [255, 255, 255, 255],
  red: [255, 0, 0, 255],
  green: [0, 128, 0, 255],
  blue: [0, 0, 255, 255],
  yellow: [255, 255, 0, 255],
  cyan: [0, 255, 255, 255],
  magenta: [255, 0, 255, 255],
  gray: [128, 128, 128, 255],
  grey: [128, 128, 128, 255],
  orange: [255, 165, 0, 255],
  purple: [128, 0, 128, 255],
  pink: [255, 192, 203, 255],
  brown: [165, 42, 42, 255],
  lime: [0, 255, 0, 255],
  navy: [0, 0, 128, 255],
  teal: [0, 128, 128, 255],
  olive: [128, 128, 0, 255],
  maroon: [128, 0, 0, 255],
  silver: [192, 192, 192, 255],
  gold: [255, 215, 0, 255]
};
function hlsToRgb(h, l, s) {
  if (s === 0) return [l, l, l];
  const m2 = l <= 0.5 ? l * (1 + s) : l + s - l * s;
  const m1 = 2 * l - m2;
  const v = (hue) => {
    hue = (hue % 1 + 1) % 1;
    if (hue < 1 / 6) return m1 + (m2 - m1) * hue * 6;
    if (hue < 0.5) return m2;
    if (hue < 2 / 3) return m1 + (m2 - m1) * (2 / 3 - hue) * 6;
    return m1;
  };
  return [v(h + 1 / 3), v(h), v(h - 1 / 3)];
}
function parseColor(input) {
  const s = String(input ?? "").trim().toLowerCase();
  if (s.startsWith("#")) {
    const hex = s.slice(1);
    if (!/^[0-9a-f]{6}([0-9a-f]{2})?$/.test(hex)) return null;
    const bytes = hex.match(/../g).map((b) => parseInt(b, 16));
    return bytes.length === 3 ? [...bytes, 255] : bytes;
  }
  const byte = (x) => x >= 0 && x <= 255 ? x : NaN;
  const unit = (x) => x >= 0 && x <= 1 ? Math.trunc(x * 255) : NaN;
  let m;
  if (m = /^rgb\((\d+),\s*(\d+),\s*(\d+)\)/.exec(s)) {
    const c = [byte(+m[1]), byte(+m[2]), byte(+m[3]), 255];
    return c.some(Number.isNaN) ? null : c;
  }
  if (m = /^rgba\((\d+),\s*(\d+),\s*(\d+),\s*([\d.]+)\)/.exec(s)) {
    const c = [byte(+m[1]), byte(+m[2]), byte(+m[3]), unit(+m[4])];
    return c.some(Number.isNaN) ? null : c;
  }
  if (m = /^hsla?\((\d+),\s*(\d+)%,\s*(\d+)%(?:,\s*([\d.]+))?\)/.exec(s)) {
    const [h, sat, l] = [+m[1], +m[2], +m[3]];
    if (h > 360 || sat > 100 || l > 100) return null;
    const alpha = m[4] === void 0 ? 255 : unit(+m[4]);
    if (Number.isNaN(alpha) || s.startsWith("hsla") && m[4] === void 0) return null;
    return [...hlsToRgb(h / 360, l / 100, sat / 100).map((c) => Math.trunc(c * 255)), alpha];
  }
  return NAMED_COLORS[s] ?? null;
}
var rgbCss = ([r, g, b]) => `rgb(${r},${g},${b})`;
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
var textImage = {
  generate: true,
  usesFont: true,
  size(v) {
    const width = Math.trunc(num3(v.width, 512));
    const height = Math.trunc(num3(v.height, 512));
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
    const size = Math.trunc(num3(v.font_size, 36));
    if (size <= 0 || !text.trim()) return;
    const layout = layoutText(ctx, text, size, 4, "left");
    const [l, t, r, b] = layout.box;
    const x = Math.floor((width - (r - l)) / 2);
    const y = Math.floor((height - (b - t)) / 2);
    ctx.fillStyle = rgbCss(ink);
    drawLayout(ctx, layout, size, x, y);
  }
};
function pyStr(value) {
  if (value === true) return "True";
  if (value === false) return "False";
  if (value === null || value === void 0) return "None";
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
var blend = (m, a, b) => Math.round((a * (255 - m) + b * m) / 255);
var textOverlay = {
  usesFont: true,
  draw(ctx, img, v, view) {
    const { width: W, height: H, scale } = view;
    ctx.drawImage(img, 0, 0, W, H);
    const text = expandTemplate(String(v.text ?? ""), v.template_values);
    const ink = parseColor(v.text_color || "#ffffffff");
    const background = parseColor(v.text_background || "#000000ff");
    const size = num3(v.font_size, 36) * scale;
    if (!text.trim() || !ink || !background || size <= 0) return;
    const margin = Math.max(0, num3(v.margin, 10)) * scale;
    const pad = 4 * scale;
    const align = v.text_horizontal_alignment || "left";
    const valign = v.text_vertical_alignment || "top";
    const scratch = document.createElement("canvas");
    scratch.width = W;
    scratch.height = H;
    const sctx = scratch.getContext("2d", { willReadFrequently: true });
    const layout = layoutText(sctx, text, size, 4 * scale, align);
    const [l, t, r, b] = layout.box;
    const textW = Math.trunc(r - l), textH = Math.trunc(b - t);
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
    sctx.fillStyle = "#fff";
    drawLayout(sctx, layout, size, x - Math.trunc(l), y - Math.trunc(t));
    const coverage = sctx.getImageData(0, 0, W, H).data;
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
  }
};

// src/ImageProcessorPreview/_pixel_nodes.ts
var num4 = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
function pyRound(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
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
var invert = {
  pixels(data, v) {
    const keepAlpha = modeOf(v, data) === "RGBA";
    for (let i = 0; i < data.length; i += 4) {
      data[i] = 255 - data[i];
      data[i + 1] = 255 - data[i + 1];
      data[i + 2] = 255 - data[i + 2];
      if (!keepAlpha) data[i + 3] = 255;
    }
  }
};
var invertMask = {
  pixels(data, v) {
    const mode = modeOf(v, data);
    for (let i = 0; i < data.length; i += 4) {
      let source;
      if (mode === "RGBA") source = data[i + 3];
      else if (mode === "L") source = data[i];
      else source = luma(data[i], data[i + 1], data[i + 2]);
      writeGray(data, i, 255 - source);
    }
  }
};
var BANDS = { red: 0, green: 1, blue: 2, alpha: 3 };
var displayChannel = {
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
  }
};
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
var colorToTransparent = {
  pixels(data, v, view) {
    const hex = String(v.color || "#00ff00").replace(/^#/, "");
    const target = [0, 2, 4].map((at) => parseInt(hex.slice(at, at + 2), 16));
    if (target.some(Number.isNaN)) throw new Error(`"${v.color}" isn't a hex colour`);
    const tolerance = v.tolerance === null || v.tolerance === void 0 ? 10 : num4(v.tolerance, 10);
    const { width: w, height: h, scale } = view;
    let mask = new Uint8ClampedArray(w * h);
    for (let p = 0; p < w * h; p++) {
      const i = p * 4;
      const within = [0, 1, 2].every((c) => Math.abs(data[i + c] - target[c]) <= tolerance);
      mask[p] = within ? 0 : 255;
    }
    const growShrink = num4(v.grow_shrink, 0);
    const iterations = Math.round(Math.abs(growShrink) * scale);
    if (iterations > 0) mask = rankFilter(mask, w, h, iterations, growShrink > 0 ? Math.min : Math.max);
    const blur = num4(v.blur_edges, 0);
    if (blur > 0) {
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let p = 0; p < w * h; p++) rgba[p * 4] = mask[p];
      gaussianBlur(rgba, w, h, blur * scale);
      for (let p = 0; p < w * h; p++) mask[p] = rgba[p * 4];
    }
    for (let p = 0; p < w * h; p++) data[p * 4 + 3] = Math.min(data[p * 4 + 3], mask[p]);
  }
};
var clampGrid = (value) => Math.max(1, Math.min(12, Math.trunc(num4(value, 2))));
function gridSize(v) {
  if (v.grid_detection_mode === "manual") return [clampGrid(v.rows), clampGrid(v.cols)];
  return [Math.max(1, num4(v.detected_rows, 1)), Math.max(1, num4(v.detected_cols, 1))];
}
var gridSplit = {
  label(v) {
    const [rows, cols] = gridSize(v);
    return `${v.grid_detection_mode === "manual" ? "" : "auto: "}${rows} × ${cols}`;
  },
  draw(ctx, img, v, view) {
    ctx.drawImage(img, 0, 0, view.width, view.height);
    const [rows, cols] = gridSize(v);
    const w = view.sourceWidth, h = view.sourceHeight;
    const thick = Math.max(2, 1 / view.scale);
    ctx.save();
    ctx.scale(view.scale, view.scale);
    ctx.fillStyle = "rgb(255,255,0)";
    for (let c = 1; c < cols; c++) {
      const x = pyRound(c * w / cols);
      for (let y = 0; y < h - 1; ) {
        const end = Math.min(y + 10, h - 1);
        ctx.fillRect(x, y, thick, end - y + 1);
        y = end + 6;
      }
    }
    for (let r = 1; r < rows; r++) {
      const y = pyRound(r * h / rows);
      for (let x = 0; x < w - 1; ) {
        const end = Math.min(x + 10, w - 1);
        ctx.fillRect(x, y, end - x + 1, thick);
        x = end + 6;
      }
    }
    ctx.restore();
  }
};
function edgeFadeMask(v, view) {
  const { width: w, height: h, scale, sourceWidth: W, sourceHeight: H } = view;
  const distance = num4(v.fade_distance, 5);
  const fp = Math.max(0, (v.fade_mode || "percentage") === "percentage" ? Math.trunc(Math.min(W, H) * distance / 100) : Math.trunc(distance));
  const curve = num4(v.fade_curve, 2);
  const on = (name) => v[name] !== false;
  const mask = new Uint8ClampedArray(w * h).fill(255);
  if (fp <= 0) return mask;
  const rounded = v.edge_shape === "rounded";
  const ft = on("apply_top") ? fp : 0, fb = on("apply_bottom") ? fp : 0;
  const fl2 = on("apply_left") ? fp : 0, fr = on("apply_right") ? fp : 0;
  for (let py = 0; py < h; py++) {
    const y = Math.floor(py / scale);
    for (let px = 0; px < w; px++) {
      const x = Math.floor(px / scale);
      let a = 255;
      if (rounded) {
        const inLeft = x < fl2, inRight = x >= W - fr, inTop = y < ft, inBottom = y >= H - fb;
        const dxl = inLeft ? x - fl2 : 0, dxr = inRight ? x - (W - fr - 1) : 0;
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
        const ramp = (t) => Math.trunc(255 * Math.pow(t / fp, curve));
        if (fp <= Math.floor(H / 2)) {
          if (ft && y < Math.min(fp, H)) a = Math.min(a, ramp(y));
          if (fb && y >= Math.max(0, H - fp)) a = Math.min(a, ramp(H - y));
        }
        if (fp <= Math.floor(W / 2)) {
          if (fl2 && x < Math.min(fp, W)) a = Math.min(a, ramp(x));
          if (fr && x >= Math.max(0, W - fp)) a = Math.min(a, ramp(W - x));
        }
      }
      mask[py * w + px] = a;
    }
  }
  return mask;
}
var edgeFade = {
  pixels(data, v, view) {
    const { width: w, height: h, scale } = view;
    const mask = edgeFadeMask(v, view);
    const blur = num4(v.blur_radius, 10);
    if (blur > 0) {
      const rgba = new Uint8ClampedArray(w * h * 4);
      for (let p = 0; p < w * h; p++) rgba[p * 4] = mask[p];
      gaussianBlur(rgba, w, h, blur * scale);
      for (let p = 0; p < w * h; p++) mask[p] = rgba[p * 4];
    }
    const keepOwn = !v.replace_mask && modeOf(v, data) === "RGBA";
    for (let p = 0; p < w * h; p++) {
      const i = p * 4 + 3;
      data[i] = keepOwn ? Math.trunc(data[i] / 255 * (mask[p] / 255) * 255) : mask[p];
    }
  }
};
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
var combineMasks = {
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
  }
};

// src/ImageProcessorPreview/_operations.ts
var num5 = (value, fallback) => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
var QUARTER_TURNS = {
  none: 0,
  "90° clockwise": 90,
  "180°": 180,
  "90° counter-clockwise": -90
};
function resolveRotation(v, w, h) {
  const turn = QUARTER_TURNS[v.rotation ?? "90° clockwise"];
  if (turn !== void 0) {
    const swap = Math.abs(turn) === 90;
    return { degrees: turn, width: swap ? h : w, height: swap ? w : h, custom: false };
  }
  const degrees = num5(v.angle, 0);
  if (v.expand_canvas === false) return { degrees, width: w, height: h, custom: true };
  const rad = degrees * Math.PI / 180;
  const cos = Math.abs(Math.cos(rad));
  const sin = Math.abs(Math.sin(rad));
  return {
    degrees,
    width: Math.max(1, Math.round(w * cos + h * sin)),
    height: Math.max(1, Math.round(w * sin + h * cos)),
    custom: true
  };
}
function fmtDegrees(d) {
  const r = Math.round(d * 100) / 100;
  return (r > 0 ? "+" : "") + r + "°";
}
var rotate = {
  size: (v, w, h) => resolveRotation(v, w, h),
  label: (v) => QUARTER_TURNS[v.rotation ?? "90° clockwise"] !== void 0 ? v.rotation : fmtDegrees(num5(v.angle, 0)),
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
    ctx.rotate(degrees * Math.PI / 180);
    ctx.drawImage(img, -w / 2, -h / 2, w, h);
    ctx.restore();
  }
};
var flip = {
  label: (v) => v.direction || "horizontal",
  draw(ctx, img, v, view) {
    const direction = v.direction || "horizontal";
    const mirror2 = direction === "horizontal" || direction === "both";
    const upsideDown = direction === "vertical" || direction === "both";
    ctx.save();
    ctx.translate(mirror2 ? view.width : 0, upsideDown ? view.height : 0);
    ctx.scale(mirror2 ? -1 : 1, upsideDown ? -1 : 1);
    ctx.drawImage(img, 0, 0, view.width, view.height);
    ctx.restore();
  }
};
var grayscale = {
  pixels(data) {
    for (let i = 0; i < data.length; i += 4) {
      const l = luma(data[i], data[i + 1], data[i + 2]);
      data[i] = l;
      data[i + 1] = l;
      data[i + 2] = l;
    }
  }
};
function levelsTable(v) {
  const highlights = Math.trunc(num5(v.highlights, 255));
  let shadows = Math.trunc(num5(v.shadows, 0));
  const midtones = num5(v.midtones, 1);
  const outShadows = Math.trunc(num5(v.output_shadows, 0));
  const outHighlights = Math.trunc(num5(v.output_highlights, 255));
  if (shadows >= highlights) shadows = Math.min(shadows, highlights - 1);
  const table = new Uint8Array(256);
  for (let i = 0; i < 256; i++) {
    let iv;
    if (i <= shadows) iv = 0;
    else if (i >= highlights) iv = 255;
    else iv = Math.trunc(255 * (i - shadows) / (highlights - shadows));
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
var levels = {
  pixels(data, v) {
    applyLut(data, levelsTable(v));
  }
};
var eq = {
  pixels(data, v) {
    const brightness = num5(v.brightness, 1);
    const contrast = num5(v.contrast, 1);
    const saturation = num5(v.saturation, 1);
    const gamma = num5(v.gamma, 1);
    if (brightness !== 1) {
      for (let i = 0; i < data.length; i += 4) {
        for (let c = 0; c < 3; c++) data[i + c] = blendByte(0, data[i + c], brightness);
      }
    }
    if (contrast !== 1) {
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
  }
};
var gaussianBlurOperation = {
  label: (v) => `radius ${num5(v.radius, 5)}`,
  pixels(data, v, view) {
    gaussianBlur(data, view.width, view.height, num5(v.radius, 5) * view.scale);
  }
};
var bloom = {
  pixels(data, v, view) {
    const amount = num5(v.bloom_amount, 0.5);
    if (amount <= 0) return;
    for (let i = 3; i < data.length; i += 4) data[i] = 255;
    const original = Uint8ClampedArray.from(data);
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) data[i + c] = blendByte(0, data[i + c], 1 + amount);
    }
    gaussianBlur(data, view.width, view.height, num5(v.bloom_radius, 5) * view.scale);
    const t = Math.min(amount, 1);
    for (let i = 0; i < data.length; i += 4) {
      for (let c = 0; c < 3; c++) data[i + c] = blendByte(original[i + c], data[i + c], t);
      data[i + 3] = 255;
    }
  }
};
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}
function parseAspectRatio(text) {
  const raw = String(text ?? "").trim();
  if (raw.includes(":")) {
    const parts = raw.split(":");
    if (parts.length !== 2) return null;
    const ratio2 = Number(parts[0].trim()) / Number(parts[1].trim());
    return Number.isFinite(ratio2) && ratio2 > 0 ? ratio2 : null;
  }
  const ratio = Number(raw);
  return raw !== "" && Number.isFinite(ratio) && ratio > 0 ? ratio : null;
}
function resizeLayout(v, w, h) {
  const mode = v.resize_mode || "percentage";
  if (mode === "width") {
    const size = Math.trunc(num5(v.target_size, 1e3));
    return { width: size, height: Math.trunc(h * (size / w)) };
  }
  if (mode === "height") {
    const size = Math.trunc(num5(v.target_size, 1e3));
    return { width: Math.trunc(w * (size / h)), height: size };
  }
  if (mode === "percentage") {
    const s = num5(v.percentage_scale, 100) / 100;
    return { width: Math.trunc(w * s), height: Math.trunc(h * s) };
  }
  let targetW;
  let targetH;
  if (mode === "width and height") {
    targetW = Math.trunc(num5(v.target_width, 1e3));
    targetH = Math.trunc(num5(v.target_height, 1e3));
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
function fitBackground(color) {
  const text = String(color ?? "");
  if (text === "transparent") return "rgba(0,0,0,0)";
  const hex = text.startsWith("#") ? text.slice(1) : "";
  if (/^[0-9a-fA-F]{3}$/.test(hex) || /^[0-9a-fA-F]{6}$/.test(hex)) return `#${hex}`;
  return "#ffffff";
}
var resize = {
  size(v, w, h) {
    const layout = resizeLayout(v, w, h);
    if (layout.error !== void 0) return layout;
    if (layout.width < 1 || layout.height < 1) return { error: "The result would be 0 pixels wide or tall" };
    return layout;
  },
  label: (v) => v.resize_mode || "percentage",
  draw(ctx, img, v, view) {
    const { fit } = resizeLayout(v, view.sourceWidth, view.sourceHeight);
    ctx.imageSmoothingEnabled = (v.resample_filter || "lanczos") !== "nearest";
    ctx.imageSmoothingQuality = "high";
    if (fit === void 0 || fit === "stretch") {
      ctx.drawImage(img, 0, 0, view.width, view.height);
      return;
    }
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
  }
};
var EXTEND_POSITIONS = {
  center: (dx, dy) => [Math.floor(dx / 2), Math.floor(dy / 2)],
  top_left: () => [0, 0],
  top_right: (dx) => [dx, 0],
  bottom_left: (dx, dy) => [0, dy],
  bottom_right: (dx, dy) => [dx, dy],
  top: (dx) => [Math.floor(dx / 2), 0],
  bottom: (dx, dy) => [Math.floor(dx / 2), dy],
  left: (dx, dy) => [0, Math.floor(dy / 2)],
  right: (dx, dy) => [dx, Math.floor(dy / 2)]
};
function extendLayout(v, w, h) {
  const preset = v.aspect_ratio_preset ?? "1:1 square";
  const ratio = preset === "custom" ? null : /^(\d+):(\d+) /.exec(preset);
  const margin = (name) => Math.max(0, Math.trunc(num5(v[name], 0)));
  let targetW;
  let targetH;
  if (ratio) {
    const rw = Number(ratio[1]);
    const rh = Number(ratio[2]);
    if (w / h > rw / rh) {
      targetW = w;
      targetH = Math.trunc(w * rh / rw);
    } else {
      targetH = h;
      targetW = Math.trunc(h * rw / rh);
    }
  } else {
    targetW = w + margin("left") + margin("right");
    targetH = h + margin("top") + margin("bottom");
  }
  const upscale = num5(v.upscale_factor, 1);
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
var extendCanvas = {
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
  }
};
var OPERATIONS = {
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
  resize_video: VIDEO_OPERATIONS.resize_video
};

// src/ImageProcessorPreview/ImageProcessorPreview.ts
var WIDGET_VERSION = "1.2.0";
var CHECKER = 10;
var SCRUB_STEPS = 1e3;
function ImageProcessorPreview(host, props) {
  const container = host;
  if (container._imagePreview?.wrapper?.isConnected) {
    container._imagePreview.handleUpdate(props);
    return { cleanup: container._imagePreview.cleanup, update: container._imagePreview.handleUpdate };
  }
  if (container._imagePreview) delete container._imagePreview;
  let value = props.value || {};
  let sourceUrl = "";
  let sourceReady = false;
  let sourcePixels = null;
  let pixelsBlocked = false;
  const wrapper = document.createElement("div");
  wrapper.className = "image-processor-preview nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;height:100%;user-select:none;-webkit-user-select:none;";
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative",
    "background:#111",
    "border-radius:6px",
    "overflow:hidden",
    "flex:1 1 0",
    "min-height:180px",
    "display:flex",
    "align-items:center",
    "justify-content:center"
  ].join(";");
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;";
  const ctx = canvas.getContext("2d");
  canvasWrap.appendChild(canvas);
  const scrubRow = document.createElement("div");
  scrubRow.style.cssText = "display:none;align-items:center;gap:8px;padding:0 2px;";
  const scrub = document.createElement("input");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(SCRUB_STEPS);
  scrub.value = "0";
  scrub.style.cssText = "flex:1;min-width:0;";
  const scrubTime = document.createElement("span");
  scrubTime.style.cssText = "font-size:10px;font-family:monospace;color:var(--muted-foreground);white-space:nowrap;";
  for (const type of ["pointerdown", "mousedown"]) scrub.addEventListener(type, (e) => e.stopPropagation());
  scrubRow.appendChild(scrub);
  scrubRow.appendChild(scrubTime);
  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px",
    "color:var(--muted-foreground)",
    "padding:4px 8px",
    "background:var(--muted)",
    "border-radius:4px",
    "display:flex",
    "justify-content:space-between",
    "align-items:center"
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;opacity:0.6;";
  statusR.textContent = `v${WIDGET_VERSION}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);
  wrapper.appendChild(canvasWrap);
  wrapper.appendChild(scrubRow);
  wrapper.appendChild(statusBar);
  container.appendChild(wrapper);
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    sourceReady = true;
    sourcePixels = null;
    pixelsBlocked = false;
    render();
  };
  img.onerror = () => {
    sourceReady = false;
    render();
  };
  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.addEventListener("loadeddata", () => {
    seekTo(Number(scrub.value) / SCRUB_STEPS);
  });
  video.addEventListener("seeked", () => {
    sourceReady = true;
    sourcePixels = null;
    pixelsBlocked = false;
    render();
  });
  video.addEventListener("error", () => {
    sourceReady = false;
    render();
  });
  let fontUrl = "";
  function loadFont(url) {
    if (!url || url === fontUrl || typeof FontFace === "undefined") return;
    fontUrl = url;
    new FontFace(PREVIEW_FONT, `url("${url}")`).load().then((face) => {
      document.fonts.add(face);
      render();
    }).catch(() => {
    });
  }
  function renderGenerated(operation) {
    const size = operation.size(value);
    if (size.error !== void 0) {
      showMessage(size.error, size.error);
      return;
    }
    const full = document.createElement("canvas");
    full.width = size.width;
    full.height = size.height;
    const fctx = full.getContext("2d");
    try {
      if (operation.pixels) {
        fctx.putImageData(new ImageData(operation.pixels(value, size.width, size.height), size.width, size.height), 0, 0);
      } else {
        operation.render(fctx, value, size);
      }
    } catch (err) {
      showMessage(err.message, err.message);
      return;
    }
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / size.width, areaH / size.height);
    canvas.width = Math.max(1, Math.round(size.width * scale));
    canvas.height = Math.max(1, Math.round(size.height * scale));
    drawChecker(canvas.width, canvas.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(full, 0, 0, canvas.width, canvas.height);
    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    statusL.textContent = `${label}${size.width} × ${size.height} px`;
  }
  let multiImages = [];
  let multiModes = [];
  let multiKey = "";
  function loadMulti(urls) {
    const key = urls.map((u) => u ? u.split("?")[0] : "").join("|");
    if (key === multiKey) return;
    multiKey = key;
    multiImages = urls.filter(Boolean).map((url) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => render();
      image.onerror = () => render();
      image.src = url;
      return image;
    });
  }
  function renderMulti(operation) {
    if (!multiImages.length) {
      showMessage("Connect masks to preview the result", "No masks connected");
      return;
    }
    if (!multiImages.every((image) => image.complete && image.naturalWidth)) {
      showMessage("Loading…", "Loading");
      return;
    }
    const [w, h] = [multiImages[0].naturalWidth, multiImages[0].naturalHeight];
    if (multiImages.some((image) => image.naturalWidth !== w || image.naturalHeight !== h)) {
      showMessage("The masks aren't all the same size", "The masks aren't all the same size");
      return;
    }
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / w, areaH / h);
    const view = { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)), scale, sourceWidth: w, sourceHeight: h };
    const scratch = document.createElement("canvas");
    scratch.width = view.width;
    scratch.height = view.height;
    const sctx = scratch.getContext("2d", { willReadFrequently: true });
    let datas;
    try {
      datas = multiImages.map((image) => {
        sctx.clearRect(0, 0, view.width, view.height);
        sctx.drawImage(image, 0, 0, view.width, view.height);
        return sctx.getImageData(0, 0, view.width, view.height).data;
      });
    } catch {
      showMessage("Preview unavailable for these images", "Preview unavailable for these images");
      return;
    }
    canvas.width = view.width;
    canvas.height = view.height;
    drawChecker(view.width, view.height);
    sctx.putImageData(new ImageData(operation.combine(datas, multiModes, view), view.width, view.height), 0, 0);
    ctx.drawImage(scratch, 0, 0);
    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    const skipped = Number(value.skipped) || 0;
    const note = skipped ? `  (${skipped} more can't be previewed)` : "";
    statusL.textContent = `${label}${w} × ${h} px${note}`;
  }
  let isVideo = false;
  const source = () => isVideo ? video : img;
  const sourceSize = () => isVideo ? [video.videoWidth, video.videoHeight] : [img.naturalWidth, img.naturalHeight];
  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00.0";
    const m = Math.floor(seconds / 60);
    const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  }
  function seekTo(fraction) {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    video.currentTime = Math.max(0, Math.min(duration - 1e-3, fraction * duration));
    scrubTime.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
  }
  scrub.addEventListener("input", () => {
    if (isVideo) seekTo(Number(scrub.value) / SCRUB_STEPS);
  });
  function drawChecker(w, h) {
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#4a4a4a";
    for (let y = 0; y < h; y += CHECKER) {
      for (let x = y / CHECKER % 2 ? 0 : CHECKER; x < w; x += CHECKER * 2) {
        ctx.fillRect(x, y, CHECKER, CHECKER);
      }
    }
  }
  function showMessage(text, status) {
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    canvas.width = areaW;
    canvas.height = areaH;
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, areaW, areaH);
    ctx.fillStyle = "#555";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, areaW / 2, areaH / 2);
    statusL.textContent = status;
  }
  function drawPixels(operation, view) {
    if (!sourcePixels || sourcePixels.width !== view.width || sourcePixels.height !== view.height) {
      const scratch = document.createElement("canvas");
      scratch.width = view.width;
      scratch.height = view.height;
      const sctx = scratch.getContext("2d", { willReadFrequently: true });
      sctx.drawImage(source(), 0, 0, view.width, view.height);
      try {
        sourcePixels = sctx.getImageData(0, 0, view.width, view.height);
        pixelsBlocked = false;
      } catch {
        sourcePixels = null;
        pixelsBlocked = true;
      }
    }
    if (pixelsBlocked) {
      ctx.drawImage(source(), 0, 0, view.width, view.height);
      return;
    }
    const result = new ImageData(new Uint8ClampedArray(sourcePixels.data), view.width, view.height);
    operation.pixels(result.data, value, view);
    const out = document.createElement("canvas");
    out.width = view.width;
    out.height = view.height;
    out.getContext("2d").putImageData(result, 0, 0);
    ctx.drawImage(out, 0, 0);
  }
  function render() {
    const operation = OPERATIONS[value.operation];
    if (!operation) {
      showMessage("No preview for this node", "");
      return;
    }
    if (operation.generate) {
      renderGenerated(operation);
      return;
    }
    if (operation.multi) {
      renderMulti(operation);
      return;
    }
    if (!sourceReady) {
      const what = isVideo ? "a video" : "an image";
      showMessage(`Connect ${what} to preview the result`, isVideo ? "No video connected" : "No image connected");
      return;
    }
    const [w, h] = sourceSize();
    const size = operation.size ? operation.size(value, w, h) : { width: w, height: h };
    if (size.error !== void 0) {
      showMessage(size.error, size.error);
      return;
    }
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / size.width, areaH / size.height);
    const view = {
      width: Math.max(1, Math.round(size.width * scale)),
      height: Math.max(1, Math.round(size.height * scale)),
      scale,
      sourceWidth: w,
      sourceHeight: h,
      outputWidth: size.width,
      outputHeight: size.height
    };
    canvas.width = view.width;
    canvas.height = view.height;
    drawChecker(view.width, view.height);
    try {
      if (operation.pixels) drawPixels(operation, view);
      else operation.draw(ctx, source(), value, view);
    } catch (err) {
      showMessage(err.message, err.message);
      return;
    }
    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    const note = pixelsBlocked && operation.pixels ? "  (preview unavailable for this source)" : "";
    statusL.textContent = `${label}${size.width} × ${size.height} px${note}`;
  }
  function handleUpdate(newProps) {
    value = newProps.value || {};
    loadFont(value.font_url);
    if (Array.isArray(value.images)) {
      const entries = value.images.filter((entry) => entry && entry.url);
      multiModes = entries.map((entry) => entry.mode || "");
      loadMulti(entries.map((entry) => entry.url));
      render();
      return;
    }
    const nextIsVideo = value.media === "video" || Boolean(value.video_url);
    const newUrl = (nextIsVideo ? value.video_url : value.image_url) || "";
    const base = (u) => u ? u.split("?")[0] : "";
    if (nextIsVideo !== isVideo || base(newUrl) !== base(sourceUrl)) {
      isVideo = nextIsVideo;
      sourceUrl = newUrl;
      sourceReady = false;
      sourcePixels = null;
      scrubRow.style.display = isVideo ? "flex" : "none";
      if (!newUrl) {
        render();
        return;
      }
      if (isVideo) {
        scrub.value = "0";
        video.src = newUrl;
        video.load();
      } else {
        img.src = newUrl;
      }
      render();
      return;
    }
    render();
  }
  const ro = new ResizeObserver(() => render());
  ro.observe(canvasWrap);
  handleUpdate(props);
  function cleanup() {
    ro.disconnect();
    video.pause();
    video.removeAttribute("src");
    video.load();
    wrapper.remove();
    delete container._imagePreview;
  }
  container._imagePreview = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
export {
  ImageProcessorPreview as default
};
