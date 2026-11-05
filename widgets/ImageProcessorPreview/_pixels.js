// Pixel helpers that reproduce Pillow's integer math, so a preview matches what the node produces.

// Pillow's RGB to L conversion (ITU-R 601-2 luma, fixed point).
export function luma(r, g, b) {
  return (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
}

// One band of Image.blend(a, b, t): truncates inside 0..1, clamps then truncates outside it.
// ImageEnhance.X(img).enhance(f) is Image.blend(degenerate, img, f).
export function blendByte(a, b, t) {
  const value = a + t * (b - a);
  if (t >= 0 && t <= 1) return Math.trunc(value);
  if (value <= 0) return 0;
  if (value >= 255) return 255;
  return Math.trunc(value);
}

// Applies a 256-entry lookup table to the RGB bands and leaves alpha alone, as the nodes do for RGBA.
export function applyLut(data, table) {
  for (let i = 0; i < data.length; i += 4) {
    data[i] = table[data[i]];
    data[i + 1] = table[data[i + 1]];
    data[i + 2] = table[data[i + 2]];
  }
}

// Box radius Pillow's GaussianBlur uses for its three box passes (ImageFilter._gaussian_blur_radius).
function boxRadius(radius, passes) {
  const sigma2 = (radius * radius) / passes;
  const l = Math.floor((Math.sqrt(12 * sigma2 + 1) - 1) / 2);
  const a = ((2 * l + 1) * (l * (l + 1) - 3 * sigma2)) / (6 * (sigma2 - (l + 1) * (l + 1)));
  return l + a;
}

// One extended box pass along rows (step 4) or columns (step 4 * width), with edge pixels repeated.
// The two pixels just outside the window count by the fractional part of the radius.
function boxPass(src, dst, width, height, radius, horizontal) {
  const whole = Math.floor(radius);
  const frac = radius - whole;
  const norm = 1 / (2 * radius + 1);
  const lines = horizontal ? height : width;
  const length = horizontal ? width : height;
  const step = horizontal ? 4 : 4 * width;
  const clamp = (k) => (k < 0 ? 0 : k >= length ? length - 1 : k);

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

// Pillow's GaussianBlur: three box passes each way, on straight (not premultiplied) RGBA, so colour
// from transparent pixels bleeds the way it does in the node.
export function gaussianBlur(data, width, height, radius) {
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
