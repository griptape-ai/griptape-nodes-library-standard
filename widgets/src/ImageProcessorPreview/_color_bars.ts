// Create Color Bars, ported from create_color_bars.py. Patterns are written straight into an RGBA buffer
// with PIL's inclusive rectangles and Python's integer maths, so the preview matches pixel for pixel.

import type { GeneratedPixelsOperation, PreviewValue } from "./_types";

type Rgb = readonly [number, number, number];

const num = (value: unknown, fallback: number): number => {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
};
const fl = Math.floor;
const tr = Math.trunc;
// Python's float %: fmod, then add the divisor when the signs differ. A tiny negative remainder becomes
// exactly the divisor, which the Zone Plate relies on.
const pymod = (a: number, m: number) => {
  const r = a % m;
  return r !== 0 && (r < 0) !== (m < 0) ? r + m : r;
};

const BLACK: Rgb = [0, 0, 0];
const WHITE: Rgb = [255, 255, 255];
const GRAY: Rgb = [128, 128, 128];
const C100: Rgb[] = [[255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0], [255, 0, 255], [255, 0, 0], [0, 0, 255]];
const C75 = C100.map((c) => c.map((v) => (v === 255 ? 191 : v)) as unknown as Rgb);
const SMALL75: Rgb[] = [[0, 0, 191], [191, 0, 191], [0, 191, 191], [191, 191, 191], [191, 191, 0], [191, 0, 0], [0, 191, 0]];
const SMALL100 = SMALL75.map((c) => c.map((v) => (v === 191 ? 255 : v)) as unknown as Rgb);
const FULL_FIELD: Record<string, Rgb> = {
  "Full Field White": [255, 255, 255],
  "Full Field Blue": [0, 0, 255],
  "Full Field Cyan": [0, 255, 255],
  "Full Field Green": [0, 255, 0],
  "Full Field Magenta": [255, 0, 255],
  "Full Field Red": [255, 0, 0],
  "Full Field Yellow": [255, 255, 0],
};

class Canvas {
  w: number;
  h: number;
  data: Uint8ClampedArray<ArrayBuffer>;

  constructor(w: number, h: number, fill: Rgb = BLACK) {
    this.w = w;
    this.h = h;
    this.data = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) this.set(i, fill);
  }
  set(i: number, [r, g, b]: Rgb): void {
    this.data[i * 4] = r;
    this.data[i * 4 + 1] = g;
    this.data[i * 4 + 2] = b;
    this.data[i * 4 + 3] = 255;
  }
  // PIL draw.rectangle: both corners inclusive, clipped to the image.
  rect(x0: number, y0: number, x1: number, y1: number, color: Rgb): void {
    const xa = Math.max(0, x0), ya = Math.max(0, y0);
    const xb = Math.min(this.w - 1, x1), yb = Math.min(this.h - 1, y1);
    for (let y = ya; y <= yb; y++) for (let x = xa; x <= xb; x++) this.set(y * this.w + x, color);
  }
  pixel(x: number, y: number, color: Rgb): void { this.set(y * this.w + x, color); }
  // Vertical bars, the last one running to the right edge.
  bars(colors: readonly Rgb[], bottom: number): void {
    const bw = fl(this.w / colors.length);
    colors.forEach((c, i) => this.rect(i * bw, 0, i < colors.length - 1 ? (i + 1) * bw : this.w, bottom, c));
  }
}

const barBottom = (h: number) => tr(h * 0.67);

function smpteBars(cv: Canvas): void {
  const { w, h } = cv;
  const by = barBottom(h);
  cv.bars(C75, by);
  const lsw = fl(w / 4);
  const sbw = fl((w - lsw) / 7);
  cv.rect(0, by, fl(lsw / 3), h, BLACK);
  cv.rect(fl(lsw / 3), by, fl((lsw * 2) / 3), h, GRAY);
  cv.rect(fl((lsw * 2) / 3), by, lsw, h, [191, 191, 191]);
  SMALL75.forEach((c, i) => cv.rect(lsw + i * sbw, by, i < 6 ? lsw + (i + 1) * sbw : w, h, c));
}

function checker(cv: Canvas, light: Rgb): void {
  const by = barBottom(cv.h);
  const cs = Math.min(fl(cv.w / 20), fl((cv.h - by) / 4));
  for (let row = 0; row < 4; row++) {
    for (let col = 0; col < 4; col++) {
      cv.rect(col * cs, by + row * cs, col * cs + cs, by + row * cs + cs, (row + col) % 2 === 0 ? light : BLACK);
    }
  }
}

function fullFieldBars(cv: Canvas, colors: readonly Rgb[], white: Rgb, small: readonly Rgb[]): void {
  const { w, h } = cv;
  const by = barBottom(h);
  cv.bars(colors, by);
  const s = fl(w / 11);
  cv.rect(0, by, 3 * s, h, BLACK);
  cv.rect(s, by, 2 * s, h, white);
  small.forEach((c, i) => cv.rect(3 * s + i * s, by, i < 6 ? 3 * s + (i + 1) * s : w, h, c));
}

function stairs(cv: Canvas, steps: number, vertical: boolean): void {
  const length = vertical ? cv.h : cv.w;
  const sw = fl(length / steps);
  for (let i = 0; i < steps; i++) {
    const g = tr((255 * i) / (steps - 1));
    const end = i < steps - 1 ? (i + 1) * sw : length;
    if (vertical) cv.rect(0, i * sw, cv.w, end, [g, g, g]);
    else cv.rect(i * sw, 0, end, cv.h, [g, g, g]);
  }
}

function eachColumn(cv: Canvas, fn: (x: number) => Rgb): void {
  for (let x = 0; x < cv.w; x++) {
    const c = fn(x);
    for (let y = 0; y < cv.h; y++) cv.pixel(x, y, c);
  }
}

function hsvFull(h: number): Rgb {
  const i = tr(h * 6);
  const f = h * 6 - i;
  const q = 1 - f;
  const rgb = [[1, f, 0], [q, 1, 0], [0, 1, f], [0, q, 1], [f, 0, 1], [1, 0, q]][pymod(i, 6)]!;
  return rgb.map((c) => tr(c * 255)) as unknown as Rgb;
}

function pluge(cv: Canvas, v: PreviewValue): void {
  const { w, h } = cv;
  const setup = (v.pluge_ire_setup as string | undefined) || "NTSC 7.5 IRE";
  const ire = setup === "PAL 0 IRE" ? [-2, 0, 2] : setup === "RGB Full Range" ? [-1, 0, 1.5] : [3.5, 7.5, 11.5];
  const [superBlack, black, above] = ire.map((x) => Math.max(0, tr((x / 100) * 255))) as [number, number, number];
  const count = Math.max(2, Math.min(5, tr(num(v.pluge_bar_count, 3))));
  const levels = count >= 3 ? [superBlack, black, above] : [black, above];
  for (let i = 0; i < count - 3; i++) levels.push(tr(black + (above - black) * ((i + 1) / (count - 2))));
  if ((v.pluge_orientation || "vertical") === "horizontal") {
    const bh = fl(h / (count + 2)), bw = fl(w / 2), cx = fl(w / 2);
    const sy = fl((h - bh * count) / 2);
    levels.forEach((g, i) => cv.rect(cx - fl(bw / 2), sy + i * bh, cx + fl(bw / 2), sy + i * bh + bh, [g, g, g]));
  } else {
    const bw = fl(w / (count + 3)), bh = fl(h / 2), by = fl((h - bh) / 2);
    const sx = fl(w / 2) - fl((bw * levels.length) / 2);
    levels.forEach((g, i) => cv.rect(sx + i * bw, by, sx + i * bw + bw, by + bh, [g, g, g]));
  }
}

// Each pattern by name. A pattern throws for sizes the node can't make either.
const PATTERNS: Record<string, (cv: Canvas, v: PreviewValue) => void> = {
  "SMPTE 219-100 Bars": (cv) => cv.bars(C100, cv.h),
  "SMPTE 75% Bars": (cv) => cv.bars(C75, cv.h),
  "SMPTE Bars": smpteBars,
  "SMPTE 219+i Bars": (cv) => { smpteBars(cv); checker(cv, WHITE); },
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
  "ARIB 28-100": (cv) => { smpteBars(cv); checker(cv, WHITE); },
  "ARIB 28-75": (cv) => {
    const { w, h } = cv;
    const by = barBottom(h);
    cv.bars(C75, by);
    const lsw = fl(w / 4);
    cv.rect(0, by, fl(lsw / 3), h, BLACK);
    cv.rect(fl(lsw / 3), by, fl((lsw * 2) / 3), h, GRAY);
    cv.rect(fl((lsw * 2) / 3), by, lsw, h, [191, 191, 191]);
    checker(cv, [191, 191, 191]);
    const sbw = fl((w - lsw) / 7);
    SMALL75.forEach((c, i) => cv.rect(lsw + i * sbw, by, i < 6 ? lsw + (i + 1) * sbw : w, h, c));
  },
  "ARIB 28+i": (cv) => { smpteBars(cv); checker(cv, WHITE); },
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
        const ring = d < r ? tr(pymod((d / r) * 10, 2) * 255) : 128;
        const radial = tr(pymod((Math.atan2(dy, dx) / Math.PI) * 10, 2) * 255);
        const g = tr((ring + radial) / 2);
        cv.pixel(x, y, [g, g, g]);
      }
    }
  },
  "Tartan Bars": (cv) => {
    const colors: Rgb[] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255], WHITE, BLACK];
    const t = fl(Math.min(cv.w, cv.h) / 16);
    if (t <= 0) throw new Error("Tartan Bars needs an image at least 16 pixels on each side");
    for (let y = 0; y < cv.h; y += t) {
      for (let x = 0; x < cv.w; x += t) {
        cv.rect(x, y, Math.min(x + t, cv.w), Math.min(y + t, cv.h), colors[(fl(x / t) + fl(y / t)) % 8]!);
      }
    }
  },
  "Stair 5 Step": (cv) => stairs(cv, 5, false),
  "Stair 5 Step Vert": (cv) => stairs(cv, 5, true),
  "Stair 10 Step": (cv) => stairs(cv, 10, false),
  "Stair 10 Step Vert": (cv) => stairs(cv, 10, true),
  "Y Ramp Up": (cv) => {
    if (cv.w < 2) throw new Error("Y Ramp Up needs an image at least 2 pixels wide");
    eachColumn(cv, (x) => { const g = tr(255 * (x / (cv.w - 1))); return [g, g, g]; });
  },
  "Y Ramp Down": (cv) => {
    if (cv.w < 2) throw new Error("Y Ramp Down needs an image at least 2 pixels wide");
    eachColumn(cv, (x) => { const g = tr(255 * (1 - x / (cv.w - 1))); return [g, g, g]; });
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
  },
};
for (const [name, color] of Object.entries(FULL_FIELD)) PATTERNS[name] = (cv) => cv.rect(0, 0, cv.w, cv.h, color);

// The biggest image the preview generates at full size; larger sizes aren't worth the memory.
const MAX_PIXELS = 16_000_000;

export const colorBars: GeneratedPixelsOperation = {
  generate: true,
  label: (v) => (v.bar_type as string | undefined) || "SMPTE 219-100 Bars",
  size(v) {
    const w = num(v.width, 1920);
    const h = num(v.height, 1080);
    if (w <= 0 || h <= 0) return { error: "Width and height must be greater than 0" };
    const width = Math.max(1, tr(w));
    const height = Math.max(1, tr(h));
    if (width * height > MAX_PIXELS) return { error: "Too large to preview" };
    return { width, height };
  },
  pixels(v, width, height) {
    const pattern = PATTERNS[v.bar_type as string] ?? PATTERNS["SMPTE 219-100 Bars"]!;
    const cv = new Canvas(width, height);
    pattern(cv, v);
    return cv.data;
  },
};
