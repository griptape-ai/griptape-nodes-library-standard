// Output size for the "divisible by" resize mode, as utils/divisible_size.py computes it.

const ROUNDING = {
  round: (q) => Math.floor(q + 0.5),
  floor: Math.floor,
  ceil: Math.ceil,
};

export function divisibleSize(v, w, h) {
  const divisor = Math.trunc(Number(v.divisor ?? 64));
  if (!Number.isFinite(divisor) || divisor < 1) return { error: `Divisor must be at least 1, got ${v.divisor}` };
  const rounding = v.rounding || "round";
  const steps = ROUNDING[rounding];
  if (!steps) return { error: `Unknown rounding "${rounding}"` };
  const snap = (x) => Math.max(1, steps(x / divisor)) * divisor;
  if (v.preserve_aspect_ratio === false) return { width: snap(w), height: snap(h) };
  const anchor = v.anchor_dimension || "auto";
  if (!["auto", "width", "height"].includes(anchor)) return { error: `Unknown anchor dimension "${anchor}"` };
  if (anchor === "width" || (anchor === "auto" && w >= h)) {
    const width = snap(w);
    return { width, height: snap((h * width) / w) };
  }
  const height = snap(h);
  return { width: snap((w * height) / h), height };
}
