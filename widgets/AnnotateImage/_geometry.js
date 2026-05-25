// Pure math / geometry utilities shared across the annotation widget.
// No DOM, no canvas context, no shared mutable state.

export function decimatePoints(points, minDist = 3) {
  if (points.length <= 2) return points;
  const out = [points[0]];
  for (let i = 1; i < points.length - 1; i++) {
    const prev = out[out.length - 1];
    if (Math.hypot(points[i][0] - prev[0], points[i][1] - prev[1]) >= minDist) out.push(points[i]);
  }
  out.push(points[points.length - 1]);
  return out;
}

export function strokeBounds(stroke) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const pt of (stroke.points || [])) {
    minX = Math.min(minX, pt[0]); minY = Math.min(minY, pt[1]);
    maxX = Math.max(maxX, pt[0]); maxY = Math.max(maxY, pt[1]);
  }
  if (!isFinite(minX)) return null;
  const r = (stroke.size || 8) / 2;
  return { minX: minX - r, minY: minY - r, maxX: maxX + r, maxY: maxY + r };
}

export function naturalBounds(ann) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const stroke of (ann.strokes || [])) {
    const b = strokeBounds(stroke);
    if (b) {
      minX = Math.min(minX, b.minX); minY = Math.min(minY, b.minY);
      maxX = Math.max(maxX, b.maxX); maxY = Math.max(maxY, b.maxY);
    }
  }
  return isFinite(minX) ? { minX, minY, maxX, maxY } : null;
}

export function paintCenter(ann) {
  if (ann.cx != null && ann.cy != null) return [ann.cx, ann.cy];
  const b = naturalBounds(ann);
  return b ? [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2] : [0, 0];
}

export function paintTransformPt(ann, nx, ny) {
  const [cx, cy] = paintCenter(ann);
  const x = ann.x || 0, y = ann.y || 0;
  const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = ann.rotation || 0;
  const cos = Math.cos(r), sin = Math.sin(r);
  const lx = (nx - cx) * sx, ly = (ny - cy) * sy;
  return [cx + x + lx * cos - ly * sin, cy + y + lx * sin + ly * cos];
}

export function paintInvTransformPt(ann, px, py) {
  const [cx, cy] = paintCenter(ann);
  const x = ann.x || 0, y = ann.y || 0;
  const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = -(ann.rotation || 0);
  const cos = Math.cos(r), sin = Math.sin(r);
  const dx = px - (cx + x), dy = py - (cy + y);
  return [(dx * cos - dy * sin) / sx + cx, (dx * sin + dy * cos) / sy + cy];
}

export function getTransformedCorners(ann, pad = 10) {
  const b = naturalBounds(ann);
  if (!b) return [];
  return [
    [b.minX - pad, b.minY - pad], [b.maxX + pad, b.minY - pad],
    [b.maxX + pad, b.maxY + pad], [b.minX - pad, b.maxY + pad],
  ].map(([nx, ny]) => paintTransformPt(ann, nx, ny));
}

export function defaultCps(ann) {
  const cp1x = ann.cp1x ?? (ann.x1 + ((ann.x2 ?? 0) - (ann.x1 ?? 0)) / 3);
  const cp1y = ann.cp1y ?? (ann.y1 + ((ann.y2 ?? 0) - (ann.y1 ?? 0)) / 3);
  const cp2x = ann.cp2x ?? (ann.x1 + ((ann.x2 ?? 0) - (ann.x1 ?? 0)) * 2 / 3);
  const cp2y = ann.cp2y ?? (ann.y1 + ((ann.y2 ?? 0) - (ann.y1 ?? 0)) * 2 / 3);
  return { cp1x, cp1y, cp2x, cp2y };
}

export function snapshotAnn(ann) {
  const [cx, cy] = paintCenter(ann);
  const cps = defaultCps(ann);
  return {
    cx, cy,
    x: ann.x ?? 0, y: ann.y ?? 0,
    scaleX: ann.scaleX ?? 1, scaleY: ann.scaleY ?? 1,
    rotation: ann.rotation ?? 0,
    x1: ann.x1 ?? 0, y1: ann.y1 ?? 0, x2: ann.x2 ?? 0, y2: ann.y2 ?? 0,
    cp1x: cps.cp1x, cp1y: cps.cp1y, cp2x: cps.cp2x, cp2y: cps.cp2y,
    font_size: ann.font_size ?? 48,
    text: ann.text || "",
    width: ann.width ?? 3,
    sizeScale: ann.sizeScale ?? 1,
    w: ann.w ?? 100, h: ann.h ?? 100,
  };
}

// ── OBB frame helpers ─────────────────────────────────────────────────────────

export function frameCorners(frame) {
  const { pivotX: px, pivotY: py, rotation: r, halfW: hw, halfH: hh } = frame;
  const cos = Math.cos(r), sin = Math.sin(r);
  return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([lx,ly]) =>
    [px + lx*cos - ly*sin, py + lx*sin + ly*cos]);
}

export function frameTopMid(frame) {
  const { pivotX: px, pivotY: py, rotation: r, halfH: hh } = frame;
  return [px + hh*Math.sin(r), py - hh*Math.cos(r)];
}

// displayScale is passed explicitly so this stays a pure function
export function frameRotHandle(frame, displayScale) {
  const [tx, ty] = frameTopMid(frame);
  const d = 28 / displayScale;
  return [tx + d*Math.sin(frame.rotation), ty - d*Math.cos(frame.rotation)];
}
