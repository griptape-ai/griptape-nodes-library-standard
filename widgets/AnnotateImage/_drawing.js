// Canvas drawing functions for the annotation widget.
// Use createDrawing(getState) to get a bound set of draw functions.
// getState() must return { ctx, displayScale, hoverId }.

import { paintCenter, defaultCps, naturalBounds, getTransformedCorners } from './_geometry.js';
import {
  DEFAULT_COLOR, DEFAULT_PAINT_SIZE, DEFAULT_ARROW_WIDTH, DEFAULT_TEXT_SIZE, MIN_TEXT_SIZE, DEFAULT_SHAPE_WIDTH,
  SEL_COLOR_RGB, HOVER_OPACITY, HANDLE_FILL, HANDLE_STROKE_OPACITY, CP_LINE_OPACITY,
  LINE_WIDTH_PRIMARY, LINE_WIDTH_SECONDARY,
  HANDLE_RADIUS, CP_HANDLE_RADIUS,
  DASH_CP_LINE, HOVER_PAD,
} from './_styles.js';

export function createDrawing(getState) {

  function isHovered(ann) {
    const { hoverId, hoverGroupId, marqueePreviewIds } = getState();
    return ann.id === hoverId
      || (hoverGroupId && ann.group_id === hoverGroupId)
      || (marqueePreviewIds && marqueePreviewIds.has(ann.id));
  }

  function renderStrokes(strokes, sizeScale = 1) {
    const { ctx } = getState();
    for (const stroke of strokes) {
      const pts = stroke.points || [];
      if (!pts.length) continue;
      ctx.fillStyle = stroke.color || DEFAULT_COLOR;
      const r0 = ((pts[0][2] ?? (stroke.size || DEFAULT_PAINT_SIZE)) * sizeScale) / 2;
      ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], r0, 0, Math.PI * 2); ctx.fill();
      for (let i = 1; i < pts.length; i++) {
        const px = pts[i-1][0], py = pts[i-1][1], pr = ((pts[i-1][2] ?? (stroke.size || DEFAULT_PAINT_SIZE)) * sizeScale) / 2;
        const qx = pts[i][0],   qy = pts[i][1],   qr = ((pts[i][2]   ?? (stroke.size || DEFAULT_PAINT_SIZE)) * sizeScale) / 2;
        ctx.beginPath(); ctx.arc(qx, qy, qr, 0, Math.PI * 2); ctx.fill();
        const dx = qx - px, dy = qy - py, len = Math.hypot(dx, dy);
        if (len > 0) {
          const nx = -dy / len, ny = dx / len;
          ctx.beginPath();
          ctx.moveTo(px + nx * pr, py + ny * pr);
          ctx.lineTo(qx + nx * qr, qy + ny * qr);
          ctx.lineTo(qx - nx * qr, qy - ny * qr);
          ctx.lineTo(px - nx * pr, py - ny * pr);
          ctx.closePath(); ctx.fill();
        }
      }
    }
  }

  function drawPaint(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();
    const [cx, cy] = paintCenter(ann);
    const x = ann.x || 0, y = ann.y || 0;
    const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = ann.rotation || 0;
    ctx.save();
    ctx.translate(cx + x, cy + y);
    ctx.rotate(r);
    ctx.scale(sx, sy);
    ctx.translate(-cx, -cy);
    renderStrokes(ann.strokes || [], ann.sizeScale ?? 1);
    ctx.restore();
    if (isHovered(ann) && !selected) {
      const corners = getTransformedCorners(ann, HOVER_PAD + 2);
      if (corners.length === 4) {
        ctx.save();
        ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HOVER_OPACITY})`;
        ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
        ctx.beginPath();
        ctx.moveTo(corners[0][0], corners[0][1]);
        for (let i = 1; i < corners.length; i++) ctx.lineTo(corners[i][0], corners[i][1]);
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  function drawText(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();
    const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
    const lineHeight = fontSize * 1.2;
    const lines = (ann.text || "").split("\n");
    ctx.save();
    ctx.font = `${fontSize}px sans-serif`;
    ctx.translate(ann.x || 0, ann.y || 0);
    ctx.rotate(ann.rotation || 0);
    ctx.fillStyle = ann.color || DEFAULT_COLOR;
    ctx.textBaseline = "top";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], 0, i * lineHeight);
    }
    if (isHovered(ann) && !selected) {
      const w = Math.max(1, ...lines.map((l) => ctx.measureText(l).width));
      const h = lineHeight * lines.length;
      ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HOVER_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
      ctx.strokeRect(-HOVER_PAD, -HOVER_PAD, w + HOVER_PAD * 2, h + HOVER_PAD * 2);
    }
    ctx.restore();
  }

  function drawArrowLine(x1, y1, x2, y2, color, width, cp1x, cp1y, cp2x, cp2y, hasStartArrow, hasEndArrow, taper) {
    const { ctx } = getState();
    if (cp1x == null) cp1x = x1 + (x2 - x1) / 3;
    if (cp1y == null) cp1y = y1 + (y2 - y1) / 3;
    if (cp2x == null) cp2x = x1 + (x2 - x1) * 2 / 3;
    if (cp2y == null) cp2y = y1 + (y2 - y1) * 2 / 3;
    if (hasEndArrow == null) hasEndArrow = true;
    const w = Math.max(1, width);
    const head = Math.max(15, w * 4);
    const setback = head * Math.cos(Math.PI / 6);

    let endAngle = 0, startAngle = 0;
    if (hasEndArrow) {
      endAngle = Math.hypot(x2 - cp2x, y2 - cp2y) < 0.1
        ? Math.atan2(y2 - y1, x2 - x1)
        : Math.atan2(y2 - cp2y, x2 - cp2x);
    }
    if (hasStartArrow) {
      startAngle = Math.hypot(cp1x - x1, cp1y - y1) < 0.1
        ? Math.atan2(y1 - y2, x1 - x2)
        : Math.atan2(y1 - cp1y, x1 - cp1x);
    }

    const lx2 = hasEndArrow   ? x2 - setback * Math.cos(endAngle)   : x2;
    const ly2 = hasEndArrow   ? y2 - setback * Math.sin(endAngle)   : y2;
    const lx1 = hasStartArrow ? x1 - setback * Math.cos(startAngle) : x1;
    const ly1 = hasStartArrow ? y1 - setback * Math.sin(startAngle) : y1;

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;

    if (taper) {
      // Velocity taper: stroke is thickest where the bezier curves most (slow parametric
      // speed) and thinnest where it runs straight (fast speed). On a perfectly straight
      // arrow the width is uniform; on a curved bezier the curves bulge out visibly.
      const N = 48;
      const bxs = [], bys = [], spds = [];
      for (let i = 0; i <= N; i++) {
        const t = i / N, mt = 1 - t;
        bxs.push(mt**3*lx1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*lx2);
        bys.push(mt**3*ly1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ly2);
        const dvx = 3*(mt**2*(cp1x-lx1) + 2*mt*t*(cp2x-cp1x) + t**2*(lx2-cp2x));
        const dvy = 3*(mt**2*(cp1y-ly1) + 2*mt*t*(cp2y-cp1y) + t**2*(ly2-cp2y));
        spds.push(Math.max(0.001, Math.hypot(dvx, dvy)));
      }
      const minSpd = Math.min(...spds);
      const left = [], right = [];
      for (let i = 0; i <= N; i++) {
        const t = i / N, mt = 1 - t;
        const spd = spds[i];
        const hw = (minSpd / spd) * w / 2;
        const dvx = 3*(mt**2*(cp1x-lx1) + 2*mt*t*(cp2x-cp1x) + t**2*(lx2-cp2x));
        const dvy = 3*(mt**2*(cp1y-ly1) + 2*mt*t*(cp2y-cp1y) + t**2*(ly2-cp2y));
        const [px, py] = spd < 0.001 ? [0, hw] : [-dvy / spd * hw, dvx / spd * hw];
        left.push([bxs[i] + px, bys[i] + py]);
        right.push([bxs[i] - px, bys[i] - py]);
      }
      ctx.beginPath();
      ctx.moveTo(left[0][0], left[0][1]);
      for (let i = 1; i <= N; i++) ctx.lineTo(left[i][0], left[i][1]);
      for (let i = N; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
      ctx.closePath();
      ctx.fill();
    } else {
      // Uniform-width stroke — simple and clean.
      ctx.lineWidth = w;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(lx1, ly1);
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, lx2, ly2);
      ctx.stroke();
    }

    if (hasEndArrow) {
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - head * Math.cos(endAngle - Math.PI/6), y2 - head * Math.sin(endAngle - Math.PI/6));
      ctx.lineTo(x2 - head * Math.cos(endAngle + Math.PI/6), y2 - head * Math.sin(endAngle + Math.PI/6));
      ctx.closePath(); ctx.fill();
    }
    if (hasStartArrow) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(startAngle - Math.PI/6), y1 - head * Math.sin(startAngle - Math.PI/6));
      ctx.lineTo(x1 - head * Math.cos(startAngle + Math.PI/6), y1 - head * Math.sin(startAngle + Math.PI/6));
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }

  function drawArrowAnnotation(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();
    const isBezier = ann.is_bezier ?? false;
    const cps = defaultCps(ann);
    const cp1x = isBezier ? cps.cp1x : null;
    const cp1y = isBezier ? cps.cp1y : null;
    const cp2x = isBezier ? cps.cp2x : null;
    const cp2y = isBezier ? cps.cp2y : null;
    drawArrowLine(ann.x1, ann.y1, ann.x2, ann.y2, ann.color || DEFAULT_COLOR, ann.width || DEFAULT_ARROW_WIDTH,
      cp1x, cp1y, cp2x, cp2y, ann.has_start_arrow ?? false, ann.has_end_arrow ?? true, ann.taper ?? false);
    if (selected) {
      const r = HANDLE_RADIUS / displayScale;
      ctx.save();
      for (const [ex, ey] of [[ann.x1, ann.y1], [ann.x2, ann.y2]]) {
        ctx.fillStyle = HANDLE_FILL;
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HANDLE_STROKE_OPACITY})`; ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.stroke();
      }
      if (isBezier) {
        const cpR = CP_HANDLE_RADIUS / displayScale;
        ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${CP_LINE_OPACITY})`;
        ctx.lineWidth = LINE_WIDTH_SECONDARY / displayScale;
        ctx.setLineDash(DASH_CP_LINE.map((v) => v / displayScale));
        ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(cps.cp1x, cps.cp1y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ann.x2, ann.y2); ctx.lineTo(cps.cp2x, cps.cp2y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HANDLE_STROKE_OPACITY})`;
        ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
        for (const [hx, hy] of [[cps.cp1x, cps.cp1y], [cps.cp2x, cps.cp2y]]) {
          ctx.fillStyle = HANDLE_FILL;
          ctx.beginPath(); ctx.arc(hx, hy, cpR, 0, Math.PI * 2); ctx.fill();
          ctx.beginPath(); ctx.arc(hx, hy, cpR, 0, Math.PI * 2); ctx.stroke();
        }
      }
      ctx.restore();
    }
    if (isHovered(ann) && !selected) {
      const r = CP_HANDLE_RADIUS / displayScale;
      ctx.save();
      ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HOVER_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
      for (const [ex, ey] of [[ann.x1, ann.y1], [ann.x2, ann.y2]]) {
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
    }
  }

  function drawRect(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();
    const hw = (ann.w || 10) / 2, hh = (ann.h || 10) / 2;
    ctx.save();
    ctx.translate(ann.x || 0, ann.y || 0);
    ctx.rotate(ann.rotation || 0);
    ctx.lineWidth = ann.width || DEFAULT_SHAPE_WIDTH;
    ctx.strokeStyle = ann.color || DEFAULT_COLOR;
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fillRect(-hw, -hh, hw * 2, hh * 2); }
    ctx.strokeRect(-hw, -hh, hw * 2, hh * 2);
    if (isHovered(ann) && !selected) {
      const pad = HOVER_PAD / displayScale;
      ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HOVER_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
      ctx.strokeRect(-hw - pad, -hh - pad, hw * 2 + pad * 2, hh * 2 + pad * 2);
    }
    ctx.restore();
  }

  function drawEllipse(ann, selected) {
    const { ctx, displayScale, hoverId } = getState();
    const rx = Math.max(0.5, (ann.w || 10) / 2), ry = Math.max(0.5, (ann.h || 10) / 2);
    ctx.save();
    ctx.translate(ann.x || 0, ann.y || 0);
    ctx.rotate(ann.rotation || 0);
    ctx.lineWidth = ann.width || DEFAULT_SHAPE_WIDTH;
    ctx.strokeStyle = ann.color || DEFAULT_COLOR;
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fill(); }
    ctx.stroke();
    if (isHovered(ann) && !selected) {
      const pad = HOVER_PAD / displayScale;
      ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${HOVER_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx + pad, ry + pad, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  return { renderStrokes, drawPaint, drawText, drawArrowLine, drawArrowAnnotation, drawRect, drawEllipse };
}
