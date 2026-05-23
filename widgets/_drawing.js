// Canvas drawing functions for the annotation widget.
// Use createDrawing(getState) to get a bound set of draw functions.
// getState() must return { ctx, displayScale, hoverId }.

import { paintCenter, defaultCps, naturalBounds, getTransformedCorners } from './_geometry.js';

export function createDrawing(getState) {

  function renderStrokes(strokes, sizeScale = 1) {
    const { ctx } = getState();
    for (const stroke of strokes) {
      const pts = stroke.points || [];
      if (!pts.length) continue;
      ctx.fillStyle = stroke.color || "#ff0000";
      const r0 = ((pts[0][2] ?? (stroke.size || 8)) * sizeScale) / 2;
      ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], r0, 0, Math.PI * 2); ctx.fill();
      for (let i = 1; i < pts.length; i++) {
        const px = pts[i-1][0], py = pts[i-1][1], pr = ((pts[i-1][2] ?? (stroke.size || 8)) * sizeScale) / 2;
        const qx = pts[i][0],   qy = pts[i][1],   qr = ((pts[i][2]   ?? (stroke.size || 8)) * sizeScale) / 2;
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
    if (ann.id === hoverId && !selected) {
      const corners = getTransformedCorners(ann, 6);
      if (corners.length === 4) {
        ctx.save();
        ctx.strokeStyle = "rgba(122,157,184,0.7)";
        ctx.lineWidth = 1.5 / displayScale;
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
    const fontSize = Math.max(8, ann.font_size || 48);
    const lineHeight = fontSize * 1.2;
    const lines = (ann.text || "").split("\n");
    const x = ann.x || 0;
    const y = ann.y || 0;
    ctx.save();
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillStyle = ann.color || "#ffffff";
    ctx.textBaseline = "top";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], x, y + i * lineHeight);
    }
    if (ann.id === hoverId && !selected) {
      const w = Math.max(...lines.map((l) => ctx.measureText(l).width));
      const h = lineHeight * lines.length;
      ctx.strokeStyle = "rgba(122,157,184,0.7)";
      ctx.lineWidth = 1.5 / displayScale;
      ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);
    }
    ctx.restore();
  }

  function drawArrowLine(x1, y1, x2, y2, color, width, cp1x, cp1y, cp2x, cp2y, hasStartArrow, hasEndArrow) {
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

    const N = 48;
    const pts = [], speeds = [], tangents = [];
    for (let i = 0; i <= N; i++) {
      const t = i / N, mt = 1 - t;
      pts.push([
        mt**3*lx1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*lx2,
        mt**3*ly1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ly2,
      ]);
      const dvx = 3*(mt**2*(cp1x-lx1) + 2*mt*t*(cp2x-cp1x) + t**2*(lx2-cp2x));
      const dvy = 3*(mt**2*(cp1y-ly1) + 2*mt*t*(cp2y-cp1y) + t**2*(ly2-cp2y));
      const spd = Math.hypot(dvx, dvy);
      tangents.push([dvx, dvy, spd]);
      speeds.push(spd);
    }

    const minSpd = Math.min(...speeds), maxSpd = Math.max(...speeds);
    const spdRange = maxSpd - minSpd;

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = color;

    if (spdRange < 0.001) {
      ctx.lineWidth = w;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(lx1, ly1);
      ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, lx2, ly2);
      ctx.stroke();
    } else {
      const minW = w * 0.18, maxW = w;
      const left = [], right = [];
      for (let i = 0; i <= N; i++) {
        const [bx, by] = pts[i];
        const [dvx, dvy, spd] = tangents[i];
        const rawNorm = (speeds[i] - minSpd) / spdRange;
        const hw = (minW + (1 - rawNorm) * (maxW - minW)) / 2;
        const [px, py] = spd < 0.001 ? [0, hw] : [-dvy / spd * hw, dvx / spd * hw];
        left.push([bx + px, by + py]);
        right.push([bx - px, by - py]);
      }
      ctx.beginPath();
      ctx.moveTo(left[0][0], left[0][1]);
      for (let i = 1; i <= N; i++) ctx.lineTo(left[i][0], left[i][1]);
      for (let i = N; i >= 0; i--) ctx.lineTo(right[i][0], right[i][1]);
      ctx.closePath();
      ctx.fill();
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
    drawArrowLine(ann.x1, ann.y1, ann.x2, ann.y2, ann.color || "#ff0000", ann.width || 3,
      cp1x, cp1y, cp2x, cp2y, ann.has_start_arrow ?? false, ann.has_end_arrow ?? true);
    if (selected) {
      const r = 5 / displayScale;
      ctx.save();
      for (const [ex, ey] of [[ann.x1, ann.y1], [ann.x2, ann.y2]]) {
        ctx.fillStyle = "white";
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(122,157,184,0.9)"; ctx.lineWidth = 1.5 / displayScale;
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.stroke();
      }
      if (isBezier) {
        const cpR = 4 / displayScale;
        ctx.strokeStyle = "rgba(122,157,184,0.5)";
        ctx.lineWidth = 1 / displayScale;
        ctx.setLineDash([3 / displayScale, 2 / displayScale]);
        ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(cps.cp1x, cps.cp1y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ann.x2, ann.y2); ctx.lineTo(cps.cp2x, cps.cp2y); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = "white";
        ctx.strokeStyle = "rgba(122,157,184,0.9)";
        ctx.lineWidth = 1.5 / displayScale;
        ctx.beginPath(); ctx.arc(cps.cp1x, cps.cp1y, cpR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cps.cp1x, cps.cp1y, cpR, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "white";
        ctx.beginPath(); ctx.arc(cps.cp2x, cps.cp2y, cpR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cps.cp2x, cps.cp2y, cpR, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
    }
    if (ann.id === hoverId && !selected) {
      const r = 4 / displayScale;
      ctx.save();
      ctx.strokeStyle = "rgba(122,157,184,0.7)";
      ctx.lineWidth = 1.5 / displayScale;
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
    ctx.lineWidth = ann.width || 2;
    ctx.strokeStyle = ann.color || "#ff0000";
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fillRect(-hw, -hh, hw * 2, hh * 2); }
    ctx.strokeRect(-hw, -hh, hw * 2, hh * 2);
    if (ann.id === hoverId && !selected) {
      const pad = 4 / displayScale;
      ctx.strokeStyle = "rgba(122,157,184,0.7)";
      ctx.lineWidth = 1.5 / displayScale;
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
    ctx.lineWidth = ann.width || 2;
    ctx.strokeStyle = ann.color || "#ff0000";
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fill(); }
    ctx.stroke();
    if (ann.id === hoverId && !selected) {
      const pad = 4 / displayScale;
      ctx.strokeStyle = "rgba(122,157,184,0.7)";
      ctx.lineWidth = 1.5 / displayScale;
      ctx.beginPath();
      ctx.ellipse(0, 0, rx + pad, ry + pad, 0, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.restore();
  }

  return { renderStrokes, drawPaint, drawText, drawArrowLine, drawArrowAnnotation, drawRect, drawEllipse };
}
