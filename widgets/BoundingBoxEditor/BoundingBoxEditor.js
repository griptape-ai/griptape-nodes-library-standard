import {
  WIDGET_VERSION,
  HANDLE_R,
  CANVAS_BG, CANVAS_GRID, OVERLAY, BOX_BORDER, BOX_FILL, GUIDE,
  HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED,
} from './_styles.js';
import { createSidebar } from './_sidebar.js';
import { createFooter }  from './_footer.js';

// Widget contract (CreateBoundingBox lists the box and canvas parameters in synced_parameters):
//   props.value carries the node's parameters merged into the widget's own value:
//     canvas_width / canvas_height : canvas size in box coordinates (origin is always 0,0)
//     left / top / width / height  : the box
//     image_url / image_revision   : the widget's own value; optional background image, empty draws a blank grid canvas
//   props.locked : subset of ["left", "top", "width", "height"] driven by incoming connections
// onChange takes the whole value; the editor writes the synced keys back to the node's parameters.

// Mirrors box_problems() in create_bounding_box.py: a box is unusable when it is empty or leaves the canvas.
function isBoxInvalid(v) {
  const l = v.left ?? 0, t = v.top ?? 0, w = v.width ?? 0, h = v.height ?? 0;
  return w <= 0 || h <= 0 || l < 0 || t < 0
    || l + w > (v.canvas_width ?? 0) || t + h > (v.canvas_height ?? 0);
}

export default function BoundingBoxEditor(container, props) {
  if (container._bboxEditor?.wrapper?.isConnected) {
    container._bboxEditor.handleUpdate(props);
    return { cleanup: container._bboxEditor.cleanup, update: container._bboxEditor.handleUpdate };
  }
  if (container._bboxEditor) delete container._bboxEditor;

  // ── State ──────────────────────────────────────────────────────────────────
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled = props.disabled || false;

  let imageUrl     = latestValue.image_url || "";
  let imageRevision = latestValue.image_revision || 0;
  let canvasW      = latestValue.canvas_width  || 0;
  let canvasH      = latestValue.canvas_height || 0;
  let boxL         = latestValue.left   ?? 0;
  let boxT         = latestValue.top    ?? 0;
  let boxW         = latestValue.width  ?? 0;
  let boxH         = latestValue.height ?? 0;
  // Box parameters with an incoming connection, worked out by the editor from synced_parameters
  let lockedParams = props.locked || [];
  let boxInvalid = isBoxInvalid(latestValue);

  function canvasReady() { return canvasW > 0 && canvasH > 0; }
  function ecL() { return boxL; }
  function ecT() { return boxT; }
  function ecW() { return boxW > 0 ? boxW : (canvasW || 1); }
  function ecH() { return boxH > 0 ? boxH : (canvasH || 1); }

  // ── Per-handle locking ─────────────────────────────────────────────────────
  function handleAffects(id) {
    const fields = [];
    if (id.includes("n")) { fields.push("top"); fields.push("height"); }
    if (id.includes("s")) fields.push("height");
    if (id.includes("w")) { fields.push("left"); fields.push("width"); }
    if (id.includes("e")) fields.push("width");
    return fields;
  }
  function isHandleLocked(id) { return handleAffects(id).some(f => lockedParams.includes(f)); }
  function isMoveLocked() { return lockedParams.includes("left") || lockedParams.includes("top"); }
  function isDrawLocked() { return isMoveLocked() || lockedParams.includes("width") || lockedParams.includes("height"); }

  let scale = 1;
  let mode = "idle";
  let activeHandle = null;
  let dragStart = null;
  let hoveredHandle = null;

  // ── DOM ────────────────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "bbox-editor nodrag nowheel";
  wrapper.style.cssText = [
    "display:flex", "flex-direction:column", "gap:6px",
    "height:100%",
    "user-select:none", "-webkit-user-select:none",
  ].join(";");

  const canvasRow = document.createElement("div");
  canvasRow.style.cssText = [
    "display:flex", "flex-direction:row", "gap:6px",
    "flex:1 1 0", "min-height:180px",
  ].join(";");

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:6px",
    "overflow:hidden", "flex:1 1 0", "min-height:0",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;cursor:crosshair;outline:1px solid rgba(255,255,255,0.5);";
  const ctx = canvas.getContext("2d");

  // ── Sidebar ────────────────────────────────────────────────────────────────
  const sidebarInst = createSidebar({
    getCanvasSize: () => ({ canvasW, canvasH }),
    getBoxRect: () => ({ l: ecL(), t: ecT(), w: ecW(), h: ecH() }),
    onApply(box) {
      if (isDisabled || !canvasReady()) return;
      mode = "idle"; activeHandle = null; dragStart = null; hoveredHandle = null;
      boxL = box.left; boxT = box.top; boxW = box.width; boxH = box.height;
      commit();
      syncUI();
    },
  });

  // ── Footer (status bar) ────────────────────────────────────────────────────
  const footerInst = createFooter({ version: WIDGET_VERSION });

  canvasWrap.appendChild(canvas);
  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);
  wrapper.appendChild(canvasRow);
  wrapper.appendChild(footerInst.statusBar);
  container.appendChild(wrapper);

  function syncUI() {
    sidebarInst.syncDisabled(lockedParams, isDisabled);
  }

  // ── Background image ───────────────────────────────────────────────────────
  let img = new Image();
  img.crossOrigin = "anonymous";
  let imageLoaded = false;
  let imageRequestId = 0;

  function initBox() {
    if (canvasReady()) {
      if (boxW === 0) boxW = canvasW;
      if (boxH === 0) boxH = canvasH;
    }
  }

  function resizeCanvas() {
    if (!canvasReady()) { canvas.width = 400; canvas.height = 300; render(); return; }
    const areaW = canvasWrap.clientWidth  || 480;
    const areaH = canvasWrap.clientHeight || 360;
    scale = Math.min(areaW / canvasW, areaH / canvasH);
    canvas.width  = Math.max(1, Math.round(canvasW * scale));
    canvas.height = Math.max(1, Math.round(canvasH * scale));
    render();
  }

  function loadImage(url) {
    const requestId = ++imageRequestId;
    imageLoaded = false;

    if (!url) {
      img = new Image();
      img.crossOrigin = "anonymous";
      render();
      return;
    }

    const nextImage = new Image();
    nextImage.crossOrigin = "anonymous";
    nextImage.onload = () => {
      if (requestId !== imageRequestId) return;
      img = nextImage;
      imageLoaded = true;
      render();
    };
    nextImage.onerror = () => {
      if (requestId !== imageRequestId) return;
      imageLoaded = false;
      render();
    };
    img = nextImage;
    img.src = url;
  }

  // ── Emit ───────────────────────────────────────────────────────────────────
  function emitAll() {
    if (typeof onChangeRef === "function") {
      onChangeRef({
        ...latestValue,
        left: boxL, top: boxT, width: boxW, height: boxH,
      });
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    const cw = canvas.width  || 400;
    const ch = canvas.height || 300;
    ctx.clearRect(0, 0, cw, ch);

    if (!canvasReady()) {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#555";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Set a canvas width and height", cw / 2, ch / 2);
      footerInst.updateStatus({ canvasReady: false, lockedParams, left: 0, top: 0, width: 0, height: 0 });
      return;
    }

    if (imageLoaded) {
      ctx.drawImage(img, 0, 0, cw, ch);
    } else {
      ctx.fillStyle = CANVAS_BG;
      ctx.fillRect(0, 0, cw, ch);
      ctx.strokeStyle = CANVAS_GRID;
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = 1; i < 10; i++) {
        ctx.moveTo(cw * i / 10, 0); ctx.lineTo(cw * i / 10, ch);
        ctx.moveTo(0, ch * i / 10); ctx.lineTo(cw, ch * i / 10);
      }
      ctx.stroke();
    }

    const strip = stripRect();
    if (strip) {
      ctx.fillStyle = OVERLAY;
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#ef4444";
      ctx.fillRect(strip.x, strip.y, strip.w, strip.h);
      drawHandles();
      footerInst.updateStatus({ canvasReady: true, lockedParams, left: ecL(), top: ecT(), width: ecW(), height: ecH() });
      return;
    }

    const x  = ecL() * scale, y  = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const rw = x2 - x, rh = y2 - y;

    ctx.fillStyle = OVERLAY;
    if (y  > 0)  ctx.fillRect(0, 0, cw, y);
    if (y2 < ch) ctx.fillRect(0, y2, cw, ch - y2);
    if (x  > 0)  ctx.fillRect(0, y, x, rh);
    if (x2 < cw) ctx.fillRect(x2, y, cw - x2, rh);

    // Rule-of-thirds guides
    ctx.save();
    ctx.strokeStyle = GUIDE;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let i = 1; i < 3; i++) {
      ctx.moveTo(x + rw * i / 3, y);  ctx.lineTo(x + rw * i / 3, y2);
      ctx.moveTo(x, y + rh * i / 3);  ctx.lineTo(x2, y + rh * i / 3);
    }
    ctx.stroke();
    ctx.restore();

    const boxColor = boxInvalid ? "#ef4444" : BOX_BORDER;
    const boxFill = boxInvalid ? "rgba(239,68,68,0.12)" : BOX_FILL;
    ctx.fillStyle = boxFill;
    ctx.fillRect(x, y, rw, rh);
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, rw, rh);

    drawHandles();

    footerInst.updateStatus({
      canvasReady: true, lockedParams,
      left: ecL(), top: ecT(), width: ecW(), height: ecH(),
    });
  }

  function drawHandles() {
    if (isDisabled) return;
    for (const h of getHandles()) {
      const locked = isHandleLocked(h.id);
      ctx.beginPath();
      ctx.arc(h.cx, h.cy, HANDLE_R, 0, Math.PI * 2);
      ctx.fillStyle = locked ? HANDLE_LOCKED : (boxInvalid ? "#ef4444" : HANDLE_FILL);
      ctx.fill();
      ctx.strokeStyle = locked ? "rgba(100,100,100,0.6)" : (hoveredHandle === h.id ? HANDLE_STROKE_HOVER : (boxInvalid ? "#ef4444" : HANDLE_STROKE));
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }

  // ── Handles ────────────────────────────────────────────────────────────────
  // An off-canvas box exposes only the handles facing the canvas.
  const FACING = { left: ["ne", "e", "se"], right: ["nw", "w", "sw"], top: ["sw", "s", "se"], bottom: ["nw", "n", "ne"] };

  function getHandles() {
    const s = stripRect();
    let x = ecL() * scale, y = ecT() * scale;
    let x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    // An off-canvas box is resized from its grab strip.
    if (s) { x = s.x; y = s.y; x2 = s.x + s.w; y2 = s.y + s.h; }
    const mx = (x + x2) / 2, my = (y + y2) / 2;
    const handles = [
      { id: "nw", cx: x,  cy: y  }, { id: "n",  cx: mx, cy: y  },
      { id: "ne", cx: x2, cy: y  }, { id: "e",  cx: x2, cy: my },
      { id: "se", cx: x2, cy: y2 }, { id: "s",  cx: mx, cy: y2 },
      { id: "sw", cx: x,  cy: y2 }, { id: "w",  cx: x,  cy: my },
    ];
    if (s) return handles.filter(h => FACING[offCanvasEdge()].includes(h.id));
    return handles;
  }

  // The canvas edge a wholly off-canvas box is furthest past, or null if it overlaps the canvas.
  function offCanvasEdge() {
    if (mode === "drawing") return null;
    const gaps = { left: -(boxL + ecW()), right: boxL - canvasW, top: -(boxT + ecH()), bottom: boxT - canvasH };
    let edge = "left";
    for (const side in gaps) if (gaps[side] > gaps[edge]) edge = side;
    return gaps[edge] >= 0 ? edge : null;
  }

  // The part of [start, start + size) on [0, limit); a span wholly past the canvas is pinned to the nearest end.
  function visibleSpan(start, size, limit) {
    const a = Math.max(0, start), b = Math.min(limit, start + size);
    if (a < b) return [a, b];
    if (start >= limit) return [Math.max(0, limit - size), limit];
    return [0, Math.min(limit, size)];
  }

  // The one-pixel grab strip on that edge, spanning the box's visible extent along it.
  function stripRect() {
    const edge = offCanvasEdge();
    if (!edge) return null;
    const [x0, x1] = visibleSpan(boxL, ecW(), canvasW);
    const [y0, y1] = visibleSpan(boxT, ecH(), canvasH);
    if (edge === "left")  return { x: 0, y: y0 * scale, w: 1, h: (y1 - y0) * scale };
    if (edge === "right") return { x: canvas.width - 1, y: y0 * scale, w: 1, h: (y1 - y0) * scale };
    if (edge === "top")   return { x: x0 * scale, y: 0, w: (x1 - x0) * scale, h: 1 };
    return { x: x0 * scale, y: canvas.height - 1, w: (x1 - x0) * scale, h: 1 };
  }

  function hitStrip(px, py) {
    const s = stripRect();
    const pad = HANDLE_R + 5;
    return s !== null && px >= s.x - pad && px <= s.x + s.w + pad && py >= s.y - pad && py <= s.y + s.h + pad;
  }

  function hitHandle(px, py) {
    const hitR = HANDLE_R + 5;
    for (const h of getHandles()) {
      if (isHandleLocked(h.id)) continue;
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h.id;
    }
    return null;
  }

  function hitBox(px, py) {
    return px >= ecL() * scale && px <= (ecL() + ecW()) * scale
        && py >= ecT() * scale && py <= (ecT() + ecH()) * scale;
  }

  const CURSORS = { nw:"nw-resize", n:"n-resize", ne:"ne-resize", e:"e-resize", se:"se-resize", s:"s-resize", sw:"sw-resize", w:"w-resize" };

  function canvasPos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }

  function toCanvasUnits(cx, cy) { return [cx / scale, cy / scale]; }

  function clampRect(l, t, w, h) {
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    l = Math.max(0, Math.min(Math.round(l), canvasW - w));
    t = Math.max(0, Math.min(Math.round(t), canvasH - h));
    return [l, t, Math.min(w, canvasW - l), Math.min(h, canvasH - t)];
  }

  // A valid box is kept inside the canvas. An invalid one keeps its coordinates as edited, rounded by edge.
  function settleRect(l, t, w, h) {
    if (!boxInvalid) return clampRect(l, t, w, h);
    const rl = Math.round(l), rt = Math.round(t);
    return [rl, rt, Math.round(l + w) - rl, Math.round(t + h) - rt];
  }

  function commit() {
    const [l, t, w, h] = settleRect(boxL, boxT, boxW || canvasW, boxH || canvasH);
    boxL = l; boxT = t; boxW = w; boxH = h;
    emitAll();
    render();
  }

  function overlapsCanvas(start, size, limit) {
    return Math.max(0, start) < Math.min(limit, start + size);
  }

  // Clamps a move to the canvas, but never pulls in a coordinate that started outside it.
  function clampMove(start, delta, size, limit) {
    return Math.max(Math.min(0, start), Math.min(start + delta, Math.max(limit - size, start)));
  }

  // A move starts from the box pulled onto the canvas so it follows the pointer at once. A box wholly off the canvas
  // is pulled in only along the axes it does not overlap. prev keeps the real coordinates of a pulled-in box.
  function moveOrigin(ix, iy) {
    let [l, t] = clampRect(boxL, boxT, ecW(), ecH());
    if (offCanvasEdge()) {
      if (overlapsCanvas(boxL, ecW(), canvasW)) l = boxL;
      if (overlapsCanvas(boxT, ecH(), canvasH)) t = boxT;
    }
    const origin = { ix, iy, l, t, w: ecW(), h: ecH() };
    if (l !== boxL || t !== boxT) origin.prev = { l: boxL, t: boxT, w: boxW, h: boxH };
    return origin;
  }

  // ── Pointer events ─────────────────────────────────────────────────────────
  canvas.addEventListener("pointerdown", (e) => {
    if (isDisabled || !canvasReady()) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toCanvasUnits(px, py);
    const handle = hitHandle(px, py);

    if (handle) {
      canvas.setPointerCapture(e.pointerId);
      mode = "resizing"; activeHandle = handle;
      dragStart = { ix, iy, l: boxL, t: boxT, w: ecW(), h: ecH() };
      // A box wholly off the canvas resizes from its real edges; prev keeps them until the drag brings it on.
      if (offCanvasEdge()) dragStart.prev = { l: boxL, t: boxT, w: boxW, h: boxH };
    } else if ((hitStrip(px, py) || hitBox(px, py)) && !isMoveLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "moving"; canvas.style.cursor = "grabbing";
      dragStart = moveOrigin(ix, iy);
    } else if (!hitBox(px, py) && !isDrawLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "drawing";
      const six = Math.max(0, Math.min(ix, canvasW));
      const siy = Math.max(0, Math.min(iy, canvasH));
      dragStart = { ix: six, iy: siy, prev: { l: boxL, t: boxT, w: boxW, h: boxH } };
      boxL = six; boxT = siy; boxW = 0; boxH = 0;
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!canvasReady()) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toCanvasUnits(px, py);

    if (mode === "idle") {
      const h = hitHandle(px, py);
      hoveredHandle = h;
      if (h) canvas.style.cursor = CURSORS[h];
      else if (hitStrip(px, py)) canvas.style.cursor = isMoveLocked() ? "not-allowed" : "grab";
      else if (hitBox(px, py)) canvas.style.cursor = isMoveLocked() ? "not-allowed" : "move";
      else canvas.style.cursor = isDrawLocked() ? "not-allowed" : "crosshair";
      render();
      return;
    }

    if (mode === "moving") {
      const dx = ix - dragStart.ix, dy = iy - dragStart.iy;
      // A click (no displacement yet) must not move the box.
      if (dx !== 0 || dy !== 0) dragStart.moved = true;
      if (dragStart.moved) {
        boxW = dragStart.w; boxH = dragStart.h;
        boxL = clampMove(dragStart.l, dx, boxW, canvasW);
        boxT = clampMove(dragStart.t, dy, boxH, canvasH);
      }
    } else if (mode === "resizing") {
      const { l, t, w, h } = dragStart;
      let nl = l, nt = t, nr = l + w, nb = t + h;
      const cix = Math.max(0, Math.min(ix, canvasW));
      const ciy = Math.max(0, Math.min(iy, canvasH));
      const id = activeHandle;
      if (id.includes("n")) nt = Math.min(ciy, nb - 1);
      if (id.includes("s")) nb = Math.max(ciy, nt + 1);
      if (id.includes("w")) nl = Math.min(cix, nr - 1);
      if (id.includes("e")) nr = Math.max(cix, nl + 1);

      if (e.shiftKey && id.length === 2) {
        const ratio = dragStart.w / dragStart.h;
        const curW = nr - nl, curH = nb - nt;
        if (Math.abs(curW - dragStart.w) >= Math.abs(curH - dragStart.h) * ratio) {
          const ch = Math.round(curW / ratio);
          if (id.includes("n")) nt = Math.max(0, nb - ch);
          else nb = Math.min(canvasH, nt + ch);
        } else {
          const cw = Math.round(curH * ratio);
          if (id.includes("w")) nl = Math.max(0, nr - cw);
          else nr = Math.min(canvasW, nl + cw);
        }
      }

      boxL = nl; boxT = nt; boxW = nr - nl; boxH = nb - nt;
    } else if (mode === "drawing") {
      const six = dragStart.ix, siy = dragStart.iy;
      const eix = Math.max(0, Math.min(ix, canvasW));
      const eiy = Math.max(0, Math.min(iy, canvasH));
      if (e.shiftKey) {
        const dx = eix - six, dy = eiy - siy;
        const side = Math.max(Math.abs(dx), Math.abs(dy));
        const maxW = dx < 0 ? six : canvasW - six;
        const maxH = dy < 0 ? siy : canvasH - siy;
        const s = Math.min(side, maxW, maxH);
        boxW = s; boxH = s;
        boxL = dx < 0 ? six - s : six;
        boxT = dy < 0 ? siy - s : siy;
      } else {
        boxL = Math.min(six, eix); boxT = Math.min(siy, eiy);
        boxW = Math.abs(eix - six); boxH = Math.abs(eiy - siy);
      }
    }
    render();
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!canvasReady() || mode === "idle") return;
    e.stopPropagation();
    const prevMode = mode;
    const prevBox = dragStart?.prev;
    const startBox = prevBox
      ? settleRect(prevBox.l, prevBox.t, prevBox.w || canvasW, prevBox.h || canvasH)
      : settleRect(dragStart.l, dragStart.t, dragStart.w, dragStart.h);
    mode = "idle"; hoveredHandle = null; canvas.style.cursor = "crosshair";
    // A click, or a drag that leaves an off-canvas box off the canvas, restores its real coordinates without emitting.
    if (prevBox && prevMode !== "drawing") {
      const unchanged = boxL === prevBox.l && boxT === prevBox.t && boxW === prevBox.w && boxH === prevBox.h;
      if (unchanged || offCanvasEdge()) {
        boxL = prevBox.l; boxT = prevBox.t; boxW = prevBox.w; boxH = prevBox.h;
        render();
      } else {
        commit();
      }
      return;
    }
    // A click without a drag leaves a zero-size box; keep the previous one instead.
    if (prevMode === "drawing" && prevBox && (boxW < 1 || boxH < 1)) {
      boxL = prevBox.l; boxT = prevBox.t; boxW = prevBox.w; boxH = prevBox.h;
      render();
      return;
    }
    const endBox = settleRect(boxL, boxT, boxW || canvasW, boxH || canvasH);
    // A plain click on the box or a handle changes nothing: skip the emit.
    if (endBox.every((value, i) => value === startBox[i])) {
      [boxL, boxT, boxW, boxH] = endBox;
      render();
      return;
    }
    commit();
  });

  canvas.addEventListener("pointercancel", () => {
    if (mode !== "idle" && dragStart?.prev) {
      boxL = dragStart.prev.l; boxT = dragStart.prev.t;
      boxW = dragStart.prev.w; boxH = dragStart.prev.h;
    } else if (mode === "moving" || mode === "resizing") {
      boxL = dragStart.l; boxT = dragStart.t;
      boxW = dragStart.w; boxH = dragStart.h;
    }
    mode = "idle"; hoveredHandle = null; canvas.style.cursor = "crosshair"; render();
  });

  canvas.addEventListener("mousedown", (e) => e.stopPropagation());
  canvas.addEventListener("keydown",   (e) => e.stopPropagation());
  canvasWrap.addEventListener("pointerdown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("mousedown",   (e) => e.stopPropagation());

  // ── ResizeObserver ─────────────────────────────────────────────────────────
  const ro = new ResizeObserver(() => { if (canvasReady()) resizeCanvas(); });
  ro.observe(canvasWrap);

  // ── Initial load ───────────────────────────────────────────────────────────
  initBox();
  loadImage(imageUrl);
  resizeCanvas();
  syncUI();

  // ── handleUpdate ───────────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    isDisabled  = newProps.disabled || false;
    latestValue = newProps.value || {};

    const v = latestValue;
    const newUrl  = v.image_url || "";
    const newImageRevision = v.image_revision || 0;
    const newCanvasW = v.canvas_width  || 0;
    const newCanvasH = v.canvas_height || 0;

    const urlChanged = newUrl !== imageUrl;
    const imageChanged = urlChanged || newImageRevision !== imageRevision;
    const sizeChanged = newCanvasW !== canvasW || newCanvasH !== canvasH;

    if (imageChanged) {
      imageUrl = newUrl;
      imageRevision = newImageRevision;
      imageLoaded = false;
      loadImage(imageUrl);
    }
    canvasW = newCanvasW;
    canvasH = newCanvasH;

    lockedParams = newProps.locked || [];

    if (mode === "idle") {
      boxInvalid = isBoxInvalid(v);
      if ("left"   in v) boxL = v.left;
      if ("top"    in v) boxT = v.top;
      if ("width"  in v) boxW = v.width;
      if ("height" in v) boxH = v.height;
    }

    syncUI();
    if (sizeChanged) resizeCanvas();
    else render();
  }

  // ── cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    ro.disconnect();
    wrapper.remove();
    delete container._bboxEditor;
  }

  container._bboxEditor = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
