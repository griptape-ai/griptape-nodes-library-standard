import {
  WIDGET_VERSION,
  HANDLE_R,
  CANVAS_BG, CANVAS_GRID, OVERLAY, BOX_BORDER, BOX_FILL, GUIDE,
  HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED,
} from './_styles.js';
import { createSidebar } from './_sidebar.js';
import { createFooter }  from './_footer.js';

// Widget value contract (written by CreateBoundingBox):
//   img_width / img_height : canvas size in box coordinates (origin is always 0,0)
//   image_url              : optional background image; empty draws a blank grid canvas
//   left / top / width / height : the box
//   locked                 : subset of ["x1", "y1", "x2", "y2"] driven by incoming connections
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
  let canvasW      = latestValue.img_width  || 0;
  let canvasH      = latestValue.img_height || 0;
  let boxL         = latestValue.left   ?? 0;
  let boxT         = latestValue.top    ?? 0;
  let boxW         = latestValue.width  ?? 0;
  let boxH         = latestValue.height ?? 0;
  let lockedParams = latestValue.locked || [];

  function canvasReady() { return canvasW > 0 && canvasH > 0; }
  function ecL() { return Math.max(0, boxL); }
  function ecT() { return Math.max(0, boxT); }
  function ecW() { return boxW > 0 ? boxW : (canvasW || 1); }
  function ecH() { return boxH > 0 ? boxH : (canvasH || 1); }

  // ── Per-handle locking ─────────────────────────────────────────────────────
  function handleAffects(id) {
    const fields = [];
    if (id.includes("n")) fields.push("y1");
    if (id.includes("s")) fields.push("y2");
    if (id.includes("w")) fields.push("x1");
    if (id.includes("e")) fields.push("x2");
    return fields;
  }
  function isHandleLocked(id) { return handleAffects(id).some(f => lockedParams.includes(f)); }
  function isAnyLocked() { return ["x1", "y1", "x2", "y2"].some(f => lockedParams.includes(f)); }
  function isMoveLocked() { return isAnyLocked(); }
  function isDrawLocked() { return isAnyLocked(); }

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
  const img = new Image();
  img.crossOrigin = "anonymous";
  let imageLoaded = false;

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

  img.onload = () => { imageLoaded = true; render(); };
  img.onerror = () => { imageLoaded = false; render(); };

  function loadImage(url) {
    if (!url) { imageLoaded = false; render(); return; }
    const base = (u) => u ? u.split("?")[0] : "";
    if (base(url) === base(img.src)) return;
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
      footerInst.updateStatus({ canvasReady: false, lockedParams, x1: 0, y1: 0, x2: 0, y2: 0 });
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

    const x  = ecL() * scale, y  = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const rw = x2 - x, rh = y2 - y;

    // Dim everything outside the box (4 rects)
    ctx.fillStyle = OVERLAY;
    if (y  > 0)  ctx.fillRect(0, 0, cw, y);
    if (y2 < ch) ctx.fillRect(0, y2, cw, ch - y2);
    if (x  > 0)  ctx.fillRect(0, y, x, rh);
    if (x2 < cw) ctx.fillRect(x2, y, cw - x2, rh);

    ctx.fillStyle = BOX_FILL;
    ctx.fillRect(x, y, rw, rh);

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

    // Box border
    ctx.strokeStyle = BOX_BORDER;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, rw, rh);

    // Resize handles
    if (!isDisabled) {
      for (const h of getHandles()) {
        const locked = isHandleLocked(h.id);
        ctx.beginPath();
        ctx.arc(h.cx, h.cy, HANDLE_R, 0, Math.PI * 2);
        ctx.fillStyle = locked ? HANDLE_LOCKED : HANDLE_FILL;
        ctx.fill();
        ctx.strokeStyle = locked ? "rgba(100,100,100,0.6)" : (hoveredHandle === h.id ? HANDLE_STROKE_HOVER : HANDLE_STROKE);
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }

    footerInst.updateStatus({
      canvasReady: true, lockedParams,
      x1: ecL(), y1: ecT(), x2: ecL() + ecW(), y2: ecT() + ecH(),
    });
  }

  // ── Handles ────────────────────────────────────────────────────────────────
  function getHandles() {
    const x  = ecL() * scale, y  = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const mx = (x + x2) / 2, my = (y + y2) / 2;
    return [
      { id: "nw", cx: x,  cy: y  }, { id: "n",  cx: mx, cy: y  },
      { id: "ne", cx: x2, cy: y  }, { id: "e",  cx: x2, cy: my },
      { id: "se", cx: x2, cy: y2 }, { id: "s",  cx: mx, cy: y2 },
      { id: "sw", cx: x,  cy: y2 }, { id: "w",  cx: x,  cy: my },
    ];
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

  function commit() {
    const [l, t, w, h] = clampRect(boxL, boxT, boxW || canvasW, boxH || canvasH);
    boxL = l; boxT = t; boxW = w; boxH = h;
    emitAll();
    render();
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
    } else if (hitBox(px, py) && !isMoveLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "moving"; canvas.style.cursor = "grabbing";
      dragStart = { ix, iy, l: boxL, t: boxT, w: ecW(), h: ecH() };
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
      else if (hitBox(px, py)) canvas.style.cursor = isMoveLocked() ? "not-allowed" : "move";
      else canvas.style.cursor = isDrawLocked() ? "not-allowed" : "crosshair";
      render();
      return;
    }

    if (mode === "moving") {
      const dx = ix - dragStart.ix, dy = iy - dragStart.iy;
      boxW = dragStart.w; boxH = dragStart.h;
      boxL = Math.max(0, Math.min(dragStart.l + dx, canvasW - boxW));
      boxT = Math.max(0, Math.min(dragStart.t + dy, canvasH - boxH));
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
    mode = "idle"; hoveredHandle = null; canvas.style.cursor = "crosshair";
    // A click without a drag leaves a zero-size box; keep the previous one instead.
    if (prevMode === "drawing" && prevBox && (boxW < 1 || boxH < 1)) {
      boxL = prevBox.l; boxT = prevBox.t; boxW = prevBox.w; boxH = prevBox.h;
      render();
      return;
    }
    commit();
  });

  canvas.addEventListener("pointercancel", () => {
    if (mode === "drawing" && dragStart?.prev) {
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
    const newCanvasW = v.img_width  || 0;
    const newCanvasH = v.img_height || 0;

    const base = (u) => u ? u.split("?")[0] : "";
    const urlChanged = base(newUrl) !== base(imageUrl);
    const sizeChanged = newCanvasW !== canvasW || newCanvasH !== canvasH;

    if (urlChanged) {
      imageUrl = newUrl;
      imageLoaded = false;
      loadImage(imageUrl);
    }
    canvasW = newCanvasW;
    canvasH = newCanvasH;

    if (v.locked !== undefined) lockedParams = v.locked || [];

    if (mode === "idle") {
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
