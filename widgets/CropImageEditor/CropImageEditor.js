import {
  WIDGET_VERSION,
  HANDLE_R,
  OVERLAY, CROP_BORDER, GUIDE, TRANSFORM_PREVIEW,
  HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED,
} from './_styles.js';
import { createSidebar } from './_sidebar.js';
import { createFooter }  from './_footer.js';

export default function CropImageEditor(container, props) {
  if (container._cropEditor?.wrapper?.isConnected) {
    container._cropEditor.handleUpdate(props);
    return { cleanup: container._cropEditor.cleanup, update: container._cropEditor.handleUpdate };
  }
  if (container._cropEditor) delete container._cropEditor;

  // ── State ──────────────────────────────────────────────────────────────────
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled = props.disabled || false;

  let imageUrl     = latestValue.image_url || "";
  let imgNatW      = latestValue.img_width  || 0;
  let imgNatH      = latestValue.img_height || 0;
  let cropL        = latestValue.left   ?? 0;
  let cropT        = latestValue.top    ?? 0;
  let cropW        = latestValue.width  ?? 0;
  let cropH        = latestValue.height ?? 0;
  let cropZoom     = latestValue.zoom   ?? 100;
  let cropRotate   = latestValue.rotate ?? 0;
  let lockedParams = latestValue.locked || [];

  function ecL() { return Math.max(0, cropL); }
  function ecT() { return Math.max(0, cropT); }
  function ecW() { return cropW > 0 ? cropW : (imgNatW || 1); }
  function ecH() { return cropH > 0 ? cropH : (imgNatH || 1); }

  // ── Per-handle locking ─────────────────────────────────────────────────────
  function handleAffects(id) {
    const fields = [];
    if (id.includes("n")) { fields.push("top"); fields.push("height"); }
    if (id.includes("s")) { fields.push("height"); }
    if (id.includes("w")) { fields.push("left"); fields.push("width"); }
    if (id.includes("e")) { fields.push("width"); }
    return fields;
  }
  function isHandleLocked(id) { return handleAffects(id).some(f => lockedParams.includes(f)); }
  function isMoveLocked()  { return lockedParams.includes("left") || lockedParams.includes("top"); }
  function isDrawLocked()  { return ["left", "top", "width", "height"].some(f => lockedParams.includes(f)); }

  let scale = 1;
  let mode = "idle";
  let activeHandle = null;
  let dragStart = null;
  let hoveredHandle = null;

  // ── DOM ────────────────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "crop-image-editor nodrag nowheel";
  wrapper.style.cssText = [
    "display:flex", "flex-direction:column", "gap:6px",
    "height:100%",
    "user-select:none", "-webkit-user-select:none",
  ].join(";");

  // Canvas row: canvas on left, sidebar on right
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
  canvas.style.cssText = "display:block;cursor:crosshair;";
  const ctx = canvas.getContext("2d");

  // ── Sidebar ────────────────────────────────────────────────────────────────
  const sidebarInst = createSidebar({
    getImgSize:  () => ({ imgNatW, imgNatH }),
    getCropRect: () => ({ l: ecL(), t: ecT(), w: ecW(), h: ecH() }),
    onApply(crop) {
      if (isDisabled || !imageLoaded || !imgNatW || !imgNatH) return;
      mode = "idle"; activeHandle = null; dragStart = null; hoveredHandle = null;
      cropL = crop.left; cropT = crop.top; cropW = crop.width; cropH = crop.height;
      commit();
    },
  });

  // ── Footer (sliders + status) ──────────────────────────────────────────────
  const footerInst = createFooter({
    getZoom:    () => cropZoom,
    setZoom:    (v) => { cropZoom = v; },
    getRotate:  () => cropRotate,
    setRotate:  (v) => { cropRotate = v; },
    isLocked:   (key) => lockedParams.includes(key),
    isDisabled: () => isDisabled,
    onRender:   () => render(),
    onEmit:     () => emitAll(),
    version:    WIDGET_VERSION,
  });

  canvasWrap.appendChild(canvas);
  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);
  wrapper.appendChild(canvasRow);
  wrapper.appendChild(footerInst.controls);
  wrapper.appendChild(footerInst.statusBar);
  container.appendChild(wrapper);

  function syncUI() {
    footerInst.sync(lockedParams, isDisabled);
    sidebarInst.syncDisabled(lockedParams, isDisabled);
  }

  // ── Image ──────────────────────────────────────────────────────────────────
  const img = new Image();
  img.crossOrigin = "anonymous";
  let imageLoaded = false;

  function initCrop() {
    if (imgNatW > 0 && imgNatH > 0) {
      if (cropW === 0) cropW = imgNatW;
      if (cropH === 0) cropH = imgNatH;
    }
  }

  function resizeCanvas() {
    if (!imageLoaded || !imgNatW || !imgNatH) return;
    const areaW = canvasWrap.clientWidth  || 480;
    const areaH = canvasWrap.clientHeight || 360;
    scale = Math.min(areaW / imgNatW, areaH / imgNatH);
    canvas.width  = Math.round(imgNatW * scale);
    canvas.height = Math.round(imgNatH * scale);
    render();
  }

  img.onload = () => {
    imageLoaded = true;
    if (!imgNatW) imgNatW = img.naturalWidth;
    if (!imgNatH) imgNatH = img.naturalHeight;
    initCrop();
    resizeCanvas();
  };
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
        left: cropL, top: cropT, width: cropW, height: cropH,
        zoom: cropZoom, rotate: cropRotate,
      });
    }
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    const cw = canvas.width  || 400;
    const ch = canvas.height || 300;
    ctx.clearRect(0, 0, cw, ch);

    if (!imageLoaded) {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#555";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Connect an image to begin", cw / 2, ch / 2);
      footerInst.updateStatus({ imageLoaded, lockedParams, ecL: 0, ecT: 0, ecW: 0, ecH: 0 });
      return;
    }

    ctx.drawImage(img, 0, 0, cw, ch);

    const x  = ecL() * scale, y  = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const rw = x2 - x, rh = y2 - y;

    // Dark overlay outside crop (4 rects)
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

    // Zoom/rotate preview: dashed blue rect = actual pixel-capture area
    const hasTransform = (cropZoom !== 100 && cropZoom > 0) || cropRotate !== 0;
    if (hasTransform) {
      const zoomFactor = Math.max(0.01, cropZoom / 100);
      const cx_px = (ecL() + ecW() / 2) * scale;
      const cy_px = (ecT() + ecH() / 2) * scale;
      const pw = (ecW() / zoomFactor) * scale;
      const ph = (ecH() / zoomFactor) * scale;
      ctx.save();
      ctx.translate(cx_px, cy_px);
      ctx.rotate(cropRotate * Math.PI / 180);
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = TRANSFORM_PREVIEW;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-pw / 2, -ph / 2, pw, ph);
      ctx.setLineDash([]);
      ctx.fillStyle = TRANSFORM_PREVIEW;
      ctx.beginPath();
      ctx.moveTo(0, -ph / 2 - 8);
      ctx.lineTo(-5, -ph / 2);
      ctx.lineTo(5, -ph / 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    // Crop border
    ctx.strokeStyle = CROP_BORDER;
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

    footerInst.updateStatus({ imageLoaded, lockedParams, ecL: ecL(), ecT: ecT(), ecW: ecW(), ecH: ecH() });
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

  function hitCrop(px, py) {
    return px >= ecL() * scale && px <= (ecL() + ecW()) * scale
        && py >= ecT() * scale && py <= (ecT() + ecH()) * scale;
  }

  const CURSORS = { nw:"nw-resize", n:"n-resize", ne:"ne-resize", e:"e-resize", se:"se-resize", s:"s-resize", sw:"sw-resize", w:"w-resize" };

  function canvasPos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }

  function toImg(cx, cy) { return [cx / scale, cy / scale]; }

  function clampRect(l, t, w, h) {
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    l = Math.max(0, Math.min(Math.round(l), imgNatW - w));
    t = Math.max(0, Math.min(Math.round(t), imgNatH - h));
    return [l, t, Math.min(w, imgNatW - l), Math.min(h, imgNatH - t)];
  }

  function commit() {
    const [l, t, w, h] = clampRect(cropL, cropT, cropW || imgNatW, cropH || imgNatH);
    cropL = l; cropT = t; cropW = w; cropH = h;
    emitAll();
    render();
  }

  // ── Pointer events ─────────────────────────────────────────────────────────
  canvas.addEventListener("pointerdown", (e) => {
    if (isDisabled || !imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toImg(px, py);
    const handle = hitHandle(px, py);

    if (handle) {
      canvas.setPointerCapture(e.pointerId);
      mode = "resizing"; activeHandle = handle;
      dragStart = { ix, iy, l: cropL, t: cropT, w: ecW(), h: ecH() };
    } else if (hitCrop(px, py) && !isMoveLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "moving"; canvas.style.cursor = "grabbing";
      dragStart = { ix, iy, l: cropL, t: cropT, w: ecW(), h: ecH() };
    } else if (!hitCrop(px, py) && !isDrawLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "drawing";
      const six = Math.max(0, Math.min(ix, imgNatW));
      const siy = Math.max(0, Math.min(iy, imgNatH));
      dragStart = { ix: six, iy: siy };
      cropL = six; cropT = siy; cropW = 0; cropH = 0;
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    if (!imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toImg(px, py);

    if (mode === "idle") {
      const h = hitHandle(px, py);
      hoveredHandle = h;
      if (h) canvas.style.cursor = CURSORS[h];
      else if (hitCrop(px, py)) canvas.style.cursor = isMoveLocked() ? "not-allowed" : "move";
      else canvas.style.cursor = isDrawLocked() ? "not-allowed" : "crosshair";
      render();
      return;
    }

    if (mode === "moving") {
      const dx = ix - dragStart.ix, dy = iy - dragStart.iy;
      cropW = dragStart.w; cropH = dragStart.h;
      cropL = Math.max(0, Math.min(dragStart.l + dx, imgNatW - cropW));
      cropT = Math.max(0, Math.min(dragStart.t + dy, imgNatH - cropH));
    } else if (mode === "resizing") {
      const { l, t, w, h } = dragStart;
      let nl = l, nt = t, nr = l + w, nb = t + h;
      const cix = Math.max(0, Math.min(ix, imgNatW));
      const ciy = Math.max(0, Math.min(iy, imgNatH));
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
          else nb = Math.min(imgNatH, nt + ch);
        } else {
          const cw = Math.round(curH * ratio);
          if (id.includes("w")) nl = Math.max(0, nr - cw);
          else nr = Math.min(imgNatW, nl + cw);
        }
      }

      cropL = nl; cropT = nt; cropW = nr - nl; cropH = nb - nt;
    } else if (mode === "drawing") {
      const six = dragStart.ix, siy = dragStart.iy;
      const eix = Math.max(0, Math.min(ix, imgNatW));
      const eiy = Math.max(0, Math.min(iy, imgNatH));
      if (e.shiftKey) {
        const dx = eix - six, dy = eiy - siy;
        const side = Math.max(Math.abs(dx), Math.abs(dy));
        const maxW = dx < 0 ? six : imgNatW - six;
        const maxH = dy < 0 ? siy : imgNatH - siy;
        const s = Math.min(side, maxW, maxH);
        cropW = s; cropH = s;
        cropL = dx < 0 ? six - s : six;
        cropT = dy < 0 ? siy - s : siy;
      } else {
        cropL = Math.min(six, eix); cropT = Math.min(siy, eiy);
        cropW = Math.abs(eix - six); cropH = Math.abs(eiy - siy);
      }
    }
    render();
  });

  canvas.addEventListener("pointerup", (e) => {
    if (!imageLoaded || mode === "idle") return;
    e.stopPropagation();
    const prevMode = mode;
    mode = "idle"; hoveredHandle = null; canvas.style.cursor = "crosshair";
    if (prevMode !== "idle") commit();
  });

  canvas.addEventListener("pointercancel", () => {
    mode = "idle"; hoveredHandle = null; canvas.style.cursor = "crosshair"; render();
  });

  canvas.addEventListener("mousedown", (e) => e.stopPropagation());
  canvas.addEventListener("keydown",   (e) => e.stopPropagation());
  canvasWrap.addEventListener("pointerdown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("mousedown",   (e) => e.stopPropagation());

  // ── ResizeObserver ─────────────────────────────────────────────────────────
  const ro = new ResizeObserver(() => { if (imageLoaded) resizeCanvas(); });
  ro.observe(canvasWrap);

  // ── Initial load ───────────────────────────────────────────────────────────
  if (imageUrl) {
    loadImage(imageUrl);
  } else {
    canvas.width = 400; canvas.height = 300; render();
  }
  syncUI();

  // ── handleUpdate ───────────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    isDisabled  = newProps.disabled || false;
    latestValue = newProps.value || {};

    const v = latestValue;
    const newUrl  = v.image_url || "";
    const newNatW = v.img_width  || 0;
    const newNatH = v.img_height || 0;

    const base = (u) => u ? u.split("?")[0] : "";
    const urlChanged = base(newUrl) !== base(imageUrl);

    if (urlChanged) {
      imageUrl = newUrl; imgNatW = newNatW; imgNatH = newNatH;
      imageLoaded = false;
      loadImage(imageUrl);
    } else {
      if (newNatW) imgNatW = newNatW;
      if (newNatH) imgNatH = newNatH;
    }

    if (v.locked !== undefined) lockedParams = v.locked || [];

    if (mode === "idle") {
      if ("left"   in v) cropL = v.left;
      if ("top"    in v) cropT = v.top;
      if ("width"  in v) cropW = v.width;
      if ("height" in v) cropH = v.height;
      if ("zoom"   in v) cropZoom   = v.zoom   ?? cropZoom;
      if ("rotate" in v) cropRotate = v.rotate ?? cropRotate;
    }

    syncUI();
    if (imageLoaded && !urlChanged) render();
  }

  // ── cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    ro.disconnect();
    wrapper.remove();
    delete container._cropEditor;
  }

  container._cropEditor = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
