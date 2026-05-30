// CropVideoEditor — interactive drag-handle crop widget for video nodes.
// Uses a <video> element as the background (frame-accurate scrubbing) and
// overlays a transparent <canvas> that draws the crop rectangle + handles.

import { HANDLE_R, WIDGET_VERSION, OVERLAY, CROP_BORDER, GUIDE,
         HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED } from './_styles.js';
import { createSidebar } from './_sidebar.js';
import { createFooter }  from './_footer.js';

export default function CropVideoEditor(container, props) {
  const { value, onChange, disabled } = props;

  // ── Re-use existing instance ───────────────────────────────────────────────
  if (container._instance?.wrapper?.isConnected) {
    container._instance.handleUpdate(props);
    return { cleanup: container._instance.cleanup, update: container._instance.handleUpdate };
  }

  // ── State ──────────────────────────────────────────────────────────────────
  let videoUrl     = value?.video_url   || "";
  let vidNatW      = value?.video_width  || 0;
  let vidNatH      = value?.video_height || 0;
  let totalFrames  = value?.total_frames || 0;
  let lockedParams = value?.locked       || [];
  let isDisabled   = !!disabled;

  // Crop rect in natural video pixels
  let ecL = value?.left   || 0;
  let ecT = value?.top    || 0;
  let ecW = value?.width  || vidNatW;
  let ecH = value?.height || vidNatH;

  let onChangeRef = onChange;
  let videoLoaded = false;
  let scale = 1;             // canvas-px / natural-px
  let mode = "idle";         // idle | moving | resizing | drawing
  let dragHandle = null;
  let dragStart = null;
  let rectAtDrag = null;
  let hoverHandle = null;
  let shiftDown = false;

  const hitR = HANDLE_R + 5;

  // ── DOM structure ──────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;gap:6px;";

  // Canvas row: video + canvas | sidebar
  const canvasRow = document.createElement("div");
  canvasRow.style.cssText = "display:flex;flex-direction:row;flex:1 1 0;min-height:180px;gap:6px;";

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "flex:1 1 0", "min-height:0", "position:relative",
    "background:#111", "border-radius:6px", "overflow:hidden",
  ].join(";");

  // Video element (background)
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "metadata";
  video.crossOrigin = "anonymous";
  video.style.cssText = [
    "position:absolute", "inset:0", "width:100%", "height:100%",
    "object-fit:contain", "pointer-events:none",
  ].join(";");

  // Canvas overlay (transparent bg — draws crop overlay + handles only)
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
  const ctx = canvas.getContext("2d");

  canvasWrap.appendChild(video);
  canvasWrap.appendChild(canvas);

  // Sidebar
  const sidebarInst = createSidebar({
    getImgSize: () => ({ imgNatW: vidNatW, imgNatH: vidNatH }),
    getCropRect: () => ({ l: ecL, t: ecT, w: ecW, h: ecH }),
    onApply(crop) {
      mode = "idle";
      ecL = crop.left; ecT = crop.top; ecW = crop.width; ecH = crop.height;
      render();
      emitAll();
    },
  });

  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);

  // Footer (frame scrubber + status bar)
  const footerInst = createFooter({
    isDisabled: () => isDisabled,
    onSeek(frame) { seekToFrame(frame); },
    version: WIDGET_VERSION,
  });

  wrapper.appendChild(canvasRow);
  wrapper.appendChild(footerInst.controls);
  wrapper.appendChild(footerInst.statusBar);
  container.appendChild(wrapper);

  // ── Video loading ──────────────────────────────────────────────────────────
  function loadVideo(url) {
    if (!url) return;
    if (video.src && stripQuery(video.src) === stripQuery(url)) return;
    videoLoaded = false;
    video.src = url;
    video.load();
  }

  function stripQuery(u) { return u ? u.split("?")[0] : ""; }

  video.addEventListener("loadedmetadata", () => {
    videoLoaded = true;
    syncCanvasSize();
    // Default crop to full frame if not yet set
    if (!ecW) ecW = vidNatW;
    if (!ecH) ecH = vidNatH;
    footerInst.sync(totalFrames, isDisabled);
    syncUI();
    render();
  });

  video.addEventListener("seeked", () => render());

  // ── Canvas sizing ──────────────────────────────────────────────────────────
  function syncCanvasSize() {
    const areaW = canvasWrap.clientWidth  || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const natW = vidNatW || video.videoWidth  || 1;
    const natH = vidNatH || video.videoHeight || 1;
    scale = Math.min(areaW / natW, areaH / natH);
    const cw = Math.round(natW * scale);
    const ch = Math.round(natH * scale);
    // Centre the canvas within the wrap (letterbox)
    canvas.width  = cw;
    canvas.height = ch;
    canvas.style.cssText = [
      "position:absolute",
      `left:${Math.round((areaW - cw) / 2)}px`,
      `top:${Math.round((areaH - ch) / 2)}px`,
      `width:${cw}px`, `height:${ch}px`,
    ].join(";");
    render();
  }

  const ro = new ResizeObserver(() => { if (videoLoaded) syncCanvasSize(); });
  ro.observe(canvasWrap);

  // ── Frame scrubbing ────────────────────────────────────────────────────────
  function seekToFrame(frame) {
    if (!video.duration || totalFrames <= 0) return;
    video.currentTime = (frame / totalFrames) * video.duration;
    footerInst.setFrame(frame, totalFrames);
  }

  // ── Handle geometry ────────────────────────────────────────────────────────
  function getHandles() {
    const l = ecL * scale, t = ecT * scale;
    const r = (ecL + ecW) * scale, b = (ecT + ecH) * scale;
    const mx = (l + r) / 2, my = (t + b) / 2;
    return [
      { id: "nw", cx: l,  cy: t,  affects: ["left","top","width","height"] },
      { id: "n",  cx: mx, cy: t,  affects: ["top","height"] },
      { id: "ne", cx: r,  cy: t,  affects: ["top","width","height"] },
      { id: "e",  cx: r,  cy: my, affects: ["width"] },
      { id: "se", cx: r,  cy: b,  affects: ["width","height"] },
      { id: "s",  cx: mx, cy: b,  affects: ["height"] },
      { id: "sw", cx: l,  cy: b,  affects: ["left","width","height"] },
      { id: "w",  cx: l,  cy: my, affects: ["left","width"] },
    ];
  }

  function isHandleLocked(h) {
    return h.affects.some(f => lockedParams.includes(f));
  }

  function hitHandle(px, py) {
    for (const h of getHandles()) {
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h;
    }
    return null;
  }

  function insideCrop(px, py) {
    return px >= ecL * scale && px <= (ecL + ecW) * scale
        && py >= ecT * scale && py <= (ecT + ecH) * scale;
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);
    if (!videoLoaded) return;

    const l = ecL * scale, t = ecT * scale;
    const w = ecW * scale, h = ecH * scale;

    // Darken outside crop rect
    ctx.fillStyle = OVERLAY;
    ctx.fillRect(0, 0, cw, ch);
    // Cut out crop area
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.fillRect(l, t, w, h);
    ctx.globalCompositeOperation = "source-over";

    // Rule-of-thirds guides
    ctx.strokeStyle = GUIDE;
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath(); ctx.moveTo(l + w * i / 3, t); ctx.lineTo(l + w * i / 3, t + h); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(l, t + h * i / 3); ctx.lineTo(l + w, t + h * i / 3); ctx.stroke();
    }

    // Crop border
    ctx.strokeStyle = CROP_BORDER;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(l, t, w, h);

    // Handles
    for (const hand of getHandles()) {
      const locked = isHandleLocked(hand);
      const hovered = !locked && hand.id === hoverHandle;
      ctx.beginPath();
      ctx.arc(hand.cx, hand.cy, HANDLE_R, 0, Math.PI * 2);
      ctx.fillStyle = HANDLE_FILL;
      ctx.fill();
      ctx.strokeStyle = locked ? HANDLE_LOCKED : (hovered ? HANDLE_STROKE_HOVER : HANDLE_STROKE);
      ctx.lineWidth = locked ? 1.5 : 2;
      ctx.stroke();
    }
  }

  // ── Drag interaction ───────────────────────────────────────────────────────
  function canvasXY(e) {
    const rect = canvas.getBoundingClientRect();
    return { x: e.clientX - rect.left, y: e.clientY - rect.top };
  }

  function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (isDisabled) return;
    const { x, y } = canvasXY(e);
    const hand = hitHandle(x, y);

    if (hand && !isHandleLocked(hand)) {
      mode = "resizing";
      dragHandle = hand;
      dragStart = { x, y };
      rectAtDrag = { l: ecL, t: ecT, w: ecW, h: ecH };
      canvas.setPointerCapture(e.pointerId);
      return;
    }

    const posLocked = lockedParams.includes("left") || lockedParams.includes("top");
    if (insideCrop(x, y) && !posLocked) {
      mode = "moving";
      dragStart = { x, y };
      rectAtDrag = { l: ecL, t: ecT, w: ecW, h: ecH };
      canvas.setPointerCapture(e.pointerId);
      return;
    }

    const sizeLocked = lockedParams.includes("width") || lockedParams.includes("height");
    if (!posLocked && !sizeLocked) {
      mode = "drawing";
      dragStart = { x: x / scale, y: y / scale };
      canvas.setPointerCapture(e.pointerId);
    }
  });

  canvas.addEventListener("pointermove", (e) => {
    e.stopPropagation();
    const { x, y } = canvasXY(e);

    if (mode === "idle") {
      const hand = hitHandle(x, y);
      const newHover = hand && !isHandleLocked(hand) ? hand.id : null;
      if (newHover !== hoverHandle) { hoverHandle = newHover; render(); }
      canvas.style.cursor = newHover ? "crosshair" : (insideCrop(x, y) ? "move" : "crosshair");
      return;
    }

    if (mode === "moving") {
      const dx = (x - dragStart.x) / scale, dy = (y - dragStart.y) / scale;
      ecL = clamp(Math.round(rectAtDrag.l + dx), 0, vidNatW - ecW);
      ecT = clamp(Math.round(rectAtDrag.t + dy), 0, vidNatH - ecH);
      render();
      return;
    }

    if (mode === "resizing") {
      const nx = x / scale, ny = y / scale;
      const { l, t, w, h } = rectAtDrag;
      let nl = l, nt = t, nw = w, nh = h;
      const id = dragHandle.id;

      if (id.includes("e")) nw = clamp(Math.round(nx - nl), 1, vidNatW - nl);
      if (id.includes("s")) nh = clamp(Math.round(ny - nt), 1, vidNatH - nt);
      if (id.includes("w")) { const r = nl + nw; nl = clamp(Math.round(nx), 0, r - 1); nw = r - nl; }
      if (id.includes("n")) { const b = nt + nh; nt = clamp(Math.round(ny), 0, b - 1); nh = b - nt; }

      if (shiftDown && (id === "se" || id === "nw" || id === "ne" || id === "sw")) {
        const asp = rectAtDrag.w / rectAtDrag.h;
        if (nw / nh > asp) nh = Math.round(nw / asp); else nw = Math.round(nh * asp);
      }

      ecL = nl; ecT = nt; ecW = nw; ecH = nh;
      render();
      return;
    }

    if (mode === "drawing") {
      const nx = clamp(x / scale, 0, vidNatW), ny = clamp(y / scale, 0, vidNatH);
      ecL = Math.round(Math.min(dragStart.x, nx));
      ecT = Math.round(Math.min(dragStart.y, ny));
      ecW = Math.max(1, Math.round(Math.abs(nx - dragStart.x)));
      ecH = shiftDown ? ecW : Math.max(1, Math.round(Math.abs(ny - dragStart.y)));
      render();
    }
  });

  canvas.addEventListener("pointerup", (e) => {
    e.stopPropagation();
    if (mode !== "idle") {
      mode = "idle";
      render();
      emitAll();
    }
  });

  canvas.addEventListener("keydown", (e) => { if (e.key === "Shift") shiftDown = true; });
  canvas.addEventListener("keyup",   (e) => { if (e.key === "Shift") shiftDown = false; });
  window.addEventListener("keydown", (e) => { if (e.key === "Shift") shiftDown = true; });
  window.addEventListener("keyup",   (e) => { if (e.key === "Shift") shiftDown = false; });

  // ── Emit ───────────────────────────────────────────────────────────────────
  function emitAll() {
    if (typeof onChangeRef !== "function") return;
    onChangeRef({
      ...(value || {}),
      left: ecL, top: ecT, width: ecW, height: ecH,
    });
  }

  // ── Sync UI state ──────────────────────────────────────────────────────────
  function syncUI() {
    sidebarInst.syncDisabled(lockedParams, isDisabled);
    footerInst.sync(totalFrames, isDisabled);
    footerInst.updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH });
  }

  // ── Update handler (called by framework on value changes) ──────────────────
  function handleUpdate(newProps) {
    const v = newProps.value || {};
    onChangeRef = newProps.onChange;
    isDisabled  = !!newProps.disabled;

    const newUrl = v.video_url || "";
    if (newUrl !== videoUrl) {
      videoUrl = newUrl;
      vidNatW  = v.video_width  || 0;
      vidNatH  = v.video_height || 0;
      totalFrames = v.total_frames || 0;
      videoLoaded = false;
      loadVideo(newUrl);
    } else {
      // Video unchanged — just update dimensions if they arrived
      if (v.video_width)  vidNatW = v.video_width;
      if (v.video_height) vidNatH = v.video_height;
      if (v.total_frames !== undefined) totalFrames = v.total_frames;
    }

    lockedParams = v.locked || [];
    ecL = v.left   ?? ecL;
    ecT = v.top    ?? ecT;
    ecW = v.width  ?? ecW;
    ecH = v.height ?? ecH;

    syncUI();
    render();
  }

  // ── Cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    ro.disconnect();
    video.pause();
    video.removeAttribute("src");
    video.load();
    wrapper.remove();
    delete container._instance;
  }

  // Initial load
  loadVideo(videoUrl);
  syncUI();

  container._instance = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
