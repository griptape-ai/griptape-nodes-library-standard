// CropVideoEditor — interactive drag-handle crop widget for video nodes.
// Uses a <video> element as the background (frame-accurate scrubbing) and
// overlays a transparent <canvas> that draws the crop rectangle + handles.

import { HANDLE_R, WIDGET_VERSION, OVERLAY, CROP_BORDER, GUIDE,
         HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED } from './_styles';
import { createSidebar } from './_sidebar';
import { createFooter }  from './_footer';
import type { CropVideoValue, LiveCropVideoValue } from './_types';
import type { WidgetHandle, WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedCropVideoValue, CropRect, CropVideoValue, CustomCrop, LiveCropVideoValue } from './_types';

export type CropVideoEditorProps = WidgetProps<CropVideoValue, LiveCropVideoValue>;

type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type Mode = "idle" | "moving" | "resizing" | "drawing";

interface Handle { id: HandleId; cx: number; cy: number; affects: string[] }
interface Point { x: number; y: number }
interface Box { l: number; t: number; w: number; h: number }

// A mounted editor, kept on its container so a repeated mount updates it instead of building
// a second one.
interface CropVideoInstance {
  handleUpdate: (props: CropVideoEditorProps) => void;
  cleanup: () => void;
  wrapper: HTMLDivElement;
}
type CropVideoContainer = HTMLElement & { _instance?: CropVideoInstance };

// The node's RESOLUTION_PRESETS and crop maths (CropVideo._get_crop_dimensions and
// _calculate_crop_coordinates). The crop settings arrive as synced_parameters, so the widget works out
// the box itself.
const RESOLUTION_PRESETS: Record<string, [number, number]> = {
  "480p (854x480)": [854, 480],
  "720p (1280x720)": [1280, 720],
  "1080p (1920x1080)": [1920, 1080],
  "1440p (2560x1440)": [2560, 1440],
  "2160p (3840x2160)": [3840, 2160],
  "Square (1080x1080)": [1080, 1080],
  "Portrait (1080x1920)": [1080, 1920],
  "4:3 (1440x1080)": [1440, 1080],
};

function cropSize(v: CropVideoValue): [number, number] | null {
  const size = v.crop_size || "Custom";
  if (size === "Custom") {
    const w = v.custom_width || 0, h = v.custom_height || 0;
    return w > 0 && h > 0 ? [w, h] : null;
  }
  const preset = Object.entries(RESOLUTION_PRESETS).find(([name]) => name.toLowerCase() === size.trim().toLowerCase());
  if (preset) return preset[1];
  const parts = size.split("x");
  if (parts.length === 2) {
    const w = parseInt(parts[0] as string, 10), h = parseInt(parts[1] as string, 10);
    if (w > 0 && h > 0) return [w, h];
  }
  return null;
}

function cropFromSettings(v: CropVideoValue, vw: number, vh: number): Box {
  const dims = cropSize(v);
  const w = dims ? Math.min(dims[0], vw) : vw;
  const h = dims ? Math.min(dims[1], vh) : vh;
  const dx = vw - w, dy = vh - h;
  const position = v.crop_position || "center";
  const positions: Record<string, [number, number]> & { center: [number, number] } = {
    "center": [Math.floor(dx / 2), Math.floor(dy / 2)],
    "top-left": [0, 0],
    "top-right": [dx, 0],
    "bottom-left": [0, dy],
    "bottom-right": [dx, dy],
    "top-center": [Math.floor(dx / 2), 0],
    "bottom-center": [Math.floor(dx / 2), dy],
    "left-center": [0, Math.floor(dy / 2)],
    "right-center": [dx, Math.floor(dy / 2)],
  };
  let [l, t]: [number, number] = position === "Custom" ? [v.custom_left || 0, v.custom_top || 0] : (positions[position] || positions.center);
  l = Math.max(0, Math.min(l, dx));
  t = Math.max(0, Math.min(t, dy));
  return { l, t, w, h };
}

// The editor locks synced parameters with an incoming connection; the widget locks box edges.
function lockedEdges(locked: readonly string[] | undefined): string[] {
  const has = (name: string) => (locked || []).includes(name);
  const edges: string[] = [];
  if (has("crop_size") || has("custom_width")) edges.push("width");
  if (has("crop_size") || has("custom_height")) edges.push("height");
  if (has("crop_position") || has("custom_left")) edges.push("left");
  if (has("crop_position") || has("custom_top")) edges.push("top");
  return edges;
}

export default function CropVideoEditor(
  host: HTMLElement,
  props: CropVideoEditorProps,
): WidgetHandle<CropVideoEditorProps> {
  const container = host as CropVideoContainer;
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
  let lockedParams = lockedEdges(props.locked);
  let latestValue: CropVideoValue = value || {};
  let isDisabled   = !!disabled;
  let currentFrame = 0;

  // Crop rect in natural video pixels
  const initialCrop = cropFromSettings(latestValue, vidNatW, vidNatH);
  let ecL = initialCrop.l;
  let ecT = initialCrop.t;
  let ecW = initialCrop.w;
  let ecH = initialCrop.h;

  let onChangeRef = onChange;
  let onLiveChangeRef = props.onLiveChange;
  let videoLoaded = false;
  let scale = 1;             // canvas-px / natural-px
  let mode: Mode = "idle";   // idle | moving | resizing | drawing
  let dragHandle: Handle | null = null;
  let dragStart: Point | null = null;
  let rectAtDrag: Box | null = null;
  let hoverHandle: HandleId | null = null;

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
  // Every browser the editor supports provides a 2D context for a fresh canvas.
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

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
  // Compare by pathname only so token refreshes in signed URLs don't cause reloads.
  function urlPathname(u: string): string {
    if (!u) return "";
    try { return new URL(u, location.href).pathname; }
    catch { return u.split("?")[0] as string; }
  }

  function loadVideo(url: string) {
    if (!url) return;
    if (video.src && urlPathname(video.src) === urlPathname(url)) return;
    videoLoaded = false;
    video.src = url;
    video.load();
  }

  video.addEventListener("loadedmetadata", () => {
    videoLoaded = true;
    // Fill in dims from the video element if Python couldn't probe them
    if (!vidNatW) vidNatW = video.videoWidth;
    if (!vidNatH) vidNatH = video.videoHeight;
    if (mode === "idle") ({ l: ecL, t: ecT, w: ecW, h: ecH } = cropFromSettings(latestValue, vidNatW, vidNatH));
    syncCanvasSize();
    // Default crop to full frame if not yet set
    if (!ecW) ecW = vidNatW;
    if (!ecH) ecH = vidNatH;
    footerInst.sync(totalFrames, isDisabled);
    // Restore the previously viewed frame (no-op for first load where currentFrame=0)
    if (currentFrame > 0 && video.duration) {
      video.currentTime = (currentFrame / totalFrames) * video.duration;
    }
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
  function seekToFrame(frame: number) {
    if (!video.duration || totalFrames <= 0) return;
    currentFrame = frame;
    video.currentTime = (frame / totalFrames) * video.duration;
    footerInst.setFrame(frame, totalFrames);
  }

  // ── Handle geometry ────────────────────────────────────────────────────────
  function getHandles(): Handle[] {
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

  function isHandleLocked(h: Handle) {
    return h.affects.some(f => lockedParams.includes(f));
  }

  function hitHandle(px: number, py: number): Handle | null {
    for (const h of getHandles()) {
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h;
    }
    return null;
  }

  function insideCrop(px: number, py: number) {
    return px >= ecL * scale && px <= (ecL + ecW) * scale
        && py >= ecT * scale && py <= (ecT + ecH) * scale;
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function render() {
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);
    if (!videoLoaded) {
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.font = `${Math.max(11, Math.round(cw / 22))}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Connect a video to begin", cw / 2, ch / 2);
      return;
    }

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
  function canvasXY(e: MouseEvent): Point {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width  / (rect.width  || canvas.width);
    const sy = canvas.height / (rect.height || canvas.height);
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }

  function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

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

    const leftLocked = lockedParams.includes("left");
    const topLocked  = lockedParams.includes("top");
    const posLocked  = leftLocked && topLocked;   // both axes must be locked to block move
    if (insideCrop(x, y) && !(leftLocked && topLocked)) {
      mode = "moving";
      dragStart = { x, y };
      rectAtDrag = { l: ecL, t: ecT, w: ecW, h: ecH };
      canvas.setPointerCapture(e.pointerId);
      return;
    }

    const sizeLocked = lockedParams.includes("width") || lockedParams.includes("height");
    if (!(leftLocked || topLocked) && !sizeLocked) {
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
      const start = dragStart as Point, rect = rectAtDrag as Box;
      const dx = (x - start.x) / scale, dy = (y - start.y) / scale;
      if (!lockedParams.includes("left")) ecL = clamp(Math.round(rect.l + dx), 0, vidNatW - ecW);
      if (!lockedParams.includes("top"))  ecT = clamp(Math.round(rect.t + dy), 0, vidNatH - ecH);
      render();
      emitLive();
      return;
    }

    if (mode === "resizing") {
      const nx = x / scale, ny = y / scale;
      const rect = rectAtDrag as Box;
      const { l, t, w, h } = rect;
      let nl = l, nt = t, nw = w, nh = h;
      const id = (dragHandle as Handle).id;

      if (id.includes("e")) nw = clamp(Math.round(nx - nl), 1, vidNatW - nl);
      if (id.includes("s")) nh = clamp(Math.round(ny - nt), 1, vidNatH - nt);
      if (id.includes("w")) { const r = nl + nw; nl = clamp(Math.round(nx), 0, r - 1); nw = r - nl; }
      if (id.includes("n")) { const b = nt + nh; nt = clamp(Math.round(ny), 0, b - 1); nh = b - nt; }

      if (e.shiftKey && (id === "se" || id === "nw" || id === "ne" || id === "sw")) {
        const asp = rect.w / rect.h;
        if (nw / nh > asp) nh = Math.round(nw / asp); else nw = Math.round(nh * asp);
      }

      ecL = nl; ecT = nt; ecW = nw; ecH = nh;
      render();
      emitLive();
      return;
    }

    if (mode === "drawing") {
      const start = dragStart as Point;
      const nx = clamp(x / scale, 0, vidNatW), ny = clamp(y / scale, 0, vidNatH);
      ecL = Math.round(Math.min(start.x, nx));
      ecT = Math.round(Math.min(start.y, ny));
      ecW = Math.max(1, Math.round(Math.abs(nx - start.x)));
      ecH = e.shiftKey ? ecW : Math.max(1, Math.round(Math.abs(ny - start.y)));
      render();
      emitLive();
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


  // ── Emit ───────────────────────────────────────────────────────────────────
  // A drag sets the box freely, so both presets switch to Custom. The editor commits the crop settings to
  // their parameters and keeps the rest (the video's details) in the widget's own value.
  function emitAll() {
    if (typeof onChangeRef !== "function") return;
    onChangeRef({
      ...latestValue,
      custom_left: ecL, custom_top: ecT, custom_width: ecW, custom_height: ecH,
      crop_size: "Custom", crop_position: "Custom",
    });
  }

  // Moves the node's crop fields along with a drag without committing them.
  function emitLive() {
    if (typeof onLiveChangeRef !== "function") return;
    onLiveChangeRef({ custom_left: ecL, custom_top: ecT, custom_width: ecW, custom_height: ecH });
  }

  // ── Sync UI state ──────────────────────────────────────────────────────────
  function syncUI() {
    sidebarInst.syncDisabled(lockedParams, isDisabled);
    footerInst.sync(totalFrames, isDisabled);
    footerInst.updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH });
  }

  // ── Update handler (called by framework on value changes) ──────────────────
  function handleUpdate(newProps: CropVideoEditorProps) {
    const v: CropVideoValue = newProps.value || {};
    latestValue = v;
    onChangeRef = newProps.onChange;
    onLiveChangeRef = newProps.onLiveChange;
    isDisabled  = !!newProps.disabled;

    const newUrl = v.video_url || "";
    const videoChanged = urlPathname(newUrl) !== urlPathname(videoUrl);
    if (videoChanged) {
      videoUrl = newUrl;
      vidNatW  = v.video_width  || 0;
      vidNatH  = v.video_height || 0;
      totalFrames = v.total_frames || 0;
      currentFrame = 0;
      videoLoaded = false;
      loadVideo(newUrl);
    } else {
      // Same video — update token + dims without reloading or resetting the frame
      if (newUrl) videoUrl = newUrl;
      if (v.video_width)  vidNatW = v.video_width;
      if (v.video_height) vidNatH = v.video_height;
      if (v.total_frames !== undefined) totalFrames = v.total_frames;
    }

    lockedParams = lockedEdges(newProps.locked);

    // Only follow the settings when idle — don't clobber an in-progress drag.
    if (mode === "idle" && vidNatW && vidNatH) {
      ({ l: ecL, t: ecT, w: ecW, h: ecH } = cropFromSettings(v, vidNatW, vidNatH));
    }

    syncUI();
    if (videoLoaded) render();
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
