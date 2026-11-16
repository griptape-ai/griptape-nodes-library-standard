import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, SyntheticEvent } from 'react';
import { reactWidget } from '@griptape-ai/nodes-widget-api/react';

import {
  WIDGET_VERSION,
  HANDLE_R,
  OVERLAY, CROP_BORDER, GUIDE, TRANSFORM_PREVIEW, TRANSFORM_OVERLAY,
  HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED,
} from './_styles';
import { type AppliedCrop, CropSidebar } from './_sidebar';
import { type SliderParam, StatusBar, TransformControls } from './_footer';
import type { CropValue, LiveCropValue } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedCropValue, CropRect, CropValue, LiveCropValue } from './_types';

export type CropImageEditorProps = WidgetProps<CropValue, LiveCropValue>;

type CropParam = "left" | "top" | "width" | "height" | "zoom" | "rotate";
type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type Mode = "idle" | "resizing" | "moving" | "drawing";

interface Handle { id: HandleId; cx: number; cy: number }
interface DragStart { ix: number; iy: number; l: number; t: number; w: number; h: number }
interface DrawStart { ix: number; iy: number }

/** The crop as the widget edits it. A width or height of 0 means the full image. */
interface Crop { l: number; t: number; w: number; h: number; zoom: number; rotate: number }
/** The crop rectangle the widget draws and hit-tests, with the full-image fallbacks applied. */
interface Rect { l: number; t: number; w: number; h: number }

// A move or resize records the crop it started from; a draw records only its anchor point.
interface Drag { mode: Mode; handle: HandleId | null; start: DragStart | DrawStart | null }

/** The image the widget last loaded or failed to load, keyed by its URL without the query. */
interface ImageState { key: string; loaded: boolean; width: number; height: number }

const IDLE: Drag = { mode: "idle", handle: null, start: null };
const COORD_PARAMS: readonly CropParam[] = ["left", "top", "width", "height"];
const CURSORS: Record<HandleId, string> = { nw:"nw-resize", n:"n-resize", ne:"ne-resize", e:"e-resize", se:"se-resize", s:"s-resize", sw:"sw-resize", w:"w-resize" };

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();
const baseUrl = (u: string) => u ? u.split("?")[0] ?? "" : "";

// ── Geometry ───────────────────────────────────────────────────────────────────
function handleAffects(id: HandleId): CropParam[] {
  const fields: CropParam[] = [];
  if (id.includes("n")) { fields.push("top"); fields.push("height"); }
  if (id.includes("s")) { fields.push("height"); }
  if (id.includes("w")) { fields.push("left"); fields.push("width"); }
  if (id.includes("e")) { fields.push("width"); }
  return fields;
}

// Applies the crop fields the value carries over the current crop.
function cropFromValue(v: CropValue, prev: Crop): Crop {
  return {
    l: v.left ?? prev.l, t: v.top ?? prev.t, w: v.width ?? prev.w, h: v.height ?? prev.h,
    zoom: v.zoom ?? prev.zoom, rotate: v.rotate ?? prev.rotate,
  };
}

function sameCrop(a: Crop, b: Crop) {
  return a.l === b.l && a.t === b.t && a.w === b.w && a.h === b.h && a.zoom === b.zoom && a.rotate === b.rotate;
}

function effectiveRect(c: Crop, imgW: number, imgH: number): Rect {
  return {
    l: Math.max(0, c.l), t: Math.max(0, c.t),
    w: c.w > 0 ? c.w : (imgW || 1), h: c.h > 0 ? c.h : (imgH || 1),
  };
}

function clampRect(l: number, t: number, w: number, h: number, imgW: number, imgH: number): [number, number, number, number] {
  w = Math.max(1, Math.round(w));
  h = Math.max(1, Math.round(h));
  l = Math.max(0, Math.min(Math.round(l), imgW - w));
  t = Math.max(0, Math.min(Math.round(t), imgH - h));
  return [l, t, Math.min(w, imgW - l), Math.min(h, imgH - t)];
}

function getHandles(r: Rect, scale: number): Handle[] {
  const x  = r.l * scale, y  = r.t * scale;
  const x2 = (r.l + r.w) * scale, y2 = (r.t + r.h) * scale;
  const mx = (x + x2) / 2, my = (y + y2) / 2;
  return [
    { id: "nw", cx: x,  cy: y  }, { id: "n",  cx: mx, cy: y  },
    { id: "ne", cx: x2, cy: y  }, { id: "e",  cx: x2, cy: my },
    { id: "se", cx: x2, cy: y2 }, { id: "s",  cx: mx, cy: y2 },
    { id: "sw", cx: x,  cy: y2 }, { id: "w",  cx: x,  cy: my },
  ];
}

function hitHandle(r: Rect, scale: number, px: number, py: number, isLocked: (id: HandleId) => boolean): HandleId | null {
  const hitR = HANDLE_R + 5;
  for (const h of getHandles(r, scale)) {
    if (isLocked(h.id)) continue;
    if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h.id;
  }
  return null;
}

function hitCrop(r: Rect, scale: number, px: number, py: number) {
  return px >= r.l * scale && px <= (r.l + r.w) * scale
      && py >= r.t * scale && py <= (r.t + r.h) * scale;
}

// ── Drawing ────────────────────────────────────────────────────────────────────
interface Scene {
  img: HTMLImageElement | null;
  loaded: boolean;
  rect: Rect;
  scale: number;
  zoom: number;
  rotate: number;
  showHandles: boolean;
  isHandleLocked: (id: HandleId) => boolean;
  hovered: HandleId | null;
}

function drawScene(ctx: CanvasRenderingContext2D, cw: number, ch: number, s: Scene) {
  ctx.clearRect(0, 0, cw, ch);

  if (!s.loaded || !s.img) {
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, cw, ch);
    ctx.fillStyle = "#555";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("Connect an image to begin", cw / 2, ch / 2);
    return;
  }

  ctx.drawImage(s.img, 0, 0, cw, ch);

  const { rect: r, scale } = s;
  const x  = r.l * scale, y  = r.t * scale;
  const x2 = (r.l + r.w) * scale, y2 = (r.t + r.h) * scale;
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

  // Zoom/rotate preview: dashed rect = actual pixel-capture area
  const hasTransform = (s.zoom !== 100 && s.zoom > 0) || s.rotate !== 0;
  if (hasTransform) {
    const zoomFactor = Math.max(0.01, s.zoom / 100);
    const cx_px = (r.l + r.w / 2) * scale;
    const cy_px = (r.t + r.h / 2) * scale;
    const pw = (r.w / zoomFactor) * scale;
    const ph = (r.h / zoomFactor) * scale;

    // Darken everything outside the capture zone using even-odd hole punch
    ctx.save();
    ctx.fillStyle = TRANSFORM_OVERLAY;
    ctx.beginPath();
    ctx.rect(0, 0, cw, ch);           // outer: full canvas
    ctx.translate(cx_px, cy_px);
    ctx.rotate(s.rotate * Math.PI / 180);
    ctx.rect(-pw / 2, -ph / 2, pw, ph); // inner hole: the capture rect
    ctx.restore();                     // restore transform; path coords are already baked
    ctx.fill("evenodd");

    // Dashed preview border + up-arrow indicator
    ctx.save();
    ctx.translate(cx_px, cy_px);
    ctx.rotate(s.rotate * Math.PI / 180);
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
  if (s.showHandles) {
    for (const h of getHandles(r, scale)) {
      const locked = s.isHandleLocked(h.id);
      ctx.beginPath();
      ctx.arc(h.cx, h.cy, HANDLE_R, 0, Math.PI * 2);
      ctx.fillStyle = locked ? HANDLE_LOCKED : HANDLE_FILL;
      ctx.fill();
      ctx.strokeStyle = locked ? "rgba(100,100,100,0.6)" : (s.hovered === h.id ? HANDLE_STROKE_HOVER : HANDLE_STROKE);
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }
}

// ── Component ──────────────────────────────────────────────────────────────────
function CropImageEditor({ value, onChange, onLiveChange, disabled = false, locked }: CropImageEditorProps) {
  const v: CropValue = value || {};
  // Crop parameters with an incoming connection, worked out by the editor from synced_parameters
  const lockedParams: readonly string[] = locked || [];
  const isLocked = (p: string) => lockedParams.includes(p);
  const isHandleLocked = (id: HandleId) => handleAffects(id).some(isLocked);
  const moveLocked = isLocked("left") || isLocked("top");
  const coordLocked = COORD_PARAMS.filter(isLocked);

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const dragRef = useRef<Drag>(IDLE);

  // ── Image ──────────────────────────────────────────────────────────────────
  const imageUrl = v.image_url || "";
  const imageKey = baseUrl(imageUrl);
  const [image, setImage] = useState<ImageState | null>(null);
  const current = image && imageKey && image.key === imageKey ? image : null;
  const imageLoaded = !!current?.loaded;
  const imgNatW = v.img_width  || current?.width  || 0;
  const imgNatH = v.img_height || current?.height || 0;

  // Keyed on the URL without its query string, so a cache-busting query change keeps the image.
  useEffect(() => {
    if (!imageUrl) return;
    if (!imgRef.current) {
      imgRef.current = new Image();
      imgRef.current.crossOrigin = "anonymous";
    }
    const img = imgRef.current;
    img.onload  = () => setImage({ key: imageKey, loaded: true, width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => setImage({ key: imageKey, loaded: false, width: 0, height: 0 });
    img.src = imageUrl;
    return () => { img.onload = null; img.onerror = null; };
  }, [imageKey]);

  // ── Crop ───────────────────────────────────────────────────────────────────
  // The crop lives in state for rendering and in a ref for pointer handlers that run before the
  // previous change has rendered.
  const [crop, setCropState] = useState<Crop>(() => cropFromValue(v, { l: 0, t: 0, w: 0, h: 0, zoom: 100, rotate: 0 }));
  const cropRef = useRef(crop);
  function setCrop(next: Crop) {
    cropRef.current = next;
    setCropState(next);
  }

  // A value from the engine replaces the crop, except mid-drag, where the drag wins.
  useLayoutEffect(() => {
    if (dragRef.current.mode !== "idle") return;
    const next = cropFromValue(value || {}, cropRef.current);
    if (!sameCrop(next, cropRef.current)) setCrop(next);
  }, [value]);

  // Once the image is loaded, a crop with no size covers the whole image.
  const fits = imageLoaded && imgNatW > 0 && imgNatH > 0;
  useLayoutEffect(() => {
    if (!fits) return;
    const c = cropRef.current;
    if (c.w === 0 || c.h === 0) setCrop({ ...c, w: c.w || imgNatW, h: c.h || imgNatH });
  }, [fits, imgNatW, imgNatH]);

  // ── Layout ─────────────────────────────────────────────────────────────────
  // The canvas area's size, measured on every resize; a new object each time so the canvas redraws.
  const [area, setArea] = useState({ width: 0, height: 0 });
  useEffect(() => {
    const wrap = canvasWrapRef.current;
    if (!wrap) return;
    const ro = new ResizeObserver(() => setArea({ width: wrap.clientWidth, height: wrap.clientHeight }));
    ro.observe(wrap);
    return () => ro.disconnect();
  }, []);

  const scale = fits ? Math.min((area.width || 480) / imgNatW, (area.height || 360) / imgNatH) : 1;
  const canvasW = fits ? Math.round(imgNatW * scale) : 400;
  const canvasH = fits ? Math.round(imgNatH * scale) : 300;
  const rect = effectiveRect(crop, imgNatW, imgNatH);

  const [hovered, setHovered] = useState<HandleId | null>(null);
  const [cursor, setCursor] = useState("crosshair");

  // Redraws after every render: any prop or state change can alter the picture.
  useLayoutEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Every browser the editor supports provides a 2D context for a fresh canvas.
    ctxRef.current ??= canvas.getContext("2d") as CanvasRenderingContext2D;
    drawScene(ctxRef.current, canvas.width, canvas.height, {
      img: imgRef.current, loaded: imageLoaded, rect, scale,
      zoom: crop.zoom, rotate: crop.rotate,
      showHandles: !disabled, isHandleLocked, hovered,
    });
  });

  // ── Emit ───────────────────────────────────────────────────────────────────
  function emitAll(c: Crop) {
    if (typeof onChange === "function") {
      onChange({
        ...v,
        left: c.l, top: c.t, width: c.w, height: c.h,
        zoom: c.zoom, rotate: c.rotate,
      });
    }
  }

  // Moves the node's crop parameters along with a drag. The editor shows these values without
  // sending them to the engine; emitAll commits on release.
  function emitLive(c: Crop) {
    if (typeof onLiveChange !== "function") return;
    const [l, t, w, h] = clampRect(c.l, c.t, c.w, c.h, imgNatW, imgNatH);
    onLiveChange({ left: l, top: t, width: w, height: h, zoom: c.zoom, rotate: c.rotate });
  }

  function commit(c: Crop) {
    const [l, t, w, h] = clampRect(c.l, c.t, c.w || imgNatW, c.h || imgNatH, imgNatW, imgNatH);
    const next = { ...c, l, t, w, h };
    setCrop(next);
    emitAll(next);
  }

  function applyPreset(applied: AppliedCrop) {
    if (disabled || !imageLoaded || !imgNatW || !imgNatH) return;
    dragRef.current = IDLE;
    setHovered(null);
    commit({
      ...cropRef.current,
      l: applied.left, t: applied.top, w: applied.width, h: applied.height,
      ...("zoom" in applied ? { zoom: applied.zoom, rotate: applied.rotate } : {}),
    });
  }

  function previewTransform(param: SliderParam, val: number) {
    const next = { ...cropRef.current, [param]: val };
    setCrop(next);
    emitLive(next);
  }

  // ── Pointer events ─────────────────────────────────────────────────────────
  function canvasPos(e: ReactPointerEvent<HTMLCanvasElement>): [number, number] {
    const canvas = e.currentTarget;
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }

  function onCanvasPointerDown(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (disabled || !imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const ix = px / scale, iy = py / scale;
    const c = cropRef.current;
    const r = effectiveRect(c, imgNatW, imgNatH);
    const handle = hitHandle(r, scale, px, py, isHandleLocked);
    const inCrop = hitCrop(r, scale, px, py);

    if (handle) {
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = { mode: "resizing", handle, start: { ix, iy, l: c.l, t: c.t, w: r.w, h: r.h } };
    } else if (inCrop && !moveLocked) {
      e.currentTarget.setPointerCapture(e.pointerId);
      dragRef.current = { mode: "moving", handle: null, start: { ix, iy, l: c.l, t: c.t, w: r.w, h: r.h } };
      setCursor("grabbing");
    } else if (!inCrop && coordLocked.length === 0) {
      e.currentTarget.setPointerCapture(e.pointerId);
      const six = Math.max(0, Math.min(ix, imgNatW));
      const siy = Math.max(0, Math.min(iy, imgNatH));
      dragRef.current = { mode: "drawing", handle: null, start: { ix: six, iy: siy } };
      setCrop({ ...c, l: six, t: siy, w: 0, h: 0 });
    }
  }

  function onCanvasPointerMove(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (!imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const ix = px / scale, iy = py / scale;
    const { mode, handle, start } = dragRef.current;
    const c = cropRef.current;

    if (mode === "idle") {
      const r = effectiveRect(c, imgNatW, imgNatH);
      const h = hitHandle(r, scale, px, py, isHandleLocked);
      setHovered(h);
      if (h) setCursor(CURSORS[h]);
      else if (hitCrop(r, scale, px, py)) setCursor(moveLocked ? "not-allowed" : "move");
      else setCursor(coordLocked.length > 0 ? "not-allowed" : "crosshair");
      return;
    }

    let next = c;
    if (mode === "moving") {
      const s = start as DragStart;
      const dx = ix - s.ix, dy = iy - s.iy;
      next = {
        ...c, w: s.w, h: s.h,
        l: Math.max(0, Math.min(s.l + dx, imgNatW - s.w)),
        t: Math.max(0, Math.min(s.t + dy, imgNatH - s.h)),
      };
    } else if (mode === "resizing") {
      const s = start as DragStart;
      const { l, t, w, h } = s;
      let nl = l, nt = t, nr = l + w, nb = t + h;
      const cix = Math.max(0, Math.min(ix, imgNatW));
      const ciy = Math.max(0, Math.min(iy, imgNatH));
      const id = handle as HandleId;
      if (id.includes("n")) nt = Math.min(ciy, nb - 1);
      if (id.includes("s")) nb = Math.max(ciy, nt + 1);
      if (id.includes("w")) nl = Math.min(cix, nr - 1);
      if (id.includes("e")) nr = Math.max(cix, nl + 1);

      if (e.shiftKey && id.length === 2) {
        const ratio = s.w / s.h;
        const curW = nr - nl, curH = nb - nt;
        if (Math.abs(curW - s.w) >= Math.abs(curH - s.h) * ratio) {
          const ch = Math.round(curW / ratio);
          if (id.includes("n")) nt = Math.max(0, nb - ch);
          else nb = Math.min(imgNatH, nt + ch);
        } else {
          const cw = Math.round(curH * ratio);
          if (id.includes("w")) nl = Math.max(0, nr - cw);
          else nr = Math.min(imgNatW, nl + cw);
        }
      }

      next = { ...c, l: nl, t: nt, w: nr - nl, h: nb - nt };
    } else if (mode === "drawing") {
      const { ix: six, iy: siy } = start as DrawStart;
      const eix = Math.max(0, Math.min(ix, imgNatW));
      const eiy = Math.max(0, Math.min(iy, imgNatH));
      if (e.shiftKey) {
        const dx = eix - six, dy = eiy - siy;
        const side = Math.max(Math.abs(dx), Math.abs(dy));
        const maxW = dx < 0 ? six : imgNatW - six;
        const maxH = dy < 0 ? siy : imgNatH - siy;
        const sq = Math.min(side, maxW, maxH);
        next = { ...c, w: sq, h: sq, l: dx < 0 ? six - sq : six, t: dy < 0 ? siy - sq : siy };
      } else {
        next = {
          ...c,
          l: Math.min(six, eix), t: Math.min(siy, eiy),
          w: Math.abs(eix - six), h: Math.abs(eiy - siy),
        };
      }
    }
    setCrop(next);
    emitLive(next);
  }

  function onCanvasPointerUp(e: ReactPointerEvent<HTMLCanvasElement>) {
    if (!imageLoaded || dragRef.current.mode === "idle") return;
    e.stopPropagation();
    dragRef.current = IDLE;
    setHovered(null);
    setCursor("crosshair");
    commit(cropRef.current);
  }

  function onCanvasPointerCancel() {
    dragRef.current = IDLE;
    setHovered(null);
    setCursor("crosshair");
  }

  // ── Status ─────────────────────────────────────────────────────────────────
  let statusText = "No image connected";
  let statusColor = "";
  if (imageLoaded && coordLocked.length > 0) {
    statusColor = "#f59e0b";
    statusText = "\u{1F512} Connected: " + coordLocked.join(", ");
  } else if (imageLoaded) {
    statusText = `x: ${Math.round(rect.l)}  y: ${Math.round(rect.t)}  →  ${Math.round(rect.w)} × ${Math.round(rect.h)} px`;
  }

  return (
    <div
      className="crop-image-editor nodrag nowheel"
      style={{ display: "flex", flexDirection: "column", gap: "6px", height: "100%", userSelect: "none", WebkitUserSelect: "none" }}
    >
      {/* Canvas row: canvas on left, sidebar on right */}
      <div style={{ display: "flex", flexDirection: "row", gap: "6px", flex: "1 1 0", minHeight: "180px" }}>
        <div
          ref={canvasWrapRef}
          style={{
            position: "relative", background: "#111", borderRadius: "6px",
            overflow: "hidden", flex: "1 1 0", minHeight: "0",
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
          onPointerDown={stopPropagation}
          onMouseDown={stopPropagation}
        >
          <canvas
            ref={canvasRef}
            width={canvasW}
            height={canvasH}
            style={{ display: "block", cursor }}
            onPointerDown={onCanvasPointerDown}
            onPointerMove={onCanvasPointerMove}
            onPointerUp={onCanvasPointerUp}
            onPointerCancel={onCanvasPointerCancel}
            onMouseDown={stopPropagation}
            onKeyDown={stopPropagation}
          />
        </div>
        <CropSidebar
          imgNatW={imgNatW} imgNatH={imgNatH}
          cropRect={rect}
          off={coordLocked.length > 0 || disabled}
          onApply={applyPreset}
        />
      </div>
      <TransformControls
        zoom={crop.zoom} rotate={crop.rotate}
        isOff={(param) => isLocked(param) || disabled}
        onPreview={previewTransform}
        onCommit={() => emitAll(cropRef.current)}
      />
      <StatusBar text={statusText} color={statusColor} version={WIDGET_VERSION} />
    </div>
  );
}

export default reactWidget(CropImageEditor);
