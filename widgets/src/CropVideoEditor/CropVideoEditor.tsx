// CropVideoEditor — interactive drag-handle crop widget for video nodes.
// Uses a <video> element as the background (frame-accurate scrubbing) and
// overlays a transparent <canvas> that draws the crop rectangle + handles.

import { HANDLE_R, WIDGET_VERSION, OVERLAY, CROP_BORDER, GUIDE,
         HANDLE_FILL, HANDLE_STROKE, HANDLE_STROKE_HOVER, HANDLE_LOCKED } from './_styles';
import { type PointerEvent as ReactPointerEvent, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { reactWidget } from '@griptape-ai/nodes-widget-api/react';
import { Sidebar } from './_sidebar';
import { FrameScrubber, StatusBar } from './_footer';
import type { CropRect, CropVideoValue, LiveCropVideoValue } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedCropVideoValue, CropRect, CropVideoValue, CustomCrop, LiveCropVideoValue } from './_types';

export type CropVideoEditorProps = WidgetProps<CropVideoValue, LiveCropVideoValue>;

type HandleId = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";
type DragMode = "moving" | "resizing" | "drawing";

interface Handle { id: HandleId; cx: number; cy: number; affects: string[] }
interface Point { x: number; y: number }
interface Box { l: number; t: number; w: number; h: number }

// The canvas wrap's size and the video size the canvas is fitted to.
interface Layout { w: number; h: number; natW: number; natH: number }

// The video the editor shows: its URL, its size in natural pixels, and the frame scrubbed to.
interface Media {
  url: string;
  natW: number;
  natH: number;
  totalFrames: number;
  frame: number;
  loaded: boolean;
}

// Pointer-drag scratch: what the drag started from and the box it has reached.
interface Drag {
  mode: DragMode;
  handle: Handle | null;
  start: Point;
  rect: Box;
  box: Box;
}

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

// Compare by pathname only so token refreshes in signed URLs don't cause reloads.
function urlPathname(u: string): string {
  if (!u) return "";
  try { return new URL(u, location.href).pathname; }
  catch { return u.split("?")[0] as string; }
}

function mediaFromValue(v: CropVideoValue): Media {
  return {
    url: v.video_url || "", natW: v.video_width || 0, natH: v.video_height || 0,
    totalFrames: v.total_frames || 0, frame: 0, loaded: false,
  };
}

// A new video path starts over; the same video takes the refreshed URL and any dims the value carries,
// keeping the frame and the loaded metadata.
function followValue(m: Media, v: CropVideoValue): Media {
  const url = v.video_url || "";
  if (urlPathname(url) !== urlPathname(m.url)) return mediaFromValue(v);
  return {
    ...m,
    url: url || m.url,
    natW: v.video_width || m.natW,
    natH: v.video_height || m.natH,
    totalFrames: v.total_frames !== undefined ? v.total_frames : m.totalFrames,
  };
}

const hitR = HANDLE_R + 5;

function getHandles({ l: bl, t: bt, w: bw, h: bh }: Box, scale: number): Handle[] {
  const l = bl * scale, t = bt * scale;
  const r = (bl + bw) * scale, b = (bt + bh) * scale;
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

function clamp(v: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, v)); }

function CropVideoEditor(props: CropVideoEditorProps) {
  const { onChange, onLiveChange } = props;
  const value: CropVideoValue = props.value || {};
  const isDisabled = !!props.disabled;
  const lockedParams = lockedEdges(props.locked);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const dragRef = useRef<Drag | null>(null);

  const [media, setMedia] = useState(() => mediaFromValue(value));
  // The box a drag or preset left behind, shown until the settings next change.
  const [edited, setEdited] = useState<Box | null>(null);
  // The box an in-progress drag has reached.
  const [live, setLive] = useState<Box | null>(null);
  const [hoverHandle, setHoverHandle] = useState<HandleId | null>(null);
  const [cursor, setCursor] = useState<string | undefined>(undefined);
  // Measured when a video's metadata loads and on resize, so a new video keeps the last one's
  // canvas until its own metadata arrives.
  const [layout, setLayout] = useState<Layout | null>(null);

  // A new value replaces the edited box with the one its settings describe.
  const [followed, setFollowed] = useState(props.value);
  if (followed !== props.value) {
    setFollowed(props.value);
    setMedia((m) => followValue(m, value));
    setEdited(null);
  }

  const { natW: vidNatW, natH: vidNatH, totalFrames, loaded: videoLoaded } = media;
  const box = live ?? edited ?? cropFromSettings(value, vidNatW, vidNatH);

  // canvas-px / natural-px, with the canvas letterboxed inside the wrap
  const scale = layout ? Math.min(layout.w / layout.natW, layout.h / layout.natH) : 1;
  const canvasSize = layout && { w: Math.round(layout.natW * scale), h: Math.round(layout.natH * scale) };

  // ── Video loading ──────────────────────────────────────────────────────────
  useEffect(() => {
    const video = videoRef.current as HTMLVideoElement;
    if (!media.url) return;
    if (video.src && urlPathname(video.src) === urlPathname(media.url)) return;
    video.src = media.url;
    video.load();
  }, [media.url]);

  // Releases the video when the editor unmounts.
  useEffect(() => {
    const video = videoRef.current as HTMLVideoElement;
    return () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, []);

  function measureArea(): { w: number; h: number } {
    const wrap = canvasWrapRef.current as HTMLDivElement;
    return { w: wrap.clientWidth || 480, h: wrap.clientHeight || 360 };
  }

  function onResize() {
    const { w, h } = measureArea();
    setLayout((l) => (!l || (l.w === w && l.h === h) ? l : { ...l, w, h }));
  }

  // ── Canvas sizing ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!videoLoaded) return;
    const ro = new ResizeObserver(onResize);
    ro.observe(canvasWrapRef.current as HTMLDivElement);
    return () => ro.disconnect();
  }, [videoLoaded]);

  function onVideoLoadedMetadata() {
    const video = videoRef.current as HTMLVideoElement;
    // Fill in dims from the video element if Python couldn't probe them
    setMedia((m) => ({ ...m, loaded: true, natW: m.natW || video.videoWidth, natH: m.natH || video.videoHeight }));
    if (!dragRef.current) setEdited(null);
    setLayout({ ...measureArea(), natW: media.natW || video.videoWidth || 1, natH: media.natH || video.videoHeight || 1 });
    // Restore the previously viewed frame (no-op for first load where frame=0)
    if (media.frame > 0 && video.duration) {
      video.currentTime = (media.frame / totalFrames) * video.duration;
    }
  }

  // ── Frame scrubbing ────────────────────────────────────────────────────────
  function seekToFrame(frame: number) {
    const video = videoRef.current as HTMLVideoElement;
    if (!video.duration || totalFrames <= 0) return;
    setMedia((m) => ({ ...m, frame }));
    video.currentTime = (frame / totalFrames) * video.duration;
  }

  // ── Handle geometry ────────────────────────────────────────────────────────
  function isHandleLocked(h: Handle) {
    return h.affects.some(f => lockedParams.includes(f));
  }

  function hitHandle(px: number, py: number): Handle | null {
    for (const h of getHandles(box, scale)) {
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h;
    }
    return null;
  }

  function insideCrop(px: number, py: number) {
    return px >= box.l * scale && px <= (box.l + box.w) * scale
        && py >= box.t * scale && py <= (box.t + box.h) * scale;
  }

  // ── Render ─────────────────────────────────────────────────────────────────
  function draw() {
    const canvas = canvasRef.current as HTMLCanvasElement;
    // Every browser the editor supports provides a 2D context for a fresh canvas.
    const ctx = ctxRef.current ??= canvas.getContext("2d") as CanvasRenderingContext2D;
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

    const l = box.l * scale, t = box.t * scale;
    const w = box.w * scale, h = box.h * scale;

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
    for (const hand of getHandles(box, scale)) {
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

  // Redraws after every render; a completed seek redraws too.
  useLayoutEffect(draw);

  // ── Emit ───────────────────────────────────────────────────────────────────
  // A drag sets the box freely, so both presets switch to Custom. The editor commits the crop settings to
  // their parameters and keeps the rest (the video's details) in the widget's own value.
  function emitAll({ l, t, w, h }: Box) {
    if (typeof onChange !== "function") return;
    onChange({
      ...value,
      custom_left: l, custom_top: t, custom_width: w, custom_height: h,
      crop_size: "Custom", crop_position: "Custom",
    });
  }

  // Moves the node's crop fields along with a drag without committing them.
  function emitLive({ l, t, w, h }: Box) {
    if (typeof onLiveChange !== "function") return;
    onLiveChange({ custom_left: l, custom_top: t, custom_width: w, custom_height: h });
  }

  function commit(next: Box) {
    dragRef.current = null;
    setLive(null);
    setEdited(next);
    emitAll(next);
  }

  // ── Drag interaction ───────────────────────────────────────────────────────
  function canvasXY(e: ReactPointerEvent<HTMLCanvasElement>): Point {
    const canvas = e.currentTarget;
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width  / (rect.width  || canvas.width);
    const sy = canvas.height / (rect.height || canvas.height);
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }

  function startDrag(e: ReactPointerEvent<HTMLCanvasElement>, mode: DragMode, handle: Handle | null, start: Point) {
    dragRef.current = { mode, handle, start, rect: box, box };
    e.currentTarget.setPointerCapture(e.pointerId);
  }

  function onCanvasPointerDown(e: ReactPointerEvent<HTMLCanvasElement>) {
    e.stopPropagation();
    if (isDisabled) return;
    const { x, y } = canvasXY(e);
    const hand = hitHandle(x, y);

    if (hand && !isHandleLocked(hand)) {
      startDrag(e, "resizing", hand, { x, y });
      return;
    }

    const leftLocked = lockedParams.includes("left");
    const topLocked  = lockedParams.includes("top");
    // both axes must be locked to block move
    if (insideCrop(x, y) && !(leftLocked && topLocked)) {
      startDrag(e, "moving", null, { x, y });
      return;
    }

    const sizeLocked = lockedParams.includes("width") || lockedParams.includes("height");
    if (!(leftLocked || topLocked) && !sizeLocked) {
      startDrag(e, "drawing", null, { x: x / scale, y: y / scale });
    }
  }

  // The box a drag reaches at canvas point (x, y).
  function dragBox(drag: Drag, x: number, y: number, shiftKey: boolean): Box {
    const { start, rect } = drag;

    if (drag.mode === "moving") {
      const dx = (x - start.x) / scale, dy = (y - start.y) / scale;
      return {
        ...rect,
        l: lockedParams.includes("left") ? rect.l : clamp(Math.round(rect.l + dx), 0, vidNatW - rect.w),
        t: lockedParams.includes("top")  ? rect.t : clamp(Math.round(rect.t + dy), 0, vidNatH - rect.h),
      };
    }

    if (drag.mode === "resizing") {
      const nx = x / scale, ny = y / scale;
      const { l, t, w, h } = rect;
      let nl = l, nt = t, nw = w, nh = h;
      const id = (drag.handle as Handle).id;

      if (id.includes("e")) nw = clamp(Math.round(nx - nl), 1, vidNatW - nl);
      if (id.includes("s")) nh = clamp(Math.round(ny - nt), 1, vidNatH - nt);
      if (id.includes("w")) { const r = nl + nw; nl = clamp(Math.round(nx), 0, r - 1); nw = r - nl; }
      if (id.includes("n")) { const b = nt + nh; nt = clamp(Math.round(ny), 0, b - 1); nh = b - nt; }

      if (shiftKey && (id === "se" || id === "nw" || id === "ne" || id === "sw")) {
        const asp = rect.w / rect.h;
        if (nw / nh > asp) nh = Math.round(nw / asp); else nw = Math.round(nh * asp);
      }
      return { l: nl, t: nt, w: nw, h: nh };
    }

    const nx = clamp(x / scale, 0, vidNatW), ny = clamp(y / scale, 0, vidNatH);
    const w = Math.max(1, Math.round(Math.abs(nx - start.x)));
    return {
      l: Math.round(Math.min(start.x, nx)),
      t: Math.round(Math.min(start.y, ny)),
      w,
      h: shiftKey ? w : Math.max(1, Math.round(Math.abs(ny - start.y))),
    };
  }

  function onCanvasPointerMove(e: ReactPointerEvent<HTMLCanvasElement>) {
    e.stopPropagation();
    const { x, y } = canvasXY(e);
    const drag = dragRef.current;

    if (!drag) {
      const hand = hitHandle(x, y);
      const newHover = hand && !isHandleLocked(hand) ? hand.id : null;
      setHoverHandle(newHover);
      setCursor(newHover ? "crosshair" : (insideCrop(x, y) ? "move" : "crosshair"));
      return;
    }

    drag.box = dragBox(drag, x, y, e.shiftKey);
    setLive(drag.box);
    emitLive(drag.box);
  }

  function onCanvasPointerUp(e: ReactPointerEvent<HTMLCanvasElement>) {
    e.stopPropagation();
    if (dragRef.current) commit(dragRef.current.box);
  }

  function onApply(crop: CropRect) {
    commit({ l: crop.left, t: crop.top, w: crop.width, h: crop.height });
  }

  // ── DOM structure ──────────────────────────────────────────────────────────
  return (
    <div className="nodrag nowheel" style={{ display: "flex", flexDirection: "column", height: "100%", gap: "6px" }}>
      {/* Canvas row: video + canvas | sidebar */}
      <div style={{ display: "flex", flexDirection: "row", flex: "1 1 0", minHeight: "180px", gap: "6px" }}>
        <div
          ref={canvasWrapRef}
          style={{
            flex: "1 1 0", minHeight: "0", position: "relative",
            background: "#111", borderRadius: "6px", overflow: "hidden",
          }}
        >
          {/* Video element (background) */}
          <video
            ref={videoRef}
            muted preload="metadata" crossOrigin="anonymous"
            style={{
              position: "absolute", inset: "0", width: "100%", height: "100%",
              objectFit: "contain", pointerEvents: "none",
            }}
            onLoadedMetadata={onVideoLoadedMetadata}
            onSeeked={draw}
          />
          {/* Canvas overlay (transparent bg — draws crop overlay + handles only) */}
          <canvas
            ref={canvasRef}
            width={canvasSize?.w}
            height={canvasSize?.h}
            style={layout && canvasSize
              ? {
                  position: "absolute",
                  left: `${Math.round((layout.w - canvasSize.w) / 2)}px`,
                  top: `${Math.round((layout.h - canvasSize.h) / 2)}px`,
                  width: `${canvasSize.w}px`, height: `${canvasSize.h}px`, cursor,
                }
              : { position: "absolute", inset: "0", width: "100%", height: "100%", cursor }}
            onPointerDown={onCanvasPointerDown}
            onPointerMove={onCanvasPointerMove}
            onPointerUp={onCanvasPointerUp}
          />
        </div>
        <Sidebar
          imgNatW={vidNatW} imgNatH={vidNatH} crop={box}
          lockedParams={lockedParams} disabled={isDisabled} onApply={onApply}
        />
      </div>
      <FrameScrubber frame={media.frame} totalFrames={totalFrames} disabled={isDisabled} onSeek={seekToFrame} />
      <StatusBar
        videoLoaded={videoLoaded} lockedParams={lockedParams}
        ecL={box.l} ecT={box.t} ecW={box.w} ecH={box.h} version={WIDGET_VERSION}
      />
    </div>
  );
}

export default reactWidget(CropVideoEditor);
