/**
 * MaskAdjustmentPreview Widget
 *
 * Provides interactive preview of mask adjustments on video frames.
 * Features:
 * - Timeline scrubbing to view different frames
 * - Adjustment slider to preview dilation/erosion
 * - Semi-transparent yellow mask overlay on original video
 *
 * Performance design:
 * - applyMorphologicalOp is O(W·H) via 2D prefix-sum box morphology,
 *   regardless of radius.
 * - The cached mask frame (ImageData of the current frame) means
 *   adjustment-slider changes never re-draw from the video element.
 * - The scratch canvas is created once per mount and reused.
 * - loadeddata + requestVideoFrameCallback ensure the first frame is
 *   captured and displayed as soon as the browser has decoded it.
 */

import { useEffect, useLayoutEffect, useReducer, useRef } from "react";
import type { CSSProperties } from "react";

import type { MaskPreviewValue, LiveMaskPreviewValue } from "./_types";
import type { WidgetProps } from "@griptape-ai/nodes-widget-api";
import { reactWidget } from "@griptape-ai/nodes-widget-api/react";

export type { CommittedMaskPreviewValue, LiveMaskPreviewValue, MaskPreviewValue } from "./_types";

export type MaskAdjustmentPreviewProps = WidgetProps<MaskPreviewValue, LiveMaskPreviewValue>;

const WIDGET_VERSION = "0.9.0";
const MIN_ADJUSTMENT = -25;
const MAX_ADJUSTMENT = 25;
const STYLE_ID = "mask-preview-slider-styles";

// Shared slider styling to match the editor's native Radix-style sliders
const SLIDER_STYLE: CSSProperties = {
  WebkitAppearance: "none",
  appearance: "none",
  width: "100%",
  height: "6px",
  background: "rgba(255,255,255,0.25)",
  borderRadius: "9999px",
  outline: "none",
  cursor: "pointer",
};
const SLIDER_THUMB_CSS = `
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }
  input[type=range]::-webkit-slider-thumb:hover {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
  }
  input[type=range]::-moz-range-thumb {
    width: 16px;
    height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }
  input[type=range]::-moz-range-thumb:hover {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
  }
  input[type=range]:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }
  input[type=range]:disabled::-webkit-slider-thumb {
    cursor: not-allowed;
  }
  .mask-adjustment-preview input[type=number]::-webkit-outer-spin-button,
  .mask-adjustment-preview input[type=number]::-webkit-inner-spin-button {
    -webkit-appearance: none;
    margin: 0;
  }
`;

const NUMBER_INPUT_STYLE: CSSProperties = {
  width: "52px",
  flexShrink: 0,
  background: "transparent",
  border: "none",
  color: "#fff",
  fontSize: "12px",
  textAlign: "right",
  outline: "none",
  MozAppearance: "textfield",
  padding: "2px 0",
};

const stopPropagation = (e: { stopPropagation: () => void }) => e.stopPropagation();
const stopHandlers = { onPointerDown: stopPropagation, onMouseDown: stopPropagation };

// The slider styles apply document-wide, so they're added once and left in place.
function injectStyles(): void {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = SLIDER_THUMB_CSS;
  document.head.append(style);
}

const clampAdjustment = (val: number) => Math.max(MIN_ADJUSTMENT, Math.min(MAX_ADJUSTMENT, val));

// A video's src is the fully resolved URL and the node appends a cache buster (?t=...) that
// changes each time, so sources are compared without their query.
function urlBase(u: string): string {
  return u ? u.split("?")[0] as string : "";
}

// O(W·H) 2D prefix-sum box morphology.
//
// The kernel is a square (box) rather than a circle, which is an acceptable
// approximation for the live preview — the server-side FFmpeg processing uses
// the true filter.
//
// Algorithm:
//   1. Extract binary mask → Uint8Array                         O(W·H)
//   2. Build 2D prefix-sum table (Int32Array, 1-indexed)         O(W·H)
//   3. For each pixel, query neighbourhood sum in O(1)           O(W·H)
//   4. Dilation: sum > 0 → white; Erosion: sum == area → white   O(W·H)
//
// At r=25 on 1920×1080 this is ~4M operations.
function applyMorphologicalOp(imageData: ImageData, adj: number): ImageData {
  if (adj === 0) return imageData;

  const w = imageData.width;
  const h = imageData.height;
  const src = imageData.data;
  const radius = Math.abs(adj);
  const isDilation = adj > 0;
  const threshold = 127;

  // Extract single-channel binary mask
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) {
    mask[i] = (src[i * 4] as number) > threshold ? 1 : 0;
  }

  // Build 2D prefix-sum table.
  // prefix[(y+1)*stride + (x+1)] = sum of mask[0..y][0..x]
  const stride = w + 1;
  const prefix = new Int32Array((h + 1) * stride);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      prefix[(y + 1) * stride + (x + 1)] =
        (mask[y * w + x] as number) +
        (prefix[y * stride + (x + 1)] as number) +
        (prefix[(y + 1) * stride + x] as number) -
        (prefix[y * stride + x] as number);
    }
  }

  // Apply box erosion/dilation using O(1) neighbourhood sum queries
  const result = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const r1 = Math.max(0, y - radius);
    const r2 = Math.min(h - 1, y + radius);
    const rowLen = r2 - r1 + 1;
    for (let x = 0; x < w; x++) {
      const c1 = Math.max(0, x - radius);
      const c2 = Math.min(w - 1, x + radius);
      const sum =
        (prefix[(r2 + 1) * stride + (c2 + 1)] as number) -
        (prefix[r1 * stride + (c2 + 1)] as number) -
        (prefix[(r2 + 1) * stride + c1] as number) +
        (prefix[r1 * stride + c1] as number);
      if (isDilation) {
        result[y * w + x] = sum > 0 ? 1 : 0;
      } else {
        result[y * w + x] = sum === rowLen * (c2 - c1 + 1) ? 1 : 0;
      }
    }
  }

  // Write back to RGBA
  const output = new Uint8ClampedArray(src.length);
  for (let i = 0; i < result.length; i++) {
    const v = result[i] ? 255 : 0;
    const j = i * 4;
    output[j] = output[j + 1] = output[j + 2] = v;
    output[j + 3] = 255;
  }

  return new ImageData(output, w, h);
}

// Detect FPS using requestVideoFrameCallback when available
function detectFps(videoEl: HTMLVideoElement, onDetected: (fps: number) => void) {
  if (!videoEl.requestVideoFrameCallback) return;
  let firstTime: number | null = null;
  videoEl.requestVideoFrameCallback((now, metadata) => {
    firstTime = metadata.mediaTime;
    videoEl.requestVideoFrameCallback((now2, metadata2) => {
      const delta = metadata2.mediaTime - (firstTime as number);
      if (delta > 0) {
        const fps = Math.round(1 / delta);
        if (fps > 0 && fps < 240) {
          onDetected(fps);
        }
      }
    });
  });
}

// ── Preview state ─────────────────────────────────────────────────────────────

interface PreviewState {
  /** The value prop last applied. Each new value object from the editor is applied once. */
  value: MaskPreviewValue | null | undefined;
  /** The URLs commits carry. An empty URL in the value leaves the last one in place. */
  originalUrl: string;
  maskUrl: string;
  /** What each video element loads; changes only when the URL's path does. */
  originalSrc: string;
  maskSrc: string;
  videoReady: boolean;
  maskReady: boolean;
  adjustment: number;
  /** Text typed into the number field and not yet committed. */
  draft: string | null;
  currentFrame: number;
  totalFrames: number;
  /** The latest render or error message; when null the info bar shows the load status. */
  message: string | null;
}

type PreviewAction =
  | { type: "value"; value: MaskPreviewValue | null | undefined }
  | { type: "adjusted"; adjustment: number }
  | { type: "typed"; draft: string | null }
  | { type: "scrubbed"; frame: number }
  | { type: "videoLoaded"; totalFrames?: number }
  | { type: "maskLoaded"; totalFrames?: number }
  | { type: "framesCounted"; totalFrames: number }
  | { type: "message"; message: string };

// Points each video at its URL when the URL's path differs from what it loads.
function loadSources(state: PreviewState): PreviewState {
  let next = state;
  if (next.originalUrl && urlBase(next.originalSrc) !== urlBase(next.originalUrl)) {
    next = { ...next, originalSrc: next.originalUrl, videoReady: false, message: null };
  }
  if (next.maskUrl && urlBase(next.maskSrc) !== urlBase(next.maskUrl)) {
    next = { ...next, maskSrc: next.maskUrl, maskReady: false, message: null };
  }
  return next;
}

function initState(value: MaskPreviewValue | null | undefined): PreviewState {
  return loadSources({
    value,
    originalUrl: value?.original_video_url || "",
    maskUrl: value?.mask_video_url || "",
    originalSrc: "",
    maskSrc: "",
    videoReady: false,
    maskReady: false,
    adjustment: value?.adjustment ?? 0,
    draft: null,
    currentFrame: value?.current_frame ?? 0,
    totalFrames: value?.total_frames ?? 0,
    message: "Waiting for video inputs...",
  });
}

function previewReducer(state: PreviewState, action: PreviewAction): PreviewState {
  switch (action.type) {
    case "value": {
      const { value } = action;
      const adjustment = value?.adjustment ?? 0;
      return loadSources({
        ...state,
        value,
        originalUrl: value?.original_video_url || state.originalUrl,
        maskUrl: value?.mask_video_url || state.maskUrl,
        adjustment,
        draft: adjustment === state.adjustment ? state.draft : null,
      });
    }
    case "adjusted":
      return { ...state, adjustment: action.adjustment, draft: null };
    case "typed":
      return { ...state, draft: action.draft };
    case "scrubbed":
      return { ...state, currentFrame: action.frame };
    case "videoLoaded":
      return { ...state, videoReady: true, message: null, totalFrames: action.totalFrames ?? state.totalFrames };
    case "maskLoaded":
      return { ...state, maskReady: true, message: null, totalFrames: action.totalFrames ?? state.totalFrames };
    case "framesCounted":
      return { ...state, totalFrames: action.totalFrames };
    case "message":
      return { ...state, message: action.message };
  }
}

function loadStatus(state: PreviewState, maskOnly: boolean): string {
  const parts: string[] = [];
  if (!maskOnly) {
    if (!state.originalUrl) {
      parts.push("No original video");
    } else if (!state.videoReady) {
      parts.push("Loading video...");
    } else {
      parts.push("Video: OK");
    }
  }
  if (!state.maskUrl) {
    parts.push("No mask video");
  } else if (!state.maskReady) {
    parts.push("Loading mask...");
  } else {
    parts.push("Mask: OK");
  }
  if (maskOnly) {
    parts.push("Mask-only preview");
  }
  return parts.join(" | ");
}

// ── Component ─────────────────────────────────────────────────────────────────

function MaskAdjustmentPreview(props: MaskAdjustmentPreviewProps) {
  const { value, disabled, onChange, onLiveChange, locked } = props;

  const [state, dispatch] = useReducer(previewReducer, value, initState);
  if (state.value !== value) dispatch({ type: "value", value });

  // With no original video the mask video is shown and the canvas draws the adjusted mask.
  const maskOnly = !value?.original_video_url;
  // adjustment is a node parameter listed in synced_parameters; locked while a wire drives it
  const controlsDisabled = disabled || (locked || []).includes("adjustment");
  const ready = state.maskReady && (maskOnly || state.videoReady);

  const videoRef = useRef<HTMLVideoElement>(null);
  const maskVideoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sliderRef = useRef<HTMLInputElement>(null);
  // Reusable scratch canvas and the overlay's context, created once per mount
  const surfaces = useRef<{ tmpCvs: HTMLCanvasElement; tmpCtx: CanvasRenderingContext2D; ctx: CanvasRenderingContext2D } | null>(null);
  // Cached ImageData of the current mask frame — updated only on seek or new video
  const cachedMaskData = useRef<ImageData | null>(null);
  const isSeeking = useRef(false);
  const renderPending = useRef(false);
  const detectedFps = useRef(30);
  // Commits go to the latest onChange; an update without one keeps the previous.
  const onChangeRef = useRef(onChange);
  // The overlay renders in an animation frame and the slider commits from a native listener;
  // both read the latest render's state through this.
  const latest = useRef({ renderMaskOverlay: () => {}, commitAdjustment: (_val: number) => {} });

  // The display video is whichever element is currently visible
  function displayVideo() {
    return (maskOnly ? maskVideoRef.current : videoRef.current) as HTMLVideoElement;
  }

  // Keep canvas CSS size in sync with the visible video element's rendered size
  function syncCanvasSize() {
    const canvas = canvasRef.current;
    const el = displayVideo();
    if (!canvas || !el) return;
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (w && h) {
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
    }
  }

  function sizeCanvasTo(el: HTMLVideoElement) {
    const canvas = canvasRef.current as HTMLCanvasElement;
    canvas.width = el.videoWidth;
    canvas.height = el.videoHeight;
    syncCanvasSize();
  }

  // Capture the current mask video frame into cachedMaskData.
  // Call this after any seek or on initial frame load — NOT on adjustment changes.
  function captureFrame() {
    const maskVideo = maskVideoRef.current;
    const s = surfaces.current;
    if (!maskVideo || !s || !maskVideo.videoWidth) return false;
    const mw = maskVideo.videoWidth;
    const mh = maskVideo.videoHeight;
    if (s.tmpCvs.width !== mw || s.tmpCvs.height !== mh) {
      s.tmpCvs.width = mw;
      s.tmpCvs.height = mh;
    }
    try {
      s.tmpCtx.drawImage(maskVideo, 0, 0);
      cachedMaskData.current = s.tmpCtx.getImageData(0, 0, mw, mh);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Coalesce rapid render requests into one rAF
  function scheduleRender() {
    if (renderPending.current) return;
    renderPending.current = true;
    requestAnimationFrame(() => {
      renderPending.current = false;
      latest.current.renderMaskOverlay();
    });
  }

  // Render mask overlay on canvas.
  // Uses cachedMaskData — never re-draws from the video element directly.
  function renderMaskOverlay() {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const s = surfaces.current;
    if (!state.maskReady || !video || !canvas || !s) return;

    // Ensure we have a cached frame (capture on first render after load)
    if (!cachedMaskData.current && !captureFrame()) return;
    // captureFrame fills the cache whenever it succeeds
    const maskData = cachedMaskData.current as ImageData;

    // In overlay mode we also need the original video to be ready
    if (!maskOnly && (!state.videoReady || !video.videoWidth)) return;

    const refVideo = displayVideo();
    const { tmpCvs, tmpCtx, ctx } = s;
    const { adjustment } = state;

    // Sync canvas pixel dimensions to the reference video resolution
    if (canvas.width !== refVideo.videoWidth || canvas.height !== refVideo.videoHeight) {
      canvas.width = refVideo.videoWidth;
      canvas.height = refVideo.videoHeight;
    }
    syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    try {
      const processedData = adjustment !== 0
        ? applyMorphologicalOp(maskData, adjustment)
        : maskData;

      if (maskOnly) {
        // Mask-only mode: draw the adjusted B&W mask on the canvas.
        // When adjustment is 0, canvas stays clear and the raw video shows through.
        if (adjustment !== 0) {
          const mw = maskData.width;
          const mh = maskData.height;
          if (tmpCvs.width !== mw || tmpCvs.height !== mh) {
            tmpCvs.width = mw;
            tmpCvs.height = mh;
          }
          tmpCtx.putImageData(processedData, 0, 0);
          ctx.drawImage(tmpCvs, 0, 0, canvas.width, canvas.height);
        }
        dispatch({
          type: "message",
          message: `Mask: OK | ${adjustment !== 0 ? `Adjusted preview (${adjustment}px)` : "No adjustment"}`,
        });
      } else {
        // Overlay mode: semi-transparent yellow mask on top of original video
        const sw = maskData.width;
        const sh = maskData.height;
        const srcData = processedData.data;
        const overlayBuf = new Uint8ClampedArray(sw * sh * 4);

        for (let i = 0; i < srcData.length; i += 4) {
          if ((srcData[i] as number) > 127) {
            overlayBuf[i] = 255;     // R
            overlayBuf[i + 1] = 255; // G
            overlayBuf[i + 2] = 0;   // B
            overlayBuf[i + 3] = 128; // A (50% opacity)
          }
          // else stays 0 (transparent)
        }

        if (tmpCvs.width !== sw || tmpCvs.height !== sh) {
          tmpCvs.width = sw;
          tmpCvs.height = sh;
        }
        tmpCtx.putImageData(new ImageData(overlayBuf, sw, sh), 0, 0);
        ctx.drawImage(tmpCvs, 0, 0, canvas.width, canvas.height);
        dispatch({ type: "message", message: `Video: OK | Mask: OK | Overlay active (adj: ${adjustment}px)` });
      }
    } catch (error) {
      dispatch({ type: "message", message: `Overlay error: ${(error as Error).message}` });
      console.error("Error rendering mask overlay:", error);
    }
  }

  // Seek both videos to a frame, unless a seek is still under way
  function seekToFrame(frameNum: number) {
    const primary = displayVideo();
    const video = videoRef.current as HTMLVideoElement;
    const maskVideo = maskVideoRef.current as HTMLVideoElement;
    if (!primary.duration || isSeeking.current) return;

    const frames = state.totalFrames > 0 ? state.totalFrames : Math.floor(primary.duration * detectedFps.current);
    const time = (frameNum / frames) * primary.duration;

    isSeeking.current = true;
    primary.currentTime = time;
    // Also seek the hidden video to keep them in sync
    if (!maskOnly && maskVideo.duration) {
      maskVideo.currentTime = time;
    }
    if (maskOnly && video.duration) {
      video.currentTime = time;
    }
  }

  function commitAdjustment(raw: number) {
    const val = clampAdjustment(raw);
    dispatch({ type: "adjusted", adjustment: val });
    onChangeRef.current({
      original_video_url: state.originalUrl,
      mask_video_url: state.maskUrl,
      adjustment: val,
      current_frame: state.currentFrame,
      total_frames: state.totalFrames,
    });
  }

  // Counts frames from the video's duration, then again once its real frame rate is known
  function countFrames(el: HTMLVideoElement): number {
    detectFps(el, (fps) => {
      detectedFps.current = fps;
      dispatch({ type: "framesCounted", totalFrames: Math.floor(el.duration * fps) });
    });
    return Math.floor(el.duration * detectedFps.current);
  }

  useLayoutEffect(() => {
    latest.current = { renderMaskOverlay, commitAdjustment };
    if (onChange) onChangeRef.current = onChange;
  });

  // Create the canvases' contexts, and fully release video resources on unmount to avoid
  // WebMediaPlayer exhaustion
  useEffect(() => {
    injectStyles();
    const tmpCvs = document.createElement("canvas");
    // Every browser the editor supports provides a 2D context for a fresh canvas.
    const tmpCtx = tmpCvs.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
    const canvas = canvasRef.current as HTMLCanvasElement;
    const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
    surfaces.current = { tmpCvs, tmpCtx, ctx };

    const videos = [videoRef.current, maskVideoRef.current] as HTMLVideoElement[];
    return () => {
      surfaces.current = null;
      for (const el of videos) el.pause();
      for (const el of videos) el.removeAttribute("src");
      for (const el of videos) el.load();
    };
  }, []);

  // The slider commits when released (native change), not on every drag tick
  useEffect(() => {
    const slider = sliderRef.current as HTMLInputElement;
    const commit = () => latest.current.commitAdjustment(parseInt(slider.value, 10));
    slider.addEventListener("change", commit);
    return () => slider.removeEventListener("change", commit);
  }, []);

  // Stale frame data must not survive a source change
  useEffect(() => {
    cachedMaskData.current = null;
  }, [state.maskSrc]);

  // Re-render the overlay from the cached frame whenever what it shows changes
  useEffect(() => {
    scheduleRender();
  }, [state.adjustment, maskOnly, state.maskReady, state.videoReady]);

  // Restore the frame position once the videos needed for the preview have loaded
  useEffect(() => {
    if (ready && state.currentFrame > 0) seekToFrame(state.currentFrame);
  }, [ready]);

  const frames = state.totalFrames;

  return (
    <div
      className="mask-adjustment-preview nodrag nowheel"
      style={{
        display: "flex", flexDirection: "column", gap: "12px", padding: "12px",
        background: "#1a1a1a", borderRadius: "8px", minHeight: "400px",
      }}
    >
      {/* Video preview container with overlay; stops drags from reaching the canvas behind */}
      <div
        style={{ position: "relative", width: "100%", background: "#000", borderRadius: "4px", overflow: "hidden" }}
        {...stopHandlers}
      >
        <video
          ref={videoRef}
          src={state.originalSrc || undefined}
          style={{ width: "100%", display: maskOnly ? "none" : "block" }}
          preload="auto"
          crossOrigin="anonymous"
          onLoadedMetadata={(e) => {
            const video = e.currentTarget;
            let totalFrames: number | undefined;
            if (!maskOnly) {
              sizeCanvasTo(video);
              totalFrames = countFrames(video);
            }
            dispatch({ type: "videoLoaded", totalFrames });
          }}
          // loadeddata fires when the first frame is available for drawing.
          // This is more reliable than loadedmetadata for initial canvas render.
          onLoadedData={() => {
            if (!maskOnly && cachedMaskData.current) scheduleRender();
          }}
          // Re-sync canvas when the visible video element resizes
          onResize={syncCanvasSize}
          onSeeked={() => {
            isSeeking.current = false;
            if (!maskOnly) scheduleRender();
          }}
          onError={(e) => {
            const error = e.currentTarget.error;
            dispatch({ type: "message", message: `Video error: ${error?.message || "unknown"}` });
            console.error("Original video error:", e.nativeEvent, error);
          }}
        />
        {/* Mask video (hidden when original video is present, visible otherwise) */}
        <video
          ref={maskVideoRef}
          src={state.maskSrc || undefined}
          style={maskOnly ? { width: "100%", display: "block" } : { display: "none" }}
          preload="auto"
          crossOrigin="anonymous"
          onLoadedMetadata={(e) => {
            const maskVideo = e.currentTarget;
            let totalFrames: number | undefined;
            if (maskOnly) {
              sizeCanvasTo(maskVideo);
              totalFrames = countFrames(maskVideo);
            }
            dispatch({ type: "maskLoaded", totalFrames });
            // Use requestVideoFrameCallback when available for accurate first-frame capture.
            // Falls back to loadeddata below.
            if (maskVideo.requestVideoFrameCallback) {
              maskVideo.requestVideoFrameCallback(() => {
                if (captureFrame()) scheduleRender();
              });
            }
          }}
          // loadeddata fires when the first frame is decoded — reliable fallback for
          // browsers that don't support requestVideoFrameCallback.
          onLoadedData={() => {
            if (!cachedMaskData.current) captureFrame();
            scheduleRender();
          }}
          onResize={() => { if (maskOnly) syncCanvasSize(); }}
          // After a seek completes, capture the new frame then re-render.
          onSeeked={() => {
            isSeeking.current = false;
            captureFrame();
            scheduleRender();
          }}
          onError={(e) => {
            const error = e.currentTarget.error;
            dispatch({ type: "message", message: `Mask video error: ${error?.message || "unknown"}` });
            console.error("Mask video error:", e.nativeEvent, error);
          }}
        />
        {/* Canvas for mask overlay — sized to match the video element exactly */}
        <canvas ref={canvasRef} style={{ position: "absolute", top: "0", left: "0", pointerEvents: "none" }} />
      </div>

      {/* Controls container */}
      <div
        style={{
          display: "flex", flexDirection: "column", gap: "16px", padding: "12px",
          background: "#252525", borderRadius: "4px",
        }}
      >
        {/* Adjustment slider control */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <span style={{ fontSize: "12px", color: "#ccc", flexShrink: 0 }}>Mask Adjustment</span>
          {/* Slider + number input row (matches editor layout) */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", width: "100%" }}>
            <input
              ref={sliderRef}
              type="range"
              min={MIN_ADJUSTMENT}
              max={MAX_ADJUSTMENT}
              value={state.adjustment}
              disabled={controlsDisabled}
              style={{ ...SLIDER_STYLE, flex: 1, minWidth: 0 }}
              {...stopHandlers}
              // Live preview on every drag tick — re-renders from cache, no video re-draw. The node's
              // adjustment slider follows through onLiveChange; the value is committed on release.
              onInput={(e) => {
                const val = parseInt(e.currentTarget.value, 10);
                dispatch({ type: "adjusted", adjustment: val });
                if (typeof onLiveChange === "function") onLiveChange({ adjustment: val });
              }}
            />
            <input
              type="number"
              min={MIN_ADJUSTMENT}
              max={MAX_ADJUSTMENT}
              step="1"
              value={state.draft ?? String(state.adjustment)}
              disabled={controlsDisabled}
              style={NUMBER_INPUT_STYLE}
              {...stopHandlers}
              onChange={(e) => dispatch({ type: "typed", draft: e.currentTarget.value })}
              // Number input commits on blur or Enter
              onBlur={(e) => {
                const input = e.currentTarget;
                const val = parseInt(input.value, 10);
                if (!isNaN(val)) {
                  commitAdjustment(val);
                } else {
                  dispatch({ type: "typed", draft: null });
                  // The field can hold text no input event reported, so restore it directly.
                  input.value = String(state.adjustment);
                }
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  e.currentTarget.blur();
                }
              }}
            />
          </div>
        </div>

        {/* Timeline scrubber control */}
        <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
          <div
            style={{
              display: "flex", justifyContent: "space-between", alignItems: "center",
              fontSize: "12px", color: "#ccc",
            }}
          >
            <span>Frame</span>
            <span style={{ fontWeight: 600, color: "#fff" }}>{`${state.currentFrame} / ${frames}`}</span>
          </div>
          <input
            type="range"
            min={0}
            max={Math.max(0, frames - 1)}
            value={state.currentFrame}
            style={SLIDER_STYLE}
            {...stopHandlers}
            onInput={(e) => {
              const frameNum = parseInt(e.currentTarget.value, 10);
              dispatch({ type: "scrubbed", frame: frameNum });
              seekToFrame(frameNum);
            }}
          />
        </div>
      </div>

      {/* Info text with version number */}
      <div
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          fontSize: "11px", color: "#888", padding: "8px", background: "#1f1f1f", borderRadius: "4px",
        }}
      >
        <span>{state.message ?? loadStatus(state, maskOnly)}</span>
        <span style={{ fontSize: "10px", color: "#666", fontFamily: "monospace" }}>{`v${WIDGET_VERSION}`}</span>
      </div>
    </div>
  );
}

export default reactWidget(MaskAdjustmentPreview);
