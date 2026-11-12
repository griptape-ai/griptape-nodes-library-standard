/**
 * MaskAdjustmentPreview Widget
 *
 * Provides interactive preview of mask adjustments on video frames.
 * Features:
 * - Timeline scrubbing to view different frames
 * - Adjustment slider to preview dilation/erosion
 * - Semi-transparent yellow mask overlay on original video
 *
 * The widget caches its instance on the container element so that
 * framework re-invocations (triggered by value changes) update the
 * existing DOM instead of rebuilding from scratch. This preserves
 * video playback position and avoids the "jump to frame 0" bug.
 *
 * Performance design:
 * - applyMorphologicalOp is O(W·H) via 2D prefix-sum box morphology,
 *   regardless of radius. ~1000× faster than the previous O(W·H·r²) loop.
 * - cachedMaskData stores the ImageData of the current frame so that
 *   adjustment-slider changes never re-draw from the video element.
 * - tmpCvs / tmpCtx are created once at closure scope and reused.
 * - loadeddata + requestVideoFrameCallback ensure the first frame is
 *   captured and displayed as soon as the browser has decoded it.
 */

import type { MaskPreviewValue, LiveMaskPreviewValue } from './_types';
import type { WidgetHandle, WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedMaskPreviewValue, LiveMaskPreviewValue, MaskPreviewValue } from './_types';

export type MaskAdjustmentPreviewProps = WidgetProps<MaskPreviewValue, LiveMaskPreviewValue>;

// A mounted preview, kept on its container so a repeated mount updates it instead of building
// a second one.
interface MaskPreviewInstance {
  handleUpdate: (props: MaskAdjustmentPreviewProps) => void;
  cleanup: () => void;
  wrapper: HTMLDivElement;
}
type MaskPreviewContainer = HTMLElement & { _maskPreview?: MaskPreviewInstance };

const WIDGET_VERSION = "0.9.0";

// Shared slider styling to match the editor's native Radix-style sliders
const SLIDER_STYLE = `
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: rgba(255,255,255,0.25);
  border-radius: 9999px;
  outline: none;
  cursor: pointer;
`;
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

const stopPropagation = (e: Event) => e.stopPropagation();

export default function MaskAdjustmentPreview(
  host: HTMLElement,
  props: MaskAdjustmentPreviewProps,
): WidgetHandle<MaskAdjustmentPreviewProps> {
  const container = host as MaskPreviewContainer;
  const { value, disabled, onChange } = props;

  const newOriginalUrl = value?.original_video_url || "";
  const newMaskUrl = value?.mask_video_url || "";
  const newAdjustment = value?.adjustment ?? 0;

  // If the widget was already built and its DOM is still attached, just update
  if (container._maskPreview) {
    const inst = container._maskPreview;
    if (inst.wrapper && inst.wrapper.isConnected) {
      inst.handleUpdate(props);
      return { cleanup: inst.cleanup, update: inst.handleUpdate };
    }
    // DOM was detached (framework called cleanup between renders) — full rebuild
    delete container._maskPreview;
  }

  // --- First-time build ---

  // Mutable state held in closure
  let originalVideoUrl = newOriginalUrl;
  let maskVideoUrl = newMaskUrl;
  let adjustment = newAdjustment;
  let currentFrame = value?.current_frame ?? 0;
  let totalFrames = value?.total_frames ?? 0;
  let videoReady = false;
  let maskReady = false;
  let isSeeking = false;
  let onChangeRef = onChange;
  let onLiveChangeRef = props.onLiveChange;
  // adjustment is a node parameter listed in synced_parameters; locked while a wire drives it
  let adjustmentLocked = (props.locked || []).includes("adjustment");
  let maskOnlyMode = !newOriginalUrl;
  let detectedFps = 30;

  // Reusable temp canvas — created once, resized as needed, never recreated
  const tmpCvs = document.createElement("canvas");
  // Every browser the editor supports provides a 2D context for a fresh canvas.
  const tmpCtx = tmpCvs.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
  // Cached ImageData of the current mask frame — updated only on seek or new video
  let cachedMaskData: ImageData | null = null;

  // Inject scoped slider styles
  const styleId = "mask-preview-slider-styles";
  if (!document.getElementById(styleId)) {
    document.head.append(<style id={styleId}>{SLIDER_THUMB_CSS}</style>);
  }

  // Original video element
  const video = (
    <video
      style={{ width: "100%", display: maskOnlyMode ? "none" : "block" }}
      preload="auto"
      crossOrigin="anonymous"
    />
  ) as HTMLVideoElement;

  // Mask video element (hidden when original video is present, visible otherwise)
  const maskVideo = (
    <video
      style={maskOnlyMode ? "width: 100%; display: block;" : "display: none;"}
      preload="auto"
      crossOrigin="anonymous"
    />
  ) as HTMLVideoElement;

  // Canvas for mask overlay — sized to match the video element exactly
  const canvas = (
    <canvas style={{ position: "absolute", top: "0", left: "0", pointerEvents: "none" }} />
  ) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;

  // The display video is whichever element is currently visible
  function displayVideo() {
    return maskOnlyMode ? maskVideo : video;
  }

  // Keep canvas CSS size in sync with the visible video element's rendered size
  function syncCanvasSize() {
    const el = displayVideo();
    const w = el.clientWidth;
    const h = el.clientHeight;
    if (w && h) {
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
    }
  }

  function commitAdjustment(val: number) {
    val = Math.max(-25, Math.min(25, val));
    adjustment = val;
    adjustmentSlider.value = String(val);
    adjustmentInput.value = String(val);
    scheduleRender();
    if (typeof onChangeRef === "function") {
      onChangeRef({
        original_video_url: originalVideoUrl,
        mask_video_url: maskVideoUrl,
        adjustment: val,
        current_frame: currentFrame,
        total_frames: totalFrames,
      });
    }
  }

  const adjustmentSlider = (
    <input
      type="range"
      min="-25"
      max="25"
      value={String(adjustment)}
      disabled={disabled || adjustmentLocked}
      style={SLIDER_STYLE + "flex: 1; min-width: 0;"}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      // Live preview on every drag tick — re-renders from cache, no video re-draw. The node's
      // adjustment slider follows through onLiveChange; the value is committed on release.
      onInput={(e: Event) => {
        const val = parseInt((e.target as HTMLInputElement).value, 10);
        adjustment = val;
        adjustmentInput.value = String(val);
        scheduleRender();
        if (typeof onLiveChangeRef === "function") onLiveChangeRef({ adjustment: val });
      }}
      // Persist only on commit (mouseup / end of drag)
      onChange={(e: Event) => commitAdjustment(parseInt((e.target as HTMLInputElement).value, 10))}
    />
  ) as HTMLInputElement;

  const adjustmentInput = (
    <input
      type="number"
      min="-25"
      max="25"
      step="1"
      value={String(adjustment)}
      disabled={disabled || adjustmentLocked}
      style={`
        width: 52px;
        flex-shrink: 0;
        background: transparent;
        border: none;
        color: #fff;
        font-size: 12px;
        text-align: right;
        outline: none;
        -moz-appearance: textfield;
        padding: 2px 0;
      `}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      // Number input commits on blur or Enter
      onBlur={() => {
        const val = parseInt(adjustmentInput.value, 10);
        if (!isNaN(val)) {
          commitAdjustment(val);
        } else {
          adjustmentInput.value = String(adjustment);
        }
      }}
      onKeyDown={(e: KeyboardEvent) => {
        if (e.key === "Enter") {
          e.preventDefault();
          adjustmentInput.blur();
        }
      }}
    />
  ) as HTMLInputElement;

  const frameDisplay = (<span style={{ fontWeight: "600", color: "#fff" }} />) as HTMLSpanElement;

  const timelineSlider = (
    <input
      type="range"
      min="0"
      max="100"
      value="0"
      style={SLIDER_STYLE}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onInput={(e: Event) => {
        const frameNum = parseInt((e.target as HTMLInputElement).value, 10);
        currentFrame = frameNum;
        updateTimelineDisplay();
        seekToFrame(frameNum);
      }}
    />
  ) as HTMLInputElement;

  function updateTimelineDisplay() {
    const dur = displayVideo().duration;
    const frames = totalFrames > 0 ? totalFrames : (dur ? Math.floor(dur * detectedFps) : 0);
    frameDisplay.textContent = `${currentFrame} / ${frames}`;
    timelineSlider.max = String(Math.max(0, frames - 1));
  }

  const infoText = (<span>Waiting for video inputs...</span>) as HTMLSpanElement;

  const wrapper = (
    <div
      class="mask-adjustment-preview nodrag nowheel"
      style={{
        display: "flex", flexDirection: "column", gap: "12px", padding: "12px",
        background: "#1a1a1a", borderRadius: "8px", minHeight: "400px",
      }}
    >
      {/* Video preview container with overlay */}
      <div
        style={{ position: "relative", width: "100%", background: "#000", borderRadius: "4px", overflow: "hidden" }}
        // Prevent drag interference
        onPointerDown={stopPropagation}
        onMouseDown={stopPropagation}
      >
        {video}
        {maskVideo}
        {canvas}
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
          <span style={{ fontSize: "12px", color: "#ccc", flexShrink: "0" }}>Mask Adjustment</span>
          {/* Slider + number input row (matches editor layout) */}
          <div style={{ display: "flex", alignItems: "center", gap: "8px", width: "100%" }}>
            {adjustmentSlider}
            {adjustmentInput}
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
            {frameDisplay}
          </div>
          {timelineSlider}
        </div>
      </div>

      {/* Info text with version number */}
      <div
        style={{
          display: "flex", justifyContent: "space-between", alignItems: "center",
          fontSize: "11px", color: "#888", padding: "8px", background: "#1f1f1f", borderRadius: "4px",
        }}
      >
        {infoText}
        <span style={{ fontSize: "10px", color: "#666", fontFamily: "monospace" }}>{`v${WIDGET_VERSION}`}</span>
      </div>
    </div>
  ) as HTMLDivElement;

  // Helper function to seek to a specific frame
  function seekToFrame(frameNum: number) {
    const primary = displayVideo();
    if (!primary.duration || isSeeking) return;

    const frames = totalFrames > 0 ? totalFrames : Math.floor(primary.duration * detectedFps);
    const time = (frameNum / frames) * primary.duration;

    isSeeking = true;
    primary.currentTime = time;
    // Also seek the hidden video to keep them in sync
    if (!maskOnlyMode && maskVideo.duration) {
      maskVideo.currentTime = time;
    }
    if (maskOnlyMode && video.duration) {
      video.currentTime = time;
    }
  }

  // Capture the current mask video frame into cachedMaskData.
  // Call this after any seek or on initial frame load — NOT on adjustment changes.
  function captureFrame() {
    if (!maskVideo.videoWidth) return false;
    const mw = maskVideo.videoWidth;
    const mh = maskVideo.videoHeight;
    if (tmpCvs.width !== mw || tmpCvs.height !== mh) {
      tmpCvs.width = mw;
      tmpCvs.height = mh;
    }
    try {
      tmpCtx.drawImage(maskVideo, 0, 0);
      cachedMaskData = tmpCtx.getImageData(0, 0, mw, mh);
      return true;
    } catch (e) {
      return false;
    }
  }

  // Coalesce rapid render requests into one rAF
  let renderPending = false;
  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      renderMaskOverlay();
    });
  }

  // Render mask overlay on canvas.
  // Uses cachedMaskData — never re-draws from the video element directly.
  function renderMaskOverlay() {
    if (!maskReady) return;

    // Ensure we have a cached frame (capture on first render after load)
    if (!cachedMaskData && !captureFrame()) return;
    // captureFrame fills the cache whenever it succeeds
    const maskData = cachedMaskData as ImageData;

    // In overlay mode we also need the original video to be ready
    if (!maskOnlyMode && (!videoReady || !video.videoWidth)) return;

    const refVideo = displayVideo();

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

      if (maskOnlyMode) {
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
        infoText.textContent = `Mask: OK | ${adjustment !== 0 ? `Adjusted preview (${adjustment}px)` : "No adjustment"}`;
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
        infoText.textContent = `Video: OK | Mask: OK | Overlay active (adj: ${adjustment}px)`;
      }
    } catch (error) {
      infoText.textContent = `Overlay error: ${(error as Error).message}`;
      console.error("Error rendering mask overlay:", error);
    }
  }

  // O(W·H) 2D prefix-sum box morphology.
  //
  // Replaces the previous O(W·H·r²) circular-kernel loop. The kernel is now a
  // square (box) rather than a circle, which is an acceptable approximation for
  // the live preview — the server-side FFmpeg processing uses the true filter.
  //
  // Algorithm:
  //   1. Extract binary mask → Uint8Array                         O(W·H)
  //   2. Build 2D prefix-sum table (Int32Array, 1-indexed)         O(W·H)
  //   3. For each pixel, query neighbourhood sum in O(1)           O(W·H)
  //   4. Dilation: sum > 0 → white; Erosion: sum == area → white   O(W·H)
  //
  // At r=25 on 1920×1080 this is ~4M operations vs ~4B previously (~1000× faster).
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

  function updateStatus() {
    const parts: string[] = [];
    if (!maskOnlyMode) {
      if (!originalVideoUrl) {
        parts.push("No original video");
      } else if (!videoReady) {
        parts.push("Loading video...");
      } else {
        parts.push("Video: OK");
      }
    }
    if (!maskVideoUrl) {
      parts.push("No mask video");
    } else if (!maskReady) {
      parts.push("Loading mask...");
    } else {
      parts.push("Mask: OK");
    }
    if (maskOnlyMode) {
      parts.push("Mask-only preview");
    }
    infoText.textContent = parts.join(" | ");
  }

  // --- Load a video source only if the URL actually changed ---
  // Note: video.src returns the fully resolved URL, so strip query params
  // (cache buster ?t=...) for comparison since the timestamp changes each call.
  function urlBase(u: string): string {
    return u ? u.split("?")[0] as string : "";
  }

  function loadOriginalVideo(url: string) {
    if (!url) return;
    if (urlBase(video.src) === urlBase(url)) return;
    videoReady = false;
    updateStatus();
    video.src = url;
  }

  function loadMaskVideo(url: string) {
    if (!url) return;
    if (urlBase(maskVideo.src) === urlBase(url)) return;
    maskReady = false;
    cachedMaskData = null; // Stale frame data must not survive a source change
    updateStatus();
    maskVideo.src = url;
  }

  // Restore frame position after video metadata loads
  function restoreFramePosition() {
    if (currentFrame > 0) {
      timelineSlider.value = String(currentFrame);
      seekToFrame(currentFrame);
    }
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

  // --- Video event handlers ---

  video.addEventListener("loadedmetadata", () => {
    if (!maskOnlyMode) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      syncCanvasSize();
      detectFps(video, (fps) => {
        detectedFps = fps;
        totalFrames = Math.floor(video.duration * detectedFps);
        updateTimelineDisplay();
      });
      totalFrames = Math.floor(video.duration * detectedFps);
      updateTimelineDisplay();
    }
    videoReady = true;
    updateStatus();
    if (maskReady) {
      restoreFramePosition();
      scheduleRender();
    }
  });

  // loadeddata fires when the first frame is available for drawing.
  // This is more reliable than loadedmetadata for initial canvas render.
  video.addEventListener("loadeddata", () => {
    if (!maskOnlyMode && maskReady && cachedMaskData) {
      scheduleRender();
    }
  });

  // Re-sync canvas when the visible video element resizes
  video.addEventListener("resize", syncCanvasSize);
  maskVideo.addEventListener("resize", () => { if (maskOnlyMode) syncCanvasSize(); });

  maskVideo.addEventListener("loadedmetadata", () => {
    maskReady = true;
    if (maskOnlyMode) {
      canvas.width = maskVideo.videoWidth;
      canvas.height = maskVideo.videoHeight;
      syncCanvasSize();
      detectFps(maskVideo, (fps) => {
        detectedFps = fps;
        totalFrames = Math.floor(maskVideo.duration * detectedFps);
        updateTimelineDisplay();
      });
      totalFrames = Math.floor(maskVideo.duration * detectedFps);
      updateTimelineDisplay();
    }
    updateStatus();

    // Use requestVideoFrameCallback when available for accurate first-frame capture.
    // Falls back to loadeddata below.
    if (maskVideo.requestVideoFrameCallback) {
      maskVideo.requestVideoFrameCallback(() => {
        if (captureFrame()) scheduleRender();
      });
    }

    if (maskOnlyMode || videoReady) {
      restoreFramePosition();
    }
  });

  // loadeddata fires when the first frame is decoded — reliable fallback for
  // browsers that don't support requestVideoFrameCallback.
  maskVideo.addEventListener("loadeddata", () => {
    if (maskReady && (maskOnlyMode || videoReady)) {
      if (!cachedMaskData) captureFrame();
      scheduleRender();
    }
  });

  maskVideo.addEventListener("error", (e) => {
    infoText.textContent = `Mask video error: ${maskVideo.error?.message || "unknown"}`;
    console.error("Mask video error:", e, maskVideo.error);
  });

  video.addEventListener("error", (e) => {
    infoText.textContent = `Video error: ${video.error?.message || "unknown"}`;
    console.error("Original video error:", e, video.error);
  });

  // After a seek completes, capture the new frame then re-render.
  // This replaces the previous 50ms setTimeout hack.
  video.addEventListener("seeked", () => {
    isSeeking = false;
    if (!maskOnlyMode) scheduleRender();
  });

  maskVideo.addEventListener("seeked", () => {
    isSeeking = false;
    captureFrame(); // Update the cache with the newly seeked frame
    scheduleRender();
  });

  // Initial video load
  loadOriginalVideo(originalVideoUrl);
  loadMaskVideo(maskVideoUrl);
  updateTimelineDisplay();

  container.appendChild(wrapper);

  // --- Update function for re-invocations ---
  function update(
    newOrigUrl: string,
    newMaskUrl: string,
    newAdj: number,
    newOnChange: MaskAdjustmentPreviewProps["onChange"],
    newOnLiveChange: MaskAdjustmentPreviewProps["onLiveChange"],
    newLocked: readonly string[] | undefined,
    newDisabled: boolean | undefined,
  ) {
    // Keep callback reference fresh
    if (newOnChange) onChangeRef = newOnChange;
    onLiveChangeRef = newOnLiveChange;
    adjustmentLocked = (newLocked || []).includes("adjustment");
    adjustmentSlider.disabled = newDisabled || adjustmentLocked;
    adjustmentInput.disabled = newDisabled || adjustmentLocked;

    // Detect mode change
    const newMaskOnly = !newOrigUrl;
    if (newMaskOnly !== maskOnlyMode) {
      maskOnlyMode = newMaskOnly;
      video.style.display = maskOnlyMode ? "none" : "block";
      maskVideo.style.cssText = maskOnlyMode
        ? "width: 100%; display: block;"
        : "display: none;";
    }

    if (newOrigUrl && newOrigUrl !== originalVideoUrl) {
      originalVideoUrl = newOrigUrl;
      loadOriginalVideo(newOrigUrl);
    }
    if (newMaskUrl && newMaskUrl !== maskVideoUrl) {
      maskVideoUrl = newMaskUrl;
      // loadMaskVideo already resets cachedMaskData
      loadMaskVideo(newMaskUrl);
    }
    if (newAdj !== adjustment) {
      adjustment = newAdj;
      adjustmentSlider.value = String(newAdj);
      adjustmentInput.value = String(newAdj);
      scheduleRender(); // Uses cached frame — instant re-render
    }
  }

  // Cleanup function — fully release video resources to avoid WebMediaPlayer exhaustion.
  // IMPORTANT: must remove the wrapper from the container and delete container._maskPreview
  // so the next widget call (after framework cleanup+re-render) does a full rebuild
  // rather than calling update() on detached DOM elements.
  function cleanup() {
    video.pause();
    maskVideo.pause();
    video.removeAttribute("src");
    maskVideo.removeAttribute("src");
    video.load();
    maskVideo.load();
    video.remove();
    maskVideo.remove();
    wrapper.remove();
    delete container._maskPreview;
  }

  // Adapter: maps ParameterComponentProps → internal update signature so the
  // WidgetLoader framework can forward value changes via handleRef.current.update().
  function handleUpdate(newProps: MaskAdjustmentPreviewProps) {
    const v = newProps.value;
    update(
      v?.original_video_url || "",
      v?.mask_video_url || "",
      v?.adjustment ?? 0,
      newProps.onChange,
      newProps.onLiveChange,
      newProps.locked,
      newProps.disabled,
    );
  }

  // Store instance on container for re-invocations (includes wrapper for isConnected check)
  container._maskPreview = { handleUpdate, cleanup, wrapper };

  return { cleanup, update: handleUpdate };
}
