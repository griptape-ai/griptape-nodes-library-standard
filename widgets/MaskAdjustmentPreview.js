/**
 * MaskAdjustmentPreview Widget — Compact Slider + Floating Preview Tooltip
 *
 * Renders as a normal integer slider (-25 to +25) inside the node.
 * When the user starts dragging, a floating tooltip appears showing:
 *   - A canvas preview of the mask frame with the morphological
 *     adjustment applied in real-time (O(W·H) prefix-sum box morphology)
 *   - A timeline scrubber to pick any frame to preview
 *
 * Video elements are created once and kept alive across tooltip open/close
 * cycles so the frame doesn't have to reload on every drag.
 * Overlay mode (yellow mask on original video) activates automatically when
 * original_video_url is present in the value dict.
 *
 * Parameter value shape:
 *   { value: number, mask_video_url: string, original_video_url: string }
 */

const WIDGET_VERSION = "2.0.0";

const STYLE_ID = "mask-adj-preview-styles";
const SLIDER_BASE = `
  -webkit-appearance: none;
  appearance: none;
  width: 100%;
  height: 6px;
  background: rgba(255,255,255,0.2);
  border-radius: 9999px;
  outline: none;
  cursor: pointer;
`;
const INJECTED_CSS = `
  .map-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
    transition: box-shadow 0.15s;
  }
  .map-slider::-webkit-slider-thumb:hover {
    box-shadow: 0 0 0 4px rgba(124,58,237,0.3);
  }
  .map-slider::-moz-range-thumb {
    width: 16px; height: 16px;
    border-radius: 50%;
    background: #7c3aed;
    border: 2px solid #7c3aed;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
    cursor: pointer;
  }
  .map-slider:disabled { opacity: 0.5; cursor: not-allowed; }
  .map-frame-slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #4a9eff;
    border: 2px solid #4a9eff;
    cursor: pointer;
  }
  .map-frame-slider::-moz-range-thumb {
    width: 12px; height: 12px;
    border-radius: 50%;
    background: #4a9eff;
    border: 2px solid #4a9eff;
    cursor: pointer;
  }
`;

export default function MaskAdjustmentPreview(container, props) {
  const { value, disabled, onChange } = props;

  // Parse value — accept plain int for pipeline connections
  const parseValue = (v) => ({
    adjValue: typeof v === "number" ? v : (v?.value ?? 0),
    maskUrl: v?.mask_video_url ?? "",
    origUrl: v?.original_video_url ?? "",
  });

  let { adjValue, maskUrl, origUrl } = parseValue(value);
  let onChangeRef = onChange;

  // Re-use cached instance on subsequent framework calls
  if (container._mapInst) {
    container._mapInst.update(adjValue, maskUrl, origUrl, onChange);
    return container._mapInst.cleanup;
  }

  // ── Inject styles once ──────────────────────────────────────────────────
  if (!document.getElementById(STYLE_ID)) {
    const s = document.createElement("style");
    s.id = STYLE_ID;
    s.textContent = INJECTED_CSS;
    document.head.appendChild(s);
  }

  // ── Persistent video/canvas state (survives tooltip cycles) ─────────────
  // Video elements live in a hidden off-screen div so they stay loaded
  // even when the tooltip is closed.
  const hiddenHost = document.createElement("div");
  hiddenHost.style.cssText = "position:absolute;width:1px;height:1px;overflow:hidden;opacity:0;pointer-events:none;";
  document.body.appendChild(hiddenHost);

  const maskVideo = document.createElement("video");
  maskVideo.preload = "auto";
  maskVideo.crossOrigin = "anonymous";

  const origVideo = document.createElement("video");
  origVideo.preload = "auto";
  origVideo.crossOrigin = "anonymous";

  hiddenHost.appendChild(maskVideo);
  hiddenHost.appendChild(origVideo);

  // Reusable temp canvas for pixel operations
  const tmpCvs = document.createElement("canvas");
  const tmpCtx = tmpCvs.getContext("2d", { willReadFrequently: true });

  let cachedMaskData = null;  // ImageData of current mask frame
  let maskReady = false;
  let origReady = false;
  let currentFrame = 0;
  let totalFrames = 0;
  let detectedFps = 30;
  let isSeeking = false;

  // ── Compact slider UI (always visible in node) ──────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = `
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 4px 0;
    user-select: none;
  `;

  const labelRow = document.createElement("div");
  labelRow.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #ccc;
  `;

  const labelText = document.createElement("span");
  labelText.textContent = "Mask Adjustment";

  const valueDisplay = document.createElement("span");
  valueDisplay.style.cssText = `font-weight: 600; color: #fff; font-size: 12px;`;
  valueDisplay.textContent = String(adjValue);

  labelRow.appendChild(labelText);
  labelRow.appendChild(valueDisplay);

  const slider = document.createElement("input");
  slider.type = "range";
  slider.className = "map-slider";
  slider.min = "-25";
  slider.max = "25";
  slider.step = "1";
  slider.value = adjValue;
  slider.disabled = !!disabled;
  slider.style.cssText = SLIDER_BASE;

  const hintText = document.createElement("div");
  hintText.style.cssText = `font-size: 10px; color: #555; margin-top: 2px;`;
  hintText.textContent = "Drag to preview adjustment";

  wrapper.appendChild(labelRow);
  wrapper.appendChild(slider);
  wrapper.appendChild(hintText);
  container.appendChild(wrapper);

  // Block drag propagation to node graph
  [slider, wrapper].forEach((el) => {
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
    el.addEventListener("mousedown", (e) => e.stopPropagation());
  });

  // ── Tooltip state ───────────────────────────────────────────────────────
  let tooltipEl = null;
  let tooltipCanvas = null;
  let tooltipCtx = null;
  let tooltipFrameSlider = null;
  let tooltipFrameDisplay = null;
  let tooltipAdjDisplay = null;
  let tooltipStatusText = null;
  let isDragging = false;
  let renderPending = false;
  let hideTimer = null;

  function formatAdj(v) {
    if (v > 0) return `+${v}px  (dilate)`;
    if (v < 0) return `${v}px  (erode)`;
    return `±0  (no change)`;
  }

  function buildTooltip() {
    const el = document.createElement("div");
    el.className = "nodrag nowheel";
    el.style.cssText = `
      position: fixed;
      z-index: 99999;
      background: #1a1a1a;
      border: 1px solid #3a3a3a;
      border-radius: 10px;
      padding: 12px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.75);
      min-width: 300px;
      max-width: 520px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      pointer-events: auto;
    `;

    // Header row
    const header = document.createElement("div");
    header.style.cssText = `
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 11px;
      color: #aaa;
    `;
    const headerLabel = document.createElement("span");
    headerLabel.textContent = "Mask Preview";

    tooltipAdjDisplay = document.createElement("span");
    tooltipAdjDisplay.style.cssText = `font-weight: 700; color: #fff; font-size: 12px;`;
    tooltipAdjDisplay.textContent = formatAdj(adjValue);

    header.appendChild(headerLabel);
    header.appendChild(tooltipAdjDisplay);

    // Status / loading line
    tooltipStatusText = document.createElement("div");
    tooltipStatusText.style.cssText = `
      font-size: 11px;
      color: #666;
      min-height: 14px;
    `;
    tooltipStatusText.textContent = maskUrl ? "Loading video…" : "No mask video connected";

    // Preview canvas (hidden until frame ready)
    const cvs = document.createElement("canvas");
    cvs.style.cssText = `
      display: none;
      width: 100%;
      border-radius: 6px;
      background: #000;
    `;
    tooltipCanvas = cvs;
    tooltipCtx = cvs.getContext("2d");

    // Timeline scrubber section (hidden until video loads)
    const frameSection = document.createElement("div");
    frameSection.id = "map-frame-section";
    frameSection.style.cssText = `display: none; flex-direction: column; gap: 4px;`;

    const frameHeader = document.createElement("div");
    frameHeader.style.cssText = `
      display: flex;
      justify-content: space-between;
      font-size: 10px;
      color: #777;
    `;
    const frameHeaderLabel = document.createElement("span");
    frameHeaderLabel.textContent = "Frame";
    tooltipFrameDisplay = document.createElement("span");
    tooltipFrameDisplay.textContent = `${currentFrame} / ${totalFrames}`;
    frameHeader.appendChild(frameHeaderLabel);
    frameHeader.appendChild(tooltipFrameDisplay);

    tooltipFrameSlider = document.createElement("input");
    tooltipFrameSlider.type = "range";
    tooltipFrameSlider.className = "map-frame-slider";
    tooltipFrameSlider.min = "0";
    tooltipFrameSlider.max = Math.max(0, totalFrames - 1);
    tooltipFrameSlider.value = currentFrame;
    tooltipFrameSlider.style.cssText = SLIDER_BASE.replace("#7c3aed", "#4a9eff");

    tooltipFrameSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
    tooltipFrameSlider.addEventListener("mousedown", (e) => e.stopPropagation());

    tooltipFrameSlider.addEventListener("input", (e) => {
      currentFrame = parseInt(e.target.value, 10);
      if (tooltipFrameDisplay) {
        tooltipFrameDisplay.textContent = `${currentFrame} / ${totalFrames}`;
      }
      seekToFrame(currentFrame);
    });

    frameSection.appendChild(frameHeader);
    frameSection.appendChild(tooltipFrameSlider);

    // Version tag
    const ver = document.createElement("div");
    ver.style.cssText = `font-size: 9px; color: #444; text-align: right; font-family: monospace;`;
    ver.textContent = `v${WIDGET_VERSION}`;

    el.appendChild(header);
    el.appendChild(tooltipStatusText);
    el.appendChild(cvs);
    el.appendChild(frameSection);
    el.appendChild(ver);

    document.body.appendChild(el);
    return el;
  }

  function positionTooltip() {
    if (!tooltipEl) return;
    const rect = container.getBoundingClientRect();
    const th = tooltipEl.offsetHeight || 280;
    const tw = tooltipEl.offsetWidth || 320;
    const margin = 12;

    let top = rect.top - th - margin;
    let left = rect.left;

    // Flip below if not enough room above
    if (top < margin) top = rect.bottom + margin;
    // Clamp horizontally
    if (left + tw > window.innerWidth - margin) left = window.innerWidth - tw - margin;
    if (left < margin) left = margin;
    // Clamp vertically
    if (top + th > window.innerHeight - margin) top = window.innerHeight - th - margin;
    if (top < margin) top = margin;

    tooltipEl.style.left = left + "px";
    tooltipEl.style.top = top + "px";
  }

  function showTooltip() {
    if (tooltipEl) return;
    tooltipEl = buildTooltip();
    // Position after one frame so offsetHeight is computed
    requestAnimationFrame(() => {
      positionTooltip();
      loadVideosIfNeeded();
      // If already have a cached frame, render immediately
      if (cachedMaskData) scheduleRender();
    });
  }

  function hideTooltip() {
    clearTimeout(hideTimer);
    if (tooltipEl) {
      tooltipEl.remove();
      tooltipEl = null;
      tooltipCanvas = null;
      tooltipCtx = null;
      tooltipFrameSlider = null;
      tooltipFrameDisplay = null;
      tooltipAdjDisplay = null;
      tooltipStatusText = null;
    }
  }

  // ── Video loading ───────────────────────────────────────────────────────
  function urlBase(u) {
    return u ? u.split("?")[0] : "";
  }

  function loadVideosIfNeeded() {
    if (maskUrl && urlBase(maskVideo.src) !== urlBase(maskUrl)) {
      maskReady = false;
      cachedMaskData = null;
      maskVideo.src = maskUrl;
    }
    if (origUrl && urlBase(origVideo.src) !== urlBase(origUrl)) {
      origReady = false;
      origVideo.src = origUrl;
    }
  }

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

  function seekToFrame(frameNum) {
    if (!maskVideo.duration || isSeeking) return;
    const frames = totalFrames || Math.floor(maskVideo.duration * detectedFps);
    if (frames <= 0) return;
    isSeeking = true;
    maskVideo.currentTime = (frameNum / frames) * maskVideo.duration;
    if (origUrl && origVideo.duration) {
      origVideo.currentTime = maskVideo.currentTime;
    }
  }

  function detectFps(videoEl, cb) {
    if (!videoEl.requestVideoFrameCallback) return;
    let t0 = null;
    videoEl.requestVideoFrameCallback((_, m) => {
      t0 = m.mediaTime;
      videoEl.requestVideoFrameCallback((__, m2) => {
        const d = m2.mediaTime - t0;
        if (d > 0) {
          const fps = Math.round(1 / d);
          if (fps > 0 && fps < 240) cb(fps);
        }
      });
    });
  }

  function onFirstFrame() {
    maskReady = true;
    if (!tooltipEl) return;

    if (tooltipStatusText) tooltipStatusText.textContent = origUrl ? "Overlay mode" : "Mask-only mode";
    if (tooltipCanvas) tooltipCanvas.style.display = "block";

    // Show frame section
    const sec = tooltipEl.querySelector("#map-frame-section");
    if (sec) sec.style.display = "flex";

    // Size canvas proportionally, capped at tooltip width
    if (cachedMaskData) {
      const aspect = cachedMaskData.width / cachedMaskData.height;
      const maxPx = Math.min(520, window.innerWidth * 0.35);
      const displayW = Math.max(300, Math.min(maxPx, cachedMaskData.width));
      const displayH = Math.round(displayW / aspect);
      if (tooltipCanvas) {
        tooltipCanvas.width = cachedMaskData.width;
        tooltipCanvas.height = cachedMaskData.height;
        tooltipCanvas.style.width = displayW + "px";
        tooltipCanvas.style.height = displayH + "px";
      }
    }

    positionTooltip();
    scheduleRender();
  }

  function updateFrameUI() {
    if (tooltipFrameSlider) {
      tooltipFrameSlider.max = Math.max(0, totalFrames - 1);
      tooltipFrameSlider.value = currentFrame;
    }
    if (tooltipFrameDisplay) {
      tooltipFrameDisplay.textContent = `${currentFrame} / ${totalFrames}`;
    }
  }

  // ── Video event handlers ────────────────────────────────────────────────
  maskVideo.addEventListener("loadedmetadata", () => {
    detectFps(maskVideo, (fps) => {
      detectedFps = fps;
      totalFrames = Math.floor(maskVideo.duration * fps);
      updateFrameUI();
    });
    totalFrames = Math.floor(maskVideo.duration * detectedFps);
    updateFrameUI();

    // Use requestVideoFrameCallback when available for accurate first-frame capture
    if (maskVideo.requestVideoFrameCallback) {
      maskVideo.requestVideoFrameCallback(() => {
        if (captureFrame()) onFirstFrame();
      });
    }
    // Restore frame position if we had one
    if (currentFrame > 0) seekToFrame(currentFrame);
  });

  maskVideo.addEventListener("loadeddata", () => {
    if (!cachedMaskData && captureFrame()) onFirstFrame();
  });

  maskVideo.addEventListener("seeked", () => {
    isSeeking = false;
    if (captureFrame()) scheduleRender();
  });

  maskVideo.addEventListener("error", () => {
    if (tooltipStatusText) {
      tooltipStatusText.textContent = `Video error: ${maskVideo.error?.message || "unknown"}`;
    }
  });

  origVideo.addEventListener("loadeddata", () => {
    origReady = true;
    if (tooltipStatusText && maskReady) tooltipStatusText.textContent = "Overlay mode";
    scheduleRender();
  });

  // ── Rendering ───────────────────────────────────────────────────────────
  function scheduleRender() {
    if (renderPending || !tooltipCanvas) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      renderPreview();
    });
  }

  function renderPreview() {
    if (!tooltipCtx || !cachedMaskData) return;

    const mw = cachedMaskData.width;
    const mh = cachedMaskData.height;

    if (tooltipCanvas.width !== mw || tooltipCanvas.height !== mh) {
      tooltipCanvas.width = mw;
      tooltipCanvas.height = mh;
    }

    const processed = adjValue !== 0
      ? applyMorphologicalOp(cachedMaskData, adjValue)
      : cachedMaskData;

    if (origReady && origVideo.videoWidth) {
      // Overlay mode: draw original video frame, then semi-transparent yellow mask
      tooltipCtx.drawImage(origVideo, 0, 0, mw, mh);

      const overlayBuf = new Uint8ClampedArray(mw * mh * 4);
      const srcData = processed.data;
      for (let i = 0; i < srcData.length; i += 4) {
        if (srcData[i] > 127) {
          overlayBuf[i] = 255;
          overlayBuf[i + 1] = 255;
          overlayBuf[i + 2] = 0;
          overlayBuf[i + 3] = 128;
        }
      }
      if (tmpCvs.width !== mw || tmpCvs.height !== mh) { tmpCvs.width = mw; tmpCvs.height = mh; }
      tmpCtx.putImageData(new ImageData(overlayBuf, mw, mh), 0, 0);
      tooltipCtx.drawImage(tmpCvs, 0, 0);
    } else {
      // Mask-only mode: draw the B&W mask (adjusted or raw)
      if (tmpCvs.width !== mw || tmpCvs.height !== mh) { tmpCvs.width = mw; tmpCvs.height = mh; }
      tmpCtx.putImageData(processed, 0, 0);
      tooltipCtx.drawImage(tmpCvs, 0, 0);
    }

    if (tooltipAdjDisplay) tooltipAdjDisplay.textContent = formatAdj(adjValue);
  }

  // O(W·H) 2D prefix-sum box morphology — ~1000× faster than O(W·H·r²) loop.
  // Box kernel (square) is an acceptable approximation for the live preview;
  // the server-side FFmpeg processing uses the exact morphological filter.
  function applyMorphologicalOp(imageData, adj) {
    if (adj === 0) return imageData;
    const w = imageData.width;
    const h = imageData.height;
    const src = imageData.data;
    const radius = Math.abs(adj);
    const isDilation = adj > 0;

    const mask = new Uint8Array(w * h);
    for (let i = 0; i < mask.length; i++) {
      mask[i] = src[i * 4] > 127 ? 1 : 0;
    }

    const stride = w + 1;
    const prefix = new Int32Array((h + 1) * stride);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        prefix[(y + 1) * stride + (x + 1)] =
          mask[y * w + x] +
          prefix[y * stride + (x + 1)] +
          prefix[(y + 1) * stride + x] -
          prefix[y * stride + x];
      }
    }

    const result = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const r1 = Math.max(0, y - radius);
      const r2 = Math.min(h - 1, y + radius);
      const rowLen = r2 - r1 + 1;
      for (let x = 0; x < w; x++) {
        const c1 = Math.max(0, x - radius);
        const c2 = Math.min(w - 1, x + radius);
        const sum =
          prefix[(r2 + 1) * stride + (c2 + 1)] -
          prefix[r1 * stride + (c2 + 1)] -
          prefix[(r2 + 1) * stride + c1] +
          prefix[r1 * stride + c1];
        result[y * w + x] = isDilation
          ? (sum > 0 ? 1 : 0)
          : (sum === rowLen * (c2 - c1 + 1) ? 1 : 0);
      }
    }

    const output = new Uint8ClampedArray(src.length);
    for (let i = 0; i < result.length; i++) {
      const v = result[i] ? 255 : 0;
      const j = i * 4;
      output[j] = output[j + 1] = output[j + 2] = v;
      output[j + 3] = 255;
    }
    return new ImageData(output, w, h);
  }

  // ── Slider interactions ─────────────────────────────────────────────────
  slider.addEventListener("mousedown", () => {
    clearTimeout(hideTimer);
    isDragging = true;
    showTooltip();
  });

  slider.addEventListener("input", (e) => {
    adjValue = parseInt(e.target.value, 10);
    valueDisplay.textContent = String(adjValue);
    if (tooltipAdjDisplay) tooltipAdjDisplay.textContent = formatAdj(adjValue);
    scheduleRender();
  });

  slider.addEventListener("change", (e) => {
    adjValue = parseInt(e.target.value, 10);
    commitValue();
  });

  function commitValue() {
    if (typeof onChangeRef === "function") {
      onChangeRef({ value: adjValue, mask_video_url: maskUrl, original_video_url: origUrl });
    }
  }

  function handlePointerUp() {
    if (!isDragging) return;
    isDragging = false;
    // Keep tooltip visible briefly so user can read the result
    hideTimer = setTimeout(hideTooltip, 600);
  }

  document.addEventListener("pointerup", handlePointerUp);

  // ── Update function for subsequent framework calls ──────────────────────
  function update(newAdj, newMaskUrl, newOrigUrl, newOnChange) {
    if (newOnChange) onChangeRef = newOnChange;

    if (newAdj !== adjValue) {
      adjValue = newAdj;
      slider.value = newAdj;
      valueDisplay.textContent = String(newAdj);
      scheduleRender();
    }

    if (urlBase(newMaskUrl) !== urlBase(maskUrl)) {
      maskUrl = newMaskUrl;
      maskReady = false;
      cachedMaskData = null;
      if (tooltipEl) loadVideosIfNeeded();
    }

    if (urlBase(newOrigUrl) !== urlBase(origUrl)) {
      origUrl = newOrigUrl;
      origReady = false;
      if (tooltipEl) loadVideosIfNeeded();
    }
  }

  // ── Cleanup ─────────────────────────────────────────────────────────────
  function cleanup() {
    document.removeEventListener("pointerup", handlePointerUp);
    clearTimeout(hideTimer);
    maskVideo.pause();
    maskVideo.removeAttribute("src");
    maskVideo.load();
    origVideo.pause();
    origVideo.removeAttribute("src");
    origVideo.load();
    hiddenHost.remove();
    hideTooltip();
  }

  container._mapInst = { update, cleanup };
  return cleanup;
}
