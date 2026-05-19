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

const WIDGET_VERSION = "0.7.0";

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

export default function MaskAdjustmentPreview(container, props) {
  const { value, disabled, onChange } = props;

  const newOriginalUrl = value?.original_video_url || "";
  const newMaskUrl = value?.mask_video_url || "";
  const newAdjustment = value?.adjustment ?? 0;

  // If the widget was already built, just update what changed
  if (container._maskPreview) {
    const inst = container._maskPreview;
    inst.update(newOriginalUrl, newMaskUrl, newAdjustment, onChange);
    return inst.cleanup;
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
  let maskOnlyMode = !newOriginalUrl;
  let detectedFps = 30;

  // Reusable temp canvas — created once, resized as needed, never recreated
  const tmpCvs = document.createElement("canvas");
  const tmpCtx = tmpCvs.getContext("2d", { willReadFrequently: true });
  // Cached ImageData of the current mask frame — updated only on seek or new video
  let cachedMaskData = null;

  // Inject scoped slider styles
  const styleId = "mask-preview-slider-styles";
  if (!document.getElementById(styleId)) {
    const styleEl = document.createElement("style");
    styleEl.id = styleId;
    styleEl.textContent = SLIDER_THUMB_CSS;
    document.head.appendChild(styleEl);
  }

  // Create wrapper
  const wrapper = document.createElement("div");
  wrapper.className = "mask-adjustment-preview nodrag nowheel";
  wrapper.style.cssText = `
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    background: #1a1a1a;
    border-radius: 8px;
    min-height: 400px;
  `;

  // Video preview container with overlay
  const previewContainer = document.createElement("div");
  previewContainer.style.cssText = `
    position: relative;
    width: 100%;
    background: #000;
    border-radius: 4px;
    overflow: hidden;
  `;

  // Original video element
  const video = document.createElement("video");
  video.style.cssText = `
    width: 100%;
    display: ${maskOnlyMode ? "none" : "block"};
  `;
  video.preload = "auto";
  video.crossOrigin = "anonymous";

  // Mask video element (hidden when original video is present, visible otherwise)
  const maskVideo = document.createElement("video");
  maskVideo.style.cssText = maskOnlyMode
    ? "width: 100%; display: block;"
    : "display: none;";
  maskVideo.preload = "auto";
  maskVideo.crossOrigin = "anonymous";

  // Canvas for mask overlay — sized to match the video element exactly
  const canvas = document.createElement("canvas");
  canvas.style.cssText = `
    position: absolute;
    top: 0;
    left: 0;
    pointer-events: none;
  `;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

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

  previewContainer.appendChild(video);
  previewContainer.appendChild(maskVideo);
  previewContainer.appendChild(canvas);

  // Prevent drag interference
  previewContainer.addEventListener("pointerdown", (e) => e.stopPropagation());
  previewContainer.addEventListener("mousedown", (e) => e.stopPropagation());

  // Controls container
  const controlsContainer = document.createElement("div");
  controlsContainer.style.cssText = `
    display: flex;
    flex-direction: column;
    gap: 16px;
    padding: 12px;
    background: #252525;
    border-radius: 4px;
  `;

  // Adjustment slider control
  const adjustmentControl = document.createElement("div");
  adjustmentControl.style.cssText = `
    display: flex;
    flex-direction: column;
    gap: 8px;
  `;

  const adjustmentLabelText = document.createElement("span");
  adjustmentLabelText.style.cssText = `
    font-size: 12px;
    color: #ccc;
    flex-shrink: 0;
  `;
  adjustmentLabelText.textContent = "Mask Adjustment";

  // Slider + number input row (matches editor layout)
  const sliderRow = document.createElement("div");
  sliderRow.style.cssText = `
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
  `;

  const adjustmentSlider = document.createElement("input");
  adjustmentSlider.type = "range";
  adjustmentSlider.min = "-25";
  adjustmentSlider.max = "25";
  adjustmentSlider.value = adjustment;
  adjustmentSlider.disabled = disabled;
  adjustmentSlider.style.cssText = SLIDER_STYLE + "flex: 1; min-width: 0;";

  const adjustmentInput = document.createElement("input");
  adjustmentInput.type = "number";
  adjustmentInput.min = "-25";
  adjustmentInput.max = "25";
  adjustmentInput.step = "1";
  adjustmentInput.value = adjustment;
  adjustmentInput.disabled = disabled;
  adjustmentInput.style.cssText = `
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
  `;

  adjustmentSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  adjustmentSlider.addEventListener("mousedown", (e) => e.stopPropagation());
  adjustmentInput.addEventListener("pointerdown", (e) => e.stopPropagation());
  adjustmentInput.addEventListener("mousedown", (e) => e.stopPropagation());

  function commitAdjustment(val) {
    val = Math.max(-25, Math.min(25, val));
    adjustment = val;
    adjustmentSlider.value = val;
    adjustmentInput.value = val;
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

  // Live preview on every drag tick — re-renders from cache, no video re-draw
  adjustmentSlider.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    adjustment = val;
    adjustmentInput.value = val;
    scheduleRender();
  });

  // Persist only on commit (mouseup / end of drag)
  adjustmentSlider.addEventListener("change", (e) => {
    commitAdjustment(parseInt(e.target.value, 10));
  });

  // Number input commits on blur or Enter
  adjustmentInput.addEventListener("blur", () => {
    const val = parseInt(adjustmentInput.value, 10);
    if (!isNaN(val)) {
      commitAdjustment(val);
    } else {
      adjustmentInput.value = adjustment;
    }
  });
  adjustmentInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      adjustmentInput.blur();
    }
  });

  sliderRow.appendChild(adjustmentSlider);
  sliderRow.appendChild(adjustmentInput);

  adjustmentControl.appendChild(adjustmentLabelText);
  adjustmentControl.appendChild(sliderRow);

  // Timeline scrubber control
  const timelineControl = document.createElement("div");
  timelineControl.style.cssText = `
    display: flex;
    flex-direction: column;
    gap: 8px;
  `;

  const timelineLabel = document.createElement("div");
  timelineLabel.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12px;
    color: #ccc;
  `;

  const timelineLabelText = document.createElement("span");
  timelineLabelText.textContent = "Frame";

  const frameDisplay = document.createElement("span");
  frameDisplay.style.cssText = `
    font-weight: 600;
    color: #fff;
  `;

  const timelineSlider = document.createElement("input");
  timelineSlider.type = "range";
  timelineSlider.min = "0";
  timelineSlider.max = "100";
  timelineSlider.value = "0";
  timelineSlider.style.cssText = SLIDER_STYLE;

  function updateTimelineDisplay() {
    const dur = displayVideo().duration;
    const frames = totalFrames > 0 ? totalFrames : (dur ? Math.floor(dur * detectedFps) : 0);
    frameDisplay.textContent = `${currentFrame} / ${frames}`;
    timelineSlider.max = Math.max(0, frames - 1);
  }

  timelineSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  timelineSlider.addEventListener("mousedown", (e) => e.stopPropagation());

  timelineSlider.addEventListener("input", (e) => {
    const frameNum = parseInt(e.target.value, 10);
    currentFrame = frameNum;
    updateTimelineDisplay();
    seekToFrame(frameNum);
  });

  timelineLabel.appendChild(timelineLabelText);
  timelineLabel.appendChild(frameDisplay);
  timelineControl.appendChild(timelineLabel);
  timelineControl.appendChild(timelineSlider);

  controlsContainer.appendChild(adjustmentControl);
  controlsContainer.appendChild(timelineControl);

  // Info text with version number
  const infoContainer = document.createElement("div");
  infoContainer.style.cssText = `
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 11px;
    color: #888;
    padding: 8px;
    background: #1f1f1f;
    border-radius: 4px;
  `;

  const infoText = document.createElement("span");
  infoText.textContent = "Waiting for video inputs...";

  const versionText = document.createElement("span");
  versionText.style.cssText = `
    font-size: 10px;
    color: #666;
    font-family: monospace;
  `;
  versionText.textContent = `v${WIDGET_VERSION}`;

  infoContainer.appendChild(infoText);
  infoContainer.appendChild(versionText);

  // Helper function to seek to a specific frame
  function seekToFrame(frameNum) {
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
        ? applyMorphologicalOp(cachedMaskData, adjustment)
        : cachedMaskData;

      if (maskOnlyMode) {
        // Mask-only mode: draw the adjusted B&W mask on the canvas.
        // When adjustment is 0, canvas stays clear and the raw video shows through.
        if (adjustment !== 0) {
          const mw = cachedMaskData.width;
          const mh = cachedMaskData.height;
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
        const sw = cachedMaskData.width;
        const sh = cachedMaskData.height;
        const srcData = processedData.data;
        const overlayBuf = new Uint8ClampedArray(sw * sh * 4);

        for (let i = 0; i < srcData.length; i += 4) {
          if (srcData[i] > 127) {
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
      infoText.textContent = `Overlay error: ${error.message}`;
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
  function applyMorphologicalOp(imageData, adj) {
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
      mask[i] = src[i * 4] > threshold ? 1 : 0;
    }

    // Build 2D prefix-sum table.
    // prefix[(y+1)*stride + (x+1)] = sum of mask[0..y][0..x]
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
          prefix[(r2 + 1) * stride + (c2 + 1)] -
          prefix[r1 * stride + (c2 + 1)] -
          prefix[(r2 + 1) * stride + c1] +
          prefix[r1 * stride + c1];
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
    const parts = [];
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
  function urlBase(u) {
    return u ? u.split("?")[0] : "";
  }

  function loadOriginalVideo(url) {
    if (!url) return;
    if (urlBase(video.src) === urlBase(url)) return;
    videoReady = false;
    updateStatus();
    video.src = url;
  }

  function loadMaskVideo(url) {
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
      timelineSlider.value = currentFrame;
      seekToFrame(currentFrame);
    }
  }

  // Detect FPS using requestVideoFrameCallback when available
  function detectFps(videoEl, onDetected) {
    if (!videoEl.requestVideoFrameCallback) return;
    let firstTime = null;
    videoEl.requestVideoFrameCallback((now, metadata) => {
      firstTime = metadata.mediaTime;
      videoEl.requestVideoFrameCallback((now2, metadata2) => {
        const delta = metadata2.mediaTime - firstTime;
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

  // Assemble widget
  wrapper.appendChild(previewContainer);
  wrapper.appendChild(controlsContainer);
  wrapper.appendChild(infoContainer);
  container.appendChild(wrapper);

  // --- Update function for re-invocations ---
  function update(newOrigUrl, newMaskUrl, newAdj, newOnChange) {
    // Keep callback reference fresh
    if (newOnChange) onChangeRef = newOnChange;

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
      adjustmentSlider.value = newAdj;
      adjustmentInput.value = newAdj;
      scheduleRender(); // Uses cached frame — instant re-render
    }
  }

  // Cleanup function — fully release video resources to avoid WebMediaPlayer exhaustion
  function cleanup() {
    video.pause();
    maskVideo.pause();
    video.removeAttribute("src");
    maskVideo.removeAttribute("src");
    video.load();
    maskVideo.load();
    video.remove();
    maskVideo.remove();
  }

  // Store instance on container for re-invocations
  container._maskPreview = { update, cleanup };

  return cleanup;
}
