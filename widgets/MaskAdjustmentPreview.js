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
 */

const WIDGET_VERSION = "0.5.1";

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
  const newAdjustment = value?.adjustment || 0;

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
  let currentFrame = value?.current_frame || 0;
  let totalFrames = value?.total_frames || 0;
  let videoReady = false;
  let maskReady = false;
  let isSeeking = false;
  let onChangeRef = onChange;
  let maskOnlyMode = !newOriginalUrl;

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

  // Live preview on every drag tick — no persistence yet
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
    const frames = totalFrames > 0 ? totalFrames : (dur ? Math.floor(dur * 30) : 0);
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

    const fps = 30;
    const frames = totalFrames > 0 ? totalFrames : Math.floor(primary.duration * fps);
    const time = (frameNum / frames) * primary.duration;

    isSeeking = true;
    primary.currentTime = time;
    // Also seek the other video if it's loaded
    if (!maskOnlyMode && maskVideo.duration) {
      maskVideo.currentTime = time;
    }
    if (maskOnlyMode && video.duration) {
      video.currentTime = time;
    }
  }

  // Coalesce rapid render requests
  let renderPending = false;
  function scheduleRender() {
    if (renderPending) return;
    renderPending = true;
    requestAnimationFrame(() => {
      renderPending = false;
      renderMaskOverlay();
    });
  }

  // Render mask overlay on canvas
  function renderMaskOverlay() {
    if (!maskReady) return;
    if (!maskVideo.videoWidth) return;
    if (!maskOnlyMode && (!videoReady || !video.videoWidth)) return;

    const refVideo = displayVideo();

    // Set canvas pixel dimensions to match the reference video resolution
    if (canvas.width !== refVideo.videoWidth || canvas.height !== refVideo.videoHeight) {
      canvas.width = refVideo.videoWidth;
      canvas.height = refVideo.videoHeight;
    }
    syncCanvasSize();
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const tempCanvas = document.createElement("canvas");
    tempCanvas.width = maskVideo.videoWidth;
    tempCanvas.height = maskVideo.videoHeight;
    const tempCtx = tempCanvas.getContext("2d", { willReadFrequently: true });

    try {
      tempCtx.drawImage(maskVideo, 0, 0);
      const maskData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
      const processedData = adjustment !== 0 ? applyMorphologicalOp(maskData, adjustment) : maskData;

      if (maskOnlyMode) {
        // Mask-only mode: draw the adjusted B&W mask directly on the canvas
        // When adjustment is 0 the canvas stays clear so the raw mask video shows through
        if (adjustment !== 0) {
          tempCtx.putImageData(processedData, 0, 0);
          ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        }
        infoText.textContent = `Mask: OK | ${adjustment !== 0 ? `Adjusted preview (${adjustment}px)` : "No adjustment"}`;
      } else {
        // Overlay mode: semi-transparent yellow mask on top of original video
        const overlayData = tempCtx.createImageData(tempCanvas.width, tempCanvas.height);
        const srcData = processedData.data;
        const dstData = overlayData.data;

        for (let i = 0; i < srcData.length; i += 4) {
          if (srcData[i] > 127) {
            dstData[i] = 255;     // R
            dstData[i + 1] = 255; // G
            dstData[i + 2] = 0;   // B
            dstData[i + 3] = 128; // A (50% opacity)
          } else {
            dstData[i + 3] = 0;
          }
        }

        tempCtx.putImageData(overlayData, 0, 0);
        ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
        infoText.textContent = `Video: OK | Mask: OK | Overlay active (adj: ${adjustment}px)`;
      }
    } catch (error) {
      infoText.textContent = `Overlay error: ${error.message}`;
      console.error("Error rendering mask overlay:", error);
    }
  }

  // Separable two-pass morphological operation (horizontal then vertical).
  // O(w*h*r) instead of O(w*h*r^2) — drastically reduces rAF frame time.
  function applyMorphologicalOp(imageData, adj) {
    if (adj === 0) return imageData;

    const w = imageData.width;
    const h = imageData.height;
    const src = imageData.data;
    const radius = Math.abs(adj);
    const isDilation = adj > 0;
    const threshold = 127;

    // Extract single-channel binary mask (1 = white, 0 = black)
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < mask.length; i++) {
      mask[i] = src[i * 4] > threshold ? 1 : 0;
    }

    // Pass 1: horizontal
    const temp = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) {
      const row = y * w;
      for (let x = 0; x < w; x++) {
        const x0 = Math.max(0, x - radius);
        const x1 = Math.min(w - 1, x + radius);
        let val = isDilation ? 0 : 1;
        for (let nx = x0; nx <= x1; nx++) {
          if (isDilation) { if (mask[row + nx]) { val = 1; break; } }
          else { if (!mask[row + nx]) { val = 0; break; } }
        }
        temp[row + x] = val;
      }
    }

    // Pass 2: vertical
    const result = new Uint8Array(w * h);
    for (let x = 0; x < w; x++) {
      for (let y = 0; y < h; y++) {
        const y0 = Math.max(0, y - radius);
        const y1 = Math.min(h - 1, y + radius);
        let val = isDilation ? 0 : 1;
        for (let ny = y0; ny <= y1; ny++) {
          if (isDilation) { if (temp[ny * w + x]) { val = 1; break; } }
          else { if (!temp[ny * w + x]) { val = 0; break; } }
        }
        result[y * w + x] = val;
      }
    }

    // Write back to RGBA
    const output = new Uint8ClampedArray(src.length);
    for (let i = 0; i < result.length; i++) {
      const v = result[i] ? 255 : 0;
      const j = i * 4;
      output[j] = v;
      output[j + 1] = v;
      output[j + 2] = v;
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

  // Video event handlers
  video.addEventListener("loadedmetadata", () => {
    if (!maskOnlyMode) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      syncCanvasSize();
      totalFrames = Math.floor(video.duration * 30);
      updateTimelineDisplay();
    }
    videoReady = true;
    updateStatus();
    if (maskReady) {
      restoreFramePosition();
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
      totalFrames = Math.floor(maskVideo.duration * 30);
      updateTimelineDisplay();
    }
    updateStatus();
    if (maskOnlyMode || videoReady) {
      restoreFramePosition();
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

  video.addEventListener("seeked", () => {
    isSeeking = false;
    if (!maskOnlyMode) {
      setTimeout(() => renderMaskOverlay(), 50);
    }
  });

  maskVideo.addEventListener("seeked", () => {
    isSeeking = false;
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
      loadMaskVideo(newMaskUrl);
    }
    if (newAdj !== adjustment) {
      adjustment = newAdj;
      adjustmentSlider.value = newAdj;
      adjustmentInput.value = newAdj;
      scheduleRender();
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
