// Live preview for the image and video processor nodes. The node names its operation in the widget's
// value and lists its settings as synced_parameters, so they arrive in `value` while the user drags.
// The result is drawn in the browser on the image, or on a frame of the video picked with the scrubber;
// the node processes the real media when it runs.

import { OPERATIONS } from "./_operations.js";

const WIDGET_VERSION = "1.1.0";
const CHECKER = 10;
const SCRUB_STEPS = 1000;

export default function ImageProcessorPreview(container, props) {
  if (container._imagePreview?.wrapper?.isConnected) {
    container._imagePreview.handleUpdate(props);
    return { cleanup: container._imagePreview.cleanup, update: container._imagePreview.handleUpdate };
  }
  if (container._imagePreview) delete container._imagePreview;

  let value = props.value || {};
  let sourceUrl = "";
  let sourceReady = false;
  // The scaled source's pixels, kept so a colour operation only redoes its own math while dragging.
  let sourcePixels = null;
  let pixelsBlocked = false;

  const wrapper = document.createElement("div");
  wrapper.className = "image-processor-preview nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;gap:6px;height:100%;user-select:none;-webkit-user-select:none;";

  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:6px", "overflow:hidden",
    "flex:1 1 0", "min-height:180px",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;";
  const ctx = canvas.getContext("2d");
  canvasWrap.appendChild(canvas);

  // Scrubber for video sources: picks the frame the preview is drawn on.
  const scrubRow = document.createElement("div");
  scrubRow.style.cssText = "display:none;align-items:center;gap:8px;padding:0 2px;";
  const scrub = document.createElement("input");
  scrub.type = "range";
  scrub.min = "0";
  scrub.max = String(SCRUB_STEPS);
  scrub.value = "0";
  scrub.style.cssText = "flex:1;min-width:0;";
  const scrubTime = document.createElement("span");
  scrubTime.style.cssText = "font-size:10px;font-family:monospace;color:var(--muted-foreground);white-space:nowrap;";
  for (const type of ["pointerdown", "mousedown"]) scrub.addEventListener(type, (e) => e.stopPropagation());
  scrubRow.appendChild(scrub);
  scrubRow.appendChild(scrubTime);

  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px", "color:var(--muted-foreground)", "padding:4px 8px",
    "background:var(--muted)", "border-radius:4px",
    "display:flex", "justify-content:space-between", "align-items:center",
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;opacity:0.6;";
  statusR.textContent = `v${WIDGET_VERSION}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);

  wrapper.appendChild(canvasWrap);
  wrapper.appendChild(scrubRow);
  wrapper.appendChild(statusBar);
  container.appendChild(wrapper);

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => { sourceReady = true; sourcePixels = null; pixelsBlocked = false; render(); };
  img.onerror = () => { sourceReady = false; render(); };

  const video = document.createElement("video");
  video.crossOrigin = "anonymous";
  video.muted = true;
  video.playsInline = true;
  video.preload = "auto";
  video.addEventListener("loadeddata", () => { seekTo(Number(scrub.value) / SCRUB_STEPS); });
  video.addEventListener("seeked", () => { sourceReady = true; sourcePixels = null; pixelsBlocked = false; render(); });
  video.addEventListener("error", () => { sourceReady = false; render(); });

  let isVideo = false;
  const source = () => (isVideo ? video : img);
  const sourceSize = () => (isVideo ? [video.videoWidth, video.videoHeight] : [img.naturalWidth, img.naturalHeight]);

  function formatTime(seconds) {
    if (!Number.isFinite(seconds)) return "0:00.0";
    const m = Math.floor(seconds / 60);
    const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  }

  function seekTo(fraction) {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    // Stay a hair inside the end, where some browsers have no frame to show.
    video.currentTime = Math.max(0, Math.min(duration - 0.001, fraction * duration));
    scrubTime.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
  }

  scrub.addEventListener("input", () => { if (isVideo) seekTo(Number(scrub.value) / SCRUB_STEPS); });

  function drawChecker(w, h) {
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#4a4a4a";
    for (let y = 0; y < h; y += CHECKER) {
      for (let x = (y / CHECKER) % 2 ? 0 : CHECKER; x < w; x += CHECKER * 2) {
        ctx.fillRect(x, y, CHECKER, CHECKER);
      }
    }
  }

  function showMessage(text, status) {
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    canvas.width = areaW; canvas.height = areaH;
    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, 0, areaW, areaH);
    ctx.fillStyle = "#555";
    ctx.font = "13px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, areaW / 2, areaH / 2);
    statusL.textContent = status;
  }

  // Runs a colour operation on the source scaled to the preview. Falls back to the plain source when the
  // browser won't hand over the pixels, which happens for media hosts that don't allow it.
  function drawPixels(operation, view) {
    if (!sourcePixels || sourcePixels.width !== view.width || sourcePixels.height !== view.height) {
      const scratch = document.createElement("canvas");
      scratch.width = view.width; scratch.height = view.height;
      const sctx = scratch.getContext("2d", { willReadFrequently: true });
      sctx.drawImage(source(), 0, 0, view.width, view.height);
      try {
        sourcePixels = sctx.getImageData(0, 0, view.width, view.height);
        pixelsBlocked = false;
      } catch {
        sourcePixels = null;
        pixelsBlocked = true;
      }
    }
    if (pixelsBlocked) {
      ctx.drawImage(source(), 0, 0, view.width, view.height);
      return;
    }
    const result = new ImageData(new Uint8ClampedArray(sourcePixels.data), view.width, view.height);
    operation.pixels(result.data, value, view);
    // putImageData replaces pixels, so draw through a scratch canvas to keep the checkerboard behind.
    const out = document.createElement("canvas");
    out.width = view.width; out.height = view.height;
    out.getContext("2d").putImageData(result, 0, 0);
    ctx.drawImage(out, 0, 0);
  }

  function render() {
    const operation = OPERATIONS[value.operation];
    if (!operation) {
      showMessage("No preview for this node", "");
      return;
    }
    if (!sourceReady) {
      const what = isVideo ? "a video" : "an image";
      showMessage(`Connect ${what} to preview the result`, isVideo ? "No video connected" : "No image connected");
      return;
    }

    const [w, h] = sourceSize();
    const size = operation.size ? operation.size(value, w, h) : { width: w, height: h };
    if (size.error) {
      showMessage(size.error, size.error);
      return;
    }

    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / size.width, areaH / size.height);
    const view = {
      width: Math.max(1, Math.round(size.width * scale)),
      height: Math.max(1, Math.round(size.height * scale)),
      scale,
      sourceWidth: w,
      sourceHeight: h,
      outputWidth: size.width,
      outputHeight: size.height,
    };
    canvas.width = view.width;
    canvas.height = view.height;
    drawChecker(view.width, view.height);

    if (operation.pixels) drawPixels(operation, view);
    else operation.draw(ctx, source(), value, view);

    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    const note = pixelsBlocked && operation.pixels ? "  (preview unavailable for this source)" : "";
    statusL.textContent = `${label}${size.width} × ${size.height} px${note}`;
  }

  function handleUpdate(newProps) {
    value = newProps.value || {};
    const nextIsVideo = Boolean(value.video_url);
    const newUrl = (nextIsVideo ? value.video_url : value.image_url) || "";
    const base = (u) => (u ? u.split("?")[0] : "");
    if (nextIsVideo !== isVideo || base(newUrl) !== base(sourceUrl)) {
      isVideo = nextIsVideo;
      sourceUrl = newUrl;
      sourceReady = false;
      sourcePixels = null;
      scrubRow.style.display = isVideo ? "flex" : "none";
      if (!newUrl) {
        render();
        return;
      }
      if (isVideo) {
        scrub.value = "0";
        video.src = newUrl;
        video.load();
      } else {
        img.src = newUrl;
      }
      render();
      return;
    }
    render();
  }

  const ro = new ResizeObserver(() => render());
  ro.observe(canvasWrap);

  handleUpdate(props);

  function cleanup() {
    ro.disconnect();
    // Release the media player; browsers cap how many a page can hold.
    video.pause();
    video.removeAttribute("src");
    video.load();
    wrapper.remove();
    delete container._imagePreview;
  }

  container._imagePreview = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
