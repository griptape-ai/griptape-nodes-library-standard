// Live preview for the image and video processor nodes. The node names its operation in the widget's
// value and lists its settings as synced_parameters, so they arrive in `value` while the user drags.
// The result is drawn in the browser on the image, or on a frame of the video picked with the scrubber;
// the node processes the real media when it runs.

import { OPERATIONS } from "./_operations";
import { PREVIEW_FONT } from "./_text";
import type {
  GeneratedPixelsOperation,
  GeneratedRenderOperation,
  MultiOperation,
  PixelsOperation,
  PreviewLiveValues,
  PreviewOperation,
  PreviewValue,
  PreviewView,
} from "./_types";
import type { WidgetHandle, WidgetProps } from "@griptape-ai/nodes-widget-api";

export type { PreviewImage, PreviewLiveValues, PreviewValue } from "./_types";

export type ImageProcessorPreviewProps = WidgetProps<PreviewValue, PreviewLiveValues>;

// A mounted preview, kept on its container so a repeated mount updates it instead of building
// a second one.
interface PreviewInstance {
  handleUpdate: (props: ImageProcessorPreviewProps) => void;
  cleanup: () => void;
  wrapper: HTMLDivElement;
}
type PreviewContainer = HTMLElement & { _imagePreview?: PreviewInstance };

const WIDGET_VERSION = "1.2.0";
const CHECKER = 10;
const SCRUB_STEPS = 1000;

const stopPropagation = (e: Event) => e.stopPropagation();

export default function ImageProcessorPreview(
  host: HTMLElement,
  props: ImageProcessorPreviewProps,
): WidgetHandle<ImageProcessorPreviewProps> {
  const container = host as PreviewContainer;
  if (container._imagePreview?.wrapper?.isConnected) {
    container._imagePreview.handleUpdate(props);
    return { cleanup: container._imagePreview.cleanup, update: container._imagePreview.handleUpdate };
  }
  if (container._imagePreview) delete container._imagePreview;

  let value: PreviewValue = props.value || {};
  let sourceUrl = "";
  let sourceReady = false;
  // The scaled source's pixels, kept so a colour operation only redoes its own math while dragging.
  let sourcePixels: ImageData | null = null;
  let pixelsBlocked = false;

  const canvas = (<canvas style="display:block;" />) as HTMLCanvasElement;
  const ctx = canvas.getContext("2d") as CanvasRenderingContext2D;

  const canvasWrap = (
    <div
      style={{
        position: "relative", background: "#111", borderRadius: "6px", overflow: "hidden",
        flex: "1 1 0", minHeight: "180px",
        display: "flex", alignItems: "center", justifyContent: "center",
      }}
    >
      {canvas}
    </div>
  ) as HTMLDivElement;

  // Scrubber for video sources: picks the frame the preview is drawn on.
  const scrub = (
    <input
      type="range"
      min="0"
      max={String(SCRUB_STEPS)}
      value="0"
      style="flex:1;min-width:0;"
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onInput={() => { if (isVideo) seekTo(Number(scrub.value) / SCRUB_STEPS); }}
    />
  ) as HTMLInputElement;
  const scrubTime = (
    <span style="font-size:10px;font-family:monospace;color:var(--muted-foreground);white-space:nowrap;" />
  ) as HTMLSpanElement;
  const scrubRow = (
    <div style="display:none;align-items:center;gap:8px;padding:0 2px;">
      {scrub}
      {scrubTime}
    </div>
  ) as HTMLDivElement;

  const statusL = (<span />) as HTMLSpanElement;

  const wrapper = (
    <div
      class="image-processor-preview nodrag nowheel"
      style="display:flex;flex-direction:column;gap:6px;height:100%;user-select:none;-webkit-user-select:none;"
    >
      {canvasWrap}
      {scrubRow}
      <div
        style={{
          fontSize: "11px", color: "var(--muted-foreground)", padding: "4px 8px",
          background: "var(--muted)", borderRadius: "4px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}
      >
        {statusL}
        <span style="font-family:monospace;font-size:10px;opacity:0.6;">{`v${WIDGET_VERSION}`}</span>
      </div>
    </div>
  ) as HTMLDivElement;
  container.appendChild(wrapper);

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => { sourceReady = true; sourcePixels = null; pixelsBlocked = false; render(); };
  img.onerror = () => { sourceReady = false; render(); };

  const video = (
    <video
      crossOrigin="anonymous"
      muted
      playsInline
      preload="auto"
      onLoadedData={() => { seekTo(Number(scrub.value) / SCRUB_STEPS); }}
      onSeeked={() => { sourceReady = true; sourcePixels = null; pixelsBlocked = false; render(); }}
      onError={() => { sourceReady = false; render(); }}
    />
  ) as HTMLVideoElement;

  // The font the node draws text with, loaded from the URL it sends. Text falls back to sans-serif until
  // it arrives.
  let fontUrl = "";
  function loadFont(url: string | undefined) {
    if (!url || url === fontUrl || typeof FontFace === "undefined") return;
    fontUrl = url;
    new FontFace(PREVIEW_FONT, `url("${url}")`)
      .load()
      .then((face) => { document.fonts.add(face); render(); })
      .catch(() => {});
  }

  // Draws a generated operation (one with no source image) at full size, then scales it into view.
  function renderGenerated(operation: GeneratedPixelsOperation | GeneratedRenderOperation) {
    const size = operation.size(value);
    if (size.error !== undefined) {
      showMessage(size.error, size.error);
      return;
    }
    const full = document.createElement("canvas");
    full.width = size.width; full.height = size.height;
    const fctx = full.getContext("2d") as CanvasRenderingContext2D;
    try {
      if (operation.pixels) {
        fctx.putImageData(new ImageData(operation.pixels(value, size.width, size.height), size.width, size.height), 0, 0);
      } else {
        operation.render(fctx, value, size);
      }
    } catch (err) {
      showMessage((err as Error).message, (err as Error).message);
      return;
    }
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / size.width, areaH / size.height);
    canvas.width = Math.max(1, Math.round(size.width * scale));
    canvas.height = Math.max(1, Math.round(size.height * scale));
    drawChecker(canvas.width, canvas.height);
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(full, 0, 0, canvas.width, canvas.height);
    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    statusL.textContent = `${label}${size.width} × ${size.height} px`;
  }

  // Several images at once, for operations that combine them (Combine Masks).
  let multiImages: HTMLImageElement[] = [];
  let multiModes: string[] = [];
  let multiKey = "";
  function loadMulti(urls: string[]) {
    const key = urls.map((u) => (u ? u.split("?")[0] : "")).join("|");
    if (key === multiKey) return;
    multiKey = key;
    multiImages = urls.filter(Boolean).map((url) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = () => render();
      image.onerror = () => render();
      image.src = url;
      return image;
    });
  }

  function renderMulti(operation: MultiOperation) {
    if (!multiImages.length) {
      showMessage("Connect masks to preview the result", "No masks connected");
      return;
    }
    if (!multiImages.every((image) => image.complete && image.naturalWidth)) {
      showMessage("Loading…", "Loading");
      return;
    }
    const [w, h] = [multiImages[0]!.naturalWidth, multiImages[0]!.naturalHeight];
    if (multiImages.some((image) => image.naturalWidth !== w || image.naturalHeight !== h)) {
      showMessage("The masks aren't all the same size", "The masks aren't all the same size");
      return;
    }
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / w, areaH / h);
    const view = { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)), scale, sourceWidth: w, sourceHeight: h };
    const scratch = document.createElement("canvas");
    scratch.width = view.width; scratch.height = view.height;
    const sctx = scratch.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
    let datas: Uint8ClampedArray[];
    try {
      datas = multiImages.map((image) => {
        sctx.clearRect(0, 0, view.width, view.height);
        sctx.drawImage(image, 0, 0, view.width, view.height);
        return sctx.getImageData(0, 0, view.width, view.height).data;
      });
    } catch {
      showMessage("Preview unavailable for these images", "Preview unavailable for these images");
      return;
    }
    canvas.width = view.width;
    canvas.height = view.height;
    drawChecker(view.width, view.height);
    sctx.putImageData(new ImageData(operation.combine(datas, multiModes, view), view.width, view.height), 0, 0);
    ctx.drawImage(scratch, 0, 0);
    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    const skipped = Number(value.skipped) || 0;
    const note = skipped ? `  (${skipped} more can't be previewed)` : "";
    statusL.textContent = `${label}${w} × ${h} px${note}`;
  }

  let isVideo = false;
  const source = () => (isVideo ? video : img);
  const sourceSize = (): [number, number] => (isVideo ? [video.videoWidth, video.videoHeight] : [img.naturalWidth, img.naturalHeight]);

  function formatTime(seconds: number) {
    if (!Number.isFinite(seconds)) return "0:00.0";
    const m = Math.floor(seconds / 60);
    const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
    return `${m}:${s}`;
  }

  function seekTo(fraction: number) {
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    // Stay a hair inside the end, where some browsers have no frame to show.
    video.currentTime = Math.max(0, Math.min(duration - 0.001, fraction * duration));
    scrubTime.textContent = `${formatTime(video.currentTime)} / ${formatTime(duration)}`;
  }

  function drawChecker(w: number, h: number) {
    ctx.fillStyle = "#3a3a3a";
    ctx.fillRect(0, 0, w, h);
    ctx.fillStyle = "#4a4a4a";
    for (let y = 0; y < h; y += CHECKER) {
      for (let x = (y / CHECKER) % 2 ? 0 : CHECKER; x < w; x += CHECKER * 2) {
        ctx.fillRect(x, y, CHECKER, CHECKER);
      }
    }
  }

  function showMessage(text: string, status: string) {
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
  function drawPixels(operation: PixelsOperation, view: PreviewView) {
    if (!sourcePixels || sourcePixels.width !== view.width || sourcePixels.height !== view.height) {
      const scratch = document.createElement("canvas");
      scratch.width = view.width; scratch.height = view.height;
      const sctx = scratch.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
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
    const result = new ImageData(new Uint8ClampedArray(sourcePixels!.data), view.width, view.height);
    operation.pixels(result.data, value, view);
    // putImageData replaces pixels, so draw through a scratch canvas to keep the checkerboard behind.
    const out = document.createElement("canvas");
    out.width = view.width; out.height = view.height;
    (out.getContext("2d") as CanvasRenderingContext2D).putImageData(result, 0, 0);
    ctx.drawImage(out, 0, 0);
  }

  function render() {
    const operation: PreviewOperation | undefined = (OPERATIONS as Record<string, PreviewOperation | undefined>)[value.operation as string];
    if (!operation) {
      showMessage("No preview for this node", "");
      return;
    }
    if (operation.generate) {
      renderGenerated(operation);
      return;
    }
    if (operation.multi) {
      renderMulti(operation);
      return;
    }
    if (!sourceReady) {
      const what = isVideo ? "a video" : "an image";
      showMessage(`Connect ${what} to preview the result`, isVideo ? "No video connected" : "No image connected");
      return;
    }

    const [w, h] = sourceSize();
    const size = operation.size ? operation.size(value, w, h) : { width: w, height: h };
    if (size.error !== undefined) {
      showMessage(size.error, size.error);
      return;
    }

    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const scale = Math.min(areaW / size.width, areaH / size.height);
    const view: PreviewView = {
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

    try {
      if (operation.pixels) drawPixels(operation, view);
      else operation.draw(ctx, source(), value, view);
    } catch (err) {
      showMessage((err as Error).message, (err as Error).message);
      return;
    }

    const label = operation.label ? `${operation.label(value)}  →  ` : "";
    const note = pixelsBlocked && operation.pixels ? "  (preview unavailable for this source)" : "";
    statusL.textContent = `${label}${size.width} × ${size.height} px${note}`;
  }

  function handleUpdate(newProps: ImageProcessorPreviewProps) {
    value = newProps.value || {};
    loadFont(value.font_url);
    if (Array.isArray(value.images)) {
      // Each entry pairs an image with its mode; keep them together so a mode never lands on the wrong image.
      const entries = value.images.filter((entry) => entry && entry.url);
      multiModes = entries.map((entry) => entry.mode || "");
      loadMulti(entries.map((entry) => entry.url));
      render();
      return;
    }
    // The node says which kind of media it previews, so an empty video node asks for a video, not an image.
    const nextIsVideo = value.media === "video" || Boolean(value.video_url);
    const newUrl = (nextIsVideo ? value.video_url : value.image_url) || "";
    const base = (u: string) => (u ? u.split("?")[0] : "");
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
