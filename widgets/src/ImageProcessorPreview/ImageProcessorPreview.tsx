// Live preview for the image and video processor nodes. The node names its operation in the widget's
// value and lists its settings as synced_parameters, so they arrive in `value` while the user drags.
// The result is drawn in the browser on the image, or on a frame of the video picked with the scrubber;
// the node processes the real media when it runs.

import { useEffect, useLayoutEffect, useMemo, useReducer, useRef, useState } from "react";
import type { SyntheticEvent } from "react";

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
import type { WidgetProps } from "@griptape-ai/nodes-widget-api";
import { reactWidget } from "@griptape-ai/nodes-widget-api/react";

export type { PreviewImage, PreviewLiveValues, PreviewValue } from "./_types";

export type ImageProcessorPreviewProps = WidgetProps<PreviewValue, PreviewLiveValues>;

const WIDGET_VERSION = "1.2.0";
const CHECKER = 10;
const SCRUB_STEPS = 1000;

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();
const baseUrl = (u: string) => (u ? u.split("?")[0] : "");

// The single image or video the preview draws on. A fresh object each time the media loads, seeks, or
// fails, so the pixel cache keyed on it knows when the source's pixels changed.
interface Source {
  isVideo: boolean;
  url: string;
  ready: boolean;
}

// The scaled source's pixels, kept so a colour operation only redoes its own math while dragging.
interface PixelCache {
  source: Source | null;
  pixels: ImageData | null;
  blocked: boolean;
}

// What one draw works with: the preview canvas, the area it fits into, and the media it reads.
interface Scene {
  canvas: HTMLCanvasElement;
  ctx: CanvasRenderingContext2D;
  areaW: number;
  areaH: number;
  value: PreviewValue;
  source: Source;
  media: HTMLImageElement | HTMLVideoElement;
  /** The masks for the current URLs, in their order; null while those haven't been created yet. */
  multiImages: HTMLImageElement[] | null;
  multiModes: string[];
  cache: PixelCache;
}

function formatTime(seconds: number) {
  if (!Number.isFinite(seconds)) return "0:00.0";
  const m = Math.floor(seconds / 60);
  const s = (seconds - m * 60).toFixed(1).padStart(4, "0");
  return `${m}:${s}`;
}

function drawChecker(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = "#3a3a3a";
  ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = "#4a4a4a";
  for (let y = 0; y < h; y += CHECKER) {
    for (let x = (y / CHECKER) % 2 ? 0 : CHECKER; x < w; x += CHECKER * 2) {
      ctx.fillRect(x, y, CHECKER, CHECKER);
    }
  }
}

// Fills the view with a centred message and returns the status to show with it.
function showMessage(scene: Scene, text: string, status: string): string {
  const { canvas, ctx, areaW, areaH } = scene;
  canvas.width = areaW; canvas.height = areaH;
  ctx.fillStyle = "#1a1a1a";
  ctx.fillRect(0, 0, areaW, areaH);
  ctx.fillStyle = "#555";
  ctx.font = "13px sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, areaW / 2, areaH / 2);
  return status;
}

// Draws a generated operation (one with no source image) at full size, then scales it into view.
function drawGenerated(scene: Scene, operation: GeneratedPixelsOperation | GeneratedRenderOperation): string {
  const { canvas, ctx, value } = scene;
  const size = operation.size(value);
  if (size.error !== undefined) return showMessage(scene, size.error, size.error);
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
    return showMessage(scene, (err as Error).message, (err as Error).message);
  }
  const scale = Math.min(scene.areaW / size.width, scene.areaH / size.height);
  canvas.width = Math.max(1, Math.round(size.width * scale));
  canvas.height = Math.max(1, Math.round(size.height * scale));
  drawChecker(ctx, canvas.width, canvas.height);
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(full, 0, 0, canvas.width, canvas.height);
  const label = operation.label ? `${operation.label(value)}  →  ` : "";
  return `${label}${size.width} × ${size.height} px`;
}

// Several images at once, for operations that combine them (Combine Masks).
function drawMulti(scene: Scene, operation: MultiOperation): string {
  const { canvas, ctx, value, multiImages } = scene;
  if (!multiImages) return showMessage(scene, "Loading…", "Loading");
  if (!multiImages.length) return showMessage(scene, "Connect masks to preview the result", "No masks connected");
  if (!multiImages.every((image) => image.complete && image.naturalWidth)) return showMessage(scene, "Loading…", "Loading");
  const [w, h] = [multiImages[0]!.naturalWidth, multiImages[0]!.naturalHeight];
  if (multiImages.some((image) => image.naturalWidth !== w || image.naturalHeight !== h)) {
    return showMessage(scene, "The masks aren't all the same size", "The masks aren't all the same size");
  }
  const scale = Math.min(scene.areaW / w, scene.areaH / h);
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
    return showMessage(scene, "Preview unavailable for these images", "Preview unavailable for these images");
  }
  canvas.width = view.width;
  canvas.height = view.height;
  drawChecker(ctx, view.width, view.height);
  sctx.putImageData(new ImageData(operation.combine(datas, scene.multiModes, view), view.width, view.height), 0, 0);
  ctx.drawImage(scratch, 0, 0);
  const label = operation.label ? `${operation.label(value)}  →  ` : "";
  const skipped = Number(value.skipped) || 0;
  const note = skipped ? `  (${skipped} more can't be previewed)` : "";
  return `${label}${w} × ${h} px${note}`;
}

// Runs a colour operation on the source scaled to the preview. Falls back to the plain source when the
// browser won't hand over the pixels, which happens for media hosts that don't allow it.
function drawPixels(scene: Scene, operation: PixelsOperation, view: PreviewView) {
  const { ctx, media, cache } = scene;
  if (cache.source !== scene.source || !cache.pixels || cache.pixels.width !== view.width || cache.pixels.height !== view.height) {
    const scratch = document.createElement("canvas");
    scratch.width = view.width; scratch.height = view.height;
    const sctx = scratch.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
    sctx.drawImage(media, 0, 0, view.width, view.height);
    cache.source = scene.source;
    try {
      cache.pixels = sctx.getImageData(0, 0, view.width, view.height);
      cache.blocked = false;
    } catch {
      cache.pixels = null;
      cache.blocked = true;
    }
  }
  if (cache.blocked) {
    ctx.drawImage(media, 0, 0, view.width, view.height);
    return;
  }
  const result = new ImageData(new Uint8ClampedArray(cache.pixels!.data), view.width, view.height);
  operation.pixels(result.data, scene.value, view);
  // putImageData replaces pixels, so draw through a scratch canvas to keep the checkerboard behind.
  const out = document.createElement("canvas");
  out.width = view.width; out.height = view.height;
  (out.getContext("2d") as CanvasRenderingContext2D).putImageData(result, 0, 0);
  ctx.drawImage(out, 0, 0);
}

// Draws the node's operation into the preview canvas and returns the status line.
function drawPreview(scene: Scene): string {
  const { canvas, ctx, value, source, media } = scene;
  const operation: PreviewOperation | undefined = (OPERATIONS as Record<string, PreviewOperation | undefined>)[value.operation as string];
  if (!operation) return showMessage(scene, "No preview for this node", "");
  if (operation.generate) return drawGenerated(scene, operation);
  if (operation.multi) return drawMulti(scene, operation);
  if (!source.ready) {
    const what = source.isVideo ? "a video" : "an image";
    return showMessage(scene, `Connect ${what} to preview the result`, source.isVideo ? "No video connected" : "No image connected");
  }

  const [w, h] = media instanceof HTMLVideoElement ? [media.videoWidth, media.videoHeight] : [media.naturalWidth, media.naturalHeight];
  const size = operation.size ? operation.size(value, w, h) : { width: w, height: h };
  if (size.error !== undefined) return showMessage(scene, size.error, size.error);

  const scale = Math.min(scene.areaW / size.width, scene.areaH / size.height);
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
  drawChecker(ctx, view.width, view.height);

  try {
    if (operation.pixels) drawPixels(scene, operation, view);
    else operation.draw(ctx, media, value, view);
  } catch (err) {
    return showMessage(scene, (err as Error).message, (err as Error).message);
  }

  const label = operation.label ? `${operation.label(value)}  →  ` : "";
  const blocked = scene.cache.source === source && scene.cache.blocked;
  const note = blocked && operation.pixels ? "  (preview unavailable for this source)" : "";
  return `${label}${size.width} × ${size.height} px${note}`;
}

function ImageProcessorPreview({ value: rawValue }: ImageProcessorPreviewProps) {
  const value = useMemo<PreviewValue>(() => rawValue || {}, [rawValue]);

  // Each entry pairs an image with its mode; keep them together so a mode never lands on the wrong image.
  const multiEntries = Array.isArray(value.images) ? value.images.filter((entry) => entry && entry.url) : null;
  const multiUrls = multiEntries?.map((entry) => entry.url) ?? [];
  const multiModes = multiEntries?.map((entry) => entry.mode || "") ?? [];
  const multiKey = multiEntries ? multiUrls.map(baseUrl).join("|") : null;

  // The node says which kind of media it previews, so an empty video node asks for a video, not an image.
  // A URL that only gains a new query string keeps the media already loaded.
  const nextIsVideo = value.media === "video" || Boolean(value.video_url);
  const nextUrl = (nextIsVideo ? value.video_url : value.image_url) || "";
  const [source, setSource] = useState<Source>({ isVideo: nextIsVideo, url: nextUrl, ready: false });
  if (!multiEntries && (nextIsVideo !== source.isVideo || baseUrl(nextUrl) !== baseUrl(source.url))) {
    setSource({ isVideo: nextIsVideo, url: nextUrl, ready: false });
  }
  const markReady = (ready: boolean) => setSource((s) => ({ ...s, ready }));

  // The masks with the key of the URLs they were made from, so a draw only pairs them with the modes
  // of the same URLs.
  const [multi, setMulti] = useState<{ key: string | null; images: HTMLImageElement[] }>({ key: null, images: [] });
  const multiImages = multi.key === multiKey ? multi.images : null;
  const [status, setStatus] = useState("");
  const [scrubTime, setScrubTime] = useState("");
  // Bumped when something the canvas reads changes outside React: a resize, a font, a mask loading.
  const [frame, redraw] = useReducer((n: number) => n + 1, 0);

  const [img] = useState(() => {
    const image = new Image();
    image.crossOrigin = "anonymous";
    return image;
  });
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
  const canvasWrapRef = useRef<HTMLDivElement>(null);
  const scrubRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cacheRef = useRef<PixelCache>({ source: null, pixels: null, blocked: false });

  useEffect(() => {
    img.onload = () => markReady(true);
    img.onerror = () => markReady(false);
    return () => { img.onload = null; img.onerror = null; };
  }, [img]);

  useEffect(() => {
    if (!source.url) return;
    if (source.isVideo) {
      if (scrubRef.current) scrubRef.current.value = "0";
    } else {
      img.src = source.url;
    }
  }, [img, source.isVideo, source.url]);

  useEffect(() => {
    if (multiKey === null) return;
    const images = multiUrls.map((url) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.onload = redraw;
      image.onerror = redraw;
      image.src = url;
      return image;
    });
    setMulti({ key: multiKey, images });
    return () => images.forEach((image) => { image.onload = null; image.onerror = null; });
    // Keyed on the URLs without their query strings, so a refreshed URL keeps the masks already loaded.
  }, [multiKey]);

  // The font the node draws text with, loaded from the URL it sends. Text falls back to sans-serif until
  // it arrives.
  const fontUrl = value.font_url;
  useEffect(() => {
    if (!fontUrl || typeof FontFace === "undefined") return;
    new FontFace(PREVIEW_FONT, `url("${fontUrl}")`)
      .load()
      .then((face) => { document.fonts.add(face); redraw(); })
      .catch(() => {});
  }, [fontUrl]);

  useEffect(() => {
    const ro = new ResizeObserver(() => redraw());
    ro.observe(canvasWrapRef.current!);
    return () => ro.disconnect();
  }, []);

  // Release the media player on unmount; browsers cap how many a page can hold.
  useEffect(() => {
    const video = videoRef.current!;
    return () => {
      video.pause();
      video.removeAttribute("src");
      video.load();
    };
  }, []);

  useLayoutEffect(() => {
    const canvas = canvasRef.current!;
    ctxRef.current ??= canvas.getContext("2d") as CanvasRenderingContext2D;
    const wrap = canvasWrapRef.current!;
    setStatus(drawPreview({
      canvas,
      ctx: ctxRef.current,
      areaW: wrap.clientWidth || 480,
      areaH: wrap.clientHeight || 360,
      value,
      source,
      media: source.isVideo ? videoRef.current! : img,
      multiImages,
      multiModes,
      cache: cacheRef.current,
    }));
    // multiModes is derived from value.
  }, [value, source, multiImages, frame, img]);

  function seekTo(fraction: number) {
    const video = videoRef.current!;
    const duration = Number.isFinite(video.duration) ? video.duration : 0;
    // Stay a hair inside the end, where some browsers have no frame to show.
    video.currentTime = Math.max(0, Math.min(duration - 0.001, fraction * duration));
    setScrubTime(`${formatTime(video.currentTime)} / ${formatTime(duration)}`);
  }
  const scrubFraction = () => Number(scrubRef.current?.value ?? 0) / SCRUB_STEPS;

  return (
    <div
      className="image-processor-preview nodrag nowheel"
      style={{ display: "flex", flexDirection: "column", gap: "6px", height: "100%", userSelect: "none", WebkitUserSelect: "none" }}
    >
      <div
        ref={canvasWrapRef}
        style={{
          position: "relative", background: "#111", borderRadius: "6px", overflow: "hidden",
          flex: "1 1 0", minHeight: "180px",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        <canvas ref={canvasRef} style={{ display: "block" }} />
        <video
          ref={videoRef}
          src={source.isVideo && source.url ? source.url : undefined}
          crossOrigin="anonymous"
          muted
          playsInline
          preload="auto"
          style={{ display: "none" }}
          onLoadedData={() => seekTo(scrubFraction())}
          onSeeked={() => markReady(true)}
          onError={() => markReady(false)}
        />
      </div>
      {/* Scrubber for video sources: picks the frame the preview is drawn on. */}
      <div style={{ display: source.isVideo ? "flex" : "none", alignItems: "center", gap: "8px", padding: "0 2px" }}>
        <input
          ref={scrubRef}
          type="range"
          min={0}
          max={SCRUB_STEPS}
          defaultValue={0}
          style={{ flex: 1, minWidth: 0 }}
          onPointerDown={stopPropagation}
          onMouseDown={stopPropagation}
          onInput={() => { if (source.isVideo) seekTo(scrubFraction()); }}
        />
        <span style={{ fontSize: "10px", fontFamily: "monospace", color: "var(--muted-foreground)", whiteSpace: "nowrap" }}>
          {scrubTime}
        </span>
      </div>
      <div
        style={{
          fontSize: "11px", color: "var(--muted-foreground)", padding: "4px 8px",
          background: "var(--muted)", borderRadius: "4px",
          display: "flex", justifyContent: "space-between", alignItems: "center",
        }}
      >
        <span>{status}</span>
        <span style={{ fontFamily: "monospace", fontSize: "10px", opacity: 0.6 }}>{`v${WIDGET_VERSION}`}</span>
      </div>
    </div>
  );
}

export default reactWidget(ImageProcessorPreview);
