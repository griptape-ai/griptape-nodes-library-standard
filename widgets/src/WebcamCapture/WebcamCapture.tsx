import { Icon } from './_icons';
import type { GalleryItem, WebcamCaptureLiveValues, WebcamCaptureValue, WebcamGallery } from './_types';
import type { WidgetHandle, WidgetProps } from '@griptape-ai/nodes-widget-api';

export type {
  GalleryItem, WebcamAccepted, WebcamCaptureLiveValues, WebcamCaptureValue, WebcamClearGallery, WebcamGallery,
  WebcamIdle, WebcamProcessed, WebcamRequestingUploadUrl, WebcamSelected, WebcamSelectionConfirmed,
  WebcamUploadReady,
} from './_types';

export type WebcamCaptureProps = WidgetProps<WebcamCaptureValue, WebcamCaptureLiveValues>;

// A mounted widget, kept on its container so a repeated mount updates it instead of building
// a second one.
interface WebcamCaptureInstance {
  handleUpdate: (props: WebcamCaptureProps) => void;
  cleanup: () => void;
  wrapper: HTMLDivElement;
}
type WebcamCaptureContainer = HTMLElement & { _wcInst?: WebcamCaptureInstance };

function injectStyles(): void {
  if (document.getElementById("wc-styles")) return;
  document.head.append(
    <style id="wc-styles">
      {"@keyframes wc-spin{to{transform:rotate(360deg)}}"}
      {".wc-thumb{width:106px;height:80px;object-fit:cover;border-radius:6px;cursor:pointer;" +
        "flex-shrink:0;border:3px solid transparent;transition:border-color 0.15s;display:block;}"}
      {".wc-thumb.selected{border-color:var(--primary);}"}
    </style>,
  );
}

const stopPropagation = (e: Event) => e.stopPropagation();

export default function WebcamCapture(
  host: HTMLElement,
  props: WebcamCaptureProps,
): WidgetHandle<WebcamCaptureProps> {
  const container = host as WebcamCaptureContainer;
  if (container._wcInst?.wrapper?.isConnected) {
    container._wcInst.handleUpdate(props);
    return { cleanup: container._wcInst.cleanup, update: container._wcInst.handleUpdate };
  }

  injectStyles();

  let { onChange } = props;
  let _emitSeq = 0;
  let isStreaming = false;

  // Any state the node sends may carry the gallery, so read it from whichever one arrived.
  const init = (props.value ?? {}) as Partial<WebcamGallery>;
  let galleryItems: GalleryItem[] = init.gallery_items  ?? [];
  let selectedIndex = init.selected_index ?? (galleryItems.length > 0 ? galleryItems.length - 1 : -1);
  let pendingThumbs: string[] = [];

  // Queue of Blob values waiting to enter the upload protocol.
  // Only one capture flows through requesting_upload_url → processed at a time
  // because all captures share a single snapshot parameter channel — concurrent
  // in-flight requests would race on upload_ready delivery.
  const captureQueue: Blob[] = [];
  let _processing = false;
  let _pendingBlob: Blob | null = null;    // blob for the capture currently in-flight
  let _lastProcessedSeq = -1; // dedupes double delivery of "processed" messages
  let _lastUploadSeq = -1;    // dedupes double delivery of "upload_ready" messages

  let stream: MediaStream | null = null;
  let currentVideoId = "";

  // Ideal capture size. The browser picks the highest resolution the camera
  // supports up to this, so 4K cameras deliver 4K and lesser ones fall back.
  const IDEAL_WIDTH = 4096;
  const IDEAL_HEIGHT = 2160;
  const JPEG_QUALITY = 0.95;

  // ── DOM ──────────────────────────────────────────────────────────────────

  const releaseShutter = () => { shutterBtn.style.transform = "translateX(-50%)"; };

  const video = (
    <video
      style="width:100%;height:100%;object-fit:contain;display:block;"
      autoplay muted playsInline
    />
  ) as HTMLVideoElement;

  const placeholder = (
    <div style="position:absolute;color:var(--muted-foreground);font-size:13px;text-align:center;padding:24px;user-select:none;">
      Starting camera…
    </div>
  ) as HTMLDivElement;

  // Shutter button — circular, overlaid at bottom center
  const shutterBtn = (
    <button
      style={{
        position: "absolute", bottom: "12px", left: "50%", transform: "translateX(-50%)",
        width: "52px", height: "52px", borderRadius: "50%",
        border: "3px solid rgba(255,255,255,0.85)", background: "rgba(20,20,20,0.55)",
        cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
        backdropFilter: "blur(4px)", color: "white", transition: "transform 0.1s,opacity 0.15s",
      }}
      onPointerDown={(e) => { e.stopPropagation(); shutterBtn.style.transform = "translateX(-50%) scale(0.88)"; }}
      onPointerUp={releaseShutter}
      onPointerLeave={releaseShutter}
      onMouseDown={stopPropagation}
      onClick={capture}
    >
      <Icon name="camera" size={22} />
    </button>
  ) as HTMLButtonElement;

  // Photo strip — oldest left, newest right
  const thumbStrip = (
    <div
      style={{
        display: "flex", gap: "6px", overflowX: "auto",
        padding: "2px 0", scrollbarWidth: "thin",
        minHeight: "86px",
      }}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
    />
  ) as HTMLDivElement;

  const videoSel = (
    <select
      style={{
        flex: "1", minWidth: "0", padding: "4px 6px", borderRadius: "5px",
        border: "1px solid var(--border)", background: "var(--background)",
        color: "var(--foreground)", fontSize: "12px", cursor: "pointer",
      }}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onChange={switchDevices}
    />
  ) as HTMLSelectElement;

  const clearSpan = (<span />) as HTMLSpanElement;
  const clearBtn = (
    <button
      style={{
        display: "flex", alignItems: "center", gap: "5px",
        padding: "4px 10px", borderRadius: "5px",
        border: "1px solid var(--destructive)", background: "transparent",
        color: "var(--destructive)", fontSize: "12px", cursor: "pointer",
        whiteSpace: "nowrap", flexShrink: "0",
      }}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onClick={clearGallery}
    >
      <Icon name="trash-2" size={13} />
      {clearSpan}
    </button>
  ) as HTMLButtonElement;

  const wrapper = (
    <div
      class="nodrag nowheel"
      style="display:flex;flex-direction:column;height:100%;padding:8px;box-sizing:border-box;gap:6px;"
    >
      {/* Preview area */}
      <div
        style={{
          position: "relative", background: "#111", borderRadius: "8px",
          flex: "1 1 0", minHeight: "160px", overflow: "hidden",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        {placeholder}
        {video}
        {shutterBtn}
      </div>
      {thumbStrip}
      {/* Bottom row: camera selector + clear */}
      <div style="display:flex;align-items:center;gap:6px;color:var(--muted-foreground);">
        <Icon name="camera" size={15} />
        {videoSel}
        {clearBtn}
      </div>
    </div>
  ) as HTMLDivElement;
  container.append(wrapper);

  // ── Thumbnails ────────────────────────────────────────────────────────────

  function renderThumbs(): void {
    thumbStrip.innerHTML = "";
    thumbStrip.hidden = galleryItems.length === 0 && pendingThumbs.length === 0;
    clearBtn.hidden   = galleryItems.length === 0;
    clearSpan.textContent = `Clear (${galleryItems.length})`;

    // Confirmed items: oldest left → newest right
    galleryItems.forEach((item, idx) => {
      thumbStrip.append(
        <img
          class={"wc-thumb" + (idx === selectedIndex ? " selected" : "")}
          src={item.url || ""}
          title={`Photo ${idx + 1}`}
          onClick={() => selectThumbnail(idx)}
          onPointerDown={stopPropagation}
          onMouseDown={stopPropagation}
        />,
      );
    });

    // Pending items on the right — image underneath, spinner on top
    pendingThumbs.forEach((dataUrl) => {
      thumbStrip.append(
        <div style="position:relative;width:106px;height:80px;flex-shrink:0;border-radius:6px;overflow:hidden;border:3px solid transparent;">
          <img style="width:100%;height:100%;object-fit:cover;display:block;opacity:0.35;" src={dataUrl} />
          <div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.35);">
            <div style="width:22px;height:22px;border:2px solid rgba(255,255,255,0.2);border-top-color:white;border-radius:50%;animation:wc-spin 0.75s linear infinite;" />
          </div>
        </div>,
      );
    });

    // Scroll newest (rightmost) into view
    thumbStrip.lastElementChild?.scrollIntoView({ block: "nearest", inline: "end" });
  }

  function selectThumbnail(index: number): void {
    selectedIndex = index;
    renderThumbs();
    _emitSeq++;
    onChange?.({ state: "selected", selected_index: index, gallery_items: galleryItems, _emitSeq });
  }

  // ── Render ───────────────────────────────────────────────────────────────

  function render(): void {
    placeholder.hidden    = isStreaming;
    video.hidden          = !isStreaming;
    shutterBtn.disabled   = !isStreaming;
    shutterBtn.style.opacity = isStreaming ? "1" : "0.4";
  }

  // ── Camera ───────────────────────────────────────────────────────────────

  async function populateDevices(): Promise<void> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((d) => d.kind === "videoinput");
      const current = videoSel.value;
      videoSel.innerHTML = "";
      videoSel.append(...cameras.map((d, i) => <option value={d.deviceId}>{d.label || `Camera ${i + 1}`}</option>));
      if (current && [...videoSel.options].some((o) => o.value === current)) videoSel.value = current;
      if (stream) {
        const vs = stream.getVideoTracks()[0]?.getSettings?.()?.deviceId;
        if (vs) { currentVideoId = vs; videoSel.value = vs; }
      }
    } catch { /* permission not yet granted */ }
  }

  async function switchDevices(): Promise<void> {
    currentVideoId = videoSel.value;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    isStreaming = false;
    render();
    await startCamera();
  }

  async function startCamera(): Promise<void> {
    try {
      const videoConstraints: MediaTrackConstraints = {
        width: { ideal: IDEAL_WIDTH },
        height: { ideal: IDEAL_HEIGHT },
      };
      if (currentVideoId) videoConstraints.deviceId = { exact: currentVideoId };
      const s = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });
      stream = s;
      video.srcObject = stream;
      isStreaming = true;
      render();
      await populateDevices();
    } catch {
      placeholder.textContent = "Camera unavailable — check browser permissions.";
    }
  }

  // ── Capture (upload protocol) ─────────────────────────────────────────────

  // toDataURL is one synchronous JPEG encode. We reuse that encoded data for both
  // the pending thumbnail (as a data URL) and the binary PUT upload (as a Blob).
  // This avoids calling canvas.toBlob(), which would kick off a second async JPEG
  // encode and force the upload to wait for it to complete before it could start.
  function dataUrlToBlob(dataUrl: string): Blob {
    const comma = dataUrl.indexOf(",");
    const mime  = dataUrl.slice(5, dataUrl.indexOf(";"));
    const bytes = atob(dataUrl.slice(comma + 1));
    const arr   = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }

  function capture(): void {
    if (!stream) return;

    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);

    // One encode — reuse for both thumbnail and binary upload
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const blob    = dataUrlToBlob(dataUrl);  // base64→binary in memory, no re-encode

    captureQueue.push(blob);
    pendingThumbs = [...pendingThumbs, dataUrl];
    renderThumbs();
    _processNext();
  }

  function _processNext(): void {
    if (_processing || captureQueue.length === 0) return;
    _processing = true;
    const blob = captureQueue.shift()!;
    _pendingBlob = blob;
    _emitSeq++;
    onChange?.({ state: "requesting_upload_url", _emitSeq });
  }

  function _doUpload(uploadUrl: string, seq: number): void {
    const blob = _pendingBlob;
    _pendingBlob = null;
    fetch(uploadUrl, { method: "PUT", body: blob })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); })
      .then(() => { onChange?.({ state: "accepted", _emitSeq: seq }); })
      .catch(() => {
        // On PUT failure, still send accepted so Python echoes "processed"
        // and the pending thumbnail clears rather than staying stuck forever.
        onChange?.({ state: "accepted", _emitSeq: seq });
      });
  }

  function clearGallery(): void {
    captureQueue.length = 0;
    _processing = false;
    _pendingBlob = null;
    pendingThumbs = [];
    _emitSeq++;
    onChange?.({ state: "clear_gallery", _emitSeq });
  }

  // ── handleUpdate ─────────────────────────────────────────────────────────

  function handleUpdate(newProps: WebcamCaptureProps): void {
    onChange = newProps.onChange;
    const v: WebcamCaptureValue | { state?: undefined } = newProps.value ?? {};

    if (v.state === "upload_ready" && v._uploadUrl) {
      // Dedupe double delivery, same as "processed" below. A repeat of this seq
      // would fall through to the else branch and release the in-flight slot
      // while the PUT is still running.
      if (v._emitSeq === _lastUploadSeq) return;
      _lastUploadSeq = v._emitSeq;
      if (_pendingBlob) {
        _doUpload(v._uploadUrl, v._emitSeq);
      } else {
        // No blob to upload (e.g., cleared while awaiting URL); abandon this
        // slot so the queue doesn't deadlock if _processing was somehow still true.
        _processing = false;
        _processNext();
      }
      return;
    }

    if (v.state === "processed") {
      // Guard against double delivery: Python echoes "processed" on duplicate seqs
      // (to prevent cross-reload deadlocks) and set_parameter_value +
      // publish_update_to_parameter can both trigger handleUpdate for the same event.
      if (v._emitSeq === _lastProcessedSeq) return;
      _lastProcessedSeq = v._emitSeq;
      pendingThumbs = pendingThumbs.slice(1);
      galleryItems  = v.gallery_items  ?? galleryItems;
      selectedIndex = v.selected_index ?? (galleryItems.length - 1);
      renderThumbs();
      _processing = false;
      _processNext();
      return;
    }

    if (v.state === "idle") {
      captureQueue.length = 0;
      _processing = false;
      _pendingBlob = null;
      _lastProcessedSeq = -1;
      _lastUploadSeq = -1;
      galleryItems  = [];
      selectedIndex = -1;
      pendingThumbs = [];
      renderThumbs();
      return;
    }
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  function cleanup(): void {
    navigator.mediaDevices.removeEventListener("devicechange", populateDevices);
    stream?.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    wrapper.remove();
    delete container._wcInst;
  }

  // ── Wire up & init ────────────────────────────────────────────────────────

  navigator.mediaDevices.addEventListener("devicechange", populateDevices);

  container._wcInst = { handleUpdate, cleanup, wrapper };

  renderThumbs();
  startCamera();
  render();
  return { cleanup, update: handleUpdate };
}
