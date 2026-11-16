import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import type { CSSProperties } from "react";

import { Icon } from "./_icons";
import type { GalleryItem, WebcamCaptureLiveValues, WebcamCaptureValue, WebcamGallery } from "./_types";
import type { WidgetProps } from "@griptape-ai/nodes-widget-api";
import { reactWidget } from "@griptape-ai/nodes-widget-api/react";

export type {
  GalleryItem, WebcamAccepted, WebcamCaptureLiveValues, WebcamCaptureValue, WebcamClearGallery, WebcamGallery,
  WebcamIdle, WebcamProcessed, WebcamRequestingUploadUrl, WebcamSelected, WebcamSelectionConfirmed,
  WebcamUploadReady,
} from "./_types";

export type WebcamCaptureProps = WidgetProps<WebcamCaptureValue, WebcamCaptureLiveValues>;

// Ideal capture size. The browser picks the highest resolution the camera
// supports up to this, so 4K cameras deliver 4K and lesser ones fall back.
const IDEAL_WIDTH = 4096;
const IDEAL_HEIGHT = 2160;
const JPEG_QUALITY = 0.95;

function injectStyles(): void {
  if (document.getElementById("wc-styles")) return;
  const style = document.createElement("style");
  style.id = "wc-styles";
  style.textContent =
    "@keyframes wc-spin{to{transform:rotate(360deg)}}" +
    ".wc-thumb{width:106px;height:80px;object-fit:cover;border-radius:6px;cursor:pointer;" +
    "flex-shrink:0;border:3px solid transparent;transition:border-color 0.15s;display:block;}" +
    ".wc-thumb.selected{border-color:var(--primary);}";
  document.head.append(style);
}

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

const stopPropagation = (e: { stopPropagation: () => void }) => e.stopPropagation();
const stopHandlers = { onPointerDown: stopPropagation, onMouseDown: stopPropagation };

// ── Gallery state ─────────────────────────────────────────────────────────────

interface GalleryState {
  items: GalleryItem[];
  selectedIndex: number;
  /** Data URLs of captures the node hasn't processed yet, oldest first. */
  pending: string[];
}

type GalleryAction =
  | { type: "captured"; dataUrl: string }
  | { type: "selected"; index: number }
  | { type: "processed"; items?: GalleryItem[]; selectedIndex?: number }
  | { type: "cleared" }
  | { type: "reset" };

function galleryReducer(state: GalleryState, action: GalleryAction): GalleryState {
  switch (action.type) {
    case "captured":
      return { ...state, pending: [...state.pending, action.dataUrl] };
    case "selected":
      return { ...state, selectedIndex: action.index };
    case "processed": {
      const items = action.items ?? state.items;
      return { items, selectedIndex: action.selectedIndex ?? (items.length - 1), pending: state.pending.slice(1) };
    }
    case "cleared":
      return { ...state, pending: [] };
    case "reset":
      return { items: [], selectedIndex: -1, pending: [] };
  }
}

// Any state the node sends may carry the gallery, so read it from whichever one arrived.
function initialGallery(value: WebcamCaptureValue | null | undefined): GalleryState {
  const init = (value ?? {}) as Partial<WebcamGallery>;
  const items = init.gallery_items ?? [];
  return { items, selectedIndex: init.selected_index ?? (items.length > 0 ? items.length - 1 : -1), pending: [] };
}

interface Camera {
  deviceId: string;
  label: string;
}

// ── Styles ────────────────────────────────────────────────────────────────────

const wrapperStyle: CSSProperties = {
  display: "flex", flexDirection: "column", height: "100%", padding: "8px", boxSizing: "border-box", gap: "6px",
};
const previewStyle: CSSProperties = {
  position: "relative", background: "#111", borderRadius: "8px",
  flex: "1 1 0", minHeight: "160px", overflow: "hidden",
  display: "flex", alignItems: "center", justifyContent: "center",
};
const placeholderStyle: CSSProperties = {
  position: "absolute", color: "var(--muted-foreground)", fontSize: "13px", textAlign: "center",
  padding: "24px", userSelect: "none",
};
const videoStyle: CSSProperties = { width: "100%", height: "100%", objectFit: "contain", display: "block" };
// Shutter button — circular, overlaid at bottom center
const shutterStyle: CSSProperties = {
  position: "absolute", bottom: "12px", left: "50%",
  width: "52px", height: "52px", borderRadius: "50%",
  border: "3px solid rgba(255,255,255,0.85)", background: "rgba(20,20,20,0.55)",
  cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
  backdropFilter: "blur(4px)", color: "white", transition: "transform 0.1s,opacity 0.15s",
};
const thumbStripStyle: CSSProperties = {
  display: "flex", gap: "6px", overflowX: "auto", padding: "2px 0", scrollbarWidth: "thin", minHeight: "86px",
};
const pendingThumbStyle: CSSProperties = {
  position: "relative", width: "106px", height: "80px", flexShrink: 0, borderRadius: "6px",
  overflow: "hidden", border: "3px solid transparent",
};
const pendingImageStyle: CSSProperties = {
  width: "100%", height: "100%", objectFit: "cover", display: "block", opacity: 0.35,
};
const spinnerOverlayStyle: CSSProperties = {
  position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center",
  background: "rgba(0,0,0,0.35)",
};
const spinnerStyle: CSSProperties = {
  width: "22px", height: "22px", border: "2px solid rgba(255,255,255,0.2)", borderTopColor: "white",
  borderRadius: "50%", animation: "wc-spin 0.75s linear infinite",
};
const bottomRowStyle: CSSProperties = {
  display: "flex", alignItems: "center", gap: "6px", color: "var(--muted-foreground)",
};
const selectStyle: CSSProperties = {
  flex: "1", minWidth: "0", padding: "4px 6px", borderRadius: "5px",
  border: "1px solid var(--border)", background: "var(--background)",
  color: "var(--foreground)", fontSize: "12px", cursor: "pointer",
};
const clearStyle: CSSProperties = {
  display: "flex", alignItems: "center", gap: "5px",
  padding: "4px 10px", borderRadius: "5px",
  border: "1px solid var(--destructive)", background: "transparent",
  color: "var(--destructive)", fontSize: "12px", cursor: "pointer",
  whiteSpace: "nowrap", flexShrink: 0,
};

// ── Component ─────────────────────────────────────────────────────────────────

function WebcamCapture({ value, onChange }: WebcamCaptureProps) {
  const [gallery, dispatch] = useReducer(galleryReducer, value, initialGallery);
  const [streaming, setStreaming] = useState(false);
  const [unavailable, setUnavailable] = useState(false);
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [selectedCamera, setSelectedCamera] = useState("");
  // The camera the stream is opened on; empty lets the browser pick.
  const [requestedCamera, setRequestedCamera] = useState("");
  const [shutterPressed, setShutterPressed] = useState(false);

  const videoRef = useRef<HTMLVideoElement>(null);
  const thumbStripRef = useRef<HTMLDivElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const onChangeRef = useRef(onChange);
  const emitSeq = useRef(0);

  // Queue of Blob values waiting to enter the upload protocol.
  // Only one capture flows through requesting_upload_url → processed at a time
  // because all captures share a single snapshot parameter channel — concurrent
  // in-flight requests would race on upload_ready delivery.
  const captureQueue = useRef<Blob[]>([]);
  const processing = useRef(false);
  const pendingBlob = useRef<Blob | null>(null); // blob for the capture currently in-flight
  const lastProcessedSeq = useRef(-1); // dedupes double delivery of "processed" messages
  const lastUploadSeq = useRef(-1);    // dedupes double delivery of "upload_ready" messages
  // The value last acted on; the one the widget mounted with only seeds the gallery.
  const handledValue = useRef(value);

  useEffect(() => {
    onChangeRef.current = onChange;
  }, [onChange]);

  useEffect(injectStyles, []);

  // ── Camera ───────────────────────────────────────────────────────────────

  const populateDevices = useCallback(async (): Promise<void> => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const found = devices
        .filter((d) => d.kind === "videoinput")
        .map((d, i) => ({ deviceId: d.deviceId, label: d.label || `Camera ${i + 1}` }));
      const open = streamRef.current?.getVideoTracks()[0]?.getSettings?.()?.deviceId;
      setCameras(found);
      setSelectedCamera((current) =>
        open || (found.some((c) => c.deviceId === current) ? current : (found[0]?.deviceId ?? "")));
    } catch { /* permission not yet granted */ }
  }, []);

  useEffect(() => {
    const video = videoRef.current!;
    let active = true;
    const videoConstraints: MediaTrackConstraints = {
      width: { ideal: IDEAL_WIDTH },
      height: { ideal: IDEAL_HEIGHT },
    };
    if (requestedCamera) videoConstraints.deviceId = { exact: requestedCamera };
    navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false }).then(
      (s) => {
        if (!active) {
          s.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = s;
        video.srcObject = s;
        setStreaming(true);
        return populateDevices();
      },
      () => { if (active) setUnavailable(true); },
    );
    return () => {
      active = false;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      video.srcObject = null;
      setStreaming(false);
    };
  }, [requestedCamera, populateDevices]);

  useEffect(() => {
    const media = navigator.mediaDevices;
    media.addEventListener("devicechange", populateDevices);
    return () => media.removeEventListener("devicechange", populateDevices);
  }, [populateDevices]);

  // Scroll newest (rightmost) into view
  useEffect(() => {
    thumbStripRef.current?.lastElementChild?.scrollIntoView({ block: "nearest", inline: "end" });
  }, [gallery]);

  // ── Capture (upload protocol) ─────────────────────────────────────────────

  function processNext(): void {
    if (processing.current || captureQueue.current.length === 0) return;
    processing.current = true;
    pendingBlob.current = captureQueue.current.shift()!;
    emitSeq.current++;
    onChange({ state: "requesting_upload_url", _emitSeq: emitSeq.current });
  }

  function upload(uploadUrl: string, seq: number): void {
    const blob = pendingBlob.current;
    pendingBlob.current = null;
    fetch(uploadUrl, { method: "PUT", body: blob })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); })
      .then(() => { onChangeRef.current({ state: "accepted", _emitSeq: seq }); })
      .catch(() => {
        // On PUT failure, still send accepted so Python echoes "processed"
        // and the pending thumbnail clears rather than staying stuck forever.
        onChangeRef.current({ state: "accepted", _emitSeq: seq });
      });
  }

  function capture(): void {
    const video = videoRef.current;
    if (!streamRef.current || !video) return;

    const canvas = document.createElement("canvas");
    canvas.width  = video.videoWidth  || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d")!.drawImage(video, 0, 0, canvas.width, canvas.height);

    // One encode — reuse for both thumbnail and binary upload
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    captureQueue.current.push(dataUrlToBlob(dataUrl)); // base64→binary in memory, no re-encode
    dispatch({ type: "captured", dataUrl });
    processNext();
  }

  function selectThumbnail(index: number): void {
    dispatch({ type: "selected", index });
    emitSeq.current++;
    onChange({ state: "selected", selected_index: index, gallery_items: gallery.items, _emitSeq: emitSeq.current });
  }

  function clearGallery(): void {
    captureQueue.current = [];
    processing.current = false;
    pendingBlob.current = null;
    dispatch({ type: "cleared" });
    emitSeq.current++;
    onChange({ state: "clear_gallery", _emitSeq: emitSeq.current });
  }

  // The node's replies arrive as new values.
  useEffect(() => {
    if (value === handledValue.current) return;
    handledValue.current = value;
    const v: WebcamCaptureValue | { state?: undefined } = value ?? {};

    if (v.state === "upload_ready" && v._uploadUrl) {
      // Dedupe double delivery, same as "processed" below. A repeat of this seq
      // would fall through to the else branch and release the in-flight slot
      // while the PUT is still running.
      if (v._emitSeq === lastUploadSeq.current) return;
      lastUploadSeq.current = v._emitSeq;
      if (pendingBlob.current) {
        upload(v._uploadUrl, v._emitSeq);
      } else {
        // No blob to upload (e.g., cleared while awaiting URL); abandon this
        // slot so the queue doesn't deadlock if processing was somehow still true.
        processing.current = false;
        processNext();
      }
      return;
    }

    if (v.state === "processed") {
      // Guard against double delivery: Python echoes "processed" on duplicate seqs
      // (to prevent cross-reload deadlocks) and set_parameter_value +
      // publish_update_to_parameter can both deliver the same event.
      if (v._emitSeq === lastProcessedSeq.current) return;
      lastProcessedSeq.current = v._emitSeq;
      dispatch({ type: "processed", items: v.gallery_items, selectedIndex: v.selected_index });
      processing.current = false;
      processNext();
      return;
    }

    if (v.state === "idle") {
      captureQueue.current = [];
      processing.current = false;
      pendingBlob.current = null;
      lastProcessedSeq.current = -1;
      lastUploadSeq.current = -1;
      dispatch({ type: "reset" });
    }
  }, [value]);

  // ── Render ───────────────────────────────────────────────────────────────

  const { items, selectedIndex, pending } = gallery;

  return (
    <div className="nodrag nowheel" style={wrapperStyle}>
      {/* Preview area */}
      <div style={previewStyle}>
        <div style={placeholderStyle} hidden={streaming}>
          {unavailable ? "Camera unavailable — check browser permissions." : "Starting camera…"}
        </div>
        <video ref={videoRef} style={videoStyle} hidden={!streaming} autoPlay muted playsInline />
        <button
          style={{
            ...shutterStyle,
            transform: shutterPressed ? "translateX(-50%) scale(0.88)" : "translateX(-50%)",
            opacity: streaming ? 1 : 0.4,
          }}
          disabled={!streaming}
          onPointerDown={(e) => { e.stopPropagation(); setShutterPressed(true); }}
          onPointerUp={() => setShutterPressed(false)}
          onPointerLeave={() => setShutterPressed(false)}
          onMouseDown={stopPropagation}
          onClick={capture}
        >
          <Icon name="camera" size={22} />
        </button>
      </div>
      {/* Photo strip — oldest left, newest right */}
      <div
        ref={thumbStripRef}
        style={thumbStripStyle}
        hidden={items.length === 0 && pending.length === 0}
        {...stopHandlers}
      >
        {items.map((item, idx) => (
          <img
            key={idx}
            className={"wc-thumb" + (idx === selectedIndex ? " selected" : "")}
            src={item.url || ""}
            title={`Photo ${idx + 1}`}
            onClick={() => selectThumbnail(idx)}
            {...stopHandlers}
          />
        ))}
        {/* Pending items on the right — image underneath, spinner on top */}
        {pending.map((dataUrl, idx) => (
          <div key={`pending-${idx}`} style={pendingThumbStyle}>
            <img style={pendingImageStyle} src={dataUrl} />
            <div style={spinnerOverlayStyle}>
              <div style={spinnerStyle} />
            </div>
          </div>
        ))}
      </div>
      {/* Bottom row: camera selector + clear */}
      <div style={bottomRowStyle}>
        <Icon name="camera" size={15} />
        <select
          style={selectStyle}
          value={selectedCamera}
          onChange={(e) => {
            setSelectedCamera(e.target.value);
            setRequestedCamera(e.target.value);
          }}
          {...stopHandlers}
        >
          {cameras.map((c, i) => <option key={i} value={c.deviceId}>{c.label}</option>)}
        </select>
        <button style={clearStyle} hidden={items.length === 0} onClick={clearGallery} {...stopHandlers}>
          <Icon name="trash-2" size={13} />
          <span>{`Clear (${items.length})`}</span>
        </button>
      </div>
    </div>
  );
}

export default reactWidget(WebcamCapture);
