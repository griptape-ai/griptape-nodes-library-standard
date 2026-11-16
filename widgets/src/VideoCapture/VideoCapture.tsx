import { useEffect, useRef, useState } from "react";
import type { CSSProperties, PointerEvent as ReactPointerEvent, SyntheticEvent } from "react";

import { reactWidget } from "@griptape-ai/nodes-widget-api/react";
import type { WidgetProps } from "@griptape-ai/nodes-widget-api";

import { Icon } from "./_icons";
import type { IconName } from "./_icons";
import type { VideoCaptureLiveValues, VideoCaptureValue } from "./_types";

export type {
  VideoCaptureAccepted, VideoCaptureError, VideoCaptureIdle, VideoCaptureLiveValues, VideoCaptureProcessed,
  VideoCaptureRequestingUploadUrl, VideoCaptureUploadReady, VideoCaptureValue,
} from "./_types";

export type VideoCaptureProps = WidgetProps<VideoCaptureValue, VideoCaptureLiveValues>;

const PREFERRED_MIMES = [
  "video/mp4;codecs=avc1,mp4a.40.2",
  "video/mp4;codecs=avc1",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];

function getSupportedMime(): string {
  if (typeof MediaRecorder === "undefined") return "video/mp4";
  for (const t of PREFERRED_MIMES) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/mp4";
}

function fmtTime(s: number): string {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function cameraErrorMessage(err: unknown): string {
  const name = (err as { name?: string } | null | undefined)?.name;
  switch (name) {
    case "NotAllowedError":
    case "SecurityError":
      return "Camera access is blocked. Allow camera access in your browser or system privacy settings, then retry.";
    case "NotFoundError":
    case "OverconstrainedError":
      return "No camera found. Connect a camera, then retry.";
    case "NotReadableError":
    case "AbortError":
      return "Camera is in use by another app. Close it, then retry.";
    default:
      return `Camera unavailable (${name || "unknown error"}).`;
  }
}

// Where the camera view is: opening the camera, unable to open it, showing the live stream,
// recording it, or reviewing the finished recording.
type Phase =
  | { kind: "starting" }
  | { kind: "failed"; message: string }
  | { kind: "streaming" }
  | { kind: "recording" }
  | { kind: "review" };

const STARTING: Phase = { kind: "starting" };
const STREAMING: Phase = { kind: "streaming" };
const RECORDING: Phase = { kind: "recording" };
const REVIEW: Phase = { kind: "review" };

// The overlay over an accepted recording: a spinner while it uploads, or the node's error.
interface EncodingOverlay {
  error?: string;
}

interface DeviceLists {
  video: MediaDeviceInfo[];
  audio: MediaDeviceInfo[];
  videoId: string;
  audioId: string;
}

// The selector's previous choice if the device is still listed, else the first device.
function keepSelection(previous: string, list: MediaDeviceInfo[]): string {
  if (previous && list.some((d) => d.deviceId === previous)) return previous;
  return list[0]?.deviceId ?? "";
}

// Media handles and handshake bookkeeping the async camera, recorder, and upload code shares.
interface MediaSession {
  stream: MediaStream | null;
  recorder: MediaRecorder | null;
  chunks: Blob[];
  blob: Blob | null;
  blobUrl: string | null;
  // The devices the camera is (re)opened on; empty means the browser's default.
  videoId: string;
  audioId: string;
  // Bumped by each camera start and by unmount, so a superseded getUserMedia stops its tracks.
  startSeq: number;
  // Guards against duplicate upload_ready deliveries within one accept.
  uploading: boolean;
  emitSeq: number;
  // Set on unmount, so a recorder stop or upload that lands afterwards touches nothing.
  unmounted: boolean;
}

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

const deviceSelectStyle: CSSProperties = {
  flex: "1", minWidth: "0", padding: "4px 6px", borderRadius: "5px",
  border: "1px solid var(--border)", background: "var(--background)",
  color: "var(--foreground)", fontSize: "12px", cursor: "pointer",
};

function DeviceSelect({ devices, value, disabled, onChange }: {
  devices: MediaDeviceInfo[];
  value: string;
  disabled: boolean;
  onChange: (deviceId: string) => void;
}) {
  return (
    <select
      style={deviceSelectStyle}
      value={value}
      disabled={disabled}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onChange={(e) => onChange(e.target.value)}
    >
      {devices.map((d, i) => (
        <option key={i} value={d.deviceId}>{d.label || `Device ${i + 1}`}</option>
      ))}
    </select>
  );
}

const overlayButtonStyle: CSSProperties = {
  display: "flex", alignItems: "center", gap: "6px",
  padding: "6px 14px", borderRadius: "6px", border: "none",
  cursor: "pointer", fontSize: "13px", fontWeight: "500",
  background: "rgba(0,0,0,0.6)", color: "white", backdropFilter: "blur(4px)",
};

function OverlayButton({ icon, label, style, hidden, disabled, onClick }: {
  icon: IconName;
  label: string;
  style?: CSSProperties;
  hidden?: boolean;
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      style={{ ...overlayButtonStyle, ...style }}
      hidden={hidden}
      disabled={disabled}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onClick={onClick}
    >
      <Icon name={icon} size={14} />
      <span>{label}</span>
    </button>
  );
}

const SPIN_KEYFRAMES = "@keyframes vc-spin{to{transform:rotate(360deg)}}";

const wrapperStyle: CSSProperties = {
  display: "flex", flexDirection: "column", height: "100%", padding: "8px", boxSizing: "border-box",
};
const viewStyle: CSSProperties = {
  position: "relative", background: "#111", borderRadius: "8px",
  flex: "1 1 0", minHeight: "180px", overflow: "hidden",
  display: "flex", alignItems: "center", justifyContent: "center",
};
const placeholderStyle: CSSProperties = {
  position: "absolute", color: "var(--muted-foreground)", fontSize: "13px", textAlign: "center",
  padding: "24px", userSelect: "none",
};
const videoStyle: CSSProperties = { width: "100%", height: "100%", objectFit: "contain", display: "block" };
const recBadgeStyle: CSSProperties = {
  position: "absolute", top: "10px", left: "10px",
  background: "rgba(200,0,0,0.85)", color: "#fff",
  fontSize: "11px", fontWeight: "700", padding: "2px 8px",
  borderRadius: "4px", fontFamily: "monospace", pointerEvents: "none",
};
const micBadgeStyle: CSSProperties = {
  position: "absolute", top: "10px", right: "10px",
  background: "rgba(0,0,0,0.6)", color: "#fff",
  fontSize: "11px", padding: "2px 8px",
  borderRadius: "4px", border: "none", cursor: "pointer",
};
const encodingOverlayStyle: CSSProperties = {
  position: "absolute", inset: "0", zIndex: "10",
  background: "rgba(0,0,0,0.75)",
  display: "flex", flexDirection: "column",
  alignItems: "center", justifyContent: "center", gap: "10px",
  color: "white", fontSize: "13px", fontWeight: "500",
  borderRadius: "8px",
};
const spinnerStyle: CSSProperties = {
  width: "28px", height: "28px", border: "3px solid rgba(255,255,255,0.25)", borderTopColor: "white",
  borderRadius: "50%", animation: "vc-spin 0.75s linear infinite",
};
const recordBtnStyle: CSSProperties = {
  position: "absolute", bottom: "12px", left: "50%",
  width: "52px", height: "52px", borderRadius: "50%",
  border: "3px solid rgba(255,255,255,0.85)", background: "rgba(20,20,20,0.55)",
  cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
  backdropFilter: "blur(4px)", transition: "transform 0.1s,opacity 0.15s",
};
const innerDotStyle: CSSProperties = {
  width: "20px", height: "20px", borderRadius: "50%", background: "#e53e3e",
  transition: "border-radius 0.2s,width 0.2s,height 0.2s",
};
// The record button's red dot becomes a stop square while recording.
const stopSquareStyle: CSSProperties = { ...innerDotStyle, width: "18px", height: "18px", borderRadius: "3px" };
const reviewOverlayStyle: CSSProperties = {
  position: "absolute", bottom: "12px", left: "0", right: "0",
  display: "flex", justifyContent: "center", gap: "8px", flexWrap: "wrap",
};
const retryBtnStyle: CSSProperties = { position: "absolute", bottom: "12px", left: "50%", transform: "translateX(-50%)" };
const deviceRowStyle: CSSProperties = {
  display: "flex", alignItems: "center", gap: "6px", marginTop: "6px", transition: "opacity 0.15s",
  color: "var(--muted-foreground)",
};

function VideoCapture({ value, onChange }: VideoCaptureProps) {
  const [mime] = useState(getSupportedMime);
  const [phase, setPhase] = useState<Phase>(STARTING);
  const [micUnavailable, setMicUnavailable] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [paused, setPaused] = useState(true);
  const [pressed, setPressed] = useState(false);
  const [overlay, setOverlay] = useState<EncodingOverlay | null>(null);
  const [devices, setDevices] = useState<DeviceLists>({ video: [], audio: [], videoId: "", audioId: "" });

  const videoRef = useRef<HTMLVideoElement>(null);
  const session = useRef<MediaSession>({
    stream: null, recorder: null, chunks: [], blob: null, blobUrl: null,
    videoId: "", audioId: "", startSeq: 0, uploading: false, emitSeq: 0, unmounted: false,
  });
  const onChangeRef = useRef(onChange);
  useEffect(() => {
    onChangeRef.current = onChange;
  });

  const streaming = phase.kind === "streaming" || phase.kind === "recording";
  const recording = phase.kind === "recording";
  const reviewing = phase.kind === "review";
  const failed = phase.kind === "failed";
  const devicesLocked = recording || reviewing;
  const busy = overlay !== null;

  // ── Camera / Recording ───────────────────────────────────────────────────

  async function populateDevices(): Promise<void> {
    try {
      const list = await navigator.mediaDevices.enumerateDevices();
      const s = session.current;
      const vs = s.stream?.getVideoTracks()[0]?.getSettings?.()?.deviceId;
      const as = s.stream?.getAudioTracks()[0]?.getSettings?.()?.deviceId;
      if (vs) s.videoId = vs;
      if (as) s.audioId = as;
      const video = list.filter((d) => d.kind === "videoinput");
      const audio = list.filter((d) => d.kind === "audioinput");
      setDevices((prev) => ({
        video,
        audio,
        videoId: vs || keepSelection(prev.videoId, video),
        audioId: as || keepSelection(prev.audioId, audio),
      }));
    } catch { /* permission not yet granted */ }
  }

  // getUserMedia fails the whole request if any track fails, so a missing or
  // blocked microphone would otherwise take the camera down with it. Fall back
  // to a video-only stream before reporting the camera as unavailable.
  async function startCamera(): Promise<void> {
    const s = session.current;
    const seq = ++s.startSeq;
    const videoConstraint = s.videoId ? { deviceId: { exact: s.videoId } } : true;
    const audioConstraint = s.audioId ? { deviceId: { exact: s.audioId } } : true;
    setPhase(STARTING);

    let stream: MediaStream | null = null;
    let audioMissing = false;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: audioConstraint });
    } catch (avErr) {
      if (seq !== s.startSeq) return;
      try {
        // A previously selected microphone may have been unplugged; try the default one first.
        if (s.audioId) {
          stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: true }).catch(() => null);
        }
        if (!stream) {
          stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
          audioMissing = true;
          console.warn("VideoCapture: microphone unavailable, recording video only:", avErr);
        }
      } catch (vErr) {
        console.warn("VideoCapture: camera unavailable:", vErr);
        if (seq !== s.startSeq) return;
        setPhase({ kind: "failed", message: cameraErrorMessage(vErr) });
        await populateDevices();
        return;
      }
    }

    if (seq !== s.startSeq) { stream.getTracks().forEach((t) => t.stop()); return; }
    s.stream = stream;
    const video = videoRef.current!;
    video.srcObject = stream;
    video.muted = true;
    setMicUnavailable(audioMissing);
    setPhase(STREAMING);
    await populateDevices();
  }

  async function switchDevices(videoId: string, audioId: string): Promise<void> {
    if (devicesLocked) return;
    const s = session.current;
    s.videoId = videoId;
    s.audioId = audioId;
    s.stream?.getTracks().forEach((t) => t.stop());
    s.stream = null;
    await startCamera();
  }

  // A previously selected camera may have been unplugged, so retry with the defaults.
  function retryCamera(): void {
    session.current.videoId = "";
    session.current.audioId = "";
    void startCamera();
  }

  function startRecording(): void {
    const s = session.current;
    if (!s.stream) return;
    s.chunks = [];
    setElapsed(0);
    const recorder = new MediaRecorder(s.stream, { mimeType: mime });
    recorder.ondataavailable = (e) => { if (e.data?.size > 0) s.chunks.push(e.data); };
    recorder.onstop = onStop;
    recorder.start(100);
    s.recorder = recorder;
    setPhase(RECORDING);
  }

  function stopRecording(): void {
    const recorder = session.current.recorder;
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  function onStop(): void {
    const s = session.current;
    if (s.unmounted) return;
    s.blob = new Blob(s.chunks, { type: mime });
    if (s.blobUrl) URL.revokeObjectURL(s.blobUrl);
    s.blobUrl = URL.createObjectURL(s.blob);
    s.stream?.getTracks().forEach((t) => t.stop());
    s.stream = null;
    const video = videoRef.current!;
    video.srcObject = null;
    video.src = s.blobUrl;
    video.muted = false;
    video.loop = false;
    video.play().catch(() => {});
    setPhase(REVIEW);
  }

  function accept(): void {
    const s = session.current;
    if (!s.blob) return;
    setOverlay({});
    // `uploading` only guards against duplicate upload_ready deliveries within
    // one accept. Clear it here so a second Accept on the same recording uploads.
    s.uploading = false;
    requestAnimationFrame(() => {
      s.emitSeq++;
      onChangeRef.current({ state: "requesting_upload_url", _mime: mime, _emitSeq: s.emitSeq });
    });
  }

  function doUpload(uploadUrl: string): void {
    const s = session.current;
    if (s.uploading || !s.blob) return;
    s.uploading = true;
    fetch(uploadUrl, { method: "PUT", body: s.blob })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); })
      .then(() => { onChangeRef.current({ state: "accepted", _emitSeq: s.emitSeq }); })
      .catch(() => { s.uploading = false; setOverlay(null); });
  }

  function discard(): void {
    const s = session.current;
    s.uploading = false;
    const video = videoRef.current!;
    video.src = "";
    video.muted = true;
    if (s.blobUrl) { URL.revokeObjectURL(s.blobUrl); s.blobUrl = null; }
    s.blob = null; s.chunks = [];
    void startCamera();
  }

  function togglePlayPause(): void {
    const video = videoRef.current!;
    if (video.paused || video.ended) { video.currentTime = 0; video.play().catch(() => {}); }
    else video.pause();
  }

  function syncPaused(): void {
    const video = videoRef.current!;
    setPaused(video.paused || video.ended);
  }

  function onRecordPointerDown(e: ReactPointerEvent): void {
    e.stopPropagation();
    setPressed(true);
  }

  // ── Effects ──────────────────────────────────────────────────────────────

  // Opens the camera on mount, follows device changes, and releases every media handle on unmount.
  useEffect(() => {
    const s = session.current;
    const video = videoRef.current!;
    const mediaDevices = navigator.mediaDevices;
    const onDeviceChange = () => { void populateDevices(); };
    mediaDevices.addEventListener("devicechange", onDeviceChange);
    void startCamera();
    return () => {
      s.startSeq++; // a getUserMedia still in flight stops its tracks instead of attaching them
      mediaDevices.removeEventListener("devicechange", onDeviceChange);
      s.unmounted = true;
      if (s.recorder) {
        s.recorder.onstop = null;
        s.recorder.ondataavailable = null;
        if (s.recorder.state !== "inactive") s.recorder.stop();
      }
      s.stream?.getTracks().forEach((t) => t.stop());
      if (s.blobUrl) URL.revokeObjectURL(s.blobUrl);
      video.src = ""; video.srcObject = null;
    };
  }, []);

  // Counts the recording's seconds for the REC badge.
  useEffect(() => {
    if (!recording) return;
    const timer = setInterval(() => setElapsed((n) => n + 1), 1000);
    return () => clearInterval(timer);
  }, [recording]);

  // Follows the node's side of the upload handshake.
  useEffect(() => {
    const v: VideoCaptureValue | { state?: undefined } = value ?? {};
    if (v.state === "processed") { setOverlay(null); return; }
    if (v.state === "error") {
      const message = v.message || "Upload failed.";
      setOverlay((o) => (o ? { error: message } : o));
      const timer = setTimeout(() => setOverlay(null), 4000);
      return () => clearTimeout(timer);
    }
    if (v.state === "upload_ready" && v._uploadUrl) doUpload(v._uploadUrl);
  }, [value]);

  // ── Render ───────────────────────────────────────────────────────────────

  return (
    <div className="nodrag nowheel" style={wrapperStyle}>
      <style>{SPIN_KEYFRAMES}</style>
      <div style={viewStyle}>
        <div style={placeholderStyle} hidden={streaming || reviewing}>
          {phase.kind === "failed" ? phase.message : "Starting camera…"}
        </div>
        <video
          ref={videoRef}
          style={videoStyle}
          hidden={!streaming && !reviewing}
          autoPlay muted playsInline
          onPlay={syncPaused}
          onPause={syncPaused}
          onEnded={syncPaused}
        />
        <div style={recBadgeStyle} hidden={!recording}>{`● REC ${fmtTime(elapsed)}`}</div>
        {/* Shown when the camera streams but the microphone could not be opened; click to retry the mic */}
        <button
          style={micBadgeStyle}
          hidden={!streaming || !micUnavailable}
          disabled={recording}
          onPointerDown={stopPropagation}
          onMouseDown={stopPropagation}
          onClick={() => void switchDevices(devices.videoId, "")}
        >
          No microphone — recording video only (retry)
        </button>
        <div style={encodingOverlayStyle} hidden={!busy}>
          <div style={spinnerStyle} hidden={overlay?.error !== undefined} />
          <span>{overlay?.error ?? "Processing…"}</span>
        </div>
        <button
          style={{
            ...recordBtnStyle,
            transform: pressed ? "translateX(-50%) scale(0.88)" : "translateX(-50%)",
            opacity: streaming ? "1" : "0.4",
          }}
          hidden={reviewing || failed}
          disabled={!streaming}
          onPointerDown={onRecordPointerDown}
          onPointerUp={() => setPressed(false)}
          onPointerLeave={() => setPressed(false)}
          onMouseDown={stopPropagation}
          onClick={recording ? stopRecording : startRecording}
        >
          <div style={recording ? stopSquareStyle : innerDotStyle} />
        </button>
        <div style={reviewOverlayStyle} hidden={!reviewing}>
          <OverlayButton
            icon={paused ? "play" : "pause"} label={paused ? "Play" : "Pause"}
            disabled={busy} onClick={togglePlayPause}
          />
          <OverlayButton icon="check" label="Accept" disabled={busy} onClick={accept} />
          <OverlayButton icon="rotate-ccw" label="Re-record" disabled={busy} onClick={discard} />
        </div>
        <OverlayButton icon="rotate-ccw" label="Retry" style={retryBtnStyle} hidden={!failed} onClick={retryCamera} />
      </div>
      <div
        style={{ ...deviceRowStyle, opacity: devicesLocked ? "0.4" : "1", pointerEvents: devicesLocked ? "none" : undefined }}
      >
        <Icon name="camera" size={15} />
        <DeviceSelect
          devices={devices.video} value={devices.videoId} disabled={devicesLocked}
          onChange={(videoId) => {
            setDevices((d) => ({ ...d, videoId }));
            void switchDevices(videoId, devices.audioId);
          }}
        />
        <Icon name="mic" size={15} />
        <DeviceSelect
          devices={devices.audio} value={devices.audioId} disabled={devicesLocked}
          onChange={(audioId) => {
            setDevices((d) => ({ ...d, audioId }));
            void switchDevices(devices.videoId, audioId);
          }}
        />
      </div>
    </div>
  );
}

export default reactWidget(VideoCapture);
