import { Icon } from './_icons';
import type { IconName } from './_icons';
import type { VideoCaptureLiveValues, VideoCaptureValue } from './_types';
import type { WidgetHandle, WidgetProps } from '@griptape-ai/nodes-widget-api';
import type { StyleProp } from '@griptape-ai/nodes-widget-api/jsx-runtime';

export type {
  VideoCaptureAccepted, VideoCaptureError, VideoCaptureIdle, VideoCaptureLiveValues, VideoCaptureProcessed,
  VideoCaptureRequestingUploadUrl, VideoCaptureUploadReady, VideoCaptureValue,
} from './_types';

export type VideoCaptureProps = WidgetProps<VideoCaptureValue, VideoCaptureLiveValues>;

// A mounted widget, kept on its container so a repeated mount updates it instead of building
// a second one.
interface VideoCaptureInstance {
  handleUpdate: (props: VideoCaptureProps) => void;
  cleanup: () => void;
  wrapper: HTMLDivElement;
}
type VideoCaptureContainer = HTMLElement & { _vcInst?: VideoCaptureInstance };

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

function injectStyles(): void {
  if (document.getElementById("vc-styles")) return;
  document.head.append(<style id="vc-styles">{"@keyframes vc-spin{to{transform:rotate(360deg)}}"}</style>);
}

const stopPropagation = (e: Event) => e.stopPropagation();

function DeviceSelect({ onChange }: { onChange: () => void }) {
  return (
    <select
      style={{
        flex: "1", minWidth: "0", padding: "4px 6px", borderRadius: "5px",
        border: "1px solid var(--border)", background: "var(--background)",
        color: "var(--foreground)", fontSize: "12px", cursor: "pointer",
      }}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onChange={onChange}
    />
  );
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

function OverlayButton({ icon, label, style, onClick }: {
  icon: IconName;
  label: string;
  style?: Exclude<StyleProp, string>;
  onClick: () => void;
}) {
  return (
    <button
      style={{
        display: "flex", alignItems: "center", gap: "6px",
        padding: "6px 14px", borderRadius: "6px", border: "none",
        cursor: "pointer", fontSize: "13px", fontWeight: "500",
        background: "rgba(0,0,0,0.6)", color: "white", backdropFilter: "blur(4px)",
        ...style,
      }}
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onClick={onClick}
    >
      <Icon name={icon} size={14} />
      <span>{label}</span>
    </button>
  );
}

export default function VideoCapture(
  host: HTMLElement,
  props: VideoCaptureProps,
): WidgetHandle<VideoCaptureProps> {
  const container = host as VideoCaptureContainer;
  if (container._vcInst?.wrapper?.isConnected) {
    container._vcInst.handleUpdate(props);
    return { cleanup: container._vcInst.cleanup, update: container._vcInst.handleUpdate };
  }

  injectStyles();

  let { onChange } = props;
  let _emitSeq = 0;

  let isStreaming  = false;
  let isRecording  = false;
  let hasRecording = false;

  let _uploading = false;
  let stream: MediaStream | null = null, recorder: MediaRecorder | null = null, chunks: Blob[] = [];
  let blob: Blob | null = null, blobUrl: string | null = null;
  let elapsed = 0, timer: ReturnType<typeof setInterval> | undefined;
  const mime = getSupportedMime();

  let currentVideoId = "";
  let currentAudioId = "";
  let cameraFailed = false;
  let micUnavailable = false;
  let _startSeq = 0;

  // ── DOM ──────────────────────────────────────────────────────────────────

  const releaseRecordBtn = () => { recordBtn.style.transform = "translateX(-50%)"; };

  const video = (
    <video
      style="width:100%;height:100%;object-fit:contain;display:block;"
      autoplay muted playsInline
      onPlay={syncPlayPauseBtn}
      onPause={syncPlayPauseBtn}
      onEnded={syncPlayPauseBtn}
    />
  ) as HTMLVideoElement;

  const placeholder = (
    <div style="position:absolute;color:var(--muted-foreground);font-size:13px;text-align:center;padding:24px;user-select:none;">
      Starting camera…
    </div>
  ) as HTMLDivElement;

  // REC badge
  const recBadge = (
    <div
      style={{
        position: "absolute", top: "10px", left: "10px",
        background: "rgba(200,0,0,0.85)", color: "#fff",
        fontSize: "11px", fontWeight: "700", padding: "2px 8px",
        borderRadius: "4px", fontFamily: "monospace", pointerEvents: "none",
      }}
      hidden
    />
  ) as HTMLDivElement;

  // Shown when the camera streams but the microphone could not be opened; click to retry the mic
  const micBadge = (
    <button
      style={{
        position: "absolute", top: "10px", right: "10px",
        background: "rgba(0,0,0,0.6)", color: "#fff",
        fontSize: "11px", padding: "2px 8px",
        borderRadius: "4px", border: "none", cursor: "pointer",
      }}
      hidden
      onPointerDown={stopPropagation}
      onMouseDown={stopPropagation}
      onClick={() => switchDevices({ defaultMic: true })}
    >
      No microphone — recording video only (retry)
    </button>
  ) as HTMLButtonElement;

  const retryBtn = (
    <OverlayButton
      icon="rotate-ccw" label="Retry"
      style={{ position: "absolute", bottom: "12px", left: "50%", transform: "translateX(-50%)" }}
      onClick={retryCamera}
    />
  ) as HTMLButtonElement;

  // Encoding overlay
  const spinner = (
    <div style="width:28px;height:28px;border:3px solid rgba(255,255,255,0.25);border-top-color:white;border-radius:50%;animation:vc-spin 0.75s linear infinite;" />
  ) as HTMLDivElement;
  const encodingOverlay = (
    <div
      style={{
        position: "absolute", inset: "0", zIndex: "10",
        background: "rgba(0,0,0,0.75)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center", gap: "10px",
        color: "white", fontSize: "13px", fontWeight: "500",
        borderRadius: "8px",
      }}
      hidden
    >
      {spinner}
      <span>Processing…</span>
    </div>
  ) as HTMLDivElement;

  // Circular record button — red dot morphs to stop square when recording
  const innerDot = (
    <div style="width:20px;height:20px;border-radius:50%;background:#e53e3e;transition:border-radius 0.2s,width 0.2s,height 0.2s;" />
  ) as HTMLDivElement;
  const recordBtn = (
    <button
      style={{
        position: "absolute", bottom: "12px", left: "50%", transform: "translateX(-50%)",
        width: "52px", height: "52px", borderRadius: "50%",
        border: "3px solid rgba(255,255,255,0.85)", background: "rgba(20,20,20,0.55)",
        cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
        backdropFilter: "blur(4px)", transition: "transform 0.1s,opacity 0.15s",
      }}
      onPointerDown={(e) => { e.stopPropagation(); recordBtn.style.transform = "translateX(-50%) scale(0.88)"; }}
      onPointerUp={releaseRecordBtn}
      onPointerLeave={releaseRecordBtn}
      onMouseDown={stopPropagation}
      onClick={toggleRecord}
    >
      {innerDot}
    </button>
  ) as HTMLButtonElement;

  // Review controls (shown after recording stops)
  const playPauseBtn = (<OverlayButton icon="play"       label="Play"      onClick={togglePlayPause} />) as HTMLButtonElement;
  const acceptBtn    = (<OverlayButton icon="check"      label="Accept"    onClick={accept} />) as HTMLButtonElement;
  const discardBtn   = (<OverlayButton icon="rotate-ccw" label="Re-record" onClick={discard} />) as HTMLButtonElement;

  const reviewOverlay = (
    <div style="position:absolute;bottom:12px;left:0;right:0;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;">
      {playPauseBtn}
      {acceptBtn}
      {discardBtn}
    </div>
  ) as HTMLDivElement;

  // Device selector row
  const videoSel = (<DeviceSelect onChange={() => switchDevices()} />) as HTMLSelectElement;
  const audioSel = (<DeviceSelect onChange={() => switchDevices()} />) as HTMLSelectElement;
  const deviceRow = (
    <div style="display:flex;align-items:center;gap:6px;margin-top:6px;transition:opacity 0.15s;color:var(--muted-foreground);">
      <Icon name="camera" size={15} />
      {videoSel}
      <Icon name="mic" size={15} />
      {audioSel}
    </div>
  ) as HTMLDivElement;

  const wrapper = (
    <div
      class="nodrag nowheel"
      style="display:flex;flex-direction:column;height:100%;padding:8px;box-sizing:border-box;"
    >
      <div
        style={{
          position: "relative", background: "#111", borderRadius: "8px",
          flex: "1 1 0", minHeight: "180px", overflow: "hidden",
          display: "flex", alignItems: "center", justifyContent: "center",
        }}
      >
        {placeholder}
        {video}
        {recBadge}
        {micBadge}
        {encodingOverlay}
        {recordBtn}
        {reviewOverlay}
        {retryBtn}
      </div>
      {deviceRow}
    </div>
  ) as HTMLDivElement;
  container.append(wrapper);

  // ── Play/pause button sync ────────────────────────────────────────────────

  function syncPlayPauseBtn() {
    const isPaused = video.paused || video.ended;
    playPauseBtn.removeChild(playPauseBtn.firstChild!);
    playPauseBtn.insertBefore(<Icon name={isPaused ? "play" : "pause"} size={14} />, playPauseBtn.firstChild);
    playPauseBtn.querySelector("span")!.textContent = isPaused ? "Play" : "Pause";
  }

  // ── Render ───────────────────────────────────────────────────────────────

  function render(): void {
    placeholder.hidden      = isStreaming || hasRecording;
    video.hidden            = !isStreaming && !hasRecording;
    recordBtn.hidden        = hasRecording || cameraFailed;
    recordBtn.disabled      = !isStreaming && !isRecording;
    recordBtn.style.opacity = (isStreaming || isRecording) ? "1" : "0.4";
    reviewOverlay.hidden    = !hasRecording;
    recBadge.hidden         = !isRecording;
    micBadge.hidden         = !isStreaming || !micUnavailable;
    micBadge.disabled       = isRecording;
    retryBtn.hidden         = !cameraFailed || isStreaming || hasRecording;

    const devicesLocked          = isRecording || hasRecording;
    videoSel.disabled            = devicesLocked;
    audioSel.disabled            = devicesLocked;
    deviceRow.style.opacity      = devicesLocked ? "0.4" : "1";
    deviceRow.style.pointerEvents = devicesLocked ? "none" : "";
  }

  // ── Camera / Recording ───────────────────────────────────────────────────

  async function populateDevices(): Promise<void> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      function repopulate(sel: HTMLSelectElement, list: MediaDeviceInfo[]) {
        const current = sel.value;
        sel.innerHTML = "";
        sel.append(...list.map((d, i) => <option value={d.deviceId}>{d.label || `Device ${i + 1}`}</option>));
        if (current && [...sel.options].some((o) => o.value === current)) sel.value = current;
      }
      repopulate(videoSel, devices.filter((d) => d.kind === "videoinput"));
      repopulate(audioSel, devices.filter((d) => d.kind === "audioinput"));
      if (stream) {
        const vs = stream.getVideoTracks()[0]?.getSettings?.()?.deviceId;
        const as = stream.getAudioTracks()[0]?.getSettings?.()?.deviceId;
        if (vs) { currentVideoId = vs; videoSel.value = vs; }
        if (as) { currentAudioId = as; audioSel.value = as; }
      }
    } catch { /* permission not yet granted */ }
  }

  async function switchDevices({ defaultMic = false }: { defaultMic?: boolean } = {}): Promise<void> {
    if (isRecording || hasRecording) return;
    currentVideoId = videoSel.value;
    currentAudioId = defaultMic ? "" : audioSel.value;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    isStreaming = false;
    render();
    await startCamera();
  }

  // getUserMedia fails the whole request if any track fails, so a missing or
  // blocked microphone would otherwise take the camera down with it. Fall back
  // to a video-only stream before reporting the camera as unavailable.
  async function startCamera(): Promise<void> {
    const seq = ++_startSeq;
    const videoConstraint = currentVideoId ? { deviceId: { exact: currentVideoId } } : true;
    const audioConstraint = currentAudioId ? { deviceId: { exact: currentAudioId } } : true;
    cameraFailed = false;
    placeholder.textContent = "Starting camera…";
    render();

    let s: MediaStream | null = null;
    let audioMissing = false;
    try {
      s = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: audioConstraint });
    } catch (avErr) {
      if (seq !== _startSeq) return;
      try {
        // A previously selected microphone may have been unplugged; try the default one first.
        if (currentAudioId) {
          s = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: true }).catch(() => null);
        }
        if (!s) {
          s = await navigator.mediaDevices.getUserMedia({ video: videoConstraint, audio: false });
          audioMissing = true;
          console.warn("VideoCapture: microphone unavailable, recording video only:", avErr);
        }
      } catch (vErr) {
        console.warn("VideoCapture: camera unavailable:", vErr);
        if (seq !== _startSeq) return;
        if (!hasRecording) {
          placeholder.textContent = cameraErrorMessage(vErr);
          cameraFailed = true;
          render();
        }
        await populateDevices();
        return;
      }
    }

    if (seq !== _startSeq || hasRecording) { s.getTracks().forEach((t) => t.stop()); return; }
    stream = s;
    micUnavailable = audioMissing;
    video.srcObject = stream;
    video.muted = true;
    isStreaming = true;
    render();
    await populateDevices();
  }

  // A previously selected camera may have been unplugged, so retry with the defaults.
  function retryCamera(): void {
    currentVideoId = "";
    currentAudioId = "";
    startCamera();
  }

  function startRecording(): void {
    if (!stream) return;
    chunks = [];
    elapsed = 0;
    recBadge.textContent = "● REC 00:00";
    recorder = new MediaRecorder(stream, { mimeType: mime });
    recorder.ondataavailable = (e) => { if (e.data?.size > 0) chunks.push(e.data); };
    recorder.onstop = onStop;
    recorder.start(100);
    isRecording = true;
    // Morph dot to stop square
    innerDot.style.borderRadius = "3px";
    innerDot.style.width = "18px";
    innerDot.style.height = "18px";
    render();
    timer = setInterval(() => { elapsed++; recBadge.textContent = `● REC ${fmtTime(elapsed)}`; }, 1000);
  }

  function stopRecording(): void {
    clearInterval(timer);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  function onStop(): void {
    blob = new Blob(chunks, { type: mime });
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    blobUrl = URL.createObjectURL(blob);
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    video.srcObject = null;
    video.src = blobUrl;
    video.muted = false;
    video.loop = false;
    video.play().catch(() => {});
    isStreaming = false;
    isRecording = false;
    hasRecording = true;
    // Reset dot back to circle for next time
    innerDot.style.borderRadius = "50%";
    innerDot.style.width = "20px";
    innerDot.style.height = "20px";
    render();
  }

  function hideEncodingOverlay(): void {
    encodingOverlay.hidden = true;
    acceptBtn.disabled = false;
    discardBtn.disabled = false;
    playPauseBtn.disabled = false;
  }

  function accept(): void {
    if (!blob) return;
    encodingOverlay.hidden = false;
    acceptBtn.disabled = true;
    discardBtn.disabled = true;
    playPauseBtn.disabled = true;
    // _uploading only guards against duplicate upload_ready deliveries within
    // one accept. Clear it here so a second Accept on the same recording uploads.
    _uploading = false;
    requestAnimationFrame(() => {
      _emitSeq++;
      onChange?.({ state: "requesting_upload_url", _mime: mime, _emitSeq });
    });
  }

  function doUpload(uploadUrl: string): void {
    if (_uploading) return;
    _uploading = true;
    fetch(uploadUrl, { method: "PUT", body: blob })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); })
      .then(() => { onChange?.({ state: "accepted", _emitSeq }); })
      .catch(() => { _uploading = false; hideEncodingOverlay(); });
  }

  function discard(): void {
    _uploading = false;
    video.src = "";
    video.muted = true;
    if (blobUrl) { URL.revokeObjectURL(blobUrl); blobUrl = null; }
    blob = null; chunks = [];
    isStreaming = false; isRecording = false; hasRecording = false;
    placeholder.textContent = "Starting camera…";
    render();
    startCamera();
  }

  function togglePlayPause(): void {
    if (video.paused || video.ended) { video.currentTime = 0; video.play().catch(() => {}); }
    else video.pause();
  }

  function toggleRecord(): void {
    if (isRecording) stopRecording();
    else startRecording();
  }

  // ── handleUpdate ─────────────────────────────────────────────────────────

  function handleUpdate(newProps: VideoCaptureProps): void {
    onChange = newProps.onChange;
    const v: VideoCaptureValue | { state?: undefined } = newProps.value ?? {};
    if (v.state === "processed") { hideEncodingOverlay(); return; }
    if (v.state === "error") {
      spinner.hidden = true;
      encodingOverlay.querySelector("span")!.textContent = v.message || "Upload failed.";
      setTimeout(hideEncodingOverlay, 4000);
      return;
    }
    if (v.state === "upload_ready" && v._uploadUrl) { doUpload(v._uploadUrl); return; }
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  function cleanup(): void {
    _startSeq++; // a getUserMedia still in flight stops its tracks instead of attaching them
    clearInterval(timer);
    navigator.mediaDevices.removeEventListener("devicechange", populateDevices);
    if (recorder && recorder.state !== "inactive") { recorder.onstop = null; recorder.stop(); }
    stream?.getTracks().forEach((t) => t.stop());
    if (blobUrl) URL.revokeObjectURL(blobUrl);
    video.src = ""; video.srcObject = null;
    wrapper.remove();
    delete container._vcInst;
  }

  // ── Wire up & init ────────────────────────────────────────────────────────

  navigator.mediaDevices.addEventListener("devicechange", populateDevices);

  container._vcInst = { handleUpdate, cleanup, wrapper };

  startCamera();
  render();
  return { cleanup, update: handleUpdate };
}
