import { mkIcon } from './_icons.js';

const PREFERRED_MIMES = [
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
  "video/mp4",
];

function getSupportedMime() {
  if (typeof MediaRecorder === "undefined") return "video/webm";
  for (const t of PREFERRED_MIMES) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/webm";
}

function fmtTime(s) {
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

function injectStyles() {
  if (document.getElementById("vc-styles")) return;
  const style = document.createElement("style");
  style.id = "vc-styles";
  style.textContent = "@keyframes vc-spin{to{transform:rotate(360deg)}}";
  document.head.appendChild(style);
}

function mkSelect() {
  const sel = document.createElement("select");
  sel.style.cssText = [
    "flex:1", "min-width:0", "padding:4px 6px", "border-radius:5px",
    "border:1px solid var(--border)", "background:var(--background)",
    "color:var(--foreground)", "font-size:12px", "cursor:pointer",
  ].join(";");
  sel.addEventListener("pointerdown", (e) => e.stopPropagation());
  sel.addEventListener("mousedown",   (e) => e.stopPropagation());
  return sel;
}

function mkBtn(iconName, label, danger) {
  const btn = document.createElement("button");
  btn.style.cssText = [
    "display:flex", "align-items:center", "gap:6px",
    "padding:6px 14px", "border-radius:6px", "border:none",
    "cursor:pointer", "font-size:13px", "font-weight:500",
    "transition:opacity 0.15s",
    danger ? "background:var(--destructive);color:white;"
           : "background:rgba(0,0,0,0.65);color:white;backdrop-filter:blur(4px);",
  ].join(";");
  btn.appendChild(mkIcon(iconName, 14));
  const span = document.createElement("span");
  span.textContent = label;
  btn.appendChild(span);
  btn.addEventListener("pointerdown", (e) => e.stopPropagation());
  btn.addEventListener("mousedown",   (e) => e.stopPropagation());
  return btn;
}

export default function VideoCapture(container, props) {
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

  let stream = null, recorder = null, chunks = [], blob = null, blobUrl = null;
  let elapsed = 0, timer = null;
  const mime = getSupportedMime();

  let currentVideoId = "";
  let currentAudioId = "";

  // ── DOM ──────────────────────────────────────────────────────────────────

  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  // height:100% makes the widget fill the canvas as it's resized
  wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;padding:8px;box-sizing:border-box;";

  // Video area — flex:1 so it grows to fill available height
  const videoWrap = document.createElement("div");
  videoWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:8px",
    "flex:1 1 0", "min-height:180px", "overflow:hidden",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const video = document.createElement("video");
  // object-fit:contain keeps the video aspect ratio; height:100% fills the flex area
  video.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;";
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;

  // Placeholder
  const placeholder = document.createElement("div");
  placeholder.style.cssText = "position:absolute;color:var(--muted-foreground);font-size:13px;text-align:center;padding:24px;user-select:none;";
  placeholder.textContent = "Starting camera…";

  // REC badge (top-left)
  const recBadge = document.createElement("div");
  recBadge.style.cssText = [
    "position:absolute", "top:10px", "left:10px",
    "background:rgba(200,0,0,0.85)", "color:#fff",
    "font-size:11px", "font-weight:700", "padding:2px 8px",
    "border-radius:4px", "font-family:monospace", "pointer-events:none",
  ].join(";");
  recBadge.hidden = true;

  // Encoding overlay — shown from Accept click until Python confirms processing complete
  const encodingOverlay = document.createElement("div");
  encodingOverlay.style.cssText = [
    "position:absolute", "inset:0", "z-index:10",
    "background:rgba(0,0,0,0.75)",
    "display:flex", "flex-direction:column",
    "align-items:center", "justify-content:center", "gap:10px",
    "color:white", "font-size:13px", "font-weight:500",
    "border-radius:8px",
  ].join(";");
  const spinner = document.createElement("div");
  spinner.style.cssText = "width:28px;height:28px;border:3px solid rgba(255,255,255,0.25);border-top-color:white;border-radius:50%;animation:vc-spin 0.75s linear infinite;";
  encodingOverlay.appendChild(spinner);
  encodingOverlay.appendChild(Object.assign(document.createElement("span"), { textContent: "Processing…" }));
  encodingOverlay.hidden = true;

  // Overlay button row (bottom of video)
  const overlay = document.createElement("div");
  overlay.style.cssText = "position:absolute;bottom:10px;left:0;right:0;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;";

  const startBtn    = mkBtn("circle",      "Start Recording", false);
  const stopBtn     = mkBtn("square",      "Stop",            true);
  const playPauseBtn = mkBtn("play",       "Play",            false);
  const acceptBtn   = mkBtn("check",       "Accept",          false);
  const discardBtn  = mkBtn("rotate-ccw",  "Re-record",       false);

  overlay.append(startBtn, stopBtn, playPauseBtn, acceptBtn, discardBtn);
  videoWrap.append(placeholder, video, recBadge, encodingOverlay, overlay);

  // Device selector row — [cam icon] [cam select] [mic icon] [mic select]
  const videoSel = mkSelect();
  const audioSel = mkSelect();
  const deviceRow = document.createElement("div");
  deviceRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-top:6px;transition:opacity 0.15s;";
  deviceRow.append(mkIcon("camera", 15), videoSel, mkIcon("mic", 15), audioSel);

  wrapper.append(videoWrap, deviceRow);
  container.append(wrapper);

  // ── Play/pause button sync ────────────────────────────────────────────────

  function syncPlayPauseBtn() {
    const isPaused = video.paused || video.ended;
    // Swap icon: remove first child (svg), insert fresh one
    playPauseBtn.removeChild(playPauseBtn.firstChild);
    playPauseBtn.insertBefore(mkIcon(isPaused ? "play" : "pause", 14), playPauseBtn.firstChild);
    playPauseBtn.querySelector("span").textContent = isPaused ? "Play" : "Pause";
  }

  video.addEventListener("play",  syncPlayPauseBtn);
  video.addEventListener("pause", syncPlayPauseBtn);
  video.addEventListener("ended", syncPlayPauseBtn);

  // ── Render ───────────────────────────────────────────────────────────────

  function render() {
    placeholder.hidden   = isStreaming || hasRecording;
    video.hidden         = !isStreaming && !hasRecording;
    startBtn.hidden      = isRecording || hasRecording;
    startBtn.disabled    = !isStreaming;
    stopBtn.hidden       = !isRecording;
    playPauseBtn.hidden  = !hasRecording;
    acceptBtn.hidden     = !hasRecording;
    discardBtn.hidden    = !hasRecording;
    recBadge.hidden      = !isRecording;
    const devicesLocked       = isRecording || hasRecording;
    videoSel.disabled         = devicesLocked;
    audioSel.disabled         = devicesLocked;
    deviceRow.style.opacity   = devicesLocked ? "0.4" : "1";
    deviceRow.style.pointerEvents = devicesLocked ? "none" : "";
  }

  // ── Camera / Recording ───────────────────────────────────────────────────

  async function populateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      function repopulate(sel, list) {
        const current = sel.value;
        sel.innerHTML = "";
        list.forEach((d, i) => {
          const opt = document.createElement("option");
          opt.value = d.deviceId;
          opt.textContent = d.label || `Device ${i + 1}`;
          sel.appendChild(opt);
        });
        if (current && [...sel.options].some((o) => o.value === current)) sel.value = current;
      }
      repopulate(videoSel, devices.filter((d) => d.kind === "videoinput"));
      repopulate(audioSel, devices.filter((d) => d.kind === "audioinput"));
      // Sync selects to the tracks the stream is actually using
      if (stream) {
        const vs = stream.getVideoTracks()[0]?.getSettings?.()?.deviceId;
        const as = stream.getAudioTracks()[0]?.getSettings?.()?.deviceId;
        if (vs) { currentVideoId = vs; videoSel.value = vs; }
        if (as) { currentAudioId = as; audioSel.value = as; }
      }
    } catch { /* permission not yet granted */ }
  }

  async function switchDevices() {
    if (isRecording || hasRecording) return;
    currentVideoId = videoSel.value;
    currentAudioId = audioSel.value;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    isStreaming = false;
    render();
    await startCamera();
  }

  async function startCamera() {
    try {
      const constraints = {
        video: currentVideoId ? { deviceId: { exact: currentVideoId } } : true,
        audio: currentAudioId ? { deviceId: { exact: currentAudioId } } : true,
      };
      const s = await navigator.mediaDevices.getUserMedia(constraints);
      if (hasRecording) { s.getTracks().forEach((t) => t.stop()); return; }
      stream = s;
      video.srcObject = stream;
      video.muted = true;
      isStreaming = true;
      render();
      await populateDevices();
    } catch {
      if (!hasRecording) placeholder.textContent = "Camera unavailable — check browser permissions.";
    }
  }

  function startRecording() {
    if (!stream) return;
    chunks = [];
    elapsed = 0;
    recBadge.textContent = "● REC 00:00";
    recorder = new MediaRecorder(stream, { mimeType: mime });
    recorder.ondataavailable = (e) => { if (e.data?.size > 0) chunks.push(e.data); };
    recorder.onstop = onStop;
    recorder.start(100);
    isRecording = true;
    render();
    timer = setInterval(() => { elapsed++; recBadge.textContent = `● REC ${fmtTime(elapsed)}`; }, 1000);
  }

  function stopRecording() {
    clearInterval(timer);
    if (recorder && recorder.state !== "inactive") recorder.stop();
  }

  function onStop() {
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
    render();
  }

  function hideEncodingOverlay() {
    encodingOverlay.hidden = true;
    acceptBtn.disabled = false;
    discardBtn.disabled = false;
    playPauseBtn.disabled = false;
  }

  function accept() {
    if (!blob) return;
    encodingOverlay.hidden = false;
    acceptBtn.disabled = true;
    discardBtn.disabled = true;
    playPauseBtn.disabled = true;
    // rAF ensures overlay paints before the FileReader (and WebSocket send) starts
    requestAnimationFrame(() => {
      const reader = new FileReader();
      reader.onload = () => {
        _emitSeq++;
        onChange?.({ state: "accepted", value: reader.result, type: mime.split(";")[0], _emitSeq });
        // Overlay stays visible — Python will echo back state:"processed" when done
      };
      reader.readAsDataURL(blob);
    });
  }

  function discard() {
    video.src = "";
    video.muted = true;
    if (blobUrl) { URL.revokeObjectURL(blobUrl); blobUrl = null; }
    blob = null; chunks = [];
    isStreaming = false; isRecording = false; hasRecording = false;
    placeholder.textContent = "Starting camera…";
    render();
    startCamera();
  }

  function togglePlayPause() {
    if (video.paused || video.ended) { video.currentTime = 0; video.play().catch(() => {}); }
    else video.pause();
  }

  // ── Restore saved state ──────────────────────────────────────────────────

  function restore(b64, type) {
    try {
      const mime64 = (type || "video/webm").split(";")[0];
      const raw = b64.includes(",") ? b64.split(",")[1] : b64;
      const bytes = Uint8Array.from(atob(raw), (c) => c.charCodeAt(0));
      blob = new Blob([bytes], { type: mime64 });
      if (blobUrl) URL.revokeObjectURL(blobUrl);
      blobUrl = URL.createObjectURL(blob);
      video.srcObject = null;
      video.src = blobUrl;
      video.muted = false;
      video.loop = false;
      video.play().catch(() => {});
      isStreaming = false; isRecording = false; hasRecording = true;
      render();
      return true;
    } catch { return false; }
  }

  // ── handleUpdate ─────────────────────────────────────────────────────────

  function handleUpdate(newProps) {
    onChange = newProps.onChange;
    const v = newProps.value ?? {};
    // Python echoes this after dict_to_video_url_artifact completes — hide overlay
    if (v.state === "processed") { hideEncodingOverlay(); return; }
    if ((v._emitSeq || 0) <= _emitSeq) return;
    if ((v.state === "recorded" || v.state === "accepted") && v.value && !hasRecording)
      restore(v.value, v.type);
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────

  function cleanup() {
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

  startBtn.addEventListener("click",    startRecording);
  stopBtn.addEventListener("click",     stopRecording);
  playPauseBtn.addEventListener("click", togglePlayPause);
  acceptBtn.addEventListener("click",   accept);
  discardBtn.addEventListener("click",  discard);
  videoSel.addEventListener("change",   switchDevices);
  audioSel.addEventListener("change",   switchDevices);
  navigator.mediaDevices.addEventListener("devicechange", populateDevices);

  container._vcInst = { handleUpdate, cleanup, wrapper };

  const init = props.value ?? {};
  if ((init.state === "recorded" || init.state === "accepted") && init.value) {
    if (!restore(init.value, init.type)) startCamera();
  } else {
    startCamera();
  }

  render();
  return { cleanup, update: handleUpdate };
}
