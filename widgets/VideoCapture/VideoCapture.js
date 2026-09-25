import { mkIcon } from './_icons.js';

const PREFERRED_MIMES = [
  "video/mp4;codecs=avc1,mp4a.40.2",
  "video/mp4;codecs=avc1",
  "video/mp4",
  "video/webm;codecs=vp9,opus",
  "video/webm;codecs=vp8,opus",
  "video/webm",
];

function getSupportedMime() {
  if (typeof MediaRecorder === "undefined") return "video/mp4";
  for (const t of PREFERRED_MIMES) {
    if (MediaRecorder.isTypeSupported(t)) return t;
  }
  return "video/mp4";
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

function mkOverlayBtn(iconName, label) {
  const btn = document.createElement("button");
  btn.style.cssText = [
    "display:flex", "align-items:center", "gap:6px",
    "padding:6px 14px", "border-radius:6px", "border:none",
    "cursor:pointer", "font-size:13px", "font-weight:500",
    "background:rgba(0,0,0,0.6);color:white;backdrop-filter:blur(4px);",
  ].join(";");
  btn.appendChild(mkIcon(iconName, 14));
  btn.appendChild(Object.assign(document.createElement("span"), { textContent: label }));
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

  let _uploading = false;
  let stream = null, recorder = null, chunks = [], blob = null, blobUrl = null;
  let elapsed = 0, timer = null;
  const mime = getSupportedMime();

  let currentVideoId = "";
  let currentAudioId = "";

  // ── DOM ──────────────────────────────────────────────────────────────────

  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;padding:8px;box-sizing:border-box;";

  const videoWrap = document.createElement("div");
  videoWrap.style.cssText = [
    "position:relative", "background:#111", "border-radius:8px",
    "flex:1 1 0", "min-height:180px", "overflow:hidden",
    "display:flex", "align-items:center", "justify-content:center",
  ].join(";");

  const video = document.createElement("video");
  video.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;";
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;

  const placeholder = document.createElement("div");
  placeholder.style.cssText = "position:absolute;color:var(--muted-foreground);font-size:13px;text-align:center;padding:24px;user-select:none;";
  placeholder.textContent = "Starting camera…";

  // REC badge
  const recBadge = document.createElement("div");
  recBadge.style.cssText = [
    "position:absolute", "top:10px", "left:10px",
    "background:rgba(200,0,0,0.85)", "color:#fff",
    "font-size:11px", "font-weight:700", "padding:2px 8px",
    "border-radius:4px", "font-family:monospace", "pointer-events:none",
  ].join(";");
  recBadge.hidden = true;

  // Encoding overlay
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

  // Circular record button — red dot morphs to stop square when recording
  const recordBtn = document.createElement("button");
  recordBtn.style.cssText = [
    "position:absolute", "bottom:12px", "left:50%", "transform:translateX(-50%)",
    "width:52px", "height:52px", "border-radius:50%",
    "border:3px solid rgba(255,255,255,0.85)", "background:rgba(20,20,20,0.55)",
    "cursor:pointer", "display:flex", "align-items:center", "justify-content:center",
    "backdrop-filter:blur(4px)", "transition:transform 0.1s,opacity 0.15s",
  ].join(";");
  const innerDot = document.createElement("div");
  innerDot.style.cssText = "width:20px;height:20px;border-radius:50%;background:#e53e3e;transition:border-radius 0.2s,width 0.2s,height 0.2s;";
  recordBtn.appendChild(innerDot);
  recordBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); recordBtn.style.transform = "translateX(-50%) scale(0.88)"; });
  recordBtn.addEventListener("pointerup",   ()  => { recordBtn.style.transform = "translateX(-50%)"; });
  recordBtn.addEventListener("pointerleave",()  => { recordBtn.style.transform = "translateX(-50%)"; });
  recordBtn.addEventListener("mousedown",   (e) => e.stopPropagation());

  // Review controls (shown after recording stops)
  const playPauseBtn = mkOverlayBtn("play",      "Play");
  const acceptBtn    = mkOverlayBtn("check",     "Accept");
  const discardBtn   = mkOverlayBtn("rotate-ccw","Re-record");

  const reviewOverlay = document.createElement("div");
  reviewOverlay.style.cssText = "position:absolute;bottom:12px;left:0;right:0;display:flex;justify-content:center;gap:8px;flex-wrap:wrap;";
  reviewOverlay.append(playPauseBtn, acceptBtn, discardBtn);

  videoWrap.append(placeholder, video, recBadge, encodingOverlay, recordBtn, reviewOverlay);

  // Device selector row
  const videoSel = mkSelect();
  const audioSel = mkSelect();
  const deviceRow = document.createElement("div");
  deviceRow.style.cssText = "display:flex;align-items:center;gap:6px;margin-top:6px;transition:opacity 0.15s;color:var(--muted-foreground);";
  deviceRow.append(mkIcon("camera", 15), videoSel, mkIcon("mic", 15), audioSel);

  wrapper.append(videoWrap, deviceRow);
  container.append(wrapper);

  // ── Play/pause button sync ────────────────────────────────────────────────

  function syncPlayPauseBtn() {
    const isPaused = video.paused || video.ended;
    playPauseBtn.removeChild(playPauseBtn.firstChild);
    playPauseBtn.insertBefore(mkIcon(isPaused ? "play" : "pause", 14), playPauseBtn.firstChild);
    playPauseBtn.querySelector("span").textContent = isPaused ? "Play" : "Pause";
  }

  video.addEventListener("play",  syncPlayPauseBtn);
  video.addEventListener("pause", syncPlayPauseBtn);
  video.addEventListener("ended", syncPlayPauseBtn);

  // ── Render ───────────────────────────────────────────────────────────────

  function render() {
    placeholder.hidden      = isStreaming || hasRecording;
    video.hidden            = !isStreaming && !hasRecording;
    recordBtn.hidden        = hasRecording;
    recordBtn.disabled      = !isStreaming && !isRecording;
    recordBtn.style.opacity = (isStreaming || isRecording) ? "1" : "0.4";
    reviewOverlay.hidden    = !hasRecording;
    recBadge.hidden         = !isRecording;

    const devicesLocked          = isRecording || hasRecording;
    videoSel.disabled            = devicesLocked;
    audioSel.disabled            = devicesLocked;
    deviceRow.style.opacity      = devicesLocked ? "0.4" : "1";
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
    // Morph dot to stop square
    innerDot.style.borderRadius = "3px";
    innerDot.style.width = "18px";
    innerDot.style.height = "18px";
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
    // Reset dot back to circle for next time
    innerDot.style.borderRadius = "50%";
    innerDot.style.width = "20px";
    innerDot.style.height = "20px";
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
    requestAnimationFrame(() => {
      _emitSeq++;
      onChange?.({ state: "requesting_upload_url", _mime: mime, _emitSeq });
    });
  }

  function doUpload(uploadUrl) {
    if (_uploading) return;
    _uploading = true;
    fetch(uploadUrl, { method: "PUT", body: blob })
      .then((r) => { if (!r.ok) throw new Error(r.statusText); })
      .then(() => { onChange?.({ state: "accepted", _emitSeq }); })
      .catch(() => { _uploading = false; hideEncodingOverlay(); });
  }

  function discard() {
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

  function togglePlayPause() {
    if (video.paused || video.ended) { video.currentTime = 0; video.play().catch(() => {}); }
    else video.pause();
  }

  function toggleRecord() {
    if (isRecording) stopRecording();
    else startRecording();
  }

  // ── handleUpdate ─────────────────────────────────────────────────────────

  function handleUpdate(newProps) {
    onChange = newProps.onChange;
    const v = newProps.value ?? {};
    if (v.state === "processed") { hideEncodingOverlay(); return; }
    if (v.state === "error") {
      spinner.hidden = true;
      encodingOverlay.querySelector("span").textContent = v.message || "Upload failed.";
      setTimeout(hideEncodingOverlay, 4000);
      return;
    }
    if (v.state === "upload_ready" && v._uploadUrl) { doUpload(v._uploadUrl); return; }
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

  recordBtn.addEventListener("click",    toggleRecord);
  playPauseBtn.addEventListener("click", togglePlayPause);
  acceptBtn.addEventListener("click",    accept);
  discardBtn.addEventListener("click",   discard);
  videoSel.addEventListener("change",    switchDevices);
  audioSel.addEventListener("change",    switchDevices);
  navigator.mediaDevices.addEventListener("devicechange", populateDevices);

  container._vcInst = { handleUpdate, cleanup, wrapper };

  startCamera();
  render();
  return { cleanup, update: handleUpdate };
}
