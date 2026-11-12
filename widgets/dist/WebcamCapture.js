// src/WebcamCapture/_icons.ts
var ICON_PATHS = {
  camera: `<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>`,
  check: `<path d="M20 6 9 17l-5-5"/>`,
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`,
  "trash-2": `<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>`
};
function mkIcon(name, size = 15) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.style.cssText = "display:block;flex-shrink:0;pointer-events:none;";
  svg.innerHTML = ICON_PATHS[name] || "";
  return svg;
}

// src/WebcamCapture/WebcamCapture.ts
function injectStyles() {
  if (document.getElementById("wc-styles")) return;
  const style = document.createElement("style");
  style.id = "wc-styles";
  style.textContent = [
    "@keyframes wc-spin{to{transform:rotate(360deg)}}",
    ".wc-thumb{width:106px;height:80px;object-fit:cover;border-radius:6px;cursor:pointer;",
    "flex-shrink:0;border:3px solid transparent;transition:border-color 0.15s;display:block;}",
    ".wc-thumb.selected{border-color:var(--primary);}"
  ].join("");
  document.head.appendChild(style);
}
function mkSelect() {
  const sel = document.createElement("select");
  sel.style.cssText = [
    "flex:1",
    "min-width:0",
    "padding:4px 6px",
    "border-radius:5px",
    "border:1px solid var(--border)",
    "background:var(--background)",
    "color:var(--foreground)",
    "font-size:12px",
    "cursor:pointer"
  ].join(";");
  sel.addEventListener("pointerdown", (e) => e.stopPropagation());
  sel.addEventListener("mousedown", (e) => e.stopPropagation());
  return sel;
}
function WebcamCapture(host, props) {
  const container = host;
  if (container._wcInst?.wrapper?.isConnected) {
    container._wcInst.handleUpdate(props);
    return { cleanup: container._wcInst.cleanup, update: container._wcInst.handleUpdate };
  }
  injectStyles();
  let { onChange } = props;
  let _emitSeq = 0;
  let isStreaming = false;
  const init = props.value ?? {};
  let galleryItems = init.gallery_items ?? [];
  let selectedIndex = init.selected_index ?? (galleryItems.length > 0 ? galleryItems.length - 1 : -1);
  let pendingThumbs = [];
  const captureQueue = [];
  let _processing = false;
  let _pendingBlob = null;
  let _lastProcessedSeq = -1;
  let _lastUploadSeq = -1;
  let stream = null;
  let currentVideoId = "";
  const IDEAL_WIDTH = 4096;
  const IDEAL_HEIGHT = 2160;
  const JPEG_QUALITY = 0.95;
  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;padding:8px;box-sizing:border-box;gap:6px;";
  const videoWrap = document.createElement("div");
  videoWrap.style.cssText = [
    "position:relative",
    "background:#111",
    "border-radius:8px",
    "flex:1 1 0",
    "min-height:160px",
    "overflow:hidden",
    "display:flex",
    "align-items:center",
    "justify-content:center"
  ].join(";");
  const video = document.createElement("video");
  video.style.cssText = "width:100%;height:100%;object-fit:contain;display:block;";
  video.autoplay = true;
  video.muted = true;
  video.playsInline = true;
  const placeholder = document.createElement("div");
  placeholder.style.cssText = "position:absolute;color:var(--muted-foreground);font-size:13px;text-align:center;padding:24px;user-select:none;";
  placeholder.textContent = "Starting camera…";
  const shutterBtn = document.createElement("button");
  shutterBtn.style.cssText = [
    "position:absolute",
    "bottom:12px",
    "left:50%",
    "transform:translateX(-50%)",
    "width:52px",
    "height:52px",
    "border-radius:50%",
    "border:3px solid rgba(255,255,255,0.85)",
    "background:rgba(20,20,20,0.55)",
    "cursor:pointer",
    "display:flex",
    "align-items:center",
    "justify-content:center",
    "backdrop-filter:blur(4px)",
    "color:white",
    "transition:transform 0.1s,opacity 0.15s"
  ].join(";");
  shutterBtn.appendChild(mkIcon("camera", 22));
  shutterBtn.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    shutterBtn.style.transform = "translateX(-50%) scale(0.88)";
  });
  shutterBtn.addEventListener("pointerup", () => {
    shutterBtn.style.transform = "translateX(-50%)";
  });
  shutterBtn.addEventListener("pointerleave", () => {
    shutterBtn.style.transform = "translateX(-50%)";
  });
  shutterBtn.addEventListener("mousedown", (e) => e.stopPropagation());
  videoWrap.append(placeholder, video, shutterBtn);
  const thumbStrip = document.createElement("div");
  thumbStrip.style.cssText = [
    "display:flex",
    "gap:6px",
    "overflow-x:auto",
    "padding:2px 0",
    "scrollbar-width:thin",
    "min-height:86px"
  ].join(";");
  thumbStrip.addEventListener("pointerdown", (e) => e.stopPropagation());
  thumbStrip.addEventListener("mousedown", (e) => e.stopPropagation());
  const videoSel = mkSelect();
  const clearBtn = document.createElement("button");
  clearBtn.style.cssText = [
    "display:flex",
    "align-items:center",
    "gap:5px",
    "padding:4px 10px",
    "border-radius:5px",
    "border:1px solid var(--destructive)",
    "background:transparent",
    "color:var(--destructive)",
    "font-size:12px",
    "cursor:pointer",
    "white-space:nowrap",
    "flex-shrink:0"
  ].join(";");
  clearBtn.appendChild(mkIcon("trash-2", 13));
  const clearSpan = document.createElement("span");
  clearBtn.appendChild(clearSpan);
  clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  clearBtn.addEventListener("mousedown", (e) => e.stopPropagation());
  const bottomRow = document.createElement("div");
  bottomRow.style.cssText = "display:flex;align-items:center;gap:6px;color:var(--muted-foreground);";
  bottomRow.append(mkIcon("camera", 15), videoSel, clearBtn);
  wrapper.append(videoWrap, thumbStrip, bottomRow);
  container.append(wrapper);
  function renderThumbs() {
    thumbStrip.innerHTML = "";
    thumbStrip.hidden = galleryItems.length === 0 && pendingThumbs.length === 0;
    clearBtn.hidden = galleryItems.length === 0;
    clearSpan.textContent = `Clear (${galleryItems.length})`;
    galleryItems.forEach((item, idx) => {
      const img = document.createElement("img");
      img.className = "wc-thumb" + (idx === selectedIndex ? " selected" : "");
      img.src = item.url || "";
      img.title = `Photo ${idx + 1}`;
      img.addEventListener("click", () => selectThumbnail(idx));
      img.addEventListener("pointerdown", (e) => e.stopPropagation());
      img.addEventListener("mousedown", (e) => e.stopPropagation());
      thumbStrip.appendChild(img);
    });
    pendingThumbs.forEach((dataUrl) => {
      const wrap = document.createElement("div");
      wrap.style.cssText = "position:relative;width:106px;height:80px;flex-shrink:0;border-radius:6px;overflow:hidden;border:3px solid transparent;";
      const img = document.createElement("img");
      img.style.cssText = "width:100%;height:100%;object-fit:cover;display:block;opacity:0.35;";
      img.src = dataUrl;
      const spinnerWrap = document.createElement("div");
      spinnerWrap.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,0.35);";
      const spin = document.createElement("div");
      spin.style.cssText = "width:22px;height:22px;border:2px solid rgba(255,255,255,0.2);border-top-color:white;border-radius:50%;animation:wc-spin 0.75s linear infinite;";
      spinnerWrap.appendChild(spin);
      wrap.append(img, spinnerWrap);
      thumbStrip.appendChild(wrap);
    });
    thumbStrip.lastElementChild?.scrollIntoView({ block: "nearest", inline: "end" });
  }
  function selectThumbnail(index) {
    selectedIndex = index;
    renderThumbs();
    _emitSeq++;
    onChange?.({ state: "selected", selected_index: index, gallery_items: galleryItems, _emitSeq });
  }
  function render() {
    placeholder.hidden = isStreaming;
    video.hidden = !isStreaming;
    shutterBtn.disabled = !isStreaming;
    shutterBtn.style.opacity = isStreaming ? "1" : "0.4";
  }
  async function populateDevices() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices.filter((d) => d.kind === "videoinput");
      const current = videoSel.value;
      videoSel.innerHTML = "";
      cameras.forEach((d, i) => {
        const opt = document.createElement("option");
        opt.value = d.deviceId;
        opt.textContent = d.label || `Camera ${i + 1}`;
        videoSel.appendChild(opt);
      });
      if (current && [...videoSel.options].some((o) => o.value === current)) videoSel.value = current;
      if (stream) {
        const vs = stream.getVideoTracks()[0]?.getSettings?.()?.deviceId;
        if (vs) {
          currentVideoId = vs;
          videoSel.value = vs;
        }
      }
    } catch {
    }
  }
  async function switchDevices() {
    currentVideoId = videoSel.value;
    stream?.getTracks().forEach((t) => t.stop());
    stream = null;
    isStreaming = false;
    render();
    await startCamera();
  }
  async function startCamera() {
    try {
      const videoConstraints = {
        width: { ideal: IDEAL_WIDTH },
        height: { ideal: IDEAL_HEIGHT }
      };
      if (currentVideoId) videoConstraints.deviceId = { exact: currentVideoId };
      const s = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false
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
  function dataUrlToBlob(dataUrl) {
    const comma = dataUrl.indexOf(",");
    const mime = dataUrl.slice(5, dataUrl.indexOf(";"));
    const bytes = atob(dataUrl.slice(comma + 1));
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return new Blob([arr], { type: mime });
  }
  function capture() {
    if (!stream) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
    const blob = dataUrlToBlob(dataUrl);
    captureQueue.push(blob);
    pendingThumbs = [...pendingThumbs, dataUrl];
    renderThumbs();
    _processNext();
  }
  function _processNext() {
    if (_processing || captureQueue.length === 0) return;
    _processing = true;
    const blob = captureQueue.shift();
    _pendingBlob = blob;
    _emitSeq++;
    onChange?.({ state: "requesting_upload_url", _emitSeq });
  }
  function _doUpload(uploadUrl, seq) {
    const blob = _pendingBlob;
    _pendingBlob = null;
    fetch(uploadUrl, { method: "PUT", body: blob }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
    }).then(() => {
      onChange?.({ state: "accepted", _emitSeq: seq });
    }).catch(() => {
      onChange?.({ state: "accepted", _emitSeq: seq });
    });
  }
  function clearGallery() {
    captureQueue.length = 0;
    _processing = false;
    _pendingBlob = null;
    pendingThumbs = [];
    _emitSeq++;
    onChange?.({ state: "clear_gallery", _emitSeq });
  }
  function handleUpdate(newProps) {
    onChange = newProps.onChange;
    const v = newProps.value ?? {};
    if (v.state === "upload_ready" && v._uploadUrl) {
      if (v._emitSeq === _lastUploadSeq) return;
      _lastUploadSeq = v._emitSeq;
      if (_pendingBlob) {
        _doUpload(v._uploadUrl, v._emitSeq);
      } else {
        _processing = false;
        _processNext();
      }
      return;
    }
    if (v.state === "processed") {
      if (v._emitSeq === _lastProcessedSeq) return;
      _lastProcessedSeq = v._emitSeq;
      pendingThumbs = pendingThumbs.slice(1);
      galleryItems = v.gallery_items ?? galleryItems;
      selectedIndex = v.selected_index ?? galleryItems.length - 1;
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
      galleryItems = [];
      selectedIndex = -1;
      pendingThumbs = [];
      renderThumbs();
      return;
    }
  }
  function cleanup() {
    navigator.mediaDevices.removeEventListener("devicechange", populateDevices);
    stream?.getTracks().forEach((t) => t.stop());
    video.srcObject = null;
    wrapper.remove();
    delete container._wcInst;
  }
  shutterBtn.addEventListener("click", capture);
  clearBtn.addEventListener("click", clearGallery);
  videoSel.addEventListener("change", switchDevices);
  navigator.mediaDevices.addEventListener("devicechange", populateDevices);
  container._wcInst = { handleUpdate, cleanup, wrapper };
  renderThumbs();
  startCamera();
  render();
  return { cleanup, update: handleUpdate };
}
export {
  WebcamCapture as default
};
