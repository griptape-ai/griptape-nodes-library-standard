// src/CropVideoEditor/_styles.ts
var WIDGET_VERSION = "0.1.0";
var HANDLE_R = 6;
var OVERLAY = "rgba(0,0,0,0.55)";
var CROP_BORDER = "rgba(255,255,255,0.9)";
var GUIDE = "rgba(255,255,255,0.25)";
var HANDLE_FILL = "black";
var HANDLE_STROKE = "#ffffff";
var HANDLE_STROKE_HOVER = "#60a5fa";
var HANDLE_LOCKED = "rgba(150,150,150,0.5)";

// src/CropVideoEditor/_icons.ts
var ICON_PATHS = {
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`,
  "film": `<rect width="18" height="18" x="3" y="3" rx="2"/><path d="M7 3v18"/><path d="M3 7.5h4"/><path d="M3 12h18"/><path d="M3 16.5h4"/><path d="M17 3v18"/><path d="M17 7.5h4"/><path d="M17 16.5h4"/>`
};
function mkIcon(name, size = 14) {
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

// src/CropVideoEditor/_sidebar.ts
var RATIO_PRESETS = [
  { label: "1:1", rw: 1, rh: 1 },
  { label: "16:9", rw: 16, rh: 9 },
  { label: "4:3", rw: 4, rh: 3 },
  { label: "3:2", rw: 3, rh: 2 },
  { label: "9:16", rw: 9, rh: 16 },
  { label: "2:3", rw: 2, rh: 3 },
  { label: "3:4", rw: 3, rh: 4 }
];
var RESOLUTION_PRESETS = [
  { label: "480p", w: 854, h: 480 },
  { label: "720p", w: 1280, h: 720 },
  { label: "1080p", w: 1920, h: 1080 },
  { label: "1440p", w: 2560, h: 1440 },
  { label: "4K", w: 3840, h: 2160 }
];
var POSITION_PRESETS = [
  { label: "↖", title: "Top Left", pos: "top-left" },
  { label: "↑", title: "Top Center", pos: "top-center" },
  { label: "↗", title: "Top Right", pos: "top-right" },
  { label: "←", title: "Left Center", pos: "left-center" },
  { label: "⊕", title: "Center", pos: "center" },
  { label: "→", title: "Right Center", pos: "right-center" },
  { label: "↙", title: "Bottom Left", pos: "bottom-left" },
  { label: "↓", title: "Bottom Center", pos: "bottom-center" },
  { label: "↘", title: "Bottom Right", pos: "bottom-right" }
];
function calcRatioRect(rw, rh, imgW, imgH, curL, curT, curW, curH) {
  const s = Math.min(curW, curH) / Math.min(rw, rh);
  let w = Math.round(rw * s);
  let h = Math.round(rh * s);
  if (w > imgW) {
    w = imgW;
    h = Math.round(w * rh / rw);
  }
  if (h > imgH) {
    h = imgH;
    w = Math.round(h * rw / rh);
  }
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}
function calcResolutionRect(newW, newH, imgW, imgH, curL, curT, curW, curH) {
  const w = Math.min(newW, imgW);
  const h = Math.min(newH, imgH);
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}
function calcPositionRect(posName, imgW, imgH, curL, curT, curW, curH) {
  const w = curW, h = curH;
  const positions = {
    "center": { l: Math.round((imgW - w) / 2), t: Math.round((imgH - h) / 2) },
    "top-left": { l: 0, t: 0 },
    "top-center": { l: Math.round((imgW - w) / 2), t: 0 },
    "top-right": { l: imgW - w, t: 0 },
    "left-center": { l: 0, t: Math.round((imgH - h) / 2) },
    "right-center": { l: imgW - w, t: Math.round((imgH - h) / 2) },
    "bottom-left": { l: 0, t: imgH - h },
    "bottom-center": { l: Math.round((imgW - w) / 2), t: imgH - h },
    "bottom-right": { l: imgW - w, t: imgH - h }
  };
  const { l, t } = positions[posName] || positions["center"];
  return { left: Math.max(0, l), top: Math.max(0, t), width: w, height: h };
}
function createSidebar({ getImgSize, getCropRect, onApply }) {
  const el = document.createElement("div");
  el.className = "nodrag nowheel";
  el.style.cssText = [
    "display:flex",
    "flex-direction:column",
    "gap:3px",
    "width:108px",
    "flex-shrink:0",
    "background:var(--muted)",
    "border-radius:6px",
    "padding:4px",
    "overflow-y:auto",
    "box-sizing:border-box"
  ].join(";");
  const allBtns = [];
  function makeBtn({ label, icon, title, onClick }) {
    const btn = document.createElement("button");
    btn.title = title;
    btn.style.cssText = [
      "width:100%",
      "padding:4px 2px",
      "border-radius:4px",
      "border:1px solid var(--border)",
      "background:var(--background)",
      "color:var(--foreground)",
      "cursor:pointer",
      "font-size:10px",
      "font-weight:500",
      "line-height:1.3",
      "white-space:nowrap",
      "box-sizing:border-box",
      "display:flex",
      "align-items:center",
      "justify-content:center",
      "gap:3px"
    ].join(";");
    if (icon) btn.appendChild(mkIcon(icon, 12));
    if (label) btn.append(label);
    btn.addEventListener("mouseenter", () => {
      if (!btn.disabled) btn.style.background = "var(--accent)";
    });
    btn.addEventListener("mouseleave", () => {
      btn.style.background = "var(--background)";
    });
    for (const ev of ["mousedown", "pointerdown"]) btn.addEventListener(ev, (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      if (!btn.disabled) onClick();
    });
    allBtns.push(btn);
    return btn;
  }
  function makeDivider() {
    const hr = document.createElement("div");
    hr.style.cssText = "height:1px;background:var(--border);margin:2px 0;flex-shrink:0;";
    return hr;
  }
  function makeLabel(text) {
    const lbl = document.createElement("div");
    lbl.textContent = text;
    lbl.style.cssText = "font-size:9px;color:var(--muted-foreground);text-align:center;padding:2px 0;";
    return lbl;
  }
  el.appendChild(makeBtn({ icon: "rotate-ccw", label: "Reset", title: "Reset crop to full video frame", onClick: () => {
    const { imgNatW: vw, imgNatH: vh } = getImgSize();
    onApply({ left: 0, top: 0, width: vw, height: vh });
  } }));
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Ratio"));
  const ratioGrid = document.createElement("div");
  ratioGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, rw, rh } of RATIO_PRESETS) {
    const btn = makeBtn({ label, title: label, onClick: () => {
      const { imgNatW: vw, imgNatH: vh } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcRatioRect(rw, rh, vw, vh, l, t, w, h));
    } });
    ratioGrid.appendChild(btn);
  }
  el.appendChild(ratioGrid);
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Res"));
  const resGrid = document.createElement("div");
  resGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, w: pw, h: ph } of RESOLUTION_PRESETS) {
    const btn = makeBtn({ label, title: `${pw}×${ph}`, onClick: () => {
      const { imgNatW: vw, imgNatH: vh } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcResolutionRect(pw, ph, vw, vh, l, t, w, h));
    } });
    resGrid.appendChild(btn);
  }
  el.appendChild(resGrid);
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Pos"));
  const posGrid = document.createElement("div");
  posGrid.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:2px;";
  for (const { label, title, pos } of POSITION_PRESETS) {
    const btn = makeBtn({ label, title, onClick: () => {
      const { imgNatW: vw, imgNatH: vh } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcPositionRect(pos, vw, vh, l, t, w, h));
    } });
    btn.style.width = "100%";
    posGrid.appendChild(btn);
  }
  el.appendChild(posGrid);
  function syncDisabled(lockedParams, disabled) {
    const coordLocked = ["left", "top", "width", "height"].some((f) => lockedParams.includes(f));
    const off = coordLocked || disabled;
    for (const btn of allBtns) {
      btn.disabled = off;
      btn.style.opacity = off ? "0.35" : "1";
      btn.style.cursor = off ? "not-allowed" : "pointer";
    }
  }
  return { el, syncDisabled };
}

// src/CropVideoEditor/_footer.ts
function createFooter({ isDisabled, onSeek, version }) {
  const controls = document.createElement("div");
  controls.className = "nodrag nowheel";
  controls.style.cssText = [
    "background:var(--muted)",
    "border-radius:4px",
    "padding:6px 8px",
    "display:flex",
    "flex-direction:column",
    "gap:5px"
  ].join(";");
  const scrubRow = document.createElement("div");
  scrubRow.style.cssText = "display:flex;align-items:center;gap:8px;";
  const filmIcon = document.createElement("span");
  filmIcon.style.cssText = "color:var(--muted-foreground);display:flex;align-items:center;flex-shrink:0;";
  filmIcon.appendChild(mkIcon("film", 13));
  const scrubSlider = document.createElement("input");
  scrubSlider.type = "range";
  scrubSlider.min = "0";
  scrubSlider.max = "0";
  scrubSlider.step = "1";
  scrubSlider.value = "0";
  scrubSlider.style.cssText = "flex:1;accent-color:#2563eb;";
  const frameDisplay = document.createElement("span");
  frameDisplay.textContent = "0 / 0";
  frameDisplay.style.cssText = "font-size:11px;color:var(--foreground);min-width:52px;text-align:right;font-family:monospace;";
  for (const ev of ["mousedown", "pointerdown"]) {
    scrubSlider.addEventListener(ev, (e) => e.stopPropagation());
  }
  scrubSlider.addEventListener("input", (e) => {
    if (isDisabled()) return;
    e.stopPropagation();
    const frame = parseInt(e.target.value, 10);
    const total = parseInt(scrubSlider.max, 10);
    frameDisplay.textContent = `${frame} / ${total}`;
    onSeek(frame);
  });
  scrubRow.appendChild(filmIcon);
  scrubRow.appendChild(scrubSlider);
  scrubRow.appendChild(frameDisplay);
  controls.appendChild(scrubRow);
  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px",
    "color:var(--muted-foreground)",
    "padding:4px 8px",
    "background:var(--muted)",
    "border-radius:4px",
    "display:flex",
    "justify-content:space-between",
    "align-items:center"
  ].join(";");
  const statusL = document.createElement("span");
  const statusR = document.createElement("span");
  statusR.style.cssText = "font-family:monospace;font-size:10px;color:var(--muted-foreground);opacity:0.6;";
  statusR.textContent = `v${version}`;
  statusBar.appendChild(statusL);
  statusBar.appendChild(statusR);
  function sync(totalFrames, disabled) {
    scrubSlider.disabled = disabled || totalFrames === 0;
    scrubSlider.max = String(Math.max(0, totalFrames - 1));
    scrubRow.style.opacity = disabled || totalFrames === 0 ? "0.4" : "1";
  }
  function setFrame(frame, totalFrames) {
    scrubSlider.max = String(Math.max(0, totalFrames - 1));
    scrubSlider.value = String(frame);
    frameDisplay.textContent = `${frame} / ${totalFrames}`;
  }
  function updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH }) {
    if (!videoLoaded) {
      statusL.textContent = "No video connected";
      statusL.style.color = "";
      return;
    }
    const coordLocked = ["left", "top", "width", "height"].filter((f) => lockedParams.includes(f));
    if (coordLocked.length > 0) {
      statusL.style.color = "#f59e0b";
      statusL.textContent = "🔒 Connected: " + coordLocked.join(", ");
      return;
    }
    statusL.style.color = "";
    statusL.textContent = `x: ${Math.round(ecL)}  y: ${Math.round(ecT)}  →  ${Math.round(ecW)} × ${Math.round(ecH)} px`;
  }
  return { controls, statusBar, sync, setFrame, updateStatus };
}

// src/CropVideoEditor/CropVideoEditor.ts
var RESOLUTION_PRESETS2 = {
  "480p (854x480)": [854, 480],
  "720p (1280x720)": [1280, 720],
  "1080p (1920x1080)": [1920, 1080],
  "1440p (2560x1440)": [2560, 1440],
  "2160p (3840x2160)": [3840, 2160],
  "Square (1080x1080)": [1080, 1080],
  "Portrait (1080x1920)": [1080, 1920],
  "4:3 (1440x1080)": [1440, 1080]
};
function cropSize(v) {
  const size = v.crop_size || "Custom";
  if (size === "Custom") {
    const w = v.custom_width || 0, h = v.custom_height || 0;
    return w > 0 && h > 0 ? [w, h] : null;
  }
  const preset = Object.entries(RESOLUTION_PRESETS2).find(([name]) => name.toLowerCase() === size.trim().toLowerCase());
  if (preset) return preset[1];
  const parts = size.split("x");
  if (parts.length === 2) {
    const w = parseInt(parts[0], 10), h = parseInt(parts[1], 10);
    if (w > 0 && h > 0) return [w, h];
  }
  return null;
}
function cropFromSettings(v, vw, vh) {
  const dims = cropSize(v);
  const w = dims ? Math.min(dims[0], vw) : vw;
  const h = dims ? Math.min(dims[1], vh) : vh;
  const dx = vw - w, dy = vh - h;
  const position = v.crop_position || "center";
  const positions = {
    "center": [Math.floor(dx / 2), Math.floor(dy / 2)],
    "top-left": [0, 0],
    "top-right": [dx, 0],
    "bottom-left": [0, dy],
    "bottom-right": [dx, dy],
    "top-center": [Math.floor(dx / 2), 0],
    "bottom-center": [Math.floor(dx / 2), dy],
    "left-center": [0, Math.floor(dy / 2)],
    "right-center": [dx, Math.floor(dy / 2)]
  };
  let [l, t] = position === "Custom" ? [v.custom_left || 0, v.custom_top || 0] : positions[position] || positions.center;
  l = Math.max(0, Math.min(l, dx));
  t = Math.max(0, Math.min(t, dy));
  return { l, t, w, h };
}
function lockedEdges(locked) {
  const has = (name) => (locked || []).includes(name);
  const edges = [];
  if (has("crop_size") || has("custom_width")) edges.push("width");
  if (has("crop_size") || has("custom_height")) edges.push("height");
  if (has("crop_position") || has("custom_left")) edges.push("left");
  if (has("crop_position") || has("custom_top")) edges.push("top");
  return edges;
}
function CropVideoEditor(host, props) {
  const container = host;
  const { value, onChange, disabled } = props;
  if (container._instance?.wrapper?.isConnected) {
    container._instance.handleUpdate(props);
    return { cleanup: container._instance.cleanup, update: container._instance.handleUpdate };
  }
  let videoUrl = value?.video_url || "";
  let vidNatW = value?.video_width || 0;
  let vidNatH = value?.video_height || 0;
  let totalFrames = value?.total_frames || 0;
  let lockedParams = lockedEdges(props.locked);
  let latestValue = value || {};
  let isDisabled = !!disabled;
  let currentFrame = 0;
  const initialCrop = cropFromSettings(latestValue, vidNatW, vidNatH);
  let ecL = initialCrop.l;
  let ecT = initialCrop.t;
  let ecW = initialCrop.w;
  let ecH = initialCrop.h;
  let onChangeRef = onChange;
  let onLiveChangeRef = props.onLiveChange;
  let videoLoaded = false;
  let scale = 1;
  let mode = "idle";
  let dragHandle = null;
  let dragStart = null;
  let rectAtDrag = null;
  let hoverHandle = null;
  const hitR = HANDLE_R + 5;
  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;height:100%;gap:6px;";
  const canvasRow = document.createElement("div");
  canvasRow.style.cssText = "display:flex;flex-direction:row;flex:1 1 0;min-height:180px;gap:6px;";
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "flex:1 1 0",
    "min-height:0",
    "position:relative",
    "background:#111",
    "border-radius:6px",
    "overflow:hidden"
  ].join(";");
  const video = document.createElement("video");
  video.muted = true;
  video.preload = "metadata";
  video.crossOrigin = "anonymous";
  video.style.cssText = [
    "position:absolute",
    "inset:0",
    "width:100%",
    "height:100%",
    "object-fit:contain",
    "pointer-events:none"
  ].join(";");
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:absolute;inset:0;width:100%;height:100%;";
  const ctx = canvas.getContext("2d");
  canvasWrap.appendChild(video);
  canvasWrap.appendChild(canvas);
  const sidebarInst = createSidebar({
    getImgSize: () => ({ imgNatW: vidNatW, imgNatH: vidNatH }),
    getCropRect: () => ({ l: ecL, t: ecT, w: ecW, h: ecH }),
    onApply(crop) {
      mode = "idle";
      ecL = crop.left;
      ecT = crop.top;
      ecW = crop.width;
      ecH = crop.height;
      render();
      emitAll();
    }
  });
  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);
  const footerInst = createFooter({
    isDisabled: () => isDisabled,
    onSeek(frame) {
      seekToFrame(frame);
    },
    version: WIDGET_VERSION
  });
  wrapper.appendChild(canvasRow);
  wrapper.appendChild(footerInst.controls);
  wrapper.appendChild(footerInst.statusBar);
  container.appendChild(wrapper);
  function urlPathname(u) {
    if (!u) return "";
    try {
      return new URL(u, location.href).pathname;
    } catch {
      return u.split("?")[0];
    }
  }
  function loadVideo(url) {
    if (!url) return;
    if (video.src && urlPathname(video.src) === urlPathname(url)) return;
    videoLoaded = false;
    video.src = url;
    video.load();
  }
  video.addEventListener("loadedmetadata", () => {
    videoLoaded = true;
    if (!vidNatW) vidNatW = video.videoWidth;
    if (!vidNatH) vidNatH = video.videoHeight;
    if (mode === "idle") ({ l: ecL, t: ecT, w: ecW, h: ecH } = cropFromSettings(latestValue, vidNatW, vidNatH));
    syncCanvasSize();
    if (!ecW) ecW = vidNatW;
    if (!ecH) ecH = vidNatH;
    footerInst.sync(totalFrames, isDisabled);
    if (currentFrame > 0 && video.duration) {
      video.currentTime = currentFrame / totalFrames * video.duration;
    }
    syncUI();
    render();
  });
  video.addEventListener("seeked", () => render());
  function syncCanvasSize() {
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    const natW = vidNatW || video.videoWidth || 1;
    const natH = vidNatH || video.videoHeight || 1;
    scale = Math.min(areaW / natW, areaH / natH);
    const cw = Math.round(natW * scale);
    const ch = Math.round(natH * scale);
    canvas.width = cw;
    canvas.height = ch;
    canvas.style.cssText = [
      "position:absolute",
      `left:${Math.round((areaW - cw) / 2)}px`,
      `top:${Math.round((areaH - ch) / 2)}px`,
      `width:${cw}px`,
      `height:${ch}px`
    ].join(";");
    render();
  }
  const ro = new ResizeObserver(() => {
    if (videoLoaded) syncCanvasSize();
  });
  ro.observe(canvasWrap);
  function seekToFrame(frame) {
    if (!video.duration || totalFrames <= 0) return;
    currentFrame = frame;
    video.currentTime = frame / totalFrames * video.duration;
    footerInst.setFrame(frame, totalFrames);
  }
  function getHandles() {
    const l = ecL * scale, t = ecT * scale;
    const r = (ecL + ecW) * scale, b = (ecT + ecH) * scale;
    const mx = (l + r) / 2, my = (t + b) / 2;
    return [
      { id: "nw", cx: l, cy: t, affects: ["left", "top", "width", "height"] },
      { id: "n", cx: mx, cy: t, affects: ["top", "height"] },
      { id: "ne", cx: r, cy: t, affects: ["top", "width", "height"] },
      { id: "e", cx: r, cy: my, affects: ["width"] },
      { id: "se", cx: r, cy: b, affects: ["width", "height"] },
      { id: "s", cx: mx, cy: b, affects: ["height"] },
      { id: "sw", cx: l, cy: b, affects: ["left", "width", "height"] },
      { id: "w", cx: l, cy: my, affects: ["left", "width"] }
    ];
  }
  function isHandleLocked(h) {
    return h.affects.some((f) => lockedParams.includes(f));
  }
  function hitHandle(px, py) {
    for (const h of getHandles()) {
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h;
    }
    return null;
  }
  function insideCrop(px, py) {
    return px >= ecL * scale && px <= (ecL + ecW) * scale && py >= ecT * scale && py <= (ecT + ecH) * scale;
  }
  function render() {
    const cw = canvas.width, ch = canvas.height;
    ctx.clearRect(0, 0, cw, ch);
    if (!videoLoaded) {
      ctx.fillStyle = "rgba(255,255,255,0.25)";
      ctx.font = `${Math.max(11, Math.round(cw / 22))}px sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Connect a video to begin", cw / 2, ch / 2);
      return;
    }
    const l = ecL * scale, t = ecT * scale;
    const w = ecW * scale, h = ecH * scale;
    ctx.fillStyle = OVERLAY;
    ctx.fillRect(0, 0, cw, ch);
    ctx.globalCompositeOperation = "destination-out";
    ctx.fillStyle = "rgba(0,0,0,1)";
    ctx.fillRect(l, t, w, h);
    ctx.globalCompositeOperation = "source-over";
    ctx.strokeStyle = GUIDE;
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(l + w * i / 3, t);
      ctx.lineTo(l + w * i / 3, t + h);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(l, t + h * i / 3);
      ctx.lineTo(l + w, t + h * i / 3);
      ctx.stroke();
    }
    ctx.strokeStyle = CROP_BORDER;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(l, t, w, h);
    for (const hand of getHandles()) {
      const locked = isHandleLocked(hand);
      const hovered = !locked && hand.id === hoverHandle;
      ctx.beginPath();
      ctx.arc(hand.cx, hand.cy, HANDLE_R, 0, Math.PI * 2);
      ctx.fillStyle = HANDLE_FILL;
      ctx.fill();
      ctx.strokeStyle = locked ? HANDLE_LOCKED : hovered ? HANDLE_STROKE_HOVER : HANDLE_STROKE;
      ctx.lineWidth = locked ? 1.5 : 2;
      ctx.stroke();
    }
  }
  function canvasXY(e) {
    const rect = canvas.getBoundingClientRect();
    const sx = canvas.width / (rect.width || canvas.width);
    const sy = canvas.height / (rect.height || canvas.height);
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }
  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }
  canvas.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (isDisabled) return;
    const { x, y } = canvasXY(e);
    const hand = hitHandle(x, y);
    if (hand && !isHandleLocked(hand)) {
      mode = "resizing";
      dragHandle = hand;
      dragStart = { x, y };
      rectAtDrag = { l: ecL, t: ecT, w: ecW, h: ecH };
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    const leftLocked = lockedParams.includes("left");
    const topLocked = lockedParams.includes("top");
    const posLocked = leftLocked && topLocked;
    if (insideCrop(x, y) && !(leftLocked && topLocked)) {
      mode = "moving";
      dragStart = { x, y };
      rectAtDrag = { l: ecL, t: ecT, w: ecW, h: ecH };
      canvas.setPointerCapture(e.pointerId);
      return;
    }
    const sizeLocked = lockedParams.includes("width") || lockedParams.includes("height");
    if (!(leftLocked || topLocked) && !sizeLocked) {
      mode = "drawing";
      dragStart = { x: x / scale, y: y / scale };
      canvas.setPointerCapture(e.pointerId);
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    e.stopPropagation();
    const { x, y } = canvasXY(e);
    if (mode === "idle") {
      const hand = hitHandle(x, y);
      const newHover = hand && !isHandleLocked(hand) ? hand.id : null;
      if (newHover !== hoverHandle) {
        hoverHandle = newHover;
        render();
      }
      canvas.style.cursor = newHover ? "crosshair" : insideCrop(x, y) ? "move" : "crosshair";
      return;
    }
    if (mode === "moving") {
      const start = dragStart, rect = rectAtDrag;
      const dx = (x - start.x) / scale, dy = (y - start.y) / scale;
      if (!lockedParams.includes("left")) ecL = clamp(Math.round(rect.l + dx), 0, vidNatW - ecW);
      if (!lockedParams.includes("top")) ecT = clamp(Math.round(rect.t + dy), 0, vidNatH - ecH);
      render();
      emitLive();
      return;
    }
    if (mode === "resizing") {
      const nx = x / scale, ny = y / scale;
      const rect = rectAtDrag;
      const { l, t, w, h } = rect;
      let nl = l, nt = t, nw = w, nh = h;
      const id = dragHandle.id;
      if (id.includes("e")) nw = clamp(Math.round(nx - nl), 1, vidNatW - nl);
      if (id.includes("s")) nh = clamp(Math.round(ny - nt), 1, vidNatH - nt);
      if (id.includes("w")) {
        const r = nl + nw;
        nl = clamp(Math.round(nx), 0, r - 1);
        nw = r - nl;
      }
      if (id.includes("n")) {
        const b = nt + nh;
        nt = clamp(Math.round(ny), 0, b - 1);
        nh = b - nt;
      }
      if (e.shiftKey && (id === "se" || id === "nw" || id === "ne" || id === "sw")) {
        const asp = rect.w / rect.h;
        if (nw / nh > asp) nh = Math.round(nw / asp);
        else nw = Math.round(nh * asp);
      }
      ecL = nl;
      ecT = nt;
      ecW = nw;
      ecH = nh;
      render();
      emitLive();
      return;
    }
    if (mode === "drawing") {
      const start = dragStart;
      const nx = clamp(x / scale, 0, vidNatW), ny = clamp(y / scale, 0, vidNatH);
      ecL = Math.round(Math.min(start.x, nx));
      ecT = Math.round(Math.min(start.y, ny));
      ecW = Math.max(1, Math.round(Math.abs(nx - start.x)));
      ecH = e.shiftKey ? ecW : Math.max(1, Math.round(Math.abs(ny - start.y)));
      render();
      emitLive();
    }
  });
  canvas.addEventListener("pointerup", (e) => {
    e.stopPropagation();
    if (mode !== "idle") {
      mode = "idle";
      render();
      emitAll();
    }
  });
  function emitAll() {
    if (typeof onChangeRef !== "function") return;
    onChangeRef({
      ...latestValue,
      custom_left: ecL,
      custom_top: ecT,
      custom_width: ecW,
      custom_height: ecH,
      crop_size: "Custom",
      crop_position: "Custom"
    });
  }
  function emitLive() {
    if (typeof onLiveChangeRef !== "function") return;
    onLiveChangeRef({ custom_left: ecL, custom_top: ecT, custom_width: ecW, custom_height: ecH });
  }
  function syncUI() {
    sidebarInst.syncDisabled(lockedParams, isDisabled);
    footerInst.sync(totalFrames, isDisabled);
    footerInst.updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH });
  }
  function handleUpdate(newProps) {
    const v = newProps.value || {};
    latestValue = v;
    onChangeRef = newProps.onChange;
    onLiveChangeRef = newProps.onLiveChange;
    isDisabled = !!newProps.disabled;
    const newUrl = v.video_url || "";
    const videoChanged = urlPathname(newUrl) !== urlPathname(videoUrl);
    if (videoChanged) {
      videoUrl = newUrl;
      vidNatW = v.video_width || 0;
      vidNatH = v.video_height || 0;
      totalFrames = v.total_frames || 0;
      currentFrame = 0;
      videoLoaded = false;
      loadVideo(newUrl);
    } else {
      if (newUrl) videoUrl = newUrl;
      if (v.video_width) vidNatW = v.video_width;
      if (v.video_height) vidNatH = v.video_height;
      if (v.total_frames !== void 0) totalFrames = v.total_frames;
    }
    lockedParams = lockedEdges(newProps.locked);
    if (mode === "idle" && vidNatW && vidNatH) {
      ({ l: ecL, t: ecT, w: ecW, h: ecH } = cropFromSettings(v, vidNatW, vidNatH));
    }
    syncUI();
    if (videoLoaded) render();
  }
  function cleanup() {
    ro.disconnect();
    video.pause();
    video.removeAttribute("src");
    video.load();
    wrapper.remove();
    delete container._instance;
  }
  loadVideo(videoUrl);
  syncUI();
  container._instance = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
export {
  CropVideoEditor as default
};
