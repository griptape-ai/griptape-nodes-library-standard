// src/CropImageEditor/_styles.ts
var WIDGET_VERSION = "1.0.0";
var HANDLE_R = 6;
var OVERLAY = "rgba(0,0,0,0.55)";
var CROP_BORDER = "rgba(255,255,255,0.9)";
var GUIDE = "rgba(255,255,255,0.25)";
var TRANSFORM_PREVIEW = "#1bb0f4";
var TRANSFORM_OVERLAY = "rgba(0,0,0,0.45)";
var SLIDER_ACCENT = "#a7c6c7";
var HANDLE_FILL = "black";
var HANDLE_STROKE = "#ffffff";
var HANDLE_STROKE_HOVER = "#60a5fa";
var HANDLE_LOCKED = "rgba(150,150,150,0.5)";

// src/CropImageEditor/_icons.ts
var ICON_PATHS = {
  "rotate-ccw": `<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>`
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

// src/CropImageEditor/_sidebar.ts
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
  el.appendChild(makeBtn({ icon: "rotate-ccw", label: "Reset", title: "Reset crop to full image", onClick: () => {
    const { imgNatW, imgNatH } = getImgSize();
    onApply({ left: 0, top: 0, width: imgNatW, height: imgNatH, zoom: 100, rotate: 0 });
  } }));
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Ratio"));
  const ratioGrid = document.createElement("div");
  ratioGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, rw, rh } of RATIO_PRESETS) {
    const btn = makeBtn({ label, title: label, onClick: () => {
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcRatioRect(rw, rh, imgNatW, imgNatH, l, t, w, h));
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
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcResolutionRect(pw, ph, imgNatW, imgNatH, l, t, w, h));
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
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcPositionRect(pos, imgNatW, imgNatH, l, t, w, h));
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

// src/CropImageEditor/_footer.ts
function makeSliderRow({ label, min, max, step, getVal, setVal, format, paramKey, defaultVal, isLocked, isDisabled, onRender, onEmit }) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:8px;";
  const lbl = document.createElement("span");
  lbl.textContent = label;
  lbl.style.cssText = "font-size:11px;color:var(--muted-foreground);min-width:44px;";
  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = String(min);
  slider.max = String(max);
  slider.step = String(step);
  slider.value = String(getVal());
  slider.style.cssText = `flex:1;accent-color:${SLIDER_ACCENT};`;
  const display = document.createElement("span");
  display.textContent = format(getVal());
  display.style.cssText = "font-size:11px;color:var(--foreground);min-width:38px;text-align:right;font-family:monospace;";
  const resetBtn = document.createElement("button");
  resetBtn.title = `Reset to ${format(defaultVal)}`;
  resetBtn.style.cssText = [
    "background:none",
    "border:none",
    "color:var(--muted-foreground)",
    "cursor:pointer",
    "padding:0 2px",
    "line-height:1",
    "flex-shrink:0",
    "display:flex",
    "align-items:center"
  ].join(";");
  resetBtn.appendChild(mkIcon("rotate-ccw", 13));
  resetBtn.addEventListener("mouseenter", () => {
    if (!resetBtn.disabled) resetBtn.style.color = "var(--foreground)";
  });
  resetBtn.addEventListener("mouseleave", () => {
    resetBtn.style.color = "var(--muted-foreground)";
  });
  for (const el of [slider, resetBtn]) {
    el.addEventListener("mousedown", (e) => e.stopPropagation());
    el.addEventListener("pointerdown", (e) => e.stopPropagation());
  }
  slider.addEventListener("input", (e) => {
    if (isLocked(paramKey) || isDisabled()) return;
    e.stopPropagation();
    setVal(parseFloat(e.target.value));
    display.textContent = format(getVal());
    onRender();
  });
  slider.addEventListener("change", (e) => {
    if (isLocked(paramKey) || isDisabled()) return;
    e.stopPropagation();
    onEmit();
  });
  resetBtn.addEventListener("click", (e) => {
    if (isLocked(paramKey) || isDisabled()) return;
    e.stopPropagation();
    setVal(defaultVal);
    slider.value = String(defaultVal);
    display.textContent = format(defaultVal);
    onRender();
    onEmit();
  });
  row.appendChild(lbl);
  row.appendChild(slider);
  row.appendChild(display);
  row.appendChild(resetBtn);
  function sync(lockedParams, disabled) {
    const locked = lockedParams.includes(paramKey) || disabled;
    slider.disabled = locked;
    resetBtn.disabled = locked;
    slider.value = String(getVal());
    display.textContent = format(getVal());
    row.style.opacity = locked ? "0.4" : "1";
  }
  return { row, sync };
}
function createFooter({ getZoom, setZoom, getRotate, setRotate, isLocked, isDisabled, onRender, onEmit, version }) {
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
  const zoomRow = makeSliderRow({
    label: "Zoom",
    min: 10,
    max: 500,
    step: 1,
    getVal: getZoom,
    setVal: setZoom,
    format: (v) => Math.round(v) + "%",
    paramKey: "zoom",
    defaultVal: 100,
    isLocked,
    isDisabled,
    onRender,
    onEmit
  });
  const rotateRow = makeSliderRow({
    label: "Rotate",
    min: -180,
    max: 180,
    step: 1,
    getVal: getRotate,
    setVal: setRotate,
    format: (v) => (v >= 0 ? "+" : "") + Math.round(v) + "°",
    paramKey: "rotate",
    defaultVal: 0,
    isLocked,
    isDisabled,
    onRender,
    onEmit
  });
  controls.appendChild(zoomRow.row);
  controls.appendChild(rotateRow.row);
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
  function sync(lockedParams, disabled) {
    zoomRow.sync(lockedParams, disabled);
    rotateRow.sync(lockedParams, disabled);
  }
  function updateStatus({ imageLoaded, lockedParams, ecL, ecT, ecW, ecH }) {
    if (!imageLoaded) {
      statusL.textContent = "No image connected";
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
  return { controls, statusBar, sync, updateStatus };
}

// src/CropImageEditor/CropImageEditor.ts
function CropImageEditor(host, props) {
  const container = host;
  if (container._cropEditor?.wrapper?.isConnected) {
    container._cropEditor.handleUpdate(props);
    return { cleanup: container._cropEditor.cleanup, update: container._cropEditor.handleUpdate };
  }
  if (container._cropEditor) delete container._cropEditor;
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let onLiveChangeRef = props.onLiveChange;
  let isDisabled = props.disabled || false;
  let imageUrl = latestValue.image_url || "";
  let imgNatW = latestValue.img_width || 0;
  let imgNatH = latestValue.img_height || 0;
  let cropL = latestValue.left ?? 0;
  let cropT = latestValue.top ?? 0;
  let cropW = latestValue.width ?? 0;
  let cropH = latestValue.height ?? 0;
  let cropZoom = latestValue.zoom ?? 100;
  let cropRotate = latestValue.rotate ?? 0;
  let lockedParams = props.locked || [];
  function ecL() {
    return Math.max(0, cropL);
  }
  function ecT() {
    return Math.max(0, cropT);
  }
  function ecW() {
    return cropW > 0 ? cropW : imgNatW || 1;
  }
  function ecH() {
    return cropH > 0 ? cropH : imgNatH || 1;
  }
  function handleAffects(id) {
    const fields = [];
    if (id.includes("n")) {
      fields.push("top");
      fields.push("height");
    }
    if (id.includes("s")) {
      fields.push("height");
    }
    if (id.includes("w")) {
      fields.push("left");
      fields.push("width");
    }
    if (id.includes("e")) {
      fields.push("width");
    }
    return fields;
  }
  function isHandleLocked(id) {
    return handleAffects(id).some((f) => lockedParams.includes(f));
  }
  function isMoveLocked() {
    return lockedParams.includes("left") || lockedParams.includes("top");
  }
  function isDrawLocked() {
    return ["left", "top", "width", "height"].some((f) => lockedParams.includes(f));
  }
  let scale = 1;
  let mode = "idle";
  let activeHandle = null;
  let dragStart = null;
  let hoveredHandle = null;
  const wrapper = document.createElement("div");
  wrapper.className = "crop-image-editor nodrag nowheel";
  wrapper.style.cssText = [
    "display:flex",
    "flex-direction:column",
    "gap:6px",
    "height:100%",
    "user-select:none",
    "-webkit-user-select:none"
  ].join(";");
  const canvasRow = document.createElement("div");
  canvasRow.style.cssText = [
    "display:flex",
    "flex-direction:row",
    "gap:6px",
    "flex:1 1 0",
    "min-height:180px"
  ].join(";");
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = [
    "position:relative",
    "background:#111",
    "border-radius:6px",
    "overflow:hidden",
    "flex:1 1 0",
    "min-height:0",
    "display:flex",
    "align-items:center",
    "justify-content:center"
  ].join(";");
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;cursor:crosshair;";
  const ctx = canvas.getContext("2d");
  const sidebarInst = createSidebar({
    getImgSize: () => ({ imgNatW, imgNatH }),
    getCropRect: () => ({ l: ecL(), t: ecT(), w: ecW(), h: ecH() }),
    onApply(crop) {
      if (isDisabled || !imageLoaded || !imgNatW || !imgNatH) return;
      mode = "idle";
      activeHandle = null;
      dragStart = null;
      hoveredHandle = null;
      cropL = crop.left;
      cropT = crop.top;
      cropW = crop.width;
      cropH = crop.height;
      if ("zoom" in crop) cropZoom = crop.zoom;
      if ("rotate" in crop) cropRotate = crop.rotate;
      commit();
      syncUI();
    }
  });
  const footerInst = createFooter({
    getZoom: () => cropZoom,
    setZoom: (v) => {
      cropZoom = v;
    },
    getRotate: () => cropRotate,
    setRotate: (v) => {
      cropRotate = v;
    },
    isLocked: (key) => lockedParams.includes(key),
    isDisabled: () => isDisabled,
    onRender: () => {
      render();
      emitLive();
    },
    onEmit: () => emitAll(),
    version: WIDGET_VERSION
  });
  canvasWrap.appendChild(canvas);
  canvasRow.appendChild(canvasWrap);
  canvasRow.appendChild(sidebarInst.el);
  wrapper.appendChild(canvasRow);
  wrapper.appendChild(footerInst.controls);
  wrapper.appendChild(footerInst.statusBar);
  container.appendChild(wrapper);
  function syncUI() {
    footerInst.sync(lockedParams, isDisabled);
    sidebarInst.syncDisabled(lockedParams, isDisabled);
  }
  const img = new Image();
  img.crossOrigin = "anonymous";
  let imageLoaded = false;
  function initCrop() {
    if (imgNatW > 0 && imgNatH > 0) {
      if (cropW === 0) cropW = imgNatW;
      if (cropH === 0) cropH = imgNatH;
    }
  }
  function resizeCanvas() {
    if (!imageLoaded || !imgNatW || !imgNatH) return;
    const areaW = canvasWrap.clientWidth || 480;
    const areaH = canvasWrap.clientHeight || 360;
    scale = Math.min(areaW / imgNatW, areaH / imgNatH);
    canvas.width = Math.round(imgNatW * scale);
    canvas.height = Math.round(imgNatH * scale);
    render();
  }
  img.onload = () => {
    imageLoaded = true;
    if (!imgNatW) imgNatW = img.naturalWidth;
    if (!imgNatH) imgNatH = img.naturalHeight;
    initCrop();
    resizeCanvas();
  };
  img.onerror = () => {
    imageLoaded = false;
    render();
  };
  function loadImage(url) {
    if (!url) {
      imageLoaded = false;
      render();
      return;
    }
    const base = (u) => u ? u.split("?")[0] : "";
    if (base(url) === base(img.src)) return;
    img.src = url;
  }
  function emitAll() {
    if (typeof onChangeRef === "function") {
      onChangeRef({
        ...latestValue,
        left: cropL,
        top: cropT,
        width: cropW,
        height: cropH,
        zoom: cropZoom,
        rotate: cropRotate
      });
    }
  }
  function emitLive() {
    if (typeof onLiveChangeRef !== "function") return;
    const [l, t, w, h] = clampRect(cropL, cropT, cropW, cropH);
    onLiveChangeRef({ left: l, top: t, width: w, height: h, zoom: cropZoom, rotate: cropRotate });
  }
  function render() {
    const cw = canvas.width || 400;
    const ch = canvas.height || 300;
    ctx.clearRect(0, 0, cw, ch);
    if (!imageLoaded) {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
      ctx.fillStyle = "#555";
      ctx.font = "13px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText("Connect an image to begin", cw / 2, ch / 2);
      footerInst.updateStatus({ imageLoaded, lockedParams, ecL: 0, ecT: 0, ecW: 0, ecH: 0 });
      return;
    }
    ctx.drawImage(img, 0, 0, cw, ch);
    const x = ecL() * scale, y = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const rw = x2 - x, rh = y2 - y;
    ctx.fillStyle = OVERLAY;
    if (y > 0) ctx.fillRect(0, 0, cw, y);
    if (y2 < ch) ctx.fillRect(0, y2, cw, ch - y2);
    if (x > 0) ctx.fillRect(0, y, x, rh);
    if (x2 < cw) ctx.fillRect(x2, y, cw - x2, rh);
    ctx.save();
    ctx.strokeStyle = GUIDE;
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    for (let i = 1; i < 3; i++) {
      ctx.moveTo(x + rw * i / 3, y);
      ctx.lineTo(x + rw * i / 3, y2);
      ctx.moveTo(x, y + rh * i / 3);
      ctx.lineTo(x2, y + rh * i / 3);
    }
    ctx.stroke();
    ctx.restore();
    const hasTransform = cropZoom !== 100 && cropZoom > 0 || cropRotate !== 0;
    if (hasTransform) {
      const zoomFactor = Math.max(0.01, cropZoom / 100);
      const cx_px = (ecL() + ecW() / 2) * scale;
      const cy_px = (ecT() + ecH() / 2) * scale;
      const pw = ecW() / zoomFactor * scale;
      const ph = ecH() / zoomFactor * scale;
      ctx.save();
      ctx.fillStyle = TRANSFORM_OVERLAY;
      ctx.beginPath();
      ctx.rect(0, 0, cw, ch);
      ctx.translate(cx_px, cy_px);
      ctx.rotate(cropRotate * Math.PI / 180);
      ctx.rect(-pw / 2, -ph / 2, pw, ph);
      ctx.restore();
      ctx.fill("evenodd");
      ctx.save();
      ctx.translate(cx_px, cy_px);
      ctx.rotate(cropRotate * Math.PI / 180);
      ctx.setLineDash([5, 3]);
      ctx.strokeStyle = TRANSFORM_PREVIEW;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(-pw / 2, -ph / 2, pw, ph);
      ctx.setLineDash([]);
      ctx.fillStyle = TRANSFORM_PREVIEW;
      ctx.beginPath();
      ctx.moveTo(0, -ph / 2 - 8);
      ctx.lineTo(-5, -ph / 2);
      ctx.lineTo(5, -ph / 2);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }
    ctx.strokeStyle = CROP_BORDER;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x, y, rw, rh);
    if (!isDisabled) {
      for (const h of getHandles()) {
        const locked = isHandleLocked(h.id);
        ctx.beginPath();
        ctx.arc(h.cx, h.cy, HANDLE_R, 0, Math.PI * 2);
        ctx.fillStyle = locked ? HANDLE_LOCKED : HANDLE_FILL;
        ctx.fill();
        ctx.strokeStyle = locked ? "rgba(100,100,100,0.6)" : hoveredHandle === h.id ? HANDLE_STROKE_HOVER : HANDLE_STROKE;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    }
    footerInst.updateStatus({ imageLoaded, lockedParams, ecL: ecL(), ecT: ecT(), ecW: ecW(), ecH: ecH() });
  }
  function getHandles() {
    const x = ecL() * scale, y = ecT() * scale;
    const x2 = (ecL() + ecW()) * scale, y2 = (ecT() + ecH()) * scale;
    const mx = (x + x2) / 2, my = (y + y2) / 2;
    return [
      { id: "nw", cx: x, cy: y },
      { id: "n", cx: mx, cy: y },
      { id: "ne", cx: x2, cy: y },
      { id: "e", cx: x2, cy: my },
      { id: "se", cx: x2, cy: y2 },
      { id: "s", cx: mx, cy: y2 },
      { id: "sw", cx: x, cy: y2 },
      { id: "w", cx: x, cy: my }
    ];
  }
  function hitHandle(px, py) {
    const hitR = HANDLE_R + 5;
    for (const h of getHandles()) {
      if (isHandleLocked(h.id)) continue;
      if (Math.hypot(px - h.cx, py - h.cy) <= hitR) return h.id;
    }
    return null;
  }
  function hitCrop(px, py) {
    return px >= ecL() * scale && px <= (ecL() + ecW()) * scale && py >= ecT() * scale && py <= (ecT() + ecH()) * scale;
  }
  const CURSORS = { nw: "nw-resize", n: "n-resize", ne: "ne-resize", e: "e-resize", se: "se-resize", s: "s-resize", sw: "sw-resize", w: "w-resize" };
  function canvasPos(e) {
    const r = canvas.getBoundingClientRect();
    return [(e.clientX - r.left) * (canvas.width / r.width), (e.clientY - r.top) * (canvas.height / r.height)];
  }
  function toImg(cx, cy) {
    return [cx / scale, cy / scale];
  }
  function clampRect(l, t, w, h) {
    w = Math.max(1, Math.round(w));
    h = Math.max(1, Math.round(h));
    l = Math.max(0, Math.min(Math.round(l), imgNatW - w));
    t = Math.max(0, Math.min(Math.round(t), imgNatH - h));
    return [l, t, Math.min(w, imgNatW - l), Math.min(h, imgNatH - t)];
  }
  function commit() {
    const [l, t, w, h] = clampRect(cropL, cropT, cropW || imgNatW, cropH || imgNatH);
    cropL = l;
    cropT = t;
    cropW = w;
    cropH = h;
    emitAll();
    render();
  }
  canvas.addEventListener("pointerdown", (e) => {
    if (isDisabled || !imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toImg(px, py);
    const handle = hitHandle(px, py);
    if (handle) {
      canvas.setPointerCapture(e.pointerId);
      mode = "resizing";
      activeHandle = handle;
      dragStart = { ix, iy, l: cropL, t: cropT, w: ecW(), h: ecH() };
    } else if (hitCrop(px, py) && !isMoveLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "moving";
      canvas.style.cursor = "grabbing";
      dragStart = { ix, iy, l: cropL, t: cropT, w: ecW(), h: ecH() };
    } else if (!hitCrop(px, py) && !isDrawLocked()) {
      canvas.setPointerCapture(e.pointerId);
      mode = "drawing";
      const six = Math.max(0, Math.min(ix, imgNatW));
      const siy = Math.max(0, Math.min(iy, imgNatH));
      dragStart = { ix: six, iy: siy };
      cropL = six;
      cropT = siy;
      cropW = 0;
      cropH = 0;
    }
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!imageLoaded) return;
    e.stopPropagation();
    const [px, py] = canvasPos(e);
    const [ix, iy] = toImg(px, py);
    if (mode === "idle") {
      const h = hitHandle(px, py);
      hoveredHandle = h;
      if (h) canvas.style.cursor = CURSORS[h];
      else if (hitCrop(px, py)) canvas.style.cursor = isMoveLocked() ? "not-allowed" : "move";
      else canvas.style.cursor = isDrawLocked() ? "not-allowed" : "crosshair";
      render();
      return;
    }
    if (mode === "moving") {
      const start = dragStart;
      const dx = ix - start.ix, dy = iy - start.iy;
      cropW = start.w;
      cropH = start.h;
      cropL = Math.max(0, Math.min(start.l + dx, imgNatW - cropW));
      cropT = Math.max(0, Math.min(start.t + dy, imgNatH - cropH));
    } else if (mode === "resizing") {
      const start = dragStart;
      const { l, t, w, h } = start;
      let nl = l, nt = t, nr = l + w, nb = t + h;
      const cix = Math.max(0, Math.min(ix, imgNatW));
      const ciy = Math.max(0, Math.min(iy, imgNatH));
      const id = activeHandle;
      if (id.includes("n")) nt = Math.min(ciy, nb - 1);
      if (id.includes("s")) nb = Math.max(ciy, nt + 1);
      if (id.includes("w")) nl = Math.min(cix, nr - 1);
      if (id.includes("e")) nr = Math.max(cix, nl + 1);
      if (e.shiftKey && id.length === 2) {
        const ratio = start.w / start.h;
        const curW = nr - nl, curH = nb - nt;
        if (Math.abs(curW - start.w) >= Math.abs(curH - start.h) * ratio) {
          const ch = Math.round(curW / ratio);
          if (id.includes("n")) nt = Math.max(0, nb - ch);
          else nb = Math.min(imgNatH, nt + ch);
        } else {
          const cw = Math.round(curH * ratio);
          if (id.includes("w")) nl = Math.max(0, nr - cw);
          else nr = Math.min(imgNatW, nl + cw);
        }
      }
      cropL = nl;
      cropT = nt;
      cropW = nr - nl;
      cropH = nb - nt;
    } else if (mode === "drawing") {
      const { ix: six, iy: siy } = dragStart;
      const eix = Math.max(0, Math.min(ix, imgNatW));
      const eiy = Math.max(0, Math.min(iy, imgNatH));
      if (e.shiftKey) {
        const dx = eix - six, dy = eiy - siy;
        const side = Math.max(Math.abs(dx), Math.abs(dy));
        const maxW = dx < 0 ? six : imgNatW - six;
        const maxH = dy < 0 ? siy : imgNatH - siy;
        const s = Math.min(side, maxW, maxH);
        cropW = s;
        cropH = s;
        cropL = dx < 0 ? six - s : six;
        cropT = dy < 0 ? siy - s : siy;
      } else {
        cropL = Math.min(six, eix);
        cropT = Math.min(siy, eiy);
        cropW = Math.abs(eix - six);
        cropH = Math.abs(eiy - siy);
      }
    }
    render();
    emitLive();
  });
  canvas.addEventListener("pointerup", (e) => {
    if (!imageLoaded || mode === "idle") return;
    e.stopPropagation();
    mode = "idle";
    hoveredHandle = null;
    canvas.style.cursor = "crosshair";
    commit();
  });
  canvas.addEventListener("pointercancel", () => {
    mode = "idle";
    hoveredHandle = null;
    canvas.style.cursor = "crosshair";
    render();
  });
  canvas.addEventListener("mousedown", (e) => e.stopPropagation());
  canvas.addEventListener("keydown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("pointerdown", (e) => e.stopPropagation());
  canvasWrap.addEventListener("mousedown", (e) => e.stopPropagation());
  const ro = new ResizeObserver(() => {
    if (imageLoaded) resizeCanvas();
  });
  ro.observe(canvasWrap);
  if (imageUrl) {
    loadImage(imageUrl);
  } else {
    canvas.width = 400;
    canvas.height = 300;
    render();
  }
  syncUI();
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    onLiveChangeRef = newProps.onLiveChange;
    isDisabled = newProps.disabled || false;
    latestValue = newProps.value || {};
    const v = latestValue;
    const newUrl = v.image_url || "";
    const newNatW = v.img_width || 0;
    const newNatH = v.img_height || 0;
    const base = (u) => u ? u.split("?")[0] : "";
    const urlChanged = base(newUrl) !== base(imageUrl);
    if (urlChanged) {
      imageUrl = newUrl;
      imgNatW = newNatW;
      imgNatH = newNatH;
      imageLoaded = false;
      loadImage(imageUrl);
    } else {
      if (newNatW) imgNatW = newNatW;
      if (newNatH) imgNatH = newNatH;
    }
    lockedParams = newProps.locked || [];
    if (mode === "idle") {
      if ("left" in v) cropL = v.left;
      if ("top" in v) cropT = v.top;
      if ("width" in v) cropW = v.width;
      if ("height" in v) cropH = v.height;
      if ("zoom" in v) cropZoom = v.zoom ?? cropZoom;
      if ("rotate" in v) cropRotate = v.rotate ?? cropRotate;
    }
    syncUI();
    if (imageLoaded && !urlChanged) render();
  }
  function cleanup() {
    ro.disconnect();
    wrapper.remove();
    delete container._cropEditor;
  }
  container._cropEditor = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
export {
  CropImageEditor as default
};
