// Zoom + rotate slider controls and status bar for CropImageEditor.

import { mkIcon } from './_icons.js';
import { SLIDER_ACCENT } from './_styles.js';

function makeSliderRow({ label, min, max, step, getVal, setVal, format, paramKey, defaultVal, isLocked, isDisabled, onRender, onEmit }) {
  const row = document.createElement("div");
  row.style.cssText = "display:flex;align-items:center;gap:8px;";

  const lbl = document.createElement("span");
  lbl.textContent = label;
  lbl.style.cssText = "font-size:11px;color:var(--muted-foreground);min-width:44px;";

  const slider = document.createElement("input");
  slider.type = "range";
  slider.min = min; slider.max = max; slider.step = step;
  slider.value = getVal();
  slider.style.cssText = `flex:1;accent-color:${SLIDER_ACCENT};`;

  const display = document.createElement("span");
  display.textContent = format(getVal());
  display.style.cssText = "font-size:11px;color:var(--foreground);min-width:38px;text-align:right;font-family:monospace;";

  const resetBtn = document.createElement("button");
  resetBtn.title = `Reset to ${format(defaultVal)}`;
  resetBtn.style.cssText = [
    "background:none", "border:none", "color:var(--muted-foreground)", "cursor:pointer",
    "padding:0 2px", "line-height:1", "flex-shrink:0",
    "display:flex", "align-items:center",
  ].join(";");
  resetBtn.appendChild(mkIcon("rotate-ccw", 13));
  resetBtn.addEventListener("mouseenter", () => { if (!resetBtn.disabled) resetBtn.style.color = "var(--foreground)"; });
  resetBtn.addEventListener("mouseleave", () => { resetBtn.style.color = "var(--muted-foreground)"; });

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
    slider.value = defaultVal;
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
    slider.value = getVal();
    display.textContent = format(getVal());
    row.style.opacity = locked ? "0.4" : "1";
  }

  return { row, sync };
}

export function createFooter({ getZoom, setZoom, getRotate, setRotate, isLocked, isDisabled, onRender, onEmit, version }) {
  const controls = document.createElement("div");
  controls.className = "nodrag nowheel";
  controls.style.cssText = [
    "background:var(--muted)", "border-radius:4px", "padding:6px 8px",
    "display:flex", "flex-direction:column", "gap:5px",
  ].join(";");

  const zoomRow = makeSliderRow({
    label: "Zoom", min: 10, max: 500, step: 1,
    getVal: getZoom, setVal: setZoom,
    format: (v) => Math.round(v) + "%",
    paramKey: "zoom", defaultVal: 100,
    isLocked, isDisabled, onRender, onEmit,
  });

  const rotateRow = makeSliderRow({
    label: "Rotate", min: -180, max: 180, step: 1,
    getVal: getRotate, setVal: setRotate,
    format: (v) => (v >= 0 ? "+" : "") + Math.round(v) + "°",
    paramKey: "rotate", defaultVal: 0,
    isLocked, isDisabled, onRender, onEmit,
  });

  controls.appendChild(zoomRow.row);
  controls.appendChild(rotateRow.row);

  // ── Status bar ─────────────────────────────────────────────────────────────
  const statusBar = document.createElement("div");
  statusBar.style.cssText = [
    "font-size:11px", "color:var(--muted-foreground)", "padding:4px 8px",
    "background:var(--muted)", "border-radius:4px",
    "display:flex", "justify-content:space-between", "align-items:center",
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
    const coordLocked = ["left", "top", "width", "height"].filter(f => lockedParams.includes(f));
    if (coordLocked.length > 0) {
      statusL.style.color = "#f59e0b";
      statusL.textContent = "\u{1F512} Connected: " + coordLocked.join(", ");
      return;
    }
    statusL.style.color = "";
    statusL.textContent = `x: ${Math.round(ecL)}  y: ${Math.round(ecT)}  →  ${Math.round(ecW)} × ${Math.round(ecH)} px`;
  }

  return { controls, statusBar, sync, updateStatus };
}
