// Frame scrubber and status bar for CropVideoEditor.

import { mkIcon } from './_icons.js';

export function createFooter({ isDisabled, onSeek, version }) {
  const controls = document.createElement("div");
  controls.className = "nodrag nowheel";
  controls.style.cssText = [
    "background:var(--muted)", "border-radius:4px", "padding:6px 8px",
    "display:flex", "flex-direction:column", "gap:5px",
  ].join(";");

  // ── Frame scrubber row ──────────────────────────────────────────────────────
  const scrubRow = document.createElement("div");
  scrubRow.style.cssText = "display:flex;align-items:center;gap:8px;";

  const filmIcon = document.createElement("span");
  filmIcon.style.cssText = "color:var(--muted-foreground);display:flex;align-items:center;flex-shrink:0;";
  filmIcon.appendChild(mkIcon("film", 13));

  const scrubSlider = document.createElement("input");
  scrubSlider.type = "range";
  scrubSlider.min = 0;
  scrubSlider.max = 0;
  scrubSlider.step = 1;
  scrubSlider.value = 0;
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

  function sync(totalFrames, disabled) {
    scrubSlider.disabled = disabled || totalFrames === 0;
    scrubSlider.max = Math.max(0, totalFrames - 1);
    scrubRow.style.opacity = (disabled || totalFrames === 0) ? "0.4" : "1";
  }

  function setFrame(frame, totalFrames) {
    scrubSlider.max = Math.max(0, totalFrames - 1);
    scrubSlider.value = frame;
    frameDisplay.textContent = `${frame} / ${totalFrames}`;
  }

  function updateStatus({ videoLoaded, lockedParams, ecL, ecT, ecW, ecH }) {
    if (!videoLoaded) {
      statusL.textContent = "No video connected";
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

  return { controls, statusBar, sync, setFrame, updateStatus };
}
