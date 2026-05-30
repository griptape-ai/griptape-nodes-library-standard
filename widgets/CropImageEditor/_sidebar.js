// Preset sidebar for CropImageEditor — aspect-ratio shortcuts and reset.

import { mkIcon } from './_icons.js';

const RATIO_PRESETS = [
  { label: "1:1",  rw: 1,  rh: 1  },
  { label: "16:9", rw: 16, rh: 9  },
  { label: "4:3",  rw: 4,  rh: 3  },
  { label: "3:2",  rw: 3,  rh: 2  },
  { label: "9:16", rw: 9,  rh: 16 },
  { label: "2:3",  rw: 2,  rh: 3  },
  { label: "3:4",  rw: 3,  rh: 4  },
];

// Scale from the shortest side of the current crop to the new ratio.
// If the result exceeds the image boundary, clamp that dimension and
// recalculate the other to maintain the ratio (i.e. as large as possible).
// Always re-centers on the current crop center.
function calcRatioRect(rw, rh, imgW, imgH, curL, curT, curW, curH) {
  const s = Math.min(curW, curH) / Math.min(rw, rh);
  let w = Math.round(rw * s);
  let h = Math.round(rh * s);

  // Clamp to image bounds while preserving ratio
  if (w > imgW) { w = imgW; h = Math.round(w * rh / rw); }
  if (h > imgH) { h = imgH; w = Math.round(h * rw / rh); }

  const cx = curL + curW / 2;
  const cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}

export function createSidebar({ getImgSize, getCropRect, onApply }) {
  const el = document.createElement("div");
  el.className = "nodrag nowheel";
  el.style.cssText = [
    "display:flex", "flex-direction:column", "gap:3px",
    "width:56px", "flex-shrink:0",
    "background:var(--muted)", "border-radius:6px", "padding:4px",
    "overflow-y:auto", "box-sizing:border-box",
  ].join(";");

  const allBtns = [];

  function makeBtn({ label, icon, title, onClick }) {
    const btn = document.createElement("button");
    btn.title = title;
    btn.style.cssText = [
      "width:100%", "padding:4px 2px", "border-radius:4px",
      "border:1px solid var(--border)",
      "background:var(--background)", "color:var(--foreground)", "cursor:pointer",
      "font-size:10px", "font-weight:500", "line-height:1.3",
      "white-space:nowrap", "box-sizing:border-box",
      "display:flex", "align-items:center", "justify-content:center", "gap:3px",
    ].join(";");
    if (icon) btn.appendChild(mkIcon(icon, 12));
    if (label) btn.append(label);
    btn.addEventListener("mouseenter", () => { if (!btn.disabled) btn.style.background = "var(--accent)"; });
    btn.addEventListener("mouseleave", () => { btn.style.background = "var(--background)"; });
    for (const ev of ["mousedown", "pointerdown"]) btn.addEventListener(ev, (e) => e.stopPropagation());
    btn.addEventListener("click", (e) => { e.stopPropagation(); if (!btn.disabled) onClick(); });
    allBtns.push(btn);
    return btn;
  }

  el.appendChild(makeBtn({ icon: "rotate-ccw", label: "Reset", title: "Reset crop to full image", onClick: () => {
    const { imgNatW, imgNatH } = getImgSize();
    onApply({ left: 0, top: 0, width: imgNatW, height: imgNatH });
  }}));

  const hr = document.createElement("div");
  hr.style.cssText = "height:1px;background:var(--border);margin:2px 0;flex-shrink:0;";
  el.appendChild(hr);

  for (const { label, rw, rh } of RATIO_PRESETS) {
    el.appendChild(makeBtn({ label, title: label, onClick: () => {
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcRatioRect(rw, rh, imgNatW, imgNatH, l, t, w, h));
    }}));
  }

  function syncDisabled(lockedParams, disabled) {
    const coordLocked = ["left", "top", "width", "height"].some(f => lockedParams.includes(f));
    const off = coordLocked || disabled;
    for (const btn of allBtns) {
      btn.disabled = off;
      btn.style.opacity = off ? "0.35" : "1";
      btn.style.cursor  = off ? "not-allowed" : "pointer";
    }
  }

  return { el, syncDisabled };
}
