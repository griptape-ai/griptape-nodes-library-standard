// Preset sidebar for CropImageEditor — aspect ratio, resolution, and position shortcuts.

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

const RESOLUTION_PRESETS = [
  { label: "480p",  w: 854,  h: 480  },
  { label: "720p",  w: 1280, h: 720  },
  { label: "1080p", w: 1920, h: 1080 },
  { label: "1440p", w: 2560, h: 1440 },
  { label: "4K",    w: 3840, h: 2160 },
];

const POSITION_PRESETS = [
  { label: "↖",    title: "Top Left",       pos: "top-left"      },
  { label: "↑",    title: "Top Center",     pos: "top-center"    },
  { label: "↗",    title: "Top Right",      pos: "top-right"     },
  { label: "←",    title: "Left Center",    pos: "left-center"   },
  { label: "⊕",    title: "Center",         pos: "center"        },
  { label: "→",    title: "Right Center",   pos: "right-center"  },
  { label: "↙",    title: "Bottom Left",    pos: "bottom-left"   },
  { label: "↓",    title: "Bottom Center",  pos: "bottom-center" },
  { label: "↘",    title: "Bottom Right",   pos: "bottom-right"  },
];

// Scale from the shortest side of the current crop to the new ratio.
function calcRatioRect(rw, rh, imgW, imgH, curL, curT, curW, curH) {
  const s = Math.min(curW, curH) / Math.min(rw, rh);
  let w = Math.round(rw * s);
  let h = Math.round(rh * s);
  if (w > imgW) { w = imgW; h = Math.round(w * rh / rw); }
  if (h > imgH) { h = imgH; w = Math.round(h * rw / rh); }
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}

// Center the new resolution on the current crop center, clamped to image bounds.
function calcResolutionRect(newW, newH, imgW, imgH, curL, curT, curW, curH) {
  const w = Math.min(newW, imgW);
  const h = Math.min(newH, imgH);
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), imgW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), imgH - h));
  return { left: l, top: t, width: w, height: h };
}

// Compute left/top for a named position, keeping current crop size.
function calcPositionRect(posName, imgW, imgH, curL, curT, curW, curH) {
  const w = curW, h = curH;
  const positions = {
    "center":        { l: Math.round((imgW - w) / 2), t: Math.round((imgH - h) / 2) },
    "top-left":      { l: 0,           t: 0            },
    "top-center":    { l: Math.round((imgW - w) / 2), t: 0 },
    "top-right":     { l: imgW - w,    t: 0            },
    "left-center":   { l: 0,           t: Math.round((imgH - h) / 2) },
    "right-center":  { l: imgW - w,    t: Math.round((imgH - h) / 2) },
    "bottom-left":   { l: 0,           t: imgH - h     },
    "bottom-center": { l: Math.round((imgW - w) / 2), t: imgH - h },
    "bottom-right":  { l: imgW - w,    t: imgH - h     },
  };
  const { l, t } = positions[posName] || positions["center"];
  return { left: Math.max(0, l), top: Math.max(0, t), width: w, height: h };
}

export function createSidebar({ getImgSize, getCropRect, onApply }) {
  const el = document.createElement("div");
  el.className = "nodrag nowheel";
  el.style.cssText = [
    "display:flex", "flex-direction:column", "gap:3px",
    "width:108px", "flex-shrink:0",
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

  // ── Reset ────────────────────────────────────────────────────────────────────
  el.appendChild(makeBtn({ icon: "rotate-ccw", label: "Reset", title: "Reset crop to full image", onClick: () => {
    const { imgNatW, imgNatH } = getImgSize();
    onApply({ left: 0, top: 0, width: imgNatW, height: imgNatH });
  }}));

  // ── Aspect ratios ─────────────────────────────────────────────────────────────
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Ratio"));
  const ratioGrid = document.createElement("div");
  ratioGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, rw, rh } of RATIO_PRESETS) {
    const btn = makeBtn({ label, title: label, onClick: () => {
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcRatioRect(rw, rh, imgNatW, imgNatH, l, t, w, h));
    }});
    ratioGrid.appendChild(btn);
  }
  el.appendChild(ratioGrid);

  // ── Resolutions ───────────────────────────────────────────────────────────────
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Res"));
  const resGrid = document.createElement("div");
  resGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, w: pw, h: ph } of RESOLUTION_PRESETS) {
    const btn = makeBtn({ label, title: `${pw}×${ph}`, onClick: () => {
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcResolutionRect(pw, ph, imgNatW, imgNatH, l, t, w, h));
    }});
    resGrid.appendChild(btn);
  }
  el.appendChild(resGrid);

  // ── Positions ─────────────────────────────────────────────────────────────────
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Pos"));
  const posGrid = document.createElement("div");
  posGrid.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:2px;";
  for (const { label, title, pos } of POSITION_PRESETS) {
    const btn = makeBtn({ label, title, onClick: () => {
      const { imgNatW, imgNatH } = getImgSize();
      const { l, t, w, h } = getCropRect();
      onApply(calcPositionRect(pos, imgNatW, imgNatH, l, t, w, h));
    }});
    btn.style.width = "100%";
    posGrid.appendChild(btn);
  }
  el.appendChild(posGrid);

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
