// Preset sidebar for BoundingBoxEditor — aspect ratio and position shortcuts.

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

// Scale from the shortest side of the current box to the new ratio.
function calcRatioRect(rw, rh, canvasW, canvasH, curL, curT, curW, curH) {
  const s = Math.min(curW, curH) / Math.min(rw, rh);
  let w = Math.round(rw * s);
  let h = Math.round(rh * s);
  if (w > canvasW) { w = canvasW; h = Math.round(w * rh / rw); }
  if (h > canvasH) { h = canvasH; w = Math.round(h * rw / rh); }
  const cx = curL + curW / 2, cy = curT + curH / 2;
  const l = Math.max(0, Math.min(Math.round(cx - w / 2), canvasW - w));
  const t = Math.max(0, Math.min(Math.round(cy - h / 2), canvasH - h));
  return { left: l, top: t, width: Math.max(1, w), height: Math.max(1, h) };
}

// Compute left/top for a named position, keeping current box size.
function calcPositionRect(posName, canvasW, canvasH, curL, curT, curW, curH) {
  const w = curW, h = curH;
  const positions = {
    "center":        { l: Math.round((canvasW - w) / 2), t: Math.round((canvasH - h) / 2) },
    "top-left":      { l: 0,           t: 0            },
    "top-center":    { l: Math.round((canvasW - w) / 2), t: 0 },
    "top-right":     { l: canvasW - w, t: 0            },
    "left-center":   { l: 0,           t: Math.round((canvasH - h) / 2) },
    "right-center":  { l: canvasW - w, t: Math.round((canvasH - h) / 2) },
    "bottom-left":   { l: 0,           t: canvasH - h  },
    "bottom-center": { l: Math.round((canvasW - w) / 2), t: canvasH - h },
    "bottom-right":  { l: canvasW - w, t: canvasH - h  },
  };
  const { l, t } = positions[posName] || positions["center"];
  return { left: Math.max(0, l), top: Math.max(0, t), width: w, height: h };
}

export function createSidebar({ getCanvasSize, getBoxRect, onApply }) {
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
  el.appendChild(makeBtn({ icon: "rotate-ccw", label: "Reset", title: "Reset box to the full canvas", onClick: () => {
    const { canvasW, canvasH } = getCanvasSize();
    onApply({ left: 0, top: 0, width: canvasW, height: canvasH });
  }}));

  // ── Aspect ratios ─────────────────────────────────────────────────────────────
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Ratio"));
  const ratioGrid = document.createElement("div");
  ratioGrid.style.cssText = "display:grid;grid-template-columns:repeat(2,1fr);gap:2px;";
  for (const { label, rw, rh } of RATIO_PRESETS) {
    const btn = makeBtn({ label, title: label, onClick: () => {
      const { canvasW, canvasH } = getCanvasSize();
      const { l, t, w, h } = getBoxRect();
      onApply(calcRatioRect(rw, rh, canvasW, canvasH, l, t, w, h));
    }});
    ratioGrid.appendChild(btn);
  }
  el.appendChild(ratioGrid);

  // ── Positions ─────────────────────────────────────────────────────────────────
  el.appendChild(makeDivider());
  el.appendChild(makeLabel("Pos"));
  const posGrid = document.createElement("div");
  posGrid.style.cssText = "display:grid;grid-template-columns:repeat(3,1fr);gap:2px;";
  for (const { label, title, pos } of POSITION_PRESETS) {
    const btn = makeBtn({ label, title, onClick: () => {
      const { canvasW, canvasH } = getCanvasSize();
      const { l, t, w, h } = getBoxRect();
      onApply(calcPositionRect(pos, canvasW, canvasH, l, t, w, h));
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
