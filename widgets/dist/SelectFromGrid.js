// src/SelectFromGrid/_styles.ts
var WIDGET_VERSION = "1.4.0";
var CARD_HUES = [202, 162, 267, 322, 42, 82, 132, 232];
var CARD_HEIGHTS = [95, 125, 105, 145, 85, 130, 115, 155];
var STYLES = `
/* ── Theme-aware accent colour ────────────────────────────────── */
:root {
  --sfg-accent: #2563eb;
  --sfg-accent-rgb: 37, 99, 235;
}
.dark {
  --sfg-accent: #7a9db8;
  --sfg-accent-rgb: 122, 157, 184;
}

.sfg-widget {
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 8px;
  height: 100%;
  box-sizing: border-box;
  overflow: hidden;
  user-select: none;
  -webkit-user-select: none;
}

.sfg-controls {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: wrap;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
}

.sfg-label {
  font-size: 11px;
  color: var(--muted-foreground);
  display: flex;
  align-items: center;
  gap: 6px;
  white-space: nowrap;
}

.sfg-slider {
  width: 80px;
  accent-color: var(--sfg-accent);
  cursor: pointer;
  flex-shrink: 0;
}
.sfg-slider:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Layout toggle — segmented control ────────────────────────── */
.sfg-layout-btns { display: flex; gap: 0; }

.sfg-layout-btn {
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 0;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  position: relative;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.sfg-layout-btns .sfg-layout-btn:first-child { border-radius: 4px 0 0 4px; }
.sfg-layout-btns .sfg-layout-btn + .sfg-layout-btn { border-radius: 0 4px 4px 0; margin-left: -1px; }
.sfg-layout-btn:hover { background: var(--muted); color: var(--foreground); z-index: 1; }
.sfg-layout-btn.active {
  background: rgba(var(--sfg-accent-rgb), 0.2);
  border-color: var(--sfg-accent);
  color: var(--foreground);
  z-index: 2;
}
.sfg-layout-btn:disabled { opacity: 0.4; cursor: not-allowed; }

/* ── Labels checkbox toggle ───────────────────────────────────── */
.sfg-labels-btn {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 2px 4px;
  border: none;
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  white-space: nowrap;
  transition: color 0.12s;
}
.sfg-labels-btn:hover { color: var(--foreground); }
.sfg-labels-btn.active { color: var(--foreground); }
.sfg-labels-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-cb-box {
  width: 12px; height: 12px;
  border: 1.5px solid var(--border);
  border-radius: 2px;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
  background: transparent;
  transition: background 0.12s, border-color 0.12s;
}
.sfg-cb-box svg { opacity: 0; transition: opacity 0.1s; color: #fff; }
.sfg-labels-btn.active .sfg-cb-box {
  background: var(--sfg-accent);
  border-color: var(--sfg-accent);
}
.sfg-labels-btn.active .sfg-cb-box svg { opacity: 1; }

/* ── Label size control ───────────────────────────────────────── */
.sfg-label-size-group {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: var(--muted-foreground);
  white-space: nowrap;
}
.sfg-label-size-group .sfg-slider { width: 52px; }
.sfg-hide-labels .sfg-label-size-group { display: none; }

.sfg-count { margin-left: auto; font-size: 10px; color: var(--muted-foreground); }
.sfg-count.active { color: var(--foreground); }

.sfg-clear-btn {
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  transition: background 0.12s, color 0.12s;
  flex-shrink: 0;
}
.sfg-clear-btn:hover { background: var(--muted); color: var(--foreground); }
.sfg-clear-btn.active { color: var(--foreground); border-color: rgba(var(--sfg-accent-rgb), 0.5); }
.sfg-clear-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-fullscreen-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 24px; height: 24px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  cursor: pointer;
  flex-shrink: 0;
  transition: background 0.12s, color 0.12s;
}
.sfg-fullscreen-btn:hover { background: var(--muted); color: var(--foreground); }

/* ── Grid layouts ─────────────────────────────────────────────── */
.sfg-grid {
  display: grid;
  gap: 5px;
  position: relative;
  flex: 1 1 0;
  min-height: 0;
  min-width: 0;
  overflow-y: auto;
  overflow-x: hidden;
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
  scrollbar-gutter: stable;
  align-content: start;
}
.sfg-grid::-webkit-scrollbar { width: 6px; }
.sfg-grid::-webkit-scrollbar-track { background: transparent; }
.sfg-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.sfg-grid.layout-masonry { display: flex; align-items: flex-start; }
.sfg-masonry-col { display: flex; flex-direction: column; flex: 1; gap: 5px; min-width: 0; }

/* ── Loading spinner ──────────────────────────────────────────── */
@keyframes sfg-spin { to { transform: rotate(360deg); } }
.sfg-spinner {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--muted);
  z-index: 2;
  pointer-events: none;
}
.sfg-spinner svg {
  color: var(--sfg-accent);
  opacity: 0.7;
  animation: sfg-spin 0.9s linear infinite;
}

/* ── Lasso / box-select rect ──────────────────────────────────── */
.sfg-lasso {
  position: absolute;
  border: 1.5px dashed var(--sfg-accent);
  background: rgba(var(--sfg-accent-rgb), 0.10);
  border-radius: 3px;
  pointer-events: none;
  z-index: 10;
}

/* ── Cells ────────────────────────────────────────────────────── */
.sfg-cell {
  position: relative;
  border-radius: 5px;
  overflow: hidden;
  cursor: pointer;
  border: 2px solid transparent;
  background: var(--muted);
  transition: border-color 0.12s, opacity 0.15s, box-shadow 0.12s;
  box-sizing: border-box;
}
.sfg-cell:hover   { border-color: rgba(var(--sfg-accent-rgb), 0.45); }
.sfg-cell.pending { border-color: rgba(var(--sfg-accent-rgb), 0.7); background: rgba(var(--sfg-accent-rgb), 0.08); }
.sfg-cell.selected {
  border-color: var(--sfg-accent);
  box-shadow: inset 0 0 0 1px var(--sfg-accent);
}

/* Dim unselected cells when a selection exists */
.sfg-grid.has-selection .sfg-cell:not(.selected):not(.pending) {
  opacity: 0.45;
}

/* Checkmark badge */
.sfg-cell.selected::after {
  content: "";
  position: absolute;
  top: 4px; right: 4px;
  width: 20px; height: 20px;
  border-radius: 50%;
  background-color: var(--sfg-accent);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='white'%3E%3Cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: center;
  background-size: 13px;
  z-index: 3;
  pointer-events: none;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5);
}

/* Fullscreen button, shown on hover (top-left, clear of the checkmark badge) */
.sfg-expand-btn {
  position: absolute;
  top: 4px; left: 4px;
  width: 22px; height: 22px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 4px;
  background: rgba(0,0,0,0.6);
  color: #fff;
  cursor: pointer;
  z-index: 4;
  opacity: 0;
  transition: opacity 0.12s, background 0.12s;
}
.sfg-cell:hover .sfg-expand-btn,
.sfg-expand-btn:focus-visible { opacity: 1; }
.sfg-expand-btn:hover { background: rgba(var(--sfg-accent-rgb), 0.9); }

/* ── Grid inner frame ─────────────────────────────────────────── */
.sfg-cell-inner {
  width: 100%;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #111;
}

/* Grid mode: grid-auto-rows (set via JS ResizeObserver) makes rows = column width.
   The inner fills the measured square cell completely. */
.sfg-grid:not(.layout-masonry) .sfg-cell-inner { height: 100%; }

/* Placeholder shape for masonry cells while their media is loading.
   Keeps the cell from collapsing to 0 so the spinner is visible and
   the eventual height change is from a reasonable size rather than zero. */
.layout-masonry .sfg-cell-inner.sfg-loading { aspect-ratio: 4 / 3; }

/* ── Media elements ───────────────────────────────────────────── */
.sfg-cell img {
  width: 100%; height: 100%; object-fit: contain; display: block; pointer-events: none;
  opacity: 0;
  transition: opacity 0.25s ease;
}
.sfg-cell img.sfg-loaded { opacity: 1; }
.layout-masonry .sfg-cell img { height: auto; object-fit: cover; }

.sfg-cell video {
  width: 100%; height: 100%; object-fit: contain; display: block; pointer-events: none;
  opacity: 0;
  transition: opacity 0.25s ease;
}
.sfg-cell video.sfg-loaded { opacity: 1; }
.layout-masonry .sfg-cell video { height: auto; object-fit: cover; }

.sfg-spinner { transition: opacity 0.2s ease; }

/* ── Error card (failed media load) ──────────────────────────── */
.sfg-error-card {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 100%;
  height: 100%;
  color: var(--muted-foreground);
  opacity: 0.35;
}

/* ── Audio card ───────────────────────────────────────────────── */
.sfg-audio-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 12px 8px;
  width: 100%;
  box-sizing: border-box;
  background: rgba(0,0,0,0.3);
  color: var(--muted-foreground);
}
.sfg-audio-card audio { width: 100%; height: 28px; accent-color: var(--sfg-accent); }

/* ── Quote card (text items) ──────────────────────────────────── */
.sfg-quote-card {
  padding: 14px 12px;
  width: 100%;
  height: 100%;
  box-sizing: border-box;
  align-self: stretch;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.sfg-quote-text {
  font-size: 11px;
  color: var(--foreground);
  line-height: 1.55;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

/* ── Dict card ────────────────────────────────────────────────── */
.sfg-dict-card {
  padding: 8px;
  font-size: 9px;
  font-family: monospace;
  color: var(--muted-foreground);
  white-space: pre;
  overflow: hidden;
  text-overflow: ellipsis;
  max-height: 110px;
  display: -webkit-box;
  -webkit-line-clamp: 7;
  -webkit-box-orient: vertical;
}

/* ── Label overlay ────────────────────────────────────────────── */
.sfg-item-label {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  padding: 3px 5px;
  font-size: var(--sfg-label-size, 10px);
  background: rgba(0,0,0,0.55);
  color: #fff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 2;
  pointer-events: none;
}
.sfg-hide-labels .sfg-item-label { display: none; }

/* ── Empty state ──────────────────────────────────────────────── */
.sfg-empty {
  padding: 24px 12px;
  text-align: center;
  color: var(--muted-foreground);
  font-size: 12px;
}

/* ── Fullscreen viewer ────────────────────────────────────────── */
.sfg-viewer {
  position: fixed;
  inset: 0;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  background: #08080a;
  color: #eee;
  font: 13px/1.4 system-ui, -apple-system, "Segoe UI", sans-serif;
  letter-spacing: normal;
  word-spacing: normal;
  text-align: left;
  outline: none;
  user-select: none;
  -webkit-user-select: none;
}

.sfg-viewer-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 14px;
  flex-shrink: 0;
}
.sfg-viewer-title {
  flex: 1;
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  color: #ccc;
}
.sfg-viewer-counter { color: #999; font-variant-numeric: tabular-nums; }

.sfg-viewer-select {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 12px;
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 5px;
  background: transparent;
  color: #eee;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.12s, border-color 0.12s;
}
.sfg-viewer-select:hover { background: rgba(255,255,255,0.08); }
.sfg-viewer-select.active { background: var(--sfg-accent); border-color: var(--sfg-accent); color: #fff; }
.sfg-viewer-select:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-viewer-icon-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px; height: 32px;
  padding: 0;
  border: none;
  border-radius: 5px;
  background: transparent;
  color: #ccc;
  cursor: pointer;
}
.sfg-viewer-icon-btn:hover { background: rgba(255,255,255,0.1); color: #fff; }

.sfg-viewer-body {
  flex: 1;
  min-height: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 0 8px;
}
.sfg-viewer-nav {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 44px; height: 88px;
  padding: 0;
  border: none;
  border-radius: 6px;
  background: transparent;
  color: #ccc;
  cursor: pointer;
  flex-shrink: 0;
}
.sfg-viewer-nav:hover:not(:disabled) { background: rgba(255,255,255,0.08); color: #fff; }
.sfg-viewer-nav:disabled { opacity: 0.2; cursor: default; }

.sfg-viewer-stage {
  flex: 1;
  min-width: 0;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  touch-action: pan-y;
}
.sfg-viewer-stage img,
.sfg-viewer-stage video {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  display: block;
}
.sfg-viewer-stage img { pointer-events: none; }
.sfg-viewer-stage .sfg-error-card { color: #999; }

.sfg-viewer-audio {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 24px;
  color: #888;
  width: min(560px, 100%);
}
.sfg-viewer-audio audio { width: 100%; accent-color: var(--sfg-accent); }

.sfg-viewer-text,
.sfg-viewer-dict {
  max-width: min(900px, 100%);
  max-height: 100%;
  overflow: auto;
  padding: 24px 28px;
  box-sizing: border-box;
  border-radius: 8px;
  background: rgba(255,255,255,0.05);
  user-select: text;
  -webkit-user-select: text;
  word-break: break-word;
}
.sfg-viewer-text { font-size: 18px; line-height: 1.6; white-space: pre-wrap; }
.sfg-viewer-dict { font-family: monospace; font-size: 13px; white-space: pre; color: #ccc; }

.sfg-viewer-strip {
  flex-shrink: 0;
  padding: 8px 14px 12px;
  border-top: 1px solid rgba(255,255,255,0.1);
}
.sfg-viewer-strip-hint { font-size: 11px; color: #999; margin-bottom: 6px; }
.sfg-viewer-strip-items {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  min-height: 72px;
  scrollbar-width: thin;
}
.sfg-viewer-thumb {
  position: relative;
  width: 72px; height: 72px;
  flex-shrink: 0;
  border-radius: 5px;
  overflow: hidden;
  border: 2px solid transparent;
  background: #1a1a1d;
  cursor: pointer;
  box-sizing: border-box;
}
.sfg-viewer-thumb.current { border-color: var(--sfg-accent); }
.sfg-viewer-thumb img,
.sfg-viewer-thumb video {
  width: 100%; height: 100%; object-fit: cover; display: block; pointer-events: none;
}
.sfg-viewer-thumb-icon { display: flex; align-items: center; justify-content: center; color: #888; }
.sfg-viewer-thumb-text {
  padding: 5px;
  font-size: 9px;
  line-height: 1.35;
  color: #ddd;
  overflow: hidden;
  word-break: break-word;
  display: -webkit-box;
  -webkit-line-clamp: 5;
  -webkit-box-orient: vertical;
}
.sfg-viewer-thumb-remove {
  position: absolute;
  top: 3px; right: 3px;
  width: 18px; height: 18px;
  padding: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  border: none;
  border-radius: 50%;
  background: rgba(0,0,0,0.75);
  color: #fff;
  cursor: pointer;
}
.sfg-viewer-thumb-remove:hover { background: #c0392b; }
`;
function injectStyles() {
  let el = document.getElementById("sfg-widget-styles");
  if (el?.dataset.version === WIDGET_VERSION) return;
  if (!el) {
    el = document.createElement("style");
    el.id = "sfg-widget-styles";
    document.head.appendChild(el);
  }
  el.dataset.version = WIDGET_VERSION;
  el.textContent = STYLES;
}

// src/SelectFromGrid/_icons.ts
var ICON_PATHS = {
  "music": `<path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>`,
  "loader-circle": `<path d="M21 12a9 9 0 1 1-6.219-8.56"/>`,
  "alert-circle": `<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>`,
  "check": `<polyline points="20 6 9 17 4 12"/>`,
  "maximize-2": `<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" x2="14" y1="3" y2="10"/><line x1="3" x2="10" y1="21" y2="14"/>`,
  "x": `<path d="M18 6 6 18"/><path d="m6 6 12 12"/>`,
  "chevron-left": `<path d="m15 18-6-6 6-6"/>`,
  "chevron-right": `<path d="m9 18 6-6-6-6"/>`
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

// src/SelectFromGrid/_toolbar.ts
function createToolbar({
  layout,
  columns,
  isDisabled,
  showLabels,
  labelSize,
  onColumnsChange,
  onColumnsCommit,
  onLayoutChange,
  onClear,
  onLabelsToggle,
  onLabelSizeChange,
  onLabelSizeCommit,
  onFullscreen
}) {
  const controls = document.createElement("div");
  controls.className = "sfg-controls";
  const colLabel = document.createElement("label");
  colLabel.className = "sfg-label";
  colLabel.textContent = "Columns ";
  const colSlider = document.createElement("input");
  colSlider.type = "range";
  colSlider.className = "sfg-slider";
  colSlider.min = "1";
  colSlider.max = "8";
  colSlider.value = String(columns);
  colSlider.disabled = isDisabled;
  colLabel.appendChild(colSlider);
  const colVal = document.createElement("span");
  colVal.textContent = String(columns);
  colLabel.appendChild(colVal);
  const layoutBtns = document.createElement("div");
  layoutBtns.className = "sfg-layout-btns";
  const gridBtn = document.createElement("button");
  gridBtn.type = "button";
  gridBtn.className = "sfg-layout-btn" + (layout === "grid" ? " active" : "");
  gridBtn.textContent = "Grid";
  gridBtn.disabled = isDisabled;
  const masonryBtn = document.createElement("button");
  masonryBtn.type = "button";
  masonryBtn.className = "sfg-layout-btn" + (layout === "masonry" ? " active" : "");
  masonryBtn.textContent = "Masonry";
  masonryBtn.disabled = isDisabled;
  layoutBtns.appendChild(gridBtn);
  layoutBtns.appendChild(masonryBtn);
  const labelsBtn = document.createElement("button");
  labelsBtn.type = "button";
  labelsBtn.className = "sfg-labels-btn" + (showLabels !== false ? " active" : "");
  labelsBtn.disabled = isDisabled;
  const cbBox = document.createElement("span");
  cbBox.className = "sfg-cb-box";
  cbBox.appendChild(mkIcon("check", 9));
  labelsBtn.appendChild(cbBox);
  labelsBtn.appendChild(document.createTextNode("Labels"));
  const labelSizeGroup = document.createElement("label");
  labelSizeGroup.className = "sfg-label sfg-label-size-group";
  labelSizeGroup.textContent = "Aa ";
  const labelSizeSlider = document.createElement("input");
  labelSizeSlider.type = "range";
  labelSizeSlider.className = "sfg-slider";
  labelSizeSlider.min = "8";
  labelSizeSlider.max = "32";
  labelSizeSlider.value = String(labelSize || 10);
  labelSizeSlider.disabled = isDisabled;
  labelSizeGroup.appendChild(labelSizeSlider);
  const labelSizeVal = document.createElement("span");
  labelSizeVal.textContent = String(labelSize || 10);
  labelSizeGroup.appendChild(labelSizeVal);
  const countEl = document.createElement("span");
  countEl.className = "sfg-count";
  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "sfg-clear-btn";
  clearBtn.textContent = "Clear";
  clearBtn.disabled = isDisabled;
  const fullscreenBtn = document.createElement("button");
  fullscreenBtn.type = "button";
  fullscreenBtn.className = "sfg-fullscreen-btn";
  fullscreenBtn.title = "Fullscreen";
  fullscreenBtn.appendChild(mkIcon("maximize-2", 12));
  controls.appendChild(colLabel);
  controls.appendChild(layoutBtns);
  controls.appendChild(labelsBtn);
  controls.appendChild(labelSizeGroup);
  controls.appendChild(countEl);
  controls.appendChild(clearBtn);
  controls.appendChild(fullscreenBtn);
  colSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  colSlider.addEventListener("input", () => {
    colVal.textContent = colSlider.value;
    onColumnsChange(parseInt(colSlider.value, 10));
  });
  colSlider.addEventListener("change", () => {
    if (onColumnsCommit) onColumnsCommit(parseInt(colSlider.value, 10));
  });
  gridBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  gridBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    gridBtn.classList.add("active");
    masonryBtn.classList.remove("active");
    onLayoutChange("grid");
  });
  masonryBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  masonryBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    masonryBtn.classList.add("active");
    gridBtn.classList.remove("active");
    onLayoutChange("masonry");
  });
  clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    onClear();
  });
  fullscreenBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  fullscreenBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    onFullscreen();
  });
  labelsBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  labelsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    labelsBtn.classList.toggle("active");
    if (onLabelsToggle) onLabelsToggle(labelsBtn.classList.contains("active"));
  });
  labelSizeSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  labelSizeSlider.addEventListener("input", () => {
    const n = parseInt(labelSizeSlider.value, 10);
    labelSizeVal.textContent = String(n);
    if (onLabelSizeChange) onLabelSizeChange(n);
  });
  labelSizeSlider.addEventListener("change", () => {
    if (onLabelSizeCommit) onLabelSizeCommit(parseInt(labelSizeSlider.value, 10));
  });
  function setDisabled(disabled) {
    colSlider.disabled = disabled;
    gridBtn.disabled = disabled;
    masonryBtn.disabled = disabled;
    clearBtn.disabled = disabled;
    labelsBtn.disabled = disabled;
    labelSizeSlider.disabled = disabled;
  }
  return {
    controls,
    colSlider,
    colVal,
    gridBtn,
    masonryBtn,
    countEl,
    clearBtn,
    labelsBtn,
    labelSizeSlider,
    labelSizeVal,
    fullscreenBtn,
    setDisabled
  };
}

// src/SelectFromGrid/_viewer.ts
var SWIPE_THRESHOLD = 50;
function createViewer({
  getItems,
  getSelected,
  isMultiSelect,
  isDisabled,
  onToggle,
  onRemove
}) {
  let current = 0;
  let open = false;
  let shownKey = null;
  const overlay = document.createElement("div");
  overlay.className = "sfg-viewer";
  overlay.tabIndex = -1;
  overlay.setAttribute("role", "dialog");
  overlay.setAttribute("aria-modal", "true");
  const header = document.createElement("div");
  header.className = "sfg-viewer-header";
  const titleEl = document.createElement("div");
  titleEl.className = "sfg-viewer-title";
  const counterEl = document.createElement("span");
  counterEl.className = "sfg-viewer-counter";
  const selectBtn = document.createElement("button");
  selectBtn.type = "button";
  selectBtn.className = "sfg-viewer-select";
  const closeBtn = document.createElement("button");
  closeBtn.type = "button";
  closeBtn.className = "sfg-viewer-icon-btn";
  closeBtn.title = "Close (Esc)";
  closeBtn.appendChild(mkIcon("x", 18));
  header.append(titleEl, counterEl, selectBtn, closeBtn);
  const body = document.createElement("div");
  body.className = "sfg-viewer-body";
  const prevBtn = document.createElement("button");
  prevBtn.type = "button";
  prevBtn.className = "sfg-viewer-nav prev";
  prevBtn.title = "Previous (←)";
  prevBtn.appendChild(mkIcon("chevron-left", 28));
  const nextBtn = document.createElement("button");
  nextBtn.type = "button";
  nextBtn.className = "sfg-viewer-nav next";
  nextBtn.title = "Next (→)";
  nextBtn.appendChild(mkIcon("chevron-right", 28));
  const stage = document.createElement("div");
  stage.className = "sfg-viewer-stage";
  body.append(prevBtn, stage, nextBtn);
  const strip = document.createElement("div");
  strip.className = "sfg-viewer-strip";
  const stripHint = document.createElement("div");
  stripHint.className = "sfg-viewer-strip-hint";
  const stripItems = document.createElement("div");
  stripItems.className = "sfg-viewer-strip-items";
  strip.append(stripHint, stripItems);
  overlay.append(header, body, strip);
  function mkTextBlock(className, text) {
    const el = document.createElement("div");
    el.className = className;
    el.textContent = text;
    return el;
  }
  function mkErrorCard() {
    const card = document.createElement("div");
    card.className = "sfg-error-card";
    card.appendChild(mkIcon("alert-circle", 40));
    return card;
  }
  function stageKey(item) {
    return item ? JSON.stringify([item.type, item.url, item.full_url, item.value]) : null;
  }
  function renderStage() {
    stage.querySelectorAll("video, audio").forEach((m) => m.pause());
    stage.innerHTML = "";
    const items = getItems();
    const item = items[current];
    shownKey = stageKey(item);
    if (!item) return;
    switch (item.type) {
      case "image": {
        const img = document.createElement("img");
        img.src = item.full_url || item.url || "";
        img.alt = item.label || "";
        let triedPreview = !item.full_url || !item.url;
        img.addEventListener("error", () => {
          if (triedPreview) {
            img.replaceWith(mkErrorCard());
            return;
          }
          triedPreview = true;
          img.src = item.url;
        });
        stage.appendChild(img);
        break;
      }
      case "video": {
        const vid = document.createElement("video");
        vid.src = item.url || "";
        if (item.thumbnail) vid.poster = item.thumbnail;
        vid.controls = true;
        vid.loop = true;
        vid.playsInline = true;
        vid.addEventListener("error", () => vid.replaceWith(mkErrorCard()));
        stage.appendChild(vid);
        void vid.play().catch(() => {
        });
        break;
      }
      case "audio": {
        const card = document.createElement("div");
        card.className = "sfg-viewer-audio";
        card.appendChild(mkIcon("music", 72));
        if (item.url) {
          const audio = document.createElement("audio");
          audio.src = item.url;
          audio.controls = true;
          audio.addEventListener("error", () => card.replaceWith(mkErrorCard()));
          card.appendChild(audio);
        }
        stage.appendChild(card);
        break;
      }
      case "dict":
        stage.appendChild(mkTextBlock("sfg-viewer-dict", item.value || "{}"));
        break;
      default:
        stage.appendChild(mkTextBlock(
          "sfg-viewer-text",
          item.value !== void 0 ? String(item.value) : item.label || ""
        ));
        break;
    }
  }
  function mkThumb(item, idx) {
    const thumb = document.createElement("div");
    thumb.className = "sfg-viewer-thumb" + (idx === current ? " current" : "");
    thumb.title = item.label || `Item ${idx + 1}`;
    thumb.dataset.idx = String(idx);
    switch (item.type) {
      case "image": {
        const img = document.createElement("img");
        img.src = item.url || "";
        img.alt = "";
        img.decoding = "async";
        thumb.appendChild(img);
        break;
      }
      case "video": {
        const mkFrame = () => {
          const vid = document.createElement("video");
          vid.src = item.url ? `${item.url}#t=0.1` : "";
          vid.muted = true;
          vid.preload = "metadata";
          return vid;
        };
        if (item.thumbnail) {
          const img = document.createElement("img");
          img.src = item.thumbnail;
          img.alt = "";
          img.addEventListener("error", () => img.replaceWith(mkFrame()), { once: true });
          thumb.appendChild(img);
        } else {
          thumb.appendChild(mkFrame());
        }
        break;
      }
      case "audio":
        thumb.classList.add("sfg-viewer-thumb-icon");
        thumb.appendChild(mkIcon("music", 22));
        break;
      default: {
        const text = item.type === "dict" ? item.value || "{}" : item.value !== void 0 ? String(item.value) : item.label || "";
        const snippet = mkTextBlock("sfg-viewer-thumb-text", text);
        thumb.style.background = `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.18)`;
        thumb.appendChild(snippet);
        break;
      }
    }
    thumb.addEventListener("click", () => show(idx));
    if (!isDisabled()) {
      const rm = document.createElement("button");
      rm.type = "button";
      rm.className = "sfg-viewer-thumb-remove";
      rm.title = "Remove from selection";
      rm.appendChild(mkIcon("x", 12));
      rm.addEventListener("click", (e) => {
        e.stopPropagation();
        onRemove(idx);
      });
      thumb.appendChild(rm);
    }
    return thumb;
  }
  function renderChrome() {
    const items = getItems();
    const selected = getSelected();
    const item = items[current];
    const isSel = selected.includes(current);
    const multi = isMultiSelect();
    titleEl.textContent = item?.label || "";
    counterEl.textContent = items.length ? `${current + 1} / ${items.length}` : "";
    prevBtn.disabled = current <= 0;
    nextBtn.disabled = current >= items.length - 1;
    selectBtn.disabled = isDisabled();
    selectBtn.classList.toggle("active", isSel);
    selectBtn.replaceChildren();
    if (isSel) selectBtn.appendChild(mkIcon("check", 14));
    let label;
    if (isSel) label = "Selected";
    else if (!multi && selected.length) label = "Replace selection";
    else label = "Select";
    selectBtn.appendChild(document.createTextNode(label));
    selectBtn.title = "Toggle selection (Space)";
    if (!selected.length) {
      stripHint.textContent = multi ? "Nothing selected · press Space to select" : "Nothing selected";
    } else if (multi) {
      stripHint.textContent = `${selected.length} selected`;
    } else {
      stripHint.textContent = "Selected · selecting another item replaces it";
    }
  }
  function renderStrip() {
    const items = getItems();
    stripItems.replaceChildren();
    getSelected().forEach((idx) => {
      if (items[idx]) stripItems.appendChild(mkThumb(items[idx], idx));
    });
  }
  function updateStripCurrent() {
    for (const thumb of stripItems.children) {
      thumb.classList.toggle("current", Number(thumb.dataset.idx) === current);
    }
  }
  function show(idx) {
    const items = getItems();
    if (!items.length) return;
    const next = Math.max(0, Math.min(items.length - 1, idx));
    const changed = next !== current;
    current = next;
    if (changed) renderStage();
    renderChrome();
    updateStripCurrent();
  }
  function step(delta) {
    show(current + delta);
  }
  function toggleCurrent() {
    if (isDisabled() || !getItems()[current]) return;
    onToggle(current);
  }
  prevBtn.addEventListener("click", () => step(-1));
  nextBtn.addEventListener("click", () => step(1));
  closeBtn.addEventListener("click", () => close());
  selectBtn.addEventListener("click", () => toggleCurrent());
  function onKeyDown(e) {
    if (!open) return;
    e.stopPropagation();
    switch (e.key) {
      case "ArrowLeft":
        e.preventDefault();
        step(-1);
        break;
      case "ArrowRight":
        e.preventDefault();
        step(1);
        break;
      case " ":
        e.preventDefault();
        if (!e.repeat) toggleCurrent();
        break;
      case "Escape":
        e.preventDefault();
        close();
        break;
    }
  }
  let swipe = null;
  stage.addEventListener("pointerdown", (e) => {
    if (e.target.closest("video, audio, .sfg-viewer-text, .sfg-viewer-dict")) return;
    swipe = { x: e.clientX, y: e.clientY };
  });
  stage.addEventListener("pointerup", (e) => {
    if (!swipe) return;
    const dx = e.clientX - swipe.x;
    const dy = e.clientY - swipe.y;
    swipe = null;
    if (Math.abs(dx) > SWIPE_THRESHOLD && Math.abs(dx) > Math.abs(dy)) step(dx < 0 ? 1 : -1);
  });
  stage.addEventListener("pointercancel", () => {
    swipe = null;
  });
  function openAt(idx) {
    const items = getItems();
    if (!items.length) return;
    current = Math.max(0, Math.min(items.length - 1, idx));
    if (!open) {
      open = true;
      document.body.appendChild(overlay);
      window.addEventListener("keydown", onKeyDown, true);
    }
    renderStage();
    renderChrome();
    renderStrip();
    overlay.focus();
  }
  function close() {
    if (!open) return;
    open = false;
    stage.querySelectorAll("video, audio").forEach((m) => m.pause());
    stage.innerHTML = "";
    shownKey = null;
    window.removeEventListener("keydown", onKeyDown, true);
    overlay.remove();
  }
  function refresh() {
    if (!open) return;
    const items = getItems();
    if (!items.length) {
      close();
      return;
    }
    current = Math.min(current, items.length - 1);
    if (stageKey(items[current]) !== shownKey) renderStage();
    else {
      const vid = stage.querySelector("video");
      const thumbnail = items[current]?.thumbnail;
      if (vid && thumbnail) vid.poster = thumbnail;
    }
    renderChrome();
    renderStrip();
  }
  return {
    open: openAt,
    close,
    refresh,
    destroy: close
  };
}

// src/SelectFromGrid/SelectFromGrid.ts
function SelectFromGrid(host, props) {
  const container = host;
  if (container._sfgInst?.wrapper?.isConnected) {
    container._sfgInst.handleUpdate(props);
    return { cleanup: container._sfgInst.cleanup, update: container._sfgInst.handleUpdate };
  }
  if (container._sfgInst) container._sfgInst.cleanup();
  injectStyles();
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled = props.disabled || false;
  let items = latestValue.items || [];
  let selectedIndices = latestValue.selected_indices || [];
  let columns = latestValue.columns || 3;
  let layout = latestValue.layout || "grid";
  let multiSelect = latestValue.settings?.multi_select !== false;
  let showLabels = latestValue.show_labels !== false;
  let labelSize = latestValue.label_size || 10;
  const wrapper = document.createElement("div");
  wrapper.className = "sfg-widget nodrag nowheel";
  const {
    controls,
    colSlider,
    colVal,
    gridBtn,
    masonryBtn,
    countEl,
    clearBtn,
    labelsBtn,
    labelSizeSlider,
    labelSizeVal,
    setDisabled
  } = createToolbar({
    layout,
    columns,
    isDisabled,
    showLabels,
    labelSize,
    onColumnsChange(n) {
      columns = n;
      renderGrid(
        /* skeleton */
        true
      );
    },
    onColumnsCommit(n) {
      columns = n;
      animateGridChange();
      emitChange();
    },
    onLayoutChange(l) {
      layout = l;
      animateGridChange();
      emitChange();
    },
    onClear() {
      if (selectedIndices.length === 0) return;
      setSelection([]);
    },
    onLabelsToggle(v) {
      showLabels = v;
      wrapper.classList.toggle("sfg-hide-labels", !showLabels);
      emitChange();
    },
    onLabelSizeChange(n) {
      labelSize = n;
      wrapper.style.setProperty("--sfg-label-size", n + "px");
    },
    onLabelSizeCommit(n) {
      labelSize = n;
      emitChange();
    },
    onFullscreen() {
      viewer.open(selectedIndices.length ? selectedIndices[0] : 0);
    }
  });
  const viewer = createViewer({
    getItems: () => items,
    getSelected: () => selectedIndices,
    isMultiSelect: () => multiSelect,
    isDisabled: () => isDisabled,
    onToggle: (idx) => toggleSelection(idx),
    onRemove: (idx) => setSelection(selectedIndices.filter((i) => i !== idx))
  });
  wrapper.appendChild(controls);
  if (!showLabels) wrapper.classList.add("sfg-hide-labels");
  wrapper.style.setProperty("--sfg-label-size", labelSize + "px");
  const grid = document.createElement("div");
  grid.className = "sfg-grid";
  wrapper.appendChild(grid);
  container.appendChild(wrapper);
  function emitChange() {
    onChangeRef({
      ...latestValue || {},
      items,
      selected_indices: selectedIndices,
      columns,
      layout,
      show_labels: showLabels,
      label_size: labelSize
    });
  }
  function syncSelectionClasses() {
    grid.querySelectorAll(".sfg-cell").forEach((cell) => {
      cell.classList.toggle("selected", selectedIndices.includes(parseInt(cell.dataset.idx, 10)));
    });
  }
  function setSelection(indices) {
    selectedIndices = indices;
    syncSelectionClasses();
    updateCount();
    emitChange();
    viewer.refresh();
  }
  function toggleSelection(idx) {
    const alreadySelected = selectedIndices.includes(idx);
    if (multiSelect) {
      setSelection(alreadySelected ? selectedIndices.filter((i) => i !== idx) : [...selectedIndices, idx]);
    } else {
      setSelection(alreadySelected ? [] : [idx]);
    }
  }
  function updateCount() {
    const has = selectedIndices.length > 0;
    countEl.textContent = has ? `${selectedIndices.length} selected` : "";
    countEl.classList.toggle("active", has);
    clearBtn.classList.toggle("active", has);
    grid.classList.toggle("has-selection", has);
  }
  function updateGridRows() {
    const gap = 5;
    const w = grid.clientWidth;
    const rowH = (w - (columns - 1) * gap) / columns;
    grid.style.gridAutoRows = rowH > 0 ? `${rowH}px` : "";
  }
  function applyGridLayout() {
    const isMasonry = layout === "masonry";
    grid.classList.toggle("layout-masonry", isMasonry);
    if (isMasonry) {
      grid.style.gridTemplateColumns = "";
      grid.style.gridAutoRows = "";
    } else {
      grid.style.gridTemplateColumns = `repeat(${columns}, 1fr)`;
      updateGridRows();
    }
  }
  const gridRO = new ResizeObserver(() => {
    if (layout !== "masonry") updateGridRows();
  });
  gridRO.observe(grid);
  const cellObserver = new IntersectionObserver(
    (entries) => entries.forEach(({ isIntersecting, target }) => {
      if (isIntersecting) hydrateCell(target);
    }),
    { root: grid, rootMargin: "300px 0px", threshold: 0 }
  );
  function mkSpinner() {
    const s = document.createElement("div");
    s.className = "sfg-spinner";
    s.appendChild(mkIcon("loader-circle", 22));
    return s;
  }
  function buildCellContent(cell, item, idx, skeleton = false) {
    cell.dataset.idx = String(idx);
    const inner = document.createElement("div");
    inner.className = "sfg-cell-inner";
    if (skeleton) {
      inner.style.background = `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.09)`;
      if (layout === "masonry") inner.style.minHeight = CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px";
      cell.appendChild(inner);
      return;
    }
    const isMedia = item.type === "image" || item.type === "video" || item.type === "audio";
    if (isMedia) {
      inner.classList.add("sfg-loading");
      inner.appendChild(mkSpinner());
    } else {
      switch (item.type) {
        case "dict": {
          const dictEl = document.createElement("div");
          dictEl.className = "sfg-dict-card";
          dictEl.textContent = item.value || "{}";
          inner.appendChild(dictEl);
          break;
        }
        default: {
          const hue = CARD_HUES[idx % CARD_HUES.length];
          const card = document.createElement("div");
          card.className = "sfg-quote-card";
          card.style.background = `hsla(${hue}, 22%, 50%, 0.09)`;
          if (layout === "masonry") card.style.minHeight = CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px";
          const textEl = document.createElement("div");
          textEl.className = "sfg-quote-text";
          textEl.textContent = item.value !== void 0 ? String(item.value) : item.label || "";
          card.appendChild(textEl);
          inner.appendChild(card);
          break;
        }
      }
    }
    cell.appendChild(inner);
    const expandBtn = document.createElement("button");
    expandBtn.type = "button";
    expandBtn.className = "sfg-expand-btn";
    expandBtn.title = "View fullscreen";
    expandBtn.appendChild(mkIcon("maximize-2", 12));
    expandBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
    expandBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      viewer.open(idx);
    });
    cell.appendChild(expandBtn);
    if (item.label) {
      const lbl = document.createElement("div");
      lbl.className = "sfg-item-label";
      lbl.textContent = item.label;
      cell.appendChild(lbl);
    }
    if (isMedia) cellObserver.observe(cell);
  }
  function hydrateCell(cell) {
    if (cell.dataset.hydrated === "true") return;
    const idx = parseInt(cell.dataset.idx, 10);
    const item = items[idx];
    if (!item) return;
    if ((item.type === "image" || item.type === "video") && !item.url) return;
    cell.dataset.hydrated = "true";
    cellObserver.unobserve(cell);
    const inner = cell.querySelector(".sfg-cell-inner");
    if (!inner) return;
    const spinner = inner.querySelector(".sfg-spinner");
    const fadeOutSpinner = () => {
      if (!spinner) return;
      spinner.style.opacity = "0";
      spinner.addEventListener("transitionend", () => spinner.remove(), { once: true });
    };
    const showError = () => {
      inner.classList.remove("sfg-loading");
      if (spinner) spinner.remove();
      const card = document.createElement("div");
      card.className = "sfg-error-card";
      card.appendChild(mkIcon("alert-circle", 28));
      inner.appendChild(card);
    };
    switch (item.type) {
      case "image": {
        const img = document.createElement("img");
        img.src = item.url;
        img.alt = item.label || "";
        img.decoding = "async";
        img.addEventListener("load", () => {
          inner.classList.remove("sfg-loading");
          img.classList.add("sfg-loaded");
          fadeOutSpinner();
        });
        img.addEventListener("error", showError);
        inner.appendChild(img);
        break;
      }
      case "video": {
        const vid = document.createElement("video");
        vid.src = item.url;
        if (item.thumbnail) vid.poster = item.thumbnail;
        vid.muted = true;
        vid.loop = true;
        vid.playsInline = true;
        vid.preload = "metadata";
        vid.addEventListener("loadedmetadata", () => {
          inner.classList.remove("sfg-loading");
          vid.currentTime = 1e-3;
          vid.classList.add("sfg-loaded");
          fadeOutSpinner();
        });
        vid.addEventListener("error", showError);
        cell.addEventListener("mouseenter", () => void vid.play().catch(() => {
        }));
        cell.addEventListener("mouseleave", () => {
          vid.pause();
          vid.currentTime = 0;
        });
        inner.appendChild(vid);
        break;
      }
      case "audio": {
        inner.classList.remove("sfg-loading");
        if (spinner) spinner.remove();
        const card = document.createElement("div");
        card.className = "sfg-audio-card";
        card.appendChild(mkIcon("music", 28));
        if (item.url) {
          const audio = document.createElement("audio");
          audio.src = item.url;
          audio.controls = true;
          audio.addEventListener("pointerdown", (e) => e.stopPropagation());
          audio.addEventListener("click", (e) => e.stopPropagation());
          audio.addEventListener("error", () => {
            card.replaceWith((() => {
              const e = document.createElement("div");
              e.className = "sfg-error-card";
              e.appendChild(mkIcon("alert-circle", 28));
              return e;
            })());
          });
          card.appendChild(audio);
        }
        inner.appendChild(card);
        break;
      }
    }
  }
  function animateGridChange() {
    const cells = [...grid.querySelectorAll(".sfg-cell")];
    const EXIT_MS = 110;
    cells.forEach((cell, i) => {
      const delay = Math.min(i * 6, 40);
      cell.style.transition = `opacity ${EXIT_MS}ms ease ${delay}ms, transform ${EXIT_MS}ms ease ${delay}ms`;
      cell.style.opacity = "0";
      cell.style.transform = "scale(0.82)";
    });
    setTimeout(() => {
      renderGrid();
      const entering = [...grid.querySelectorAll(".sfg-cell")];
      entering.forEach((cell) => {
        cell.style.transition = "none";
        cell.style.opacity = "0";
        cell.style.transform = "scale(0.88)";
      });
      requestAnimationFrame(() => requestAnimationFrame(() => {
        entering.forEach((cell, i) => {
          const delay = Math.min(i * 12, 90);
          cell.style.transition = `opacity 0.2s ease ${delay}ms, transform 0.22s cubic-bezier(0.34,1.56,0.64,1) ${delay}ms`;
          cell.style.opacity = "";
          cell.style.transform = "";
        });
      }));
    }, EXIT_MS + Math.min(cells.length * 6, 40) + 20);
  }
  function renderGrid(skeleton = false) {
    grid.innerHTML = "";
    applyGridLayout();
    if (items.length === 0) {
      const empty = document.createElement("div");
      empty.className = "sfg-empty";
      empty.textContent = "Connect a list to display items here";
      grid.appendChild(empty);
      updateCount();
      return;
    }
    if (layout === "masonry") {
      const cols = Array.from({ length: columns }, () => {
        const col = document.createElement("div");
        col.className = "sfg-masonry-col";
        grid.appendChild(col);
        return col;
      });
      items.forEach((item, idx) => {
        const cell = document.createElement("div");
        cell.className = "sfg-cell" + (selectedIndices.includes(idx) ? " selected" : "");
        buildCellContent(cell, item, idx, skeleton);
        cols[idx % columns].appendChild(cell);
      });
    } else {
      items.forEach((item, idx) => {
        const cell = document.createElement("div");
        cell.className = "sfg-cell" + (selectedIndices.includes(idx) ? " selected" : "");
        buildCellContent(cell, item, idx, skeleton);
        grid.appendChild(cell);
      });
    }
    updateCount();
  }
  grid.addEventListener("wheel", (e) => {
    e.stopPropagation();
  }, { passive: true });
  let dragState = null;
  const DRAG_THRESHOLD = 5;
  function findCellEl(el) {
    while (el && el !== grid) {
      if (el.classList.contains("sfg-cell")) return el;
      el = el.parentElement;
    }
    return null;
  }
  function clientToLocal(clientX, clientY) {
    const gr = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      x: (clientX - gr.left) / scale + grid.scrollLeft,
      y: (clientY - gr.top) / scale + grid.scrollTop
    };
  }
  function cellLocalRect(cell) {
    const cr = cell.getBoundingClientRect();
    const gr = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      left: (cr.left - gr.left) / scale + grid.scrollLeft,
      top: (cr.top - gr.top) / scale + grid.scrollTop,
      width: cr.width / scale,
      height: cr.height / scale
    };
  }
  grid.addEventListener("pointerdown", (e) => {
    if (isDisabled || e.button !== 0) return;
    e.stopPropagation();
    const { x: startX, y: startY } = clientToLocal(e.clientX, e.clientY);
    let lasso = null;
    if (multiSelect) {
      lasso = document.createElement("div");
      lasso.className = "sfg-lasso";
      Object.assign(lasso.style, { left: startX + "px", top: startY + "px", width: "0", height: "0" });
      grid.appendChild(lasso);
    }
    dragState = { startX, startY, lasso, dragging: false, startCell: findCellEl(e.target) };
    grid.setPointerCapture(e.pointerId);
  });
  grid.addEventListener("pointermove", (e) => {
    if (!dragState) return;
    const { x: curX, y: curY } = clientToLocal(e.clientX, e.clientY);
    const dx = curX - dragState.startX;
    const dy = curY - dragState.startY;
    if (!dragState.dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD) dragState.dragging = true;
    if (dragState.dragging && dragState.lasso) {
      const selLeft = Math.min(dragState.startX, curX);
      const selTop = Math.min(dragState.startY, curY);
      const selRight = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);
      Object.assign(dragState.lasso.style, {
        left: selLeft + "px",
        top: selTop + "px",
        width: Math.abs(dx) + "px",
        height: Math.abs(dy) + "px"
      });
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        const overlaps = cLeft < selRight && cLeft + cW > selLeft && cTop < selBottom && cTop + cH > selTop;
        const idx = parseInt(cell.dataset.idx, 10);
        cell.classList.toggle("pending", overlaps && !selectedIndices.includes(idx));
      });
    }
  });
  grid.addEventListener("pointerup", (e) => {
    if (!dragState) return;
    if (dragState.lasso) dragState.lasso.remove();
    grid.querySelectorAll(".sfg-cell.pending").forEach((c) => c.classList.remove("pending"));
    if (dragState.dragging && dragState.lasso) {
      const { x: curX, y: curY } = clientToLocal(e.clientX, e.clientY);
      const selLeft = Math.min(dragState.startX, curX);
      const selTop = Math.min(dragState.startY, curY);
      const selRight = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);
      let changed = false;
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        if (cLeft < selRight && cLeft + cW > selLeft && cTop < selBottom && cTop + cH > selTop) {
          const idx = parseInt(cell.dataset.idx, 10);
          if (!isNaN(idx) && !selectedIndices.includes(idx)) {
            selectedIndices = [...selectedIndices, idx];
            cell.classList.add("selected");
            changed = true;
          }
        }
      });
      if (changed) {
        updateCount();
        emitChange();
        viewer.refresh();
      }
    } else if (!dragState.dragging && dragState.startCell) {
      const idx = parseInt(dragState.startCell.dataset.idx, 10);
      if (!isNaN(idx)) toggleSelection(idx);
    }
    dragState = null;
  });
  grid.addEventListener("pointercancel", () => {
    if (dragState) {
      if (dragState.lasso) dragState.lasso.remove();
      grid.querySelectorAll(".sfg-cell.pending").forEach((c) => c.classList.remove("pending"));
      dragState = null;
    }
  });
  renderGrid();
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    const disabledChanged = (newProps.disabled || false) !== isDisabled;
    isDisabled = newProps.disabled || false;
    setDisabled(isDisabled);
    const newVal = newProps.value || {};
    const newItems = newVal.items || [];
    const newSelected = newVal.selected_indices || [];
    const newColumns = newVal.columns || 3;
    const newLayout = newVal.layout || "grid";
    const newMultiSelect = newVal.settings?.multi_select !== false;
    const newShowLabels = newVal.show_labels !== false;
    const newLabelSize = newVal.label_size || 10;
    let needsRender = false;
    if (JSON.stringify(newItems) !== JSON.stringify(items)) {
      items = newItems;
      needsRender = true;
    }
    if (JSON.stringify(newSelected) !== JSON.stringify(selectedIndices)) {
      selectedIndices = newSelected;
      needsRender = true;
    }
    if (newColumns !== columns) {
      columns = newColumns;
      colSlider.value = String(columns);
      colVal.textContent = String(columns);
      needsRender = true;
    }
    if (newLayout !== layout) {
      layout = newLayout;
      gridBtn.classList.toggle("active", layout === "grid");
      masonryBtn.classList.toggle("active", layout === "masonry");
      needsRender = true;
    }
    const multiSelectChanged = newMultiSelect !== multiSelect;
    if (multiSelectChanged) {
      multiSelect = newMultiSelect;
      if (!multiSelect && selectedIndices.length > 1) {
        selectedIndices = [];
        needsRender = true;
      }
    }
    if (newShowLabels !== showLabels) {
      showLabels = newShowLabels;
      labelsBtn.classList.toggle("active", showLabels);
      wrapper.classList.toggle("sfg-hide-labels", !showLabels);
    }
    if (newLabelSize !== labelSize) {
      labelSize = newLabelSize;
      labelSizeSlider.value = String(labelSize);
      labelSizeVal.textContent = String(labelSize);
      wrapper.style.setProperty("--sfg-label-size", labelSize + "px");
    }
    latestValue = newVal;
    if (needsRender) renderGrid();
    if (needsRender || disabledChanged || multiSelectChanged) viewer.refresh();
  }
  function cleanup() {
    gridRO.disconnect();
    cellObserver.disconnect();
    viewer.destroy();
    wrapper.remove();
    delete container._sfgInst;
  }
  container._sfgInst = { wrapper, handleUpdate, cleanup };
  return { cleanup, update: handleUpdate };
}
export {
  SelectFromGrid as default
};
