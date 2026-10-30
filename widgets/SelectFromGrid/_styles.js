export const WIDGET_VERSION = "1.4.0";

// Hue palette + masonry heights for text/quote cards — cycled by item index
export const CARD_HUES    = [202, 162, 267, 322, 42, 82, 132, 232];
export const CARD_HEIGHTS = [95, 125, 105, 145, 85, 130, 115, 155];

export const STYLES = `
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

export function injectStyles() {
  // Replace styles left by an older widget version still loaded in the page
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
