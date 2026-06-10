export const WIDGET_VERSION = "1.1.0";

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

.sfg-layout-btns { display: flex; gap: 4px; }
.sfg-layout-btn {
  padding: 2px 8px;
  border: 1px solid var(--border);
  border-radius: 4px;
  background: transparent;
  color: var(--muted-foreground);
  font-size: 10px;
  cursor: pointer;
  line-height: 18px;
  transition: background 0.12s, color 0.12s, border-color 0.12s;
}
.sfg-layout-btn:hover { background: var(--muted); color: var(--foreground); }
.sfg-layout-btn.active {
  background: rgba(var(--sfg-accent-rgb), 0.2);
  border-color: var(--sfg-accent);
  color: var(--foreground);
}
.sfg-layout-btn:disabled { opacity: 0.4; cursor: not-allowed; }

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
  transition: border-color 0.12s;
  box-sizing: border-box;
}
.sfg-cell:hover   { border-color: rgba(var(--sfg-accent-rgb), 0.45); }
.sfg-cell.pending { border-color: rgba(var(--sfg-accent-rgb), 0.7); background: rgba(var(--sfg-accent-rgb), 0.08); }
.sfg-cell.selected { border-color: var(--sfg-accent); }

/* Checkmark badge */
.sfg-cell.selected::after {
  content: "";
  position: absolute;
  top: 4px; right: 4px;
  width: 18px; height: 18px;
  border-radius: 50%;
  background-color: var(--sfg-accent);
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='white'%3E%3Cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-position: center;
  background-size: 11px;
  z-index: 3;
  pointer-events: none;
}

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
  font-size: 10px;
  background: rgba(0,0,0,0.55);
  color: #fff;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  z-index: 2;
  pointer-events: none;
}

/* ── Empty state ──────────────────────────────────────────────── */
.sfg-empty {
  padding: 24px 12px;
  text-align: center;
  color: var(--muted-foreground);
  font-size: 12px;
}
`;

export function injectStyles() {
  if (document.getElementById("sfg-widget-styles")) return;
  const el = document.createElement("style");
  el.id = "sfg-widget-styles";
  el.textContent = STYLES;
  document.head.appendChild(el);
}
