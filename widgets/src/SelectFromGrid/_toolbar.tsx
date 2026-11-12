// _toolbar.tsx — Controls bar for the SelectFromGrid widget.
//
// Layout:
//   [Columns ──●── N]  [Grid|Masonry]  [☑ Labels] [Aa ──●── N]  · · ·  [N selected]  [Clear]  [⤢]
//
// createToolbar(opts) →
//   { controls, colSlider, colVal, gridBtn, masonryBtn, countEl, clearBtn,
//     labelsBtn, labelSizeSlider, labelSizeVal, fullscreenBtn, setDisabled }

import { Icon } from './_icons';
import type { GridLayout } from './_types';

export interface ToolbarOptions {
  layout: GridLayout;
  columns: number;
  isDisabled: boolean;
  showLabels: boolean;
  labelSize: number;
  onColumnsChange: (n: number) => void;
  onColumnsCommit?: (n: number) => void;
  onLayoutChange: (layout: GridLayout) => void;
  onClear: () => void;
  onLabelsToggle?: (show: boolean) => void;
  onLabelSizeChange?: (n: number) => void;
  onLabelSizeCommit?: (n: number) => void;
  onFullscreen: () => void;
}

export interface Toolbar {
  controls: HTMLDivElement;
  colSlider: HTMLInputElement;
  colVal: HTMLSpanElement;
  gridBtn: HTMLButtonElement;
  masonryBtn: HTMLButtonElement;
  countEl: HTMLSpanElement;
  clearBtn: HTMLButtonElement;
  labelsBtn: HTMLButtonElement;
  labelSizeSlider: HTMLInputElement;
  labelSizeVal: HTMLSpanElement;
  fullscreenBtn: HTMLButtonElement;
  setDisabled: (disabled: boolean) => void;
}

export function createToolbar({
  layout, columns, isDisabled,
  showLabels, labelSize,
  onColumnsChange, onColumnsCommit,
  onLayoutChange,
  onClear,
  onLabelsToggle,
  onLabelSizeChange, onLabelSizeCommit,
  onFullscreen,
}: ToolbarOptions): Toolbar {

  const stopPropagation = (e: Event) => e.stopPropagation();

  // ── Column slider ──────────────────────────────────────────────────────────

  const colSlider = (
    <input
      type="range" class="sfg-slider" min="1" max="8"
      value={String(columns)} disabled={isDisabled}
      onPointerDown={stopPropagation}
      onInput={() => {
        colVal.textContent = colSlider.value;
        onColumnsChange(parseInt(colSlider.value, 10));
      }}
      onChange={() => {
        if (onColumnsCommit) onColumnsCommit(parseInt(colSlider.value, 10));
      }}
    />
  ) as HTMLInputElement;

  const colVal = (<span>{String(columns)}</span>) as HTMLSpanElement;

  // ── Layout toggle — segmented control ─────────────────────────────────────

  const gridBtn = (
    <button
      type="button" class={"sfg-layout-btn" + (layout === "grid" ? " active" : "")}
      disabled={isDisabled}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        gridBtn.classList.add("active");
        masonryBtn.classList.remove("active");
        onLayoutChange("grid");
      }}
    >
      Grid
    </button>
  ) as HTMLButtonElement;

  const masonryBtn = (
    <button
      type="button" class={"sfg-layout-btn" + (layout === "masonry" ? " active" : "")}
      disabled={isDisabled}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        masonryBtn.classList.add("active");
        gridBtn.classList.remove("active");
        onLayoutChange("masonry");
      }}
    >
      Masonry
    </button>
  ) as HTMLButtonElement;

  // ── Labels checkbox toggle ─────────────────────────────────────────────────

  const labelsBtn = (
    <button
      type="button" class={"sfg-labels-btn" + (showLabels !== false ? " active" : "")}
      disabled={isDisabled}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        labelsBtn.classList.toggle("active");
        if (onLabelsToggle) onLabelsToggle(labelsBtn.classList.contains("active"));
      }}
    >
      <span class="sfg-cb-box"><Icon name="check" size={9} /></span>
      Labels
    </button>
  ) as HTMLButtonElement;

  // ── Label size slider ──────────────────────────────────────────────────────

  const labelSizeSlider = (
    <input
      type="range" class="sfg-slider" min="8" max="32"
      value={String(labelSize || 10)} disabled={isDisabled}
      onPointerDown={stopPropagation}
      onInput={() => {
        const n = parseInt(labelSizeSlider.value, 10);
        labelSizeVal.textContent = String(n);
        if (onLabelSizeChange) onLabelSizeChange(n);
      }}
      onChange={() => {
        if (onLabelSizeCommit) onLabelSizeCommit(parseInt(labelSizeSlider.value, 10));
      }}
    />
  ) as HTMLInputElement;

  const labelSizeVal = (<span>{String(labelSize || 10)}</span>) as HTMLSpanElement;

  // ── Selection count + clear ────────────────────────────────────────────────

  const countEl = (<span class="sfg-count" />) as HTMLSpanElement;

  const clearBtn = (
    <button
      type="button" class="sfg-clear-btn" disabled={isDisabled}
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        onClear();
      }}
    >
      Clear
    </button>
  ) as HTMLButtonElement;

  // ── Fullscreen viewer ──────────────────────────────────────────────────────
  // Stays enabled while the node is disabled: browsing doesn't change the value.

  const fullscreenBtn = (
    <button
      type="button" class="sfg-fullscreen-btn" title="Fullscreen"
      onPointerDown={stopPropagation}
      onClick={(e) => {
        e.stopPropagation();
        onFullscreen();
      }}
    >
      <Icon name="maximize-2" size={12} />
    </button>
  ) as HTMLButtonElement;

  // ── Assemble ───────────────────────────────────────────────────────────────

  const controls = (
    <div class="sfg-controls">
      <label class="sfg-label">{"Columns "}{colSlider}{colVal}</label>
      <div class="sfg-layout-btns">{gridBtn}{masonryBtn}</div>
      {labelsBtn}
      <label class="sfg-label sfg-label-size-group">{"Aa "}{labelSizeSlider}{labelSizeVal}</label>
      {countEl}
      {clearBtn}
      {fullscreenBtn}
    </div>
  ) as HTMLDivElement;

  // ── Toolbar API ────────────────────────────────────────────────────────────

  function setDisabled(disabled: boolean) {
    colSlider.disabled      = disabled;
    gridBtn.disabled        = disabled;
    masonryBtn.disabled     = disabled;
    clearBtn.disabled       = disabled;
    labelsBtn.disabled      = disabled;
    labelSizeSlider.disabled = disabled;
  }

  return {
    controls, colSlider, colVal,
    gridBtn, masonryBtn,
    countEl, clearBtn,
    labelsBtn, labelSizeSlider, labelSizeVal,
    fullscreenBtn,
    setDisabled,
  };
}
