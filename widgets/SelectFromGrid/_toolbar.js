// _toolbar.js — Controls bar for the SelectFromGrid widget.
//
// Layout:
//   [Columns ──●── N]  [Grid|Masonry]  [☑ Labels] [Aa ──●── N]  · · ·  [N selected]  [Clear]
//
// createToolbar(opts) →
//   { controls, colSlider, colVal, gridBtn, masonryBtn, countEl, clearBtn,
//     labelsBtn, labelSizeSlider, labelSizeVal, setDisabled }

import { mkIcon } from './_icons.js';

export function createToolbar({
  layout, columns, isDisabled,
  showLabels, labelSize,
  onColumnsChange, onColumnsCommit,
  onLayoutChange,
  onClear,
  onLabelsToggle,
  onLabelSizeChange, onLabelSizeCommit,
}) {

  const controls = document.createElement("div");
  controls.className = "sfg-controls";

  // ── Column slider ──────────────────────────────────────────────────────────

  const colLabel = document.createElement("label");
  colLabel.className = "sfg-label";
  colLabel.textContent = "Columns ";

  const colSlider = document.createElement("input");
  colSlider.type = "range";
  colSlider.className = "sfg-slider";
  colSlider.min = 1;
  colSlider.max = 8;
  colSlider.value = columns;
  colSlider.disabled = isDisabled;
  colLabel.appendChild(colSlider);

  const colVal = document.createElement("span");
  colVal.textContent = columns;
  colLabel.appendChild(colVal);

  // ── Layout toggle — segmented control ─────────────────────────────────────

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

  // ── Labels checkbox toggle ─────────────────────────────────────────────────

  const labelsBtn = document.createElement("button");
  labelsBtn.type = "button";
  labelsBtn.className = "sfg-labels-btn" + (showLabels !== false ? " active" : "");
  labelsBtn.disabled = isDisabled;

  const cbBox = document.createElement("span");
  cbBox.className = "sfg-cb-box";
  cbBox.appendChild(mkIcon("check", 9));
  labelsBtn.appendChild(cbBox);
  labelsBtn.appendChild(document.createTextNode("Labels"));

  // ── Label size slider ──────────────────────────────────────────────────────

  const labelSizeGroup = document.createElement("label");
  labelSizeGroup.className = "sfg-label sfg-label-size-group";
  labelSizeGroup.textContent = "Aa ";

  const labelSizeSlider = document.createElement("input");
  labelSizeSlider.type = "range";
  labelSizeSlider.className = "sfg-slider";
  labelSizeSlider.min = 8;
  labelSizeSlider.max = 32;
  labelSizeSlider.value = labelSize || 10;
  labelSizeSlider.disabled = isDisabled;
  labelSizeGroup.appendChild(labelSizeSlider);

  const labelSizeVal = document.createElement("span");
  labelSizeVal.textContent = labelSize || 10;
  labelSizeGroup.appendChild(labelSizeVal);

  // ── Selection count + clear ────────────────────────────────────────────────

  const countEl = document.createElement("span");
  countEl.className = "sfg-count";

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "sfg-clear-btn";
  clearBtn.textContent = "Clear";
  clearBtn.disabled = isDisabled;

  // ── Assemble ───────────────────────────────────────────────────────────────

  controls.appendChild(colLabel);
  controls.appendChild(layoutBtns);
  controls.appendChild(labelsBtn);
  controls.appendChild(labelSizeGroup);
  controls.appendChild(countEl);
  controls.appendChild(clearBtn);

  // ── Wire events ────────────────────────────────────────────────────────────

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

  labelsBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  labelsBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    labelsBtn.classList.toggle("active");
    if (onLabelsToggle) onLabelsToggle(labelsBtn.classList.contains("active"));
  });

  labelSizeSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  labelSizeSlider.addEventListener("input", () => {
    const n = parseInt(labelSizeSlider.value, 10);
    labelSizeVal.textContent = n;
    if (onLabelSizeChange) onLabelSizeChange(n);
  });
  labelSizeSlider.addEventListener("change", () => {
    if (onLabelSizeCommit) onLabelSizeCommit(parseInt(labelSizeSlider.value, 10));
  });

  // ── Toolbar API ────────────────────────────────────────────────────────────

  function setDisabled(disabled) {
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
    setDisabled,
  };
}
