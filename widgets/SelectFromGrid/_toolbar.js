// _toolbar.js — Controls bar for the SelectFromGrid widget.
//
// Layout:
//   [Columns ──●── N]  [Grid] [Masonry]  · · ·  [N selected]  [Clear]
//
// createToolbar(opts) →
//   { controls, colSlider, colVal, squareBtn, masonryBtn, countEl, clearBtn, setDisabled }

export function createToolbar({ layout, columns, isDisabled, onColumnsChange, onColumnsCommit, onLayoutChange, onClear }) {

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

  // ── Layout toggle buttons ──────────────────────────────────────────────────

  const layoutBtns = document.createElement("div");
  layoutBtns.className = "sfg-layout-btns";

  const squareBtn = document.createElement("button");
  squareBtn.type = "button";
  squareBtn.className = "sfg-layout-btn" + (layout === "grid" ? " active" : "");
  squareBtn.textContent = "Grid";
  squareBtn.disabled = isDisabled;

  const masonryBtn = document.createElement("button");
  masonryBtn.type = "button";
  masonryBtn.className = "sfg-layout-btn" + (layout === "masonry" ? " active" : "");
  masonryBtn.textContent = "Masonry";
  masonryBtn.disabled = isDisabled;

  layoutBtns.appendChild(squareBtn);
  layoutBtns.appendChild(masonryBtn);

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

  squareBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  squareBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    squareBtn.classList.add("active");
    masonryBtn.classList.remove("active");
    onLayoutChange("grid");
  });

  masonryBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  masonryBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    masonryBtn.classList.add("active");
    squareBtn.classList.remove("active");
    onLayoutChange("masonry");
  });

  clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    onClear();
  });

  // ── Toolbar API ────────────────────────────────────────────────────────────

  function setDisabled(disabled) {
    colSlider.disabled  = disabled;
    squareBtn.disabled  = disabled;
    masonryBtn.disabled = disabled;
    clearBtn.disabled   = disabled;
  }

  return { controls, colSlider, colVal, squareBtn, masonryBtn, countEl, clearBtn, setDisabled };
}
