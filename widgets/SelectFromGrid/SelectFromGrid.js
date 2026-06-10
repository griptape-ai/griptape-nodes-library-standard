const WIDGET_VERSION = "1.1.0";

const ACCENT       = "#7a9db8";
const ACCENT_RGB   = "122,157,184";

const STYLES = `
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
  accent-color: ${ACCENT};
  cursor: pointer;
  flex-shrink: 0;
}
.sfg-slider:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-layout-btns {
  display: flex;
  gap: 4px;
}
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
  background: rgba(${ACCENT_RGB},0.2);
  border-color: ${ACCENT};
  color: var(--foreground);
}
.sfg-layout-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.sfg-count {
  margin-left: auto;
  font-size: 10px;
  color: var(--muted-foreground);
}

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
}
.sfg-grid::-webkit-scrollbar { width: 6px; }
.sfg-grid::-webkit-scrollbar-track { background: transparent; }
.sfg-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

.sfg-grid.layout-masonry {
  display: flex;
  align-items: flex-start;
}
.sfg-masonry-col {
  display: flex;
  flex-direction: column;
  flex: 1;
  gap: 5px;
  min-width: 0;
}

/* ── Lasso / box-select rect ──────────────────────────────────── */
.sfg-lasso {
  position: absolute;
  border: 1.5px dashed ${ACCENT};
  background: rgba(${ACCENT_RGB},0.10);
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
.sfg-cell:hover { border-color: rgba(${ACCENT_RGB},0.45); }
.sfg-cell.selected { border-color: ${ACCENT}; }

/* Checkmark badge on selected cells */
.sfg-cell.selected::after {
  content: "";
  position: absolute;
  top: 4px;
  right: 4px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: ${ACCENT}
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='white'%3E%3Cpath d='M12.207 4.793a1 1 0 010 1.414l-5 5a1 1 0 01-1.414 0l-2-2a1 1 0 011.414-1.414L6.5 9.086l4.293-4.293a1 1 0 011.414 0z'/%3E%3C/svg%3E")
    no-repeat center / 11px;
  z-index: 3;
  pointer-events: none;
}

/* ── Square inner frame ───────────────────────────────────────── */
.sfg-cell-inner {
  width: 100%;
  aspect-ratio: 1 / 1;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #111;
}
.layout-masonry .sfg-cell-inner {
  aspect-ratio: unset;
}

/* ── Media elements ───────────────────────────────────────────── */
.sfg-cell img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}
.layout-masonry .sfg-cell img {
  height: auto;
}

.sfg-cell video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}
.layout-masonry .sfg-cell video {
  height: auto;
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
}
.sfg-audio-icon {
  font-size: 28px;
  line-height: 1;
  pointer-events: none;
}
.sfg-audio-card audio {
  width: 100%;
  height: 28px;
  accent-color: ${ACCENT};
}

/* ── Text / dict cards ────────────────────────────────────────── */
.sfg-text-card {
  padding: 10px;
  font-size: 11px;
  color: var(--foreground);
  word-break: break-word;
  text-align: center;
  min-height: 60px;
  display: flex;
  align-items: center;
  justify-content: center;
}
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
  bottom: 0;
  left: 0;
  right: 0;
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

function injectStyles() {
  if (document.getElementById("sfg-widget-styles")) return;
  const el = document.createElement("style");
  el.id = "sfg-widget-styles";
  el.textContent = STYLES;
  document.head.appendChild(el);
}

export default function SelectFromGrid(container, props) {
  // ── Mount guard ───────────────────────────────────────────────────────────
  if (container._sfgInst?.wrapper?.isConnected) {
    container._sfgInst.handleUpdate(props);
    return { cleanup: container._sfgInst.cleanup, update: container._sfgInst.handleUpdate };
  }
  if (container._sfgInst) delete container._sfgInst;

  injectStyles();

  // ── State ─────────────────────────────────────────────────────────────────
  let latestValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled = props.disabled || false;

  let items = latestValue.items || [];
  let selectedIndices = latestValue.selected_indices || [];
  let columns = latestValue.columns || 3;
  let layout = latestValue.layout || "square";

  // ── DOM: wrapper ──────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "sfg-widget nodrag nowheel";

  // ── DOM: controls ─────────────────────────────────────────────────────────
  const controls = document.createElement("div");
  controls.className = "sfg-controls";

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

  const layoutBtns = document.createElement("div");
  layoutBtns.className = "sfg-layout-btns";

  const squareBtn = document.createElement("button");
  squareBtn.type = "button";
  squareBtn.className = "sfg-layout-btn" + (layout === "square" ? " active" : "");
  squareBtn.textContent = "Square";
  squareBtn.disabled = isDisabled;

  const masonryBtn = document.createElement("button");
  masonryBtn.type = "button";
  masonryBtn.className = "sfg-layout-btn" + (layout === "masonry" ? " active" : "");
  masonryBtn.textContent = "Masonry";
  masonryBtn.disabled = isDisabled;

  layoutBtns.appendChild(squareBtn);
  layoutBtns.appendChild(masonryBtn);

  const countEl = document.createElement("span");
  countEl.className = "sfg-count";

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "sfg-clear-btn";
  clearBtn.textContent = "Clear";
  clearBtn.disabled = isDisabled;

  controls.appendChild(colLabel);
  controls.appendChild(layoutBtns);
  controls.appendChild(countEl);
  controls.appendChild(clearBtn);
  wrapper.appendChild(controls);

  // ── DOM: grid ─────────────────────────────────────────────────────────────
  const grid = document.createElement("div");
  grid.className = "sfg-grid";
  wrapper.appendChild(grid);

  container.appendChild(wrapper);

  // ── Helpers ───────────────────────────────────────────────────────────────
  function emitChange() {
    onChangeRef({
      ...(latestValue || {}),
      items,
      selected_indices: selectedIndices,
      columns,
      layout,
    });
  }

  function updateCount() {
    countEl.textContent = selectedIndices.length > 0 ? `${selectedIndices.length} selected` : "";
  }

  function applyGridLayout() {
    const isMasonry = layout === "masonry";
    grid.classList.toggle("layout-masonry", isMasonry);
    grid.style.gridTemplateColumns = isMasonry ? "" : `repeat(${columns}, 1fr)`;
  }

  function buildCellContent(cell, item, idx) {
    cell.dataset.idx = idx;
    const inner = document.createElement("div");
    inner.className = "sfg-cell-inner";

    switch (item.type) {
      case "image": {
        if (item.url) {
          const img = document.createElement("img");
          img.src = item.url;
          img.alt = item.label || "";
          img.loading = "lazy";
          img.decoding = "async";
          inner.appendChild(img);
        }
        break;
      }
      case "video": {
        if (item.url) {
          const vid = document.createElement("video");
          vid.src = item.url;
          vid.muted = true;
          vid.loop = true;
          vid.playsInline = true;
          vid.preload = "metadata";
          vid.addEventListener("loadedmetadata", () => { vid.currentTime = 0.001; });
          cell.addEventListener("mouseenter", () => void vid.play().catch(() => {}));
          cell.addEventListener("mouseleave", () => { vid.pause(); vid.currentTime = 0; });
          inner.appendChild(vid);
        }
        break;
      }
      case "audio": {
        const card = document.createElement("div");
        card.className = "sfg-audio-card";
        const icon = document.createElement("div");
        icon.className = "sfg-audio-icon";
        icon.textContent = "🎵";
        card.appendChild(icon);
        if (item.url) {
          const audio = document.createElement("audio");
          audio.src = item.url;
          audio.controls = true;
          audio.addEventListener("pointerdown", (e) => e.stopPropagation());
          audio.addEventListener("click", (e) => e.stopPropagation());
          card.appendChild(audio);
        }
        inner.appendChild(card);
        break;
      }
      case "dict": {
        const dictEl = document.createElement("div");
        dictEl.className = "sfg-dict-card";
        dictEl.textContent = item.value || "{}";
        inner.appendChild(dictEl);
        break;
      }
      default: {
        const textEl = document.createElement("div");
        textEl.className = "sfg-text-card";
        textEl.textContent = item.value !== undefined ? item.value : (item.label || "");
        inner.appendChild(textEl);
      }
    }

    cell.appendChild(inner);

    if (item.label) {
      const lbl = document.createElement("div");
      lbl.className = "sfg-item-label";
      lbl.textContent = item.label;
      cell.appendChild(lbl);
    }
  }

  function renderGrid() {
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
      // Build N flex column wrappers and distribute items round-robin
      const cols = Array.from({ length: columns }, () => {
        const col = document.createElement("div");
        col.className = "sfg-masonry-col";
        grid.appendChild(col);
        return col;
      });
      items.forEach((item, idx) => {
        const cell = document.createElement("div");
        cell.className = "sfg-cell" + (selectedIndices.includes(idx) ? " selected" : "");
        buildCellContent(cell, item, idx);
        cols[idx % columns].appendChild(cell);
      });
    } else {
      items.forEach((item, idx) => {
        const cell = document.createElement("div");
        cell.className = "sfg-cell" + (selectedIndices.includes(idx) ? " selected" : "");
        buildCellContent(cell, item, idx);
        grid.appendChild(cell);
      });
    }

    updateCount();
  }

  // Allow the grid to scroll without React Flow intercepting wheel events
  grid.addEventListener("wheel", (e) => { e.stopPropagation(); }, { passive: true });

  // ── Box-select / click via event delegation on the grid ───────────────────
  let dragState = null;
  const DRAG_THRESHOLD = 5;

  function findCellEl(el) {
    while (el && el !== grid) {
      if (el.classList.contains("sfg-cell")) return el;
      el = el.parentElement;
    }
    return null;
  }

  grid.addEventListener("pointerdown", (e) => {
    if (isDisabled || e.button !== 0) return;
    e.stopPropagation();

    const gridRect = grid.getBoundingClientRect();
    const startX = e.clientX - gridRect.left + grid.scrollLeft;
    const startY = e.clientY - gridRect.top + grid.scrollTop;

    const lasso = document.createElement("div");
    lasso.className = "sfg-lasso";
    Object.assign(lasso.style, { left: startX + "px", top: startY + "px", width: "0", height: "0" });
    grid.appendChild(lasso);

    dragState = { startX, startY, lasso, dragging: false, startCell: findCellEl(e.target) };
    grid.setPointerCapture(e.pointerId);
  });

  grid.addEventListener("pointermove", (e) => {
    if (!dragState) return;
    const gridRect = grid.getBoundingClientRect();
    const curX = e.clientX - gridRect.left + grid.scrollLeft;
    const curY = e.clientY - gridRect.top + grid.scrollTop;
    const dx = curX - dragState.startX;
    const dy = curY - dragState.startY;

    if (!dragState.dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD) {
      dragState.dragging = true;
    }
    if (dragState.dragging) {
      Object.assign(dragState.lasso.style, {
        left: Math.min(dragState.startX, curX) + "px",
        top: Math.min(dragState.startY, curY) + "px",
        width: Math.abs(dx) + "px",
        height: Math.abs(dy) + "px",
      });
    }
  });

  grid.addEventListener("pointerup", (e) => {
    if (!dragState) return;
    dragState.lasso.remove();

    if (dragState.dragging) {
      const gridRect = grid.getBoundingClientRect();
      const curX = e.clientX - gridRect.left + grid.scrollLeft;
      const curY = e.clientY - gridRect.top + grid.scrollTop;
      const selLeft = Math.min(dragState.startX, curX);
      const selTop = Math.min(dragState.startY, curY);
      const selRight = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);

      let changed = false;
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const r = cell.getBoundingClientRect();
        const gridRect2 = grid.getBoundingClientRect();
        const cLeft = r.left - gridRect2.left + grid.scrollLeft;
        const cTop = r.top - gridRect2.top + grid.scrollTop;
        const cRight = cLeft + r.width;
        const cBottom = cTop + r.height;

        if (cLeft < selRight && cRight > selLeft && cTop < selBottom && cBottom > selTop) {
          const idx = parseInt(cell.dataset.idx, 10);
          if (!isNaN(idx) && !selectedIndices.includes(idx)) {
            selectedIndices = [...selectedIndices, idx];
            cell.classList.add("selected");
            changed = true;
          }
        }
      });

      if (changed) { updateCount(); emitChange(); }
    } else if (dragState.startCell) {
      const idx = parseInt(dragState.startCell.dataset.idx, 10);
      if (!isNaN(idx)) {
        const pos = selectedIndices.indexOf(idx);
        if (pos === -1) selectedIndices = [...selectedIndices, idx];
        else selectedIndices = selectedIndices.filter((i) => i !== idx);
        dragState.startCell.classList.toggle("selected", selectedIndices.includes(idx));
        updateCount();
        emitChange();
      }
    }

    dragState = null;
  });

  grid.addEventListener("pointercancel", () => {
    if (dragState) { dragState.lasso.remove(); dragState = null; }
  });

  // ── Wire controls ─────────────────────────────────────────────────────────
  colSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
  colSlider.addEventListener("input", () => {
    columns = parseInt(colSlider.value, 10);
    colVal.textContent = columns;
    applyGridLayout();
    emitChange();
  });

  squareBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  squareBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (isDisabled) return;
    layout = "square";
    squareBtn.classList.add("active");
    masonryBtn.classList.remove("active");
    renderGrid();
    emitChange();
  });

  masonryBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  masonryBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (isDisabled) return;
    layout = "masonry";
    masonryBtn.classList.add("active");
    squareBtn.classList.remove("active");
    renderGrid();
    emitChange();
  });

  clearBtn.addEventListener("pointerdown", (e) => e.stopPropagation());
  clearBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    if (isDisabled || selectedIndices.length === 0) return;
    selectedIndices = [];
    grid.querySelectorAll(".sfg-cell.selected").forEach((c) => c.classList.remove("selected"));
    updateCount();
    emitChange();
  });

  // ── Initial render ────────────────────────────────────────────────────────
  renderGrid();

  // ── Update handler ────────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    isDisabled = newProps.disabled || false;
    colSlider.disabled = isDisabled;
    squareBtn.disabled = isDisabled;
    masonryBtn.disabled = isDisabled;
    clearBtn.disabled = isDisabled;

    const newVal = newProps.value || {};
    const newItems = newVal.items || [];
    const newSelected = newVal.selected_indices || [];
    const newColumns = newVal.columns || 3;
    const newLayout = newVal.layout || "square";

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
      colSlider.value = columns;
      colVal.textContent = columns;
      needsRender = true;
    }

    if (newLayout !== layout) {
      layout = newLayout;
      squareBtn.classList.toggle("active", layout === "square");
      masonryBtn.classList.toggle("active", layout === "masonry");
      needsRender = true;
    }

    latestValue = newVal;

    if (needsRender) renderGrid();
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  function cleanup() {
    wrapper.remove();
    delete container._sfgInst;
  }

  container._sfgInst = { wrapper, handleUpdate, cleanup };
  return { cleanup, update: handleUpdate };
}
