import { CARD_HUES, CARD_HEIGHTS, injectStyles } from './_styles.js';
import { mkIcon } from './_icons.js';
import { createToolbar } from './_toolbar.js';

// ── SelectFromGrid widget ──────────────────────────────────────────────────────
//
// Displays a list of items as a clickable grid. The user selects items by
// clicking or drag-selecting (lasso); the selection is written back to the
// node via onChange as `selected_indices`.
//
// VALUE SHAPE (ParameterDict "grid")
// ─────────────────────────────────
//   items            array   Serialised list items (set by the Python node).
//   selected_indices array   Zero-based indices of currently selected items.
//   columns          number  Number of columns (1–8). User-adjustable via slider.
//   layout           string  "grid" | "masonry". User-adjustable via toggle.
//   settings         object  Node-author configuration — not shown to the user.
//
// SETTINGS (set once in the Python node, preserved across list updates)
// ─────────────────────────────────────────────────────────────────────
//   multi_select  boolean  (default: true)
//     true  — multiple items can be selected; lasso drag-select is enabled.
//     false — at most one item may be selected at a time; lasso is disabled.
//
// PYTHON USAGE EXAMPLE
// ─────────────────────
//   # Subclass SelectFromGrid and change a setting in __init__:
//
//   class PickOneImage(SelectFromGrid):
//       def __init__(self, name: str, metadata: dict | None = None) -> None:
//           super().__init__(name, metadata)
//           self.grid_param.default_value["settings"] = {"multi_select": False}
//
// ITEM TYPES (produced by select_from_grid.py _serialize_item)
// ─────────────────────────────────────────────────────────────
//   { type: "image", url: string }
//   { type: "video", url: string }
//   { type: "audio", url: string }
//   { type: "dict",  value: string }   — formatted JSON
//   { type: "text",  value: string }   — rendered as a styled quote card
//   Any item may also carry an optional "label" string displayed as an overlay.

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
  let isDisabled  = props.disabled || false;

  let items           = latestValue.items           || [];
  let selectedIndices = latestValue.selected_indices || [];
  let columns         = latestValue.columns          || 3;
  let layout          = latestValue.layout           || "grid";
  let multiSelect     = (latestValue.settings?.multi_select) !== false;

  // ── DOM: wrapper ──────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "sfg-widget nodrag nowheel";

  // ── Toolbar ───────────────────────────────────────────────────────────────
  const { controls, colSlider, colVal, squareBtn, masonryBtn, countEl, clearBtn, setDisabled } =
    createToolbar({
      layout, columns, isDisabled,
      onColumnsChange(n) { columns = n; renderGrid(/* skeleton */ true); },
      onColumnsCommit(n) { columns = n; animateGridChange(); emitChange(); },
      onLayoutChange(l)  { layout = l; animateGridChange(); emitChange(); },
      onClear()          {
        if (selectedIndices.length === 0) return;
        selectedIndices = [];
        grid.querySelectorAll(".sfg-cell.selected").forEach((c) => c.classList.remove("selected"));
        updateCount();
        emitChange();
      },
    });

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
    const has = selectedIndices.length > 0;
    countEl.textContent = has ? `${selectedIndices.length} selected` : "";
    countEl.classList.toggle("active", has);
    clearBtn.classList.toggle("active", has);
  }

  function updateGridRows() {
    // Set grid-auto-rows to the pixel column width so cells are always 1:1.
    // CSS aspect-ratio on grid items is unreliable for row-track sizing cross-browser.
    const gap = 5;
    const w   = grid.clientWidth;
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

  // Keep grid rows square when the node is resized
  const gridRO = new ResizeObserver(() => { if (layout !== "masonry") updateGridRows(); });
  gridRO.observe(grid);

  function mkSpinner() {
    const s = document.createElement("div");
    s.className = "sfg-spinner";
    s.appendChild(mkIcon("loader-circle", 22));
    return s;
  }

  function buildCellContent(cell, item, idx, skeleton = false) {
    cell.dataset.idx = idx;
    const inner = document.createElement("div");
    inner.className = "sfg-cell-inner";

    if (skeleton) {
      // During column-slider drag: hued placeholder only — no media loading.
      inner.style.background = `hsla(${CARD_HUES[idx % CARD_HUES.length]}, 22%, 50%, 0.09)`;
      if (layout === "masonry") {
        inner.style.minHeight = CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px";
      }
      // Grid cells get their height from grid-auto-rows; no extra class needed.
      cell.appendChild(inner);
      return;
    }

    switch (item.type) {
      case "image": {
        inner.classList.add("sfg-loading");
        const spinner = mkSpinner();
        inner.appendChild(spinner);
        if (item.url) {
          const img = document.createElement("img");
          img.src      = item.url;
          img.alt      = item.label || "";
          img.loading  = "lazy";
          img.decoding = "async";
          const fadeIn = () => {
            inner.classList.remove("sfg-loading");
            img.classList.add("sfg-loaded");
            spinner.style.opacity = "0";
            spinner.addEventListener("transitionend", () => spinner.remove(), { once: true });
          };
          img.addEventListener("load",  fadeIn);
          img.addEventListener("error", () => { inner.classList.remove("sfg-loading"); spinner.remove(); });
          inner.appendChild(img);
        }
        break;
      }
      case "video": {
        inner.classList.add("sfg-loading");
        const spinner = mkSpinner();
        inner.appendChild(spinner);
        if (item.url) {
          const vid = document.createElement("video");
          vid.src         = item.url;
          vid.muted       = true;
          vid.loop        = true;
          vid.playsInline = true;
          vid.preload     = "metadata";
          vid.addEventListener("loadedmetadata", () => {
            inner.classList.remove("sfg-loading");
            vid.currentTime = 0.001;
            vid.classList.add("sfg-loaded");
            spinner.style.opacity = "0";
            spinner.addEventListener("transitionend", () => spinner.remove(), { once: true });
          });
          vid.addEventListener("error", () => { inner.classList.remove("sfg-loading"); spinner.remove(); });
          cell.addEventListener("mouseenter", () => void vid.play().catch(() => {}));
          cell.addEventListener("mouseleave", () => { vid.pause(); vid.currentTime = 0; });
          inner.appendChild(vid);
        }
        break;
      }
      case "audio": {
        const card = document.createElement("div");
        card.className = "sfg-audio-card";
        card.appendChild(mkIcon("music", 28));
        if (item.url) {
          const audio = document.createElement("audio");
          audio.src      = item.url;
          audio.controls = true;
          audio.addEventListener("pointerdown", (e) => e.stopPropagation());
          audio.addEventListener("click",       (e) => e.stopPropagation());
          card.appendChild(audio);
        }
        inner.appendChild(card);
        break;
      }
      case "dict": {
        const dictEl = document.createElement("div");
        dictEl.className   = "sfg-dict-card";
        dictEl.textContent = item.value || "{}";
        inner.appendChild(dictEl);
        break;
      }
      default: {
        const displayText = item.value !== undefined ? String(item.value) : (item.label || "");

        const hue  = CARD_HUES[idx % CARD_HUES.length];
        const card = document.createElement("div");
        card.className        = "sfg-quote-card";
        card.style.background = `hsla(${hue}, 22%, 50%, 0.09)`;
        if (layout === "masonry") {
          card.style.minHeight = CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px";
        }

        const textEl = document.createElement("div");
        textEl.className   = "sfg-quote-text";
        textEl.textContent = displayText;
        card.appendChild(textEl);
        inner.appendChild(card);
        break;
      }
    }

    cell.appendChild(inner);

    if (item.label) {
      const lbl = document.createElement("div");
      lbl.className   = "sfg-item-label";
      lbl.textContent = item.label;
      cell.appendChild(lbl);
    }
  }

  function animateGridChange() {
    const cells = [...grid.querySelectorAll(".sfg-cell")];
    const EXIT_MS = 110;

    // Phase 1 — scale + fade existing cells out with a short stagger
    cells.forEach((cell, i) => {
      const delay = Math.min(i * 6, 40);
      cell.style.transition =
        `opacity ${EXIT_MS}ms ease ${delay}ms, transform ${EXIT_MS}ms ease ${delay}ms`;
      cell.style.opacity   = "0";
      cell.style.transform = "scale(0.82)";
    });

    setTimeout(() => {
      renderGrid();

      // Phase 3 — start new cells invisible/shrunk, then animate in with stagger
      const entering = [...grid.querySelectorAll(".sfg-cell")];
      entering.forEach((cell) => {
        cell.style.transition = "none";
        cell.style.opacity    = "0";
        cell.style.transform  = "scale(0.88)";
      });

      // Two rAF passes ensure the browser has committed the hidden state
      // before we start the entering transition.
      requestAnimationFrame(() => requestAnimationFrame(() => {
        entering.forEach((cell, i) => {
          const delay = Math.min(i * 12, 90);
          cell.style.transition =
            `opacity 0.2s ease ${delay}ms, transform 0.22s cubic-bezier(0.34,1.56,0.64,1) ${delay}ms`;
          cell.style.opacity   = "";
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
      empty.className   = "sfg-empty";
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

  // Allow the grid to scroll without React Flow intercepting wheel events
  grid.addEventListener("wheel", (e) => { e.stopPropagation(); }, { passive: true });

  // ── Box-select / click via event delegation ───────────────────────────────
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
    const startX   = e.clientX - gridRect.left + grid.scrollLeft;
    const startY   = e.clientY - gridRect.top  + grid.scrollTop;

    // Lasso is only created in multi-select mode
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
    const gridRect = grid.getBoundingClientRect();
    const curX = e.clientX - gridRect.left + grid.scrollLeft;
    const curY = e.clientY - gridRect.top  + grid.scrollTop;
    const dx   = curX - dragState.startX;
    const dy   = curY - dragState.startY;

    if (!dragState.dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD) dragState.dragging = true;

    if (dragState.dragging && dragState.lasso) {
      const selLeft   = Math.min(dragState.startX, curX);
      const selTop    = Math.min(dragState.startY, curY);
      const selRight  = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);

      Object.assign(dragState.lasso.style, {
        left:   selLeft  + "px",
        top:    selTop   + "px",
        width:  Math.abs(dx) + "px",
        height: Math.abs(dy) + "px",
      });

      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const r     = cell.getBoundingClientRect();
        const gRect = grid.getBoundingClientRect();
        const cLeft = r.left - gRect.left + grid.scrollLeft;
        const cTop  = r.top  - gRect.top  + grid.scrollTop;
        const overlaps = cLeft < selRight  && (cLeft + r.width)  > selLeft &&
                         cTop  < selBottom && (cTop  + r.height) > selTop;
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
      // Multi-select lasso: add all cells in the rect to the selection
      const gridRect  = grid.getBoundingClientRect();
      const curX      = e.clientX - gridRect.left + grid.scrollLeft;
      const curY      = e.clientY - gridRect.top  + grid.scrollTop;
      const selLeft   = Math.min(dragState.startX, curX);
      const selTop    = Math.min(dragState.startY, curY);
      const selRight  = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);

      let changed = false;
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const r      = cell.getBoundingClientRect();
        const gRect2 = grid.getBoundingClientRect();
        const cLeft  = r.left - gRect2.left + grid.scrollLeft;
        const cTop   = r.top  - gRect2.top  + grid.scrollTop;
        if (cLeft < selRight && (cLeft + r.width) > selLeft &&
            cTop  < selBottom && (cTop  + r.height) > selTop) {
          const idx = parseInt(cell.dataset.idx, 10);
          if (!isNaN(idx) && !selectedIndices.includes(idx)) {
            selectedIndices = [...selectedIndices, idx];
            cell.classList.add("selected");
            changed = true;
          }
        }
      });

      if (changed) { updateCount(); emitChange(); }
    } else if (!dragState.dragging && dragState.startCell) {
      // Click — toggle selection; in single-select mode clear all others first
      const idx = parseInt(dragState.startCell.dataset.idx, 10);
      if (!isNaN(idx)) {
        const alreadySelected = selectedIndices.includes(idx);
        if (multiSelect) {
          if (alreadySelected) selectedIndices = selectedIndices.filter((i) => i !== idx);
          else                 selectedIndices = [...selectedIndices, idx];
        } else {
          if (alreadySelected) {
            selectedIndices = [];
          } else {
            // Deselect the previously selected cell visually before updating state
            grid.querySelectorAll(".sfg-cell.selected").forEach((c) => c.classList.remove("selected"));
            selectedIndices = [idx];
          }
        }
        dragState.startCell.classList.toggle("selected", selectedIndices.includes(idx));
        updateCount();
        emitChange();
      }
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

  // ── Initial render ────────────────────────────────────────────────────────
  renderGrid();

  // ── Update handler ────────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    onChangeRef = newProps.onChange;
    isDisabled  = newProps.disabled || false;
    setDisabled(isDisabled);

    const newVal        = newProps.value || {};
    const newItems      = newVal.items             || [];
    const newSelected   = newVal.selected_indices   || [];
    const newColumns    = newVal.columns            || 3;
    const newLayout     = newVal.layout             || "grid";
    const newMultiSelect = (newVal.settings?.multi_select) !== false;

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
      colSlider.value   = columns;
      colVal.textContent = columns;
      needsRender = true;
    }
    if (newLayout !== layout) {
      layout = newLayout;
      squareBtn.classList.toggle("active",  layout === "grid");
      masonryBtn.classList.toggle("active", layout === "masonry");
      needsRender = true;
    }
    if (newMultiSelect !== multiSelect) {
      multiSelect = newMultiSelect;
      // If switching to single-select with multiple items selected, clear down to none
      if (!multiSelect && selectedIndices.length > 1) {
        selectedIndices = [];
        needsRender = true;
      }
    }

    latestValue = newVal;
    if (needsRender) renderGrid();
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  function cleanup() {
    gridRO.disconnect();
    wrapper.remove();
    delete container._sfgInst;
  }

  container._sfgInst = { wrapper, handleUpdate, cleanup };
  return { cleanup, update: handleUpdate };
}
