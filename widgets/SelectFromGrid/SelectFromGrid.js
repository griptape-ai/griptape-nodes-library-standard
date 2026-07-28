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
  let showLabels      = latestValue.show_labels !== false;
  let labelSize       = latestValue.label_size  || 10;

  // ── DOM: wrapper ──────────────────────────────────────────────────────────
  const wrapper = document.createElement("div");
  wrapper.className = "sfg-widget nodrag nowheel";

  // ── Toolbar ───────────────────────────────────────────────────────────────
  const { controls, colSlider, colVal, gridBtn, masonryBtn, countEl, clearBtn,
          labelsBtn, labelSizeSlider, labelSizeVal, setDisabled } =
    createToolbar({
      layout, columns, isDisabled, showLabels, labelSize,
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
      onLabelsToggle(v)  {
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
    });

  wrapper.appendChild(controls);
  if (!showLabels) wrapper.classList.add("sfg-hide-labels");
  wrapper.style.setProperty("--sfg-label-size", labelSize + "px");

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
      show_labels: showLabels,
      label_size:  labelSize,
    });
  }

  function updateCount() {
    const has = selectedIndices.length > 0;
    countEl.textContent = has ? `${selectedIndices.length} selected` : "";
    countEl.classList.toggle("active", has);
    clearBtn.classList.toggle("active", has);
    grid.classList.toggle("has-selection", has);
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

  // Lazy-hydrate media cells as they scroll into view
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
    cell.dataset.idx = idx;
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
      // Spinner placeholder only — media element created lazily by cellObserver
      inner.classList.add("sfg-loading");
      inner.appendChild(mkSpinner());
    } else {
      switch (item.type) {
        case "dict": {
          const dictEl = document.createElement("div");
          dictEl.className   = "sfg-dict-card";
          dictEl.textContent = item.value || "{}";
          inner.appendChild(dictEl);
          break;
        }
        default: {
          const hue  = CARD_HUES[idx % CARD_HUES.length];
          const card = document.createElement("div");
          card.className        = "sfg-quote-card";
          card.style.background = `hsla(${hue}, 22%, 50%, 0.09)`;
          if (layout === "masonry") card.style.minHeight = CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px";
          const textEl = document.createElement("div");
          textEl.className   = "sfg-quote-text";
          textEl.textContent = item.value !== undefined ? String(item.value) : (item.label || "");
          card.appendChild(textEl);
          inner.appendChild(card);
          break;
        }
      }
    }

    cell.appendChild(inner);

    if (item.label) {
      const lbl = document.createElement("div");
      lbl.className   = "sfg-item-label";
      lbl.textContent = item.label;
      cell.appendChild(lbl);
    }

    if (isMedia) cellObserver.observe(cell);
  }

  function hydrateCell(cell) {
    if (cell.dataset.hydrated === "true") return;

    const idx  = parseInt(cell.dataset.idx, 10);
    const item = items[idx];
    if (!item) return;

    // If the URL hasn't arrived yet (phase-1 placeholder), wait — phase 2 will
    // trigger a full renderGrid() which rebuilds the cell with the real URL.
    if ((item.type === "image" || item.type === "video") && !item.url) return;

    cell.dataset.hydrated = "true";
    cellObserver.unobserve(cell);

    const inner   = cell.querySelector(".sfg-cell-inner");
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
        const img    = document.createElement("img");
        img.src      = item.url;
        img.alt      = item.label || "";
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
        const vid       = document.createElement("video");
        vid.src         = item.url;
        if (item.thumbnail) vid.poster = item.thumbnail;
        vid.muted       = true;
        vid.loop        = true;
        vid.playsInline = true;
        vid.preload     = "metadata";
        vid.addEventListener("loadedmetadata", () => {
          inner.classList.remove("sfg-loading");
          vid.currentTime = 0.001;
          vid.classList.add("sfg-loaded");
          fadeOutSpinner();
        });
        vid.addEventListener("error", showError);
        cell.addEventListener("mouseenter", () => void vid.play().catch(() => {}));
        cell.addEventListener("mouseleave", () => { vid.pause(); vid.currentTime = 0; });
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
          const audio    = document.createElement("audio");
          audio.src      = item.url;
          audio.controls = true;
          audio.addEventListener("pointerdown", (e) => e.stopPropagation());
          audio.addEventListener("click",       (e) => e.stopPropagation());
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

  // Convert a viewport point to the grid's local (pre-transform) coordinate space.
  // When React Flow zooms the canvas it applies a CSS scale to an ancestor; all
  // getBoundingClientRect values are in post-scale screen pixels, but CSS position
  // values (left/top) and scrollLeft/scrollTop are in pre-scale local pixels.
  function clientToLocal(clientX, clientY) {
    const gr    = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      x: (clientX - gr.left) / scale + grid.scrollLeft,
      y: (clientY - gr.top)  / scale + grid.scrollTop,
    };
  }

  // Return a cell's rect in the same local coordinate space as clientToLocal.
  function cellLocalRect(cell) {
    const cr    = cell.getBoundingClientRect();
    const gr    = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      left:   (cr.left - gr.left) / scale + grid.scrollLeft,
      top:    (cr.top  - gr.top)  / scale + grid.scrollTop,
      width:  cr.width  / scale,
      height: cr.height / scale,
    };
  }

  grid.addEventListener("pointerdown", (e) => {
    if (isDisabled || e.button !== 0) return;
    e.stopPropagation();

    const { x: startX, y: startY } = clientToLocal(e.clientX, e.clientY);

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
    const { x: curX, y: curY } = clientToLocal(e.clientX, e.clientY);
    const dx = curX - dragState.startX;
    const dy = curY - dragState.startY;

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
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        const overlaps = cLeft < selRight  && (cLeft + cW) > selLeft &&
                         cTop  < selBottom && (cTop  + cH) > selTop;
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
      const { x: curX, y: curY } = clientToLocal(e.clientX, e.clientY);
      const selLeft   = Math.min(dragState.startX, curX);
      const selTop    = Math.min(dragState.startY, curY);
      const selRight  = Math.max(dragState.startX, curX);
      const selBottom = Math.max(dragState.startY, curY);

      let changed = false;
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        if (cLeft < selRight && (cLeft + cW) > selLeft &&
            cTop  < selBottom && (cTop  + cH) > selTop) {
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

    const newVal         = newProps.value || {};
    const newItems       = newVal.items             || [];
    const newSelected    = newVal.selected_indices   || [];
    const newColumns     = newVal.columns            || 3;
    const newLayout      = newVal.layout             || "grid";
    const newMultiSelect = (newVal.settings?.multi_select) !== false;
    const newShowLabels  = newVal.show_labels !== false;
    const newLabelSize   = newVal.label_size  || 10;

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
      gridBtn.classList.toggle("active",  layout === "grid");
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
    if (newShowLabels !== showLabels) {
      showLabels = newShowLabels;
      labelsBtn.classList.toggle("active", showLabels);
      wrapper.classList.toggle("sfg-hide-labels", !showLabels);
    }
    if (newLabelSize !== labelSize) {
      labelSize = newLabelSize;
      labelSizeSlider.value  = labelSize;
      labelSizeVal.textContent = labelSize;
      wrapper.style.setProperty("--sfg-label-size", labelSize + "px");
    }

    latestValue = newVal;
    if (needsRender) renderGrid();
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
  function cleanup() {
    gridRO.disconnect();
    cellObserver.disconnect();
    wrapper.remove();
    delete container._sfgInst;
  }

  container._sfgInst = { wrapper, handleUpdate, cleanup };
  return { cleanup, update: handleUpdate };
}
