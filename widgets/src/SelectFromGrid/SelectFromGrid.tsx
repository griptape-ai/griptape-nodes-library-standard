import { CARD_HUES, CARD_HEIGHTS, injectStyles } from './_styles';
import { Icon } from './_icons';
import { createToolbar } from './_toolbar';
import { createViewer } from './_viewer';
import type { GridItem, GridLayout, GridValue } from './_types';
import type { WidgetHandle, WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedGridValue, GridItem, GridLayout, GridSettings, GridValue } from './_types';

export type SelectFromGridProps = WidgetProps<GridValue>;

// ── SelectFromGrid widget ──────────────────────────────────────────────────────
//
// Displays a list of items as a clickable grid. The user selects items by
// clicking or drag-selecting (lasso); the selection is written back to the
// node via onChange as `selected_indices`. The toolbar's fullscreen button, or
// the one on each cell's hover, opens a fullscreen viewer (_viewer.js) that
// shares the same selection.
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

// A mounted grid, kept on its container so a repeated mount updates it instead of building
// a second one.
interface SfgInstance {
  wrapper: HTMLDivElement;
  handleUpdate: (props: SelectFromGridProps) => void;
  cleanup: () => void;
}
type SfgContainer = HTMLElement & { _sfgInst?: SfgInstance };

interface DragState {
  startX: number;
  startY: number;
  lasso: HTMLDivElement | null;
  dragging: boolean;
  startCell: HTMLElement | null;
}

const stopPropagation = (e: Event) => e.stopPropagation();

function Spinner() {
  return <div class="sfg-spinner"><Icon name="loader-circle" size={22} /></div>;
}

function ErrorCard() {
  return <div class="sfg-error-card"><Icon name="alert-circle" size={28} /></div>;
}

export default function SelectFromGrid(
  host: HTMLElement,
  props: SelectFromGridProps,
): WidgetHandle<SelectFromGridProps> {
  const container = host as SfgContainer;
  // ── Mount guard ───────────────────────────────────────────────────────────
  if (container._sfgInst?.wrapper?.isConnected) {
    container._sfgInst.handleUpdate(props);
    return { cleanup: container._sfgInst.cleanup, update: container._sfgInst.handleUpdate };
  }
  // A stale instance may still own an open fullscreen overlay and its key listener
  if (container._sfgInst) container._sfgInst.cleanup();

  injectStyles();

  // ── State ─────────────────────────────────────────────────────────────────
  let latestValue: GridValue = props.value || {};
  let onChangeRef = props.onChange;
  let isDisabled  = props.disabled || false;

  let items: GridItem[]        = latestValue.items            || [];
  let selectedIndices: number[] = latestValue.selected_indices || [];
  let columns         = latestValue.columns          || 3;
  let layout: GridLayout = latestValue.layout        || "grid";
  let multiSelect     = (latestValue.settings?.multi_select) !== false;
  let showLabels      = latestValue.show_labels !== false;
  let labelSize       = latestValue.label_size  || 10;

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
        setSelection([]);
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
      onFullscreen() {
        viewer.open(selectedIndices.length ? (selectedIndices[0] as number) : 0);
      },
    });

  const viewer = createViewer({
    getItems:      () => items,
    getSelected:   () => selectedIndices,
    isMultiSelect: () => multiSelect,
    isDisabled:    () => isDisabled,
    onToggle:      (idx) => toggleSelection(idx),
    onRemove:      (idx) => setSelection(selectedIndices.filter((i) => i !== idx)),
  });

  // ── DOM ───────────────────────────────────────────────────────────────────
  const grid = (
    <div
      class="sfg-grid"
      onPointerDown={onGridPointerDown}
      onPointerMove={onGridPointerMove}
      onPointerUp={onGridPointerUp}
      onPointerCancel={onGridPointerCancel}
    />
  ) as HTMLDivElement;

  const wrapper = (
    <div
      class={"sfg-widget nodrag nowheel" + (showLabels ? "" : " sfg-hide-labels")}
      style={{ "--sfg-label-size": labelSize + "px" }}
    >
      {controls}
      {grid}
    </div>
  ) as HTMLDivElement;

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

  function syncSelectionClasses() {
    grid.querySelectorAll<HTMLElement>(".sfg-cell").forEach((cell) => {
      cell.classList.toggle("selected", selectedIndices.includes(parseInt(cell.dataset.idx as string, 10)));
    });
  }

  function setSelection(indices: number[]) {
    selectedIndices = indices;
    syncSelectionClasses();
    updateCount();
    emitChange();
    viewer.refresh();
  }

  // Click semantics: toggle in multi-select; in single-select, picking another replaces it.
  function toggleSelection(idx: number) {
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
      if (isIntersecting) hydrateCell(target as HTMLElement);
    }),
    { root: grid, rootMargin: "300px 0px", threshold: 0 }
  );

  function buildCellContent(cell: HTMLElement, item: GridItem, idx: number, skeleton = false) {
    cell.dataset.idx = String(idx);
    const hue       = CARD_HUES[idx % CARD_HUES.length];
    const minHeight = layout === "masonry" ? CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px" : null;

    if (skeleton) {
      cell.appendChild(<div class="sfg-cell-inner" style={{ background: `hsla(${hue}, 22%, 50%, 0.09)`, minHeight }} />);
      return;
    }

    const isMedia = item.type === "image" || item.type === "video" || item.type === "audio";

    cell.appendChild(
      <div class={"sfg-cell-inner" + (isMedia ? " sfg-loading" : "")}>
        {isMedia ? (
          // Spinner placeholder only — media element created lazily by cellObserver
          <Spinner />
        ) : item.type === "dict" ? (
          <div class="sfg-dict-card" textContent={(item.value as string) || "{}"} />
        ) : (
          <div class="sfg-quote-card" style={{ background: `hsla(${hue}, 22%, 50%, 0.09)`, minHeight }}>
            <div
              class="sfg-quote-text"
              textContent={item.value !== undefined ? String(item.value) : (item.label || "")}
            />
          </div>
        )}
      </div>,
    );

    // Stop here so the grid doesn't treat the press as a selection click or lasso start
    cell.appendChild(
      <button
        type="button" class="sfg-expand-btn" title="View fullscreen"
        onPointerDown={stopPropagation}
        onClick={(e) => {
          e.stopPropagation();
          viewer.open(idx);
        }}
      >
        <Icon name="maximize-2" size={12} />
      </button>,
    );

    if (item.label) cell.appendChild(<div class="sfg-item-label">{item.label}</div>);

    if (isMedia) cellObserver.observe(cell);
  }

  function hydrateCell(cell: HTMLElement) {
    if (cell.dataset.hydrated === "true") return;

    const idx  = parseInt(cell.dataset.idx as string, 10);
    const item = items[idx];
    if (!item) return;

    // If the URL hasn't arrived yet (phase-1 placeholder), wait — phase 2 will
    // trigger a full renderGrid() which rebuilds the cell with the real URL.
    if ((item.type === "image" || item.type === "video") && !item.url) return;

    cell.dataset.hydrated = "true";
    cellObserver.unobserve(cell);

    const inner   = cell.querySelector(".sfg-cell-inner");
    if (!inner) return;
    const spinner = inner.querySelector<HTMLElement>(".sfg-spinner");

    const fadeOutSpinner = () => {
      if (!spinner) return;
      spinner.style.opacity = "0";
      spinner.addEventListener("transitionend", () => spinner.remove(), { once: true });
    };

    const showError = () => {
      inner.classList.remove("sfg-loading");
      if (spinner) spinner.remove();
      inner.appendChild(<ErrorCard />);
    };

    switch (item.type) {
      case "image": {
        const img = (
          <img
            src={item.url as string}
            alt={item.label || ""}
            decoding="async"
            onLoad={() => {
              inner.classList.remove("sfg-loading");
              img.classList.add("sfg-loaded");
              fadeOutSpinner();
            }}
            onError={showError}
          />
        ) as HTMLImageElement;
        inner.appendChild(img);
        break;
      }
      case "video": {
        const vid = (
          <video
            src={item.url as string}
            poster={item.thumbnail || null}
            muted loop playsInline
            preload="metadata"
            onLoadedMetadata={() => {
              inner.classList.remove("sfg-loading");
              vid.currentTime = 0.001;
              vid.classList.add("sfg-loaded");
              fadeOutSpinner();
            }}
            onError={showError}
          />
        ) as HTMLVideoElement;
        cell.addEventListener("mouseenter", () => void vid.play().catch(() => {}));
        cell.addEventListener("mouseleave", () => { vid.pause(); vid.currentTime = 0; });
        inner.appendChild(vid);
        break;
      }
      case "audio": {
        inner.classList.remove("sfg-loading");
        if (spinner) spinner.remove();
        const card = (
          <div class="sfg-audio-card">
            <Icon name="music" size={28} />
            {item.url ? (
              <audio
                src={item.url}
                controls
                onPointerDown={stopPropagation}
                onClick={stopPropagation}
                onError={() => card.replaceWith(<ErrorCard />)}
              />
            ) : null}
          </div>
        ) as HTMLDivElement;
        inner.appendChild(card);
        break;
      }
    }
  }

  function animateGridChange() {
    const cells = [...grid.querySelectorAll<HTMLElement>(".sfg-cell")];
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
      const entering = [...grid.querySelectorAll<HTMLElement>(".sfg-cell")];
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
      grid.appendChild(<div class="sfg-empty">Connect a list to display items here</div>);
      updateCount();
      return;
    }

    const mkCell = (item: GridItem, idx: number) => {
      const cell = (
        <div class={"sfg-cell" + (selectedIndices.includes(idx) ? " selected" : "")} />
      ) as HTMLDivElement;
      buildCellContent(cell, item, idx, skeleton);
      return cell;
    };

    if (layout === "masonry") {
      const cols = Array.from({ length: columns }, () => (<div class="sfg-masonry-col" />) as HTMLDivElement);
      grid.append(...cols);
      items.forEach((item, idx) => (cols[idx % columns] as HTMLDivElement).appendChild(mkCell(item, idx)));
    } else {
      grid.append(...items.map(mkCell));
    }

    updateCount();
  }

  // Allow the grid to scroll without React Flow intercepting wheel events
  grid.addEventListener("wheel", (e) => { e.stopPropagation(); }, { passive: true });

  // ── Box-select / click via event delegation ───────────────────────────────
  let dragState: DragState | null = null;
  const DRAG_THRESHOLD = 5;

  function findCellEl(el: HTMLElement | null) {
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
  function clientToLocal(clientX: number, clientY: number) {
    const gr    = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      x: (clientX - gr.left) / scale + grid.scrollLeft,
      y: (clientY - gr.top)  / scale + grid.scrollTop,
    };
  }

  // Return a cell's rect in the same local coordinate space as clientToLocal.
  function cellLocalRect(cell: Element) {
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

  function onGridPointerDown(e: PointerEvent) {
    if (isDisabled || e.button !== 0) return;
    e.stopPropagation();

    const { x: startX, y: startY } = clientToLocal(e.clientX, e.clientY);

    // Lasso is only created in multi-select mode
    let lasso: HTMLDivElement | null = null;
    if (multiSelect) {
      lasso = (
        <div class="sfg-lasso" style={{ left: startX + "px", top: startY + "px", width: "0", height: "0" }} />
      ) as HTMLDivElement;
      grid.appendChild(lasso);
    }

    dragState = { startX, startY, lasso, dragging: false, startCell: findCellEl(e.target as HTMLElement) };
    grid.setPointerCapture(e.pointerId);
  }

  function onGridPointerMove(e: PointerEvent) {
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

      grid.querySelectorAll<HTMLElement>(".sfg-cell").forEach((cell) => {
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        const overlaps = cLeft < selRight  && (cLeft + cW) > selLeft &&
                         cTop  < selBottom && (cTop  + cH) > selTop;
        const idx = parseInt(cell.dataset.idx as string, 10);
        cell.classList.toggle("pending", overlaps && !selectedIndices.includes(idx));
      });
    }
  }

  function onGridPointerUp(e: PointerEvent) {
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
      grid.querySelectorAll<HTMLElement>(".sfg-cell").forEach((cell) => {
        const { left: cLeft, top: cTop, width: cW, height: cH } = cellLocalRect(cell);
        if (cLeft < selRight && (cLeft + cW) > selLeft &&
            cTop  < selBottom && (cTop  + cH) > selTop) {
          const idx = parseInt(cell.dataset.idx as string, 10);
          if (!isNaN(idx) && !selectedIndices.includes(idx)) {
            selectedIndices = [...selectedIndices, idx];
            cell.classList.add("selected");
            changed = true;
          }
        }
      });

      if (changed) { updateCount(); emitChange(); viewer.refresh(); }
    } else if (!dragState.dragging && dragState.startCell) {
      const idx = parseInt(dragState.startCell.dataset.idx as string, 10);
      if (!isNaN(idx)) toggleSelection(idx);
    }

    dragState = null;
  }

  function onGridPointerCancel() {
    if (dragState) {
      if (dragState.lasso) dragState.lasso.remove();
      grid.querySelectorAll(".sfg-cell.pending").forEach((c) => c.classList.remove("pending"));
      dragState = null;
    }
  }

  // ── Initial render ────────────────────────────────────────────────────────
  renderGrid();

  // ── Update handler ────────────────────────────────────────────────────────
  function handleUpdate(newProps: SelectFromGridProps) {
    onChangeRef = newProps.onChange;
    const disabledChanged = (newProps.disabled || false) !== isDisabled;
    isDisabled  = newProps.disabled || false;
    setDisabled(isDisabled);

    const newVal: GridValue = newProps.value || {};
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
      colSlider.value   = String(columns);
      colVal.textContent = String(columns);
      needsRender = true;
    }
    if (newLayout !== layout) {
      layout = newLayout;
      gridBtn.classList.toggle("active",  layout === "grid");
      masonryBtn.classList.toggle("active", layout === "masonry");
      needsRender = true;
    }
    const multiSelectChanged = newMultiSelect !== multiSelect;
    if (multiSelectChanged) {
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
      labelSizeSlider.value  = String(labelSize);
      labelSizeVal.textContent = String(labelSize);
      wrapper.style.setProperty("--sfg-label-size", labelSize + "px");
    }

    latestValue = newVal;
    if (needsRender) renderGrid();
    if (needsRender || disabledChanged || multiSelectChanged) viewer.refresh();
  }

  // ── Cleanup ───────────────────────────────────────────────────────────────
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
