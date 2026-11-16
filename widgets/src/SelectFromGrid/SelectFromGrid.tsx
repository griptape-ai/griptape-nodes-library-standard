import { useCallback, useEffect, useInsertionEffect, useLayoutEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent, type RefObject, type SyntheticEvent } from 'react';
import { reactWidget } from '@griptape-ai/nodes-widget-api/react';

import { CARD_HUES, CARD_HEIGHTS, injectStyles } from './_styles';
import { Icon } from './_icons';
import { Toolbar } from './_toolbar';
import { Viewer } from './_viewer';
import type { GridItem, GridLayout, GridValue } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { CommittedGridValue, GridItem, GridLayout, GridSettings, GridValue } from './_types';

export type SelectFromGridProps = WidgetProps<GridValue>;

// ── SelectFromGrid widget ──────────────────────────────────────────────────────
//
// Displays a list of items as a clickable grid. The user selects items by
// clicking or drag-selecting (lasso); the selection is written back to the
// node via onChange as `selected_indices`. The toolbar's fullscreen button, or
// the one on each cell's hover, opens a fullscreen viewer (_viewer.tsx) that
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

// The grid's working copy of the value. Each update from the editor replaces it; user edits
// change it and are committed through onChange.
interface GridState {
  items: GridItem[];
  selected: number[];
  columns: number;
  layout: GridLayout;
  multiSelect: boolean;
  showLabels: boolean;
  labelSize: number;
  // The layout the cells are arranged in: a layout change takes effect after the exit animation
  shownLayout: GridLayout;
  // Cells show placeholders only, while the column slider is being dragged
  skeleton: boolean;
}

interface DragState {
  startX: number;
  startY: number;
  lasso: boolean;
  dragging: boolean;
  startCell: number | null;
}

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

const DRAG_THRESHOLD = 5;
const GRID_GAP = 5;
const EXIT_MS = 110;

const stopPropagation = (e: SyntheticEvent) => e.stopPropagation();

function stateFromValue(value: GridValue | null | undefined): GridState {
  const v = value || {};
  const layout = v.layout || "grid";
  return {
    items:       v.items            || [],
    selected:    v.selected_indices || [],
    columns:     v.columns          || 3,
    layout,
    multiSelect: v.settings?.multi_select !== false,
    showLabels:  v.show_labels !== false,
    labelSize:   v.label_size  || 10,
    shownLayout: layout,
    skeleton:    false,
  };
}

// Applies an update from the editor over the working copy, changing only the fields that differ.
// The cells are rearranged at once only when what they show changes, so an echo of the grid's own
// commit leaves a running layout animation or a column-drag skeleton alone.
function reconcile(prev: GridState, value: GridValue | null | undefined): GridState {
  const next = stateFromValue(value);
  const sameItems    = JSON.stringify(next.items) === JSON.stringify(prev.items);
  const sameSelected = JSON.stringify(next.selected) === JSON.stringify(prev.selected);
  let selected = sameSelected ? prev.selected : next.selected;
  let rearrange = !sameItems || !sameSelected || next.columns !== prev.columns || next.layout !== prev.layout;
  // If switching to single-select with multiple items selected, clear down to none
  if (next.multiSelect !== prev.multiSelect && !next.multiSelect && selected.length > 1) {
    selected = [];
    rearrange = true;
  }
  return {
    ...next,
    items:       sameItems ? prev.items : next.items,
    selected,
    shownLayout: rearrange ? next.layout : prev.shownLayout,
    skeleton:    rearrange ? false : prev.skeleton,
  };
}

const isMediaItem = (item: GridItem) => item.type === "image" || item.type === "video" || item.type === "audio";

const overlaps = (a: Rect, sel: { left: number; top: number; right: number; bottom: number }) =>
  a.left < sel.right && (a.left + a.width) > sel.left && a.top < sel.bottom && (a.top + a.height) > sel.top;

function cellIndex(cell: Element) {
  return parseInt((cell as HTMLElement).dataset.idx as string, 10);
}

// ── Lazy hydration ────────────────────────────────────────────────────────────

type WatchCell = (el: Element, onVisible: () => void) => () => void;

// One IntersectionObserver over the grid's media cells. Cells register through the returned
// function, which may be called before the observer exists: it observes everything registered so far.
function useCellObserver(gridRef: RefObject<HTMLDivElement | null>): WatchCell {
  const watched  = useRef(new Map<Element, () => void>());
  const observer = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    const io = new IntersectionObserver(
      (entries) => entries.forEach(({ isIntersecting, target }) => {
        if (isIntersecting) watched.current.get(target)?.();
      }),
      { root: gridRef.current, rootMargin: "300px 0px", threshold: 0 },
    );
    observer.current = io;
    watched.current.forEach((_, el) => io.observe(el));
    return () => {
      io.disconnect();
      observer.current = null;
    };
  }, [gridRef]);

  return useCallback((el, onVisible) => {
    watched.current.set(el, onVisible);
    observer.current?.observe(el);
    return () => {
      watched.current.delete(el);
      observer.current?.unobserve(el);
    };
  }, []);
}

// ── Cells ─────────────────────────────────────────────────────────────────────

function Spinner({ fading }: { fading: boolean }) {
  const [gone, setGone] = useState(false);
  if (gone) return null;
  return (
    <div
      className="sfg-spinner"
      style={fading ? { opacity: 0 } : undefined}
      onTransitionEnd={() => { if (fading) setGone(true); }}
    >
      <Icon name="loader-circle" size={22} />
    </div>
  );
}

function ErrorCard() {
  return <div className="sfg-error-card"><Icon name="alert-circle" size={28} /></div>;
}

interface CellMediaProps {
  item: GridItem;
  hydrated: boolean;
  videoRef: RefObject<HTMLVideoElement | null>;
}

// A media cell's content: a spinner until the cell scrolls into view and its media loads.
function CellMedia({ item, hydrated, videoRef }: CellMediaProps) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);

  const isAudio = item.type === "audio";
  const loading = !failed && !(hydrated && isAudio) && !loaded;

  let media = null;
  if (hydrated) {
    switch (item.type) {
      case "image":
        media = (
          <img
            className={loaded ? "sfg-loaded" : undefined}
            src={item.url as string}
            alt={item.label || ""}
            decoding="async"
            onLoad={() => setLoaded(true)}
            onError={() => setFailed(true)}
          />
        );
        break;
      case "video":
        media = (
          <video
            ref={videoRef}
            className={loaded ? "sfg-loaded" : undefined}
            src={item.url as string}
            poster={item.thumbnail || undefined}
            muted loop playsInline
            preload="metadata"
            onLoadedMetadata={(e) => {
              e.currentTarget.currentTime = 0.001;
              setLoaded(true);
            }}
            onError={() => setFailed(true)}
          />
        );
        break;
      case "audio":
        media = failed ? null : (
          <div className="sfg-audio-card">
            <Icon name="music" size={28} />
            {item.url ? (
              <audio
                src={item.url}
                controls
                onPointerDown={stopPropagation}
                onClick={stopPropagation}
                onError={() => setFailed(true)}
              />
            ) : null}
          </div>
        );
        break;
    }
  }

  // Audio drops its spinner as soon as it's shown; images and video fade theirs out on load
  const showSpinner = !failed && !(hydrated && isAudio);

  return (
    <div className={"sfg-cell-inner" + (loading ? " sfg-loading" : "")}>
      {showSpinner ? <Spinner fading={loaded} /> : null}
      {media}
      {failed ? <ErrorCard /> : null}
    </div>
  );
}

interface CellProps {
  item: GridItem;
  idx: number;
  masonry: boolean;
  selected: boolean;
  pending: boolean;
  skeleton: boolean;
  watch: WatchCell;
  onExpand: (idx: number) => void;
}

function Cell({ item, idx, masonry, selected, pending, skeleton, watch, onExpand }: CellProps) {
  const cellRef  = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [hydrated, setHydrated] = useState(false);

  const isMedia   = isMediaItem(item);
  const hue       = CARD_HUES[idx % CARD_HUES.length];
  const minHeight = masonry ? CARD_HEIGHTS[idx % CARD_HEIGHTS.length] + "px" : undefined;

  // Media elements are created once the cell scrolls into view
  useEffect(() => {
    const cell = cellRef.current;
    if (!cell || !isMedia || hydrated || skeleton) return;
    return watch(cell, () => {
      // If the URL hasn't arrived yet (phase-1 placeholder), wait — phase 2
      // sends the item again with the real URL, which mounts a fresh cell.
      if ((item.type === "image" || item.type === "video") && !item.url) return;
      setHydrated(true);
    });
  }, [watch, isMedia, hydrated, skeleton, item.type, item.url]);

  let content;
  if (skeleton) {
    content = <div className="sfg-cell-inner" style={{ background: `hsla(${hue}, 22%, 50%, 0.09)`, minHeight }} />;
  } else {
    content = (
      <>
        {isMedia ? (
          <CellMedia item={item} hydrated={hydrated} videoRef={videoRef} />
        ) : (
          <div className="sfg-cell-inner">
            {item.type === "dict" ? (
              <div className="sfg-dict-card">{(item.value as string) || "{}"}</div>
            ) : (
              <div className="sfg-quote-card" style={{ background: `hsla(${hue}, 22%, 50%, 0.09)`, minHeight }}>
                <div className="sfg-quote-text">
                  {item.value !== undefined ? String(item.value) : (item.label || "")}
                </div>
              </div>
            )}
          </div>
        )}
        {/* Stop here so the grid doesn't treat the press as a selection click or lasso start */}
        <button
          type="button" className="sfg-expand-btn" title="View fullscreen"
          onPointerDown={stopPropagation}
          onClick={(e) => {
            e.stopPropagation();
            onExpand(idx);
          }}
        >
          <Icon name="maximize-2" size={12} />
        </button>
        {item.label ? <div className="sfg-item-label">{item.label}</div> : null}
      </>
    );
  }

  return (
    <div
      ref={cellRef}
      className={"sfg-cell" + (selected ? " selected" : "") + (pending ? " pending" : "")}
      data-idx={String(idx)}
      data-hydrated={hydrated ? "true" : undefined}
      onMouseEnter={() => void videoRef.current?.play().catch(() => {})}
      onMouseLeave={() => {
        const vid = videoRef.current;
        if (!vid) return;
        vid.pause();
        vid.currentTime = 0;
      }}
    >
      {content}
    </div>
  );
}

// ── Widget ────────────────────────────────────────────────────────────────────

function SelectFromGrid(props: SelectFromGridProps) {
  const { value, onChange } = props;

  useInsertionEffect(() => injectStyles(), []);
  const disabled = props.disabled || false;

  const [state, setState] = useState(() => stateFromValue(value));
  const [syncedProps, setSyncedProps] = useState(props);
  if (syncedProps !== props) {
    setSyncedProps(props);
    setState((prev) => reconcile(prev, value));
  }

  const { items, selected, columns, layout, multiSelect, showLabels, labelSize, shownLayout, skeleton } = state;

  const [viewerAt, setViewerAt] = useState<number | null>(null);
  if (viewerAt !== null && items.length === 0) setViewerAt(null);

  const [lasso, setLasso] = useState<Rect | null>(null);
  const [pending, setPending] = useState<ReadonlySet<number>>(() => new Set());
  const [gridWidth, setGridWidth] = useState(0);
  const [enterTick, setEnterTick] = useState(0);

  const gridRef   = useRef<HTMLDivElement>(null);
  const dragRef   = useRef<DragState | null>(null);
  const exitTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const watch     = useCellObserver(gridRef);

  // ── Commit ──────────────────────────────────────────────────────────────

  function commit(next: GridState) {
    setState(next);
    onChange({
      ...(value || {}),
      items:            next.items,
      selected_indices: next.selected,
      columns:          next.columns,
      layout:           next.layout,
      show_labels:      next.showLabels,
      label_size:       next.labelSize,
    });
  }

  function setSelection(indices: number[]) {
    commit({ ...state, selected: indices });
  }

  // Click semantics: toggle in multi-select; in single-select, picking another replaces it.
  function toggleSelection(idx: number) {
    const alreadySelected = selected.includes(idx);
    if (multiSelect) {
      setSelection(alreadySelected ? selected.filter((i) => i !== idx) : [...selected, idx]);
    } else {
      setSelection(alreadySelected ? [] : [idx]);
    }
  }

  // ── Layout animation ────────────────────────────────────────────────────

  // Fades the current cells out, then rearranges the grid and lets the new cells animate in.
  function animateGridChange() {
    const grid = gridRef.current;
    const cells = grid ? [...grid.querySelectorAll<HTMLElement>(".sfg-cell")] : [];

    // Phase 1 — scale + fade existing cells out with a short stagger
    cells.forEach((cell, i) => {
      const delay = Math.min(i * 6, 40);
      cell.style.transition =
        `opacity ${EXIT_MS}ms ease ${delay}ms, transform ${EXIT_MS}ms ease ${delay}ms`;
      cell.style.opacity   = "0";
      cell.style.transform = "scale(0.82)";
    });

    clearTimeout(exitTimer.current);
    exitTimer.current = setTimeout(() => {
      setState((s) => ({ ...s, shownLayout: s.layout, skeleton: false }));
      setEnterTick((t) => t + 1);
    }, EXIT_MS + Math.min(cells.length * 6, 40) + 20);
  }

  useEffect(() => () => clearTimeout(exitTimer.current), []);

  // Phase 3 — start the rearranged cells invisible/shrunk, then animate them in with stagger
  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!enterTick || !grid) return;
    const entering = [...grid.querySelectorAll<HTMLElement>(".sfg-cell")];
    entering.forEach((cell) => {
      cell.style.transition = "none";
      cell.style.opacity    = "0";
      cell.style.transform  = "scale(0.88)";
    });

    // Two rAF passes ensure the browser has committed the hidden state
    // before we start the entering transition.
    let frame = requestAnimationFrame(() => {
      frame = requestAnimationFrame(() => {
        entering.forEach((cell, i) => {
          const delay = Math.min(i * 12, 90);
          cell.style.transition =
            `opacity 0.2s ease ${delay}ms, transform 0.22s cubic-bezier(0.34,1.56,0.64,1) ${delay}ms`;
          cell.style.opacity   = "";
          cell.style.transform = "";
        });
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [enterTick]);

  // Keep grid rows square when the node is resized
  useLayoutEffect(() => {
    const grid = gridRef.current;
    if (!grid) return;
    setGridWidth(grid.clientWidth);
    const ro = new ResizeObserver(() => setGridWidth(grid.clientWidth));
    ro.observe(grid);
    return () => ro.disconnect();
  }, []);

  // ── Box-select / click ──────────────────────────────────────────────────

  // Convert a viewport point to the grid's local (pre-transform) coordinate space.
  // When React Flow zooms the canvas it applies a CSS scale to an ancestor; all
  // getBoundingClientRect values are in post-scale screen pixels, but CSS position
  // values (left/top) and scrollLeft/scrollTop are in pre-scale local pixels.
  function clientToLocal(grid: HTMLElement, clientX: number, clientY: number) {
    const gr    = grid.getBoundingClientRect();
    const scale = gr.width > 0 ? gr.width / grid.offsetWidth : 1;
    return {
      x: (clientX - gr.left) / scale + grid.scrollLeft,
      y: (clientY - gr.top)  / scale + grid.scrollTop,
    };
  }

  // Return a cell's rect in the same local coordinate space as clientToLocal.
  function cellLocalRect(grid: HTMLElement, cell: Element): Rect {
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

  function selectionBox(drag: DragState, curX: number, curY: number) {
    return {
      left:   Math.min(drag.startX, curX),
      top:    Math.min(drag.startY, curY),
      right:  Math.max(drag.startX, curX),
      bottom: Math.max(drag.startY, curY),
    };
  }

  function onGridPointerDown(e: ReactPointerEvent<HTMLDivElement>) {
    if (disabled || e.button !== 0) return;
    e.stopPropagation();
    const grid = e.currentTarget;

    const { x: startX, y: startY } = clientToLocal(grid, e.clientX, e.clientY);

    // Lasso is only created in multi-select mode
    if (multiSelect) setLasso({ left: startX, top: startY, width: 0, height: 0 });

    const startCellEl = (e.target as Element).closest(".sfg-cell");
    const startCell = startCellEl && grid.contains(startCellEl) ? cellIndex(startCellEl) : null;
    dragRef.current = { startX, startY, lasso: multiSelect, dragging: false, startCell };
    grid.setPointerCapture(e.pointerId);
  }

  function onGridPointerMove(e: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const grid = e.currentTarget;
    const { x: curX, y: curY } = clientToLocal(grid, e.clientX, e.clientY);
    const dx = curX - drag.startX;
    const dy = curY - drag.startY;

    if (!drag.dragging && Math.hypot(dx, dy) > DRAG_THRESHOLD) drag.dragging = true;

    if (drag.dragging && drag.lasso) {
      const box = selectionBox(drag, curX, curY);
      setLasso({ left: box.left, top: box.top, width: Math.abs(dx), height: Math.abs(dy) });

      const next = new Set<number>();
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        const idx = cellIndex(cell);
        if (overlaps(cellLocalRect(grid, cell), box) && !selected.includes(idx)) next.add(idx);
      });
      setPending(next);
    }
  }

  function endDrag() {
    dragRef.current = null;
    setLasso(null);
    setPending(new Set());
  }

  function onGridPointerUp(e: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (!drag) return;
    const grid = e.currentTarget;
    endDrag();

    if (drag.dragging && drag.lasso) {
      // Multi-select lasso: add all cells in the rect to the selection
      const { x: curX, y: curY } = clientToLocal(grid, e.clientX, e.clientY);
      const box = selectionBox(drag, curX, curY);

      const next = [...selected];
      grid.querySelectorAll(".sfg-cell").forEach((cell) => {
        if (!overlaps(cellLocalRect(grid, cell), box)) return;
        const idx = cellIndex(cell);
        if (!isNaN(idx) && !next.includes(idx)) next.push(idx);
      });

      if (next.length !== selected.length) setSelection(next);
    } else if (!drag.dragging && drag.startCell !== null) {
      if (!isNaN(drag.startCell)) toggleSelection(drag.startCell);
    }
  }

  // ── Viewer ──────────────────────────────────────────────────────────────

  function openViewer(idx: number) {
    if (!items.length) return;
    setViewerAt(Math.max(0, Math.min(items.length - 1, idx)));
  }

  // ── Render ──────────────────────────────────────────────────────────────

  const masonry = shownLayout === "masonry";
  let gridStyle: CSSProperties | undefined;
  if (!masonry) {
    // Set grid-auto-rows to the pixel column width so cells are always 1:1.
    // CSS aspect-ratio on grid items is unreliable for row-track sizing cross-browser.
    const rowH = (gridWidth - (columns - 1) * GRID_GAP) / columns;
    gridStyle = {
      gridTemplateColumns: `repeat(${columns}, 1fr)`,
      gridAutoRows: rowH > 0 ? `${rowH}px` : undefined,
    };
  }

  const renderCell = (item: GridItem, idx: number) => (
    <Cell
      key={`${idx}|${item.type}|${item.url ?? ""}`}
      item={item} idx={idx} masonry={masonry}
      selected={selected.includes(idx)} pending={pending.has(idx)} skeleton={skeleton}
      watch={watch} onExpand={openViewer}
    />
  );

  let cells;
  if (items.length === 0) {
    cells = <div className="sfg-empty">Connect a list to display items here</div>;
  } else if (masonry) {
    cells = Array.from({ length: columns }, (_, col) => (
      <div key={col} className="sfg-masonry-col">
        {items.map((item, idx) => (idx % columns === col ? renderCell(item, idx) : null))}
      </div>
    ));
  } else {
    cells = items.map(renderCell);
  }

  return (
    <div
      className={"sfg-widget nodrag nowheel" + (showLabels ? "" : " sfg-hide-labels")}
      style={{ "--sfg-label-size": labelSize + "px" } as CSSProperties}
    >
      <Toolbar
        layout={layout} columns={columns} disabled={disabled}
        showLabels={showLabels} labelSize={labelSize} selectedCount={selected.length}
        onColumnsChange={(n) => setState({ ...state, columns: n, shownLayout: layout, skeleton: true })}
        onColumnsCommit={(n) => { animateGridChange(); commit({ ...state, columns: n }); }}
        onLayoutChange={(l) => { animateGridChange(); commit({ ...state, layout: l }); }}
        onClear={() => { if (selected.length) setSelection([]); }}
        onLabelsToggle={(v) => commit({ ...state, showLabels: v })}
        onLabelSizeChange={(n) => setState({ ...state, labelSize: n })}
        onLabelSizeCommit={(n) => commit({ ...state, labelSize: n })}
        onFullscreen={() => openViewer(selected[0] ?? 0)}
      />
      <div
        ref={gridRef}
        className={"sfg-grid" + (masonry ? " layout-masonry" : "") + (selected.length ? " has-selection" : "")}
        style={gridStyle}
        onPointerDown={onGridPointerDown}
        onPointerMove={onGridPointerMove}
        onPointerUp={onGridPointerUp}
        onPointerCancel={() => { if (dragRef.current) endDrag(); }}
        // Allow the grid to scroll without React Flow intercepting wheel events
        onWheel={stopPropagation}
      >
        {cells}
        {lasso ? (
          <div
            className="sfg-lasso"
            style={{ left: lasso.left + "px", top: lasso.top + "px", width: lasso.width + "px", height: lasso.height + "px" }}
          />
        ) : null}
      </div>
      {viewerAt !== null && items.length > 0 ? (
        <Viewer
          items={items} selected={selected} multiSelect={multiSelect} disabled={disabled}
          current={Math.min(viewerAt, items.length - 1)}
          onShow={openViewer}
          onClose={() => setViewerAt(null)}
          onToggle={toggleSelection}
          onRemove={(idx) => setSelection(selected.filter((i) => i !== idx))}
        />
      ) : null}
    </div>
  );
}

export default reactWidget(SelectFromGrid);
