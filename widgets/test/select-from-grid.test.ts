import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { GridItem, GridValue, SelectFromGridProps } from "../src/SelectFromGrid/SelectFromGrid";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type DomFakes, installDomFakes, pointer } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<SelectFromGridProps>;
type Props = Partial<SelectFromGridProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/SelectFromGrid/SelectFromGrid")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["SelectFromGrid"]);
      bundleDirs.push(outDir);
      return importMount<SelectFromGridProps>(join(outDir, "SelectFromGrid.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

// An IntersectionObserver the test drives: the DOM env never reports intersections itself.
class FakeIntersectionObserver {
  static instances: FakeIntersectionObserver[] = [];

  observed: Element[] = [];
  disconnected = false;

  constructor(private readonly callback: IntersectionObserverCallback) {
    FakeIntersectionObserver.instances.push(this);
  }

  observe(target: Element): void {
    this.observed.push(target);
  }

  unobserve(target: Element): void {
    this.observed = this.observed.filter((t) => t !== target);
  }

  disconnect(): void {
    this.disconnected = true;
    this.observed = [];
  }

  // Reports every observed target as scrolled into view.
  intersectAll(): void {
    const entries = this.observed.map((target) => ({ isIntersecting: true, target }));
    this.callback(entries as unknown as IntersectionObserverEntry[], this as unknown as IntersectionObserver);
  }
}

// The DOM env lays nothing out. Cells are placed on a 3-column grid of 100px tracks (90px cells),
// by their index, and the grid itself reports an empty rect so local and client coordinates match.
function layoutCells(): void {
  vi.spyOn(Element.prototype, "getBoundingClientRect").mockImplementation(function (this: Element) {
    if (this instanceof HTMLElement && this.classList.contains("sfg-cell")) {
      const idx = Number(this.dataset.idx);
      return new DOMRect((idx % 3) * 100, Math.floor(idx / 3) * 100, 90, 90);
    }
    return new DOMRect(0, 0, 0, 0);
  });
}

function textItems(n: number): GridItem[] {
  return Array.from({ length: n }, (_, i) => ({ type: "text", value: `item ${i}` }));
}

function baseValue(overrides: Partial<GridValue> = {}): GridValue {
  return {
    items: textItems(6),
    selected_indices: [],
    columns: 3,
    layout: "grid",
    settings: { multi_select: true },
    ...overrides,
  };
}

describe.each(implementations)("SelectFromGrid (%s)", (_label, load) => {
  let SelectFromGrid: Mount;
  let fakes: DomFakes;
  let container: HTMLElement;
  let handles: MountedWidget<SelectFromGridProps>[];

  beforeAll(async () => {
    SelectFromGrid = await load();
  });

  beforeEach(() => {
    fakes = installDomFakes();
    FakeIntersectionObserver.instances = [];
    vi.stubGlobal("IntersectionObserver", FakeIntersectionObserver);
    layoutCells();
    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(() => {
    for (const h of handles) h.cleanup();
    container.remove();
    document.getElementById("sfg-widget-styles")?.remove();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props) {
    const mounted = mountWidget(SelectFromGrid, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  function mountWith(value: GridValue, props: Props = {}) {
    const onChange = vi.fn<(value: GridValue) => void>();
    const handle = mount({ value, onChange, ...props });
    return { handle, onChange };
  }

  const cells = () => [...container.querySelectorAll<HTMLElement>(".sfg-cell")];
  const cell = (idx: number) => {
    const el = cells().find((c) => c.dataset.idx === String(idx));
    if (!el) throw new Error(`no cell ${idx}`);
    return el;
  };
  const grid = () => {
    const el = container.querySelector<HTMLElement>(".sfg-grid");
    if (!el) throw new Error("grid not mounted");
    return el;
  };
  const selected = () => cells().filter((c) => c.classList.contains("selected")).map((c) => Number(c.dataset.idx));
  const button = (cls: string, root: ParentNode = container) => {
    const el = root.querySelector<HTMLButtonElement>(`button.${cls}`);
    if (!el) throw new Error(`no button .${cls}`);
    return el;
  };
  const layoutButton = (text: string) => {
    const el = [...container.querySelectorAll<HTMLButtonElement>(".sfg-layout-btn")].find((b) => b.textContent === text);
    if (!el) throw new Error(`no layout button ${text}`);
    return el;
  };
  const sliders = () => [...container.querySelectorAll<HTMLInputElement>('input[type="range"]')];
  const viewer = () => document.body.querySelector<HTMLElement>(".sfg-viewer");
  const lastCommit = (onChange: ReturnType<typeof vi.fn>) => onChange.mock.lastCall?.[0] as GridValue;

  // A press and release on the cell's content, without moving: a click.
  function clickCell(idx: number) {
    const target = cell(idx).querySelector(".sfg-cell-inner") ?? cell(idx);
    pointer(target, "pointerdown", 10, 10);
    pointer(grid(), "pointerup", 10, 10);
  }

  function lasso(from: [number, number], to: [number, number]) {
    pointer(grid(), "pointerdown", ...from);
    pointer(grid(), "pointermove", ...to);
    pointer(grid(), "pointerup", ...to);
  }

  function setSlider(slider: HTMLInputElement, value: number) {
    slider.value = String(value);
    slider.dispatchEvent(new Event("input"));
    slider.dispatchEvent(new Event("change"));
  }

  function key(k: string) {
    window.dispatchEvent(new KeyboardEvent("keydown", { key: k, bubbles: true, cancelable: true }));
  }

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the toolbar and a cell per item", () => {
      const handle = mount({ value: baseValue() });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(container.querySelectorAll(".sfg-widget")).toHaveLength(1);
      expect(container.querySelector(".sfg-controls")).not.toBeNull();
      expect(cells()).toHaveLength(6);
      expect(cell(2).querySelector(".sfg-quote-text")?.textContent).toBe("item 2");
      expect(grid().style.gridTemplateColumns).toBe("repeat(3, 1fr)");
      expect(document.getElementById("sfg-widget-styles")).not.toBeNull();
    });

    it("prompts for a list when the value has no items", () => {
      mount({ value: null });
      expect(cells()).toHaveLength(0);
      expect(container.querySelector(".sfg-empty")?.textContent).toBe("Connect a list to display items here");
    });

    it("hydrates media cells once they scroll into view", () => {
      // Given an image item and one whose URL hasn't been resolved yet
      mount({ value: baseValue({ items: [{ type: "image", url: "https://example.test/a.png", label: "A" }, { type: "image", url: "" }] }) });
      expect(cell(0).querySelector(".sfg-spinner")).not.toBeNull();
      expect(cell(0).querySelector(".sfg-item-label")?.textContent).toBe("A");

      // When both scroll into view
      FakeIntersectionObserver.instances.at(-1)?.intersectAll();

      // Then the resolved one gets its image and the placeholder keeps waiting
      expect(cell(0).querySelector("img")?.getAttribute("src")).toBe("https://example.test/a.png");
      expect(cell(0).dataset.hydrated).toBe("true");
      expect(cell(1).querySelector("img")).toBeNull();
    });

    it("updates the existing grid when mounted again into the same container", () => {
      // Given a mounted grid
      mount({ value: baseValue() });

      // When the editor mounts into the same container with new items
      mount({ value: baseValue({ items: textItems(2) }) });

      // Then there is still one grid, showing the new items
      expect(container.querySelectorAll(".sfg-widget")).toHaveLength(1);
      expect(cells()).toHaveLength(2);
    });
  });

  describe("update", () => {
    it("applies new items, selection, columns, and label settings", () => {
      // Given a mounted grid
      const handle = mount({ value: baseValue() });

      // When the value changes
      handle.update({ value: baseValue({ items: textItems(4), selected_indices: [1, 3], columns: 2, show_labels: false, label_size: 14 }) });

      // Then the grid and toolbar reflect it
      expect(cells()).toHaveLength(4);
      expect(selected()).toEqual([1, 3]);
      expect(container.querySelector(".sfg-count")?.textContent).toBe("2 selected");
      expect(grid().style.gridTemplateColumns).toBe("repeat(2, 1fr)");
      expect(sliders()[0]?.value).toBe("2");
      expect(container.querySelector(".sfg-widget")?.classList.contains("sfg-hide-labels")).toBe(true);
      expect(sliders()[1]?.value).toBe("14");
    });

    it("switches to masonry columns when the layout changes", () => {
      const handle = mount({ value: baseValue() });
      handle.update({ value: baseValue({ layout: "masonry" }) });
      expect(grid().classList.contains("layout-masonry")).toBe(true);
      expect(container.querySelectorAll(".sfg-masonry-col")).toHaveLength(3);
      expect(layoutButton("Masonry").classList.contains("active")).toBe(true);
    });

    it("clears a multiple selection when the node switches to single-select", () => {
      const handle = mount({ value: baseValue({ selected_indices: [0, 1] }) });
      handle.update({ value: baseValue({ selected_indices: [0, 1], settings: { multi_select: false } }) });
      expect(selected()).toEqual([]);
    });
  });

  describe("selection", () => {
    it("toggles a cell on click and commits the value with the node's other fields kept", () => {
      // Given a grid whose value carries a field the widget doesn't know
      const { onChange } = mountWith(baseValue({ extra: "kept" }));

      // When the user clicks a cell
      clickCell(1);

      // Then the selection is committed with the rest of the value
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(lastCommit(onChange)).toEqual({
        ...baseValue({ extra: "kept" }),
        selected_indices: [1],
        show_labels: true,
        label_size: 10,
      });
      expect(selected()).toEqual([1]);

      // And clicking it again deselects it
      clickCell(1);
      expect(lastCommit(onChange).selected_indices).toEqual([]);
    });

    it("adds to the selection in multi-select and replaces it in single-select", () => {
      const multi = mountWith(baseValue({ selected_indices: [0] }));
      clickCell(2);
      expect(lastCommit(multi.onChange).selected_indices).toEqual([0, 2]);
      multi.handle.cleanup();

      const single = mountWith(baseValue({ selected_indices: [0], settings: { multi_select: false } }));
      clickCell(2);
      expect(lastCommit(single.onChange).selected_indices).toEqual([2]);
    });

    it("lasso-selects every cell the dragged box overlaps", () => {
      // Given a multi-select grid with item 4 already selected
      const { onChange } = mountWith(baseValue({ selected_indices: [4] }));

      // When the user drags a box over the first two columns of both rows
      lasso([5, 5], [150, 150]);

      // Then the overlapped cells join the selection and the lasso is gone
      expect(lastCommit(onChange).selected_indices).toEqual([4, 0, 1, 3]);
      expect(selected()).toEqual([0, 1, 3, 4]);
      expect(grid().querySelector(".sfg-lasso")).toBeNull();
    });

    it("has no lasso in single-select", () => {
      const { onChange } = mountWith(baseValue({ settings: { multi_select: false } }));
      pointer(grid(), "pointerdown", 5, 5);
      expect(grid().querySelector(".sfg-lasso")).toBeNull();
      pointer(grid(), "pointermove", 150, 150);
      pointer(grid(), "pointerup", 150, 150);
      expect(onChange).not.toHaveBeenCalled();
    });

    it("clears the selection from the toolbar", () => {
      const { onChange } = mountWith(baseValue({ selected_indices: [0, 2] }));
      button("sfg-clear-btn").click();
      expect(lastCommit(onChange).selected_indices).toEqual([]);
      expect(selected()).toEqual([]);
    });
  });

  describe("toolbar", () => {
    it("commits the column count when the slider is released", () => {
      const { onChange } = mountWith(baseValue());
      const colSlider = sliders()[0];
      if (!colSlider) throw new Error("no column slider");
      setSlider(colSlider, 5);
      expect(lastCommit(onChange).columns).toBe(5);
      expect(grid().style.gridTemplateColumns).toBe("repeat(5, 1fr)");
    });

    it("commits the layout and rebuilds the grid after the exit animation", () => {
      vi.useFakeTimers();
      const { onChange } = mountWith(baseValue());
      layoutButton("Masonry").click();
      expect(lastCommit(onChange).layout).toBe("masonry");
      vi.advanceTimersByTime(500);
      expect(container.querySelectorAll(".sfg-masonry-col")).toHaveLength(3);
    });

    it("commits the labels toggle and the label size", () => {
      const { onChange } = mountWith(baseValue());
      button("sfg-labels-btn").click();
      expect(lastCommit(onChange).show_labels).toBe(false);
      expect(container.querySelector(".sfg-widget")?.classList.contains("sfg-hide-labels")).toBe(true);

      const sizeSlider = sliders()[1];
      if (!sizeSlider) throw new Error("no label size slider");
      setSlider(sizeSlider, 20);
      expect(lastCommit(onChange).label_size).toBe(20);
      expect(container.querySelector<HTMLElement>(".sfg-widget")?.style.getPropertyValue("--sfg-label-size")).toBe("20px");
    });
  });

  describe("disabled", () => {
    it("blocks selection and disables the editing controls, but not the fullscreen button", () => {
      // Given a disabled grid
      const { onChange } = mountWith(baseValue(), { disabled: true });

      // When the user clicks and drags over cells
      clickCell(0);
      lasso([5, 5], [150, 150]);

      // Then nothing is committed
      expect(onChange).not.toHaveBeenCalled();
      expect(selected()).toEqual([]);
      for (const s of sliders()) expect(s.disabled).toBe(true);
      expect(button("sfg-clear-btn").disabled).toBe(true);
      expect(layoutButton("Grid").disabled).toBe(true);
      expect(button("sfg-fullscreen-btn").disabled).toBe(false);
    });

    it("re-enables selection when an update lifts it", () => {
      const onChange = vi.fn<(value: GridValue) => void>();
      const handle = mount({ value: baseValue(), onChange, disabled: true });
      handle.update({ value: baseValue(), onChange, disabled: false });
      expect(button("sfg-clear-btn").disabled).toBe(false);
      clickCell(0);
      expect(lastCommit(onChange).selected_indices).toEqual([0]);
    });

    it("blocks selection from the fullscreen viewer", () => {
      const { onChange } = mountWith(baseValue(), { disabled: true });
      button("sfg-fullscreen-btn").click();
      key(" ");
      expect(onChange).not.toHaveBeenCalled();
      expect(button("sfg-viewer-select", document.body).disabled).toBe(true);
    });
  });

  describe("fullscreen viewer", () => {
    it("opens on the first selected item, steps with the arrow keys, and closes on Escape", () => {
      mountWith(baseValue({ selected_indices: [2] }));
      button("sfg-fullscreen-btn").click();
      expect(viewer()).not.toBeNull();
      const counter = () => document.body.querySelector(".sfg-viewer-counter")?.textContent;
      expect(counter()).toBe("3 / 6");
      expect(document.body.querySelector(".sfg-viewer-text")?.textContent).toBe("item 2");

      key("ArrowRight");
      expect(counter()).toBe("4 / 6");
      key("ArrowLeft");
      key("ArrowLeft");
      expect(counter()).toBe("2 / 6");

      key("Escape");
      expect(viewer()).toBeNull();
    });

    it("toggles the shown item with Space and removes items from the strip", () => {
      // Given the viewer opened from a cell's expand button
      const { onChange } = mountWith(baseValue({ selected_indices: [0] }));
      button("sfg-expand-btn", cell(3)).click();
      expect(document.body.querySelector(".sfg-viewer-counter")?.textContent).toBe("4 / 6");

      // When the user presses Space
      key(" ");

      // Then the item joins the selection in both views
      expect(lastCommit(onChange).selected_indices).toEqual([0, 3]);
      expect(selected()).toEqual([0, 3]);
      expect(document.body.querySelectorAll(".sfg-viewer-thumb")).toHaveLength(2);

      // And removing a thumbnail from the strip deselects it
      const thumb = document.body.querySelector<HTMLElement>('.sfg-viewer-thumb[data-idx="0"]');
      if (!thumb) throw new Error("no thumbnail for item 0");
      button("sfg-viewer-thumb-remove", thumb).click();
      expect(lastCommit(onChange).selected_indices).toEqual([3]);
    });
  });

  describe("cleanup", () => {
    it("removes the grid and the open viewer and stops its observers and key listener", () => {
      // Given a grid with its fullscreen viewer open
      const onChange = vi.fn<(value: GridValue) => void>();
      const mounted = mountWidget(SelectFromGrid, container, widgetProps<GridValue>({ value: baseValue(), onChange }));
      button("sfg-fullscreen-btn").click();
      const ro = fakes.lastResizeObserver();
      const io = FakeIntersectionObserver.instances.at(-1);

      // When the widget is cleaned up
      mounted.cleanup();

      // Then everything it added is gone and nothing reacts any more
      expect(container.querySelector(".sfg-widget")).toBeNull();
      expect(viewer()).toBeNull();
      expect(ro.disconnected).toBe(true);
      expect(io?.disconnected).toBe(true);
      key(" ");
      expect(onChange).not.toHaveBeenCalled();
    });
  });
});
