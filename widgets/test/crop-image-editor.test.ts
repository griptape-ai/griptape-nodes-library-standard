import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { CropImageEditorProps, CropValue, LiveCropValue } from "../src/CropImageEditor/CropImageEditor";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type DomFakes, installDomFakes, pointer } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<CropImageEditorProps>;
type Props = Partial<CropImageEditorProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/CropImageEditor/CropImageEditor")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["CropImageEditor"]);
      bundleDirs.push(outDir);
      return importMount<CropImageEditorProps>(join(outDir, "CropImageEditor.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

// A 200×100 image in a container the DOM env reports as 0×0, so the widget falls back to a
// 480×360 area: scale 2.4 and a 480×240 canvas. Image (x, y) sits at canvas (2.4x, 2.4y).
const SCALE = 2.4;
const IMAGE_URL = "https://example.test/image.png?v=1";

function baseValue(overrides: Partial<CropValue> = {}): CropValue {
  return {
    image_url: IMAGE_URL,
    img_width: 200,
    img_height: 100,
    left: 50,
    top: 25,
    width: 100,
    height: 50,
    zoom: 100,
    rotate: 0,
    ...overrides,
  };
}

function at(ix: number, iy: number): [number, number] {
  return [ix * SCALE, iy * SCALE];
}

describe.each(implementations)("CropImageEditor (%s)", (_label, load) => {
  let CropImageEditor: Mount;
  let fakes: DomFakes;
  let container: HTMLElement;
  let handles: MountedWidget<CropImageEditorProps>[];

  beforeAll(async () => {
    CropImageEditor = await load();
  });

  beforeEach(() => {
    fakes = installDomFakes();
    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(() => {
    for (const h of handles) h.cleanup();
    container.remove();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props) {
    const mounted = mountWidget(CropImageEditor, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  function mountLoaded(props: Partial<CropImageEditorProps> = {}) {
    const onChange = vi.fn<(value: CropValue) => void>();
    const onLiveChange = vi.fn<(value: LiveCropValue) => void>();
    const handle = mount({ value: baseValue(), onChange, onLiveChange, ...props });
    fakes.lastImage().load(200, 100);
    return { handle, onChange, onLiveChange };
  }

  const canvas = () => {
    const el = container.querySelector("canvas");
    if (!el) throw new Error("canvas not mounted");
    return el;
  };
  const editors = () => container.querySelectorAll(".crop-image-editor");
  const statusText = () => container.querySelector(".crop-image-editor > div:last-child > span")?.textContent ?? "";
  const sliders = () => [...container.querySelectorAll<HTMLInputElement>('input[type="range"]')];
  const sidebarButton = (title: string) => {
    const btn = [...container.querySelectorAll<HTMLButtonElement>("button")].find((b) => b.title === title);
    if (!btn) throw new Error(`no button titled ${title}`);
    return btn;
  };

  function drag(from: [number, number], to: [number, number], init: { shiftKey?: boolean } = {}) {
    pointer(canvas(), "pointerdown", ...from);
    pointer(canvas(), "pointermove", ...to, init);
    pointer(canvas(), "pointerup", ...to);
  }

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the editor inside the container", () => {
      const handle = mount({ value: baseValue() });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(editors()).toHaveLength(1);
      expect(container.querySelector("canvas")).not.toBeNull();
      expect(sliders()).toHaveLength(2);
    });

    it("prompts for an image when the value has none", () => {
      mount({ value: null });
      expect(canvas().width).toBe(400);
      expect(canvas().height).toBe(300);
      expect(fakes.lastContext().texts()).toContain("Connect an image to begin");
      expect(statusText()).toBe("No image connected");
    });

    it("loads the image anonymously and sizes the canvas to fit once it loads", () => {
      mount({ value: baseValue() });
      const img = fakes.lastImage();
      expect(img.crossOrigin).toBe("anonymous");
      expect(img.src).toBe(IMAGE_URL);

      img.load(200, 100);

      expect(canvas().width).toBe(480);
      expect(canvas().height).toBe(240);
      expect(statusText()).toBe("x: 50  y: 25  →  100 × 50 px");
    });

    it("takes the image size from the loaded image when the value carries none, and defaults the crop to it", () => {
      mount({ value: { image_url: IMAGE_URL } });
      fakes.lastImage().load(300, 150);
      expect(statusText()).toBe("x: 0  y: 0  →  300 × 150 px");
    });

    it("shows the prompt again when the image fails to load", () => {
      mount({ value: baseValue() });
      fakes.lastContext().reset();
      fakes.lastImage().fail();
      expect(fakes.lastContext().texts()).toContain("Connect an image to begin");
    });
  });

  describe("repeated mount on the same container", () => {
    it("reuses the editor and applies the new props", () => {
      // Given a mounted, loaded editor
      const first = mountLoaded();
      const contextsBefore = fakes.contexts.length;

      // When the loader mounts again on the same container with new props
      const onChange = vi.fn<(value: CropValue) => void>();
      const second = mount({ value: baseValue({ left: 10, top: 5 }), onChange });

      // Then no second editor is built and the new value is shown
      expect(editors()).toHaveLength(1);
      expect(fakes.contexts.length).toBe(contextsBefore);
      expect(statusText()).toBe("x: 10  y: 5  →  100 × 50 px");
      expect(second.result).toEqual(first.handle.result);

      // And the new callback receives commits, the old one does not
      sidebarButton("Reset crop to full image").click();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(first.onChange).not.toHaveBeenCalled();
    });

    it("builds a fresh editor when the previous one was removed from the document", () => {
      // Given an editor whose DOM the host cleared without calling cleanup
      mount({ value: baseValue() });
      const stale = editors()[0];
      stale?.remove();

      // When the loader mounts again
      mount({ value: baseValue() });

      // Then a new editor replaces it
      expect(editors()).toHaveLength(1);
      expect(editors()[0]).not.toBe(stale);
    });
  });

  describe("server value updates", () => {
    it("applies new crop coordinates, zoom, and rotation while idle", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ left: 20, top: 10, width: 60, height: 40, zoom: 150, rotate: -30 }) });
      expect(statusText()).toBe("x: 20  y: 10  →  60 × 40 px");
      const [zoom, rotate] = sliders();
      expect(zoom?.value).toBe("150");
      expect(rotate?.value).toBe("-30");
    });

    it("keeps the in-progress crop while a drag is under way", () => {
      const { handle, onChange } = mountLoaded();
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(110, 60));

      handle.update({ value: baseValue({ left: 0, top: 0 }), onChange });
      pointer(canvas(), "pointerup", ...at(110, 60));

      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 60, top: 35 }));
    });

    it("loads a new image when the URL changes, ignoring query-string-only changes", () => {
      const { handle } = mountLoaded();
      const img = fakes.lastImage();

      handle.update({ value: baseValue({ image_url: "https://example.test/image.png?v=2" }) });
      expect(img.src).toBe(IMAGE_URL);

      handle.update({ value: baseValue({ image_url: "https://example.test/other.png", img_width: 400, img_height: 200 }) });
      expect(img.src).toBe("https://example.test/other.png");
    });

    it("keeps the rest of the value when committing a crop", () => {
      const { onChange } = mountLoaded({ value: { ...baseValue(), extra: "kept" } });
      sidebarButton("Reset crop to full image").click();
      expect(onChange).toHaveBeenCalledWith({
        ...baseValue(),
        extra: "kept",
        left: 0,
        top: 0,
        width: 200,
        height: 100,
        zoom: 100,
        rotate: 0,
      });
    });
  });

  describe("callback replacement", () => {
    it("sends commits and live changes to the callbacks from the latest update only", () => {
      // Given an editor mounted with one pair of callbacks
      const { handle, onChange: oldChange, onLiveChange: oldLive } = mountLoaded();

      // When update hands it a new pair
      const onChange = vi.fn<(value: CropValue) => void>();
      const onLiveChange = vi.fn<(value: LiveCropValue) => void>();
      handle.update({ value: baseValue(), onChange, onLiveChange });
      drag(at(100, 50), at(110, 60));

      // Then only the new callbacks fire
      expect(onLiveChange).toHaveBeenCalled();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(oldChange).not.toHaveBeenCalled();
      expect(oldLive).not.toHaveBeenCalled();
    });

    it("tolerates updates with no callbacks", () => {
      const { handle, onChange } = mountLoaded();
      handle.update({ value: baseValue() });
      expect(() => drag(at(100, 50), at(110, 60))).not.toThrow();
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe("crop drag and resize", () => {
    it("moves the crop by dragging inside it, emitting live changes and committing on release", () => {
      const { onChange, onLiveChange } = mountLoaded();
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(110, 60));

      expect(onLiveChange).toHaveBeenLastCalledWith({ left: 60, top: 35, width: 100, height: 50, zoom: 100, rotate: 0 });
      expect(onChange).not.toHaveBeenCalled();

      pointer(canvas(), "pointerup", ...at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 60, top: 35, width: 100, height: 50 }));
    });

    it("keeps a moved crop inside the image", () => {
      const { onChange } = mountLoaded();
      drag(at(100, 50), at(190, 95));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 100, top: 50, width: 100, height: 50 }));
    });

    it("resizes from the south-east handle", () => {
      const { onChange } = mountLoaded();
      drag(at(150, 75), at(170, 90));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 50, top: 25, width: 120, height: 65 }));
    });

    it("resizes from the north-west handle", () => {
      const { onChange } = mountLoaded();
      drag(at(50, 25), at(40, 20));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 40, top: 20, width: 110, height: 55 }));
    });

    it("keeps the aspect ratio when resizing a corner with shift held", () => {
      const { onChange } = mountLoaded();
      drag(at(150, 75), at(170, 80), { shiftKey: true });
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 50, top: 25, width: 120, height: 60 }));
    });

    it("draws a new crop by dragging outside the current one", () => {
      const { onChange } = mountLoaded();
      drag(at(10, 10), at(40, 30));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 10, top: 10, width: 30, height: 20 }));
    });

    it("draws a square with shift held", () => {
      const { onChange } = mountLoaded();
      drag(at(10, 10), at(40, 20), { shiftKey: true });
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 10, top: 10, width: 30, height: 30 }));
    });

    it("sets a resize cursor over a handle and a move cursor inside the crop", () => {
      mountLoaded();
      pointer(canvas(), "pointermove", ...at(150, 75));
      expect(canvas().style.cursor).toBe("se-resize");
      pointer(canvas(), "pointermove", ...at(100, 50));
      expect(canvas().style.cursor).toBe("move");
      pointer(canvas(), "pointermove", ...at(10, 10));
      expect(canvas().style.cursor).toBe("crosshair");
    });

    it("abandons the drag without committing when the pointer is cancelled", () => {
      const { onChange } = mountLoaded();
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(110, 60));
      pointer(canvas(), "pointercancel", ...at(110, 60));
      pointer(canvas(), "pointerup", ...at(110, 60));
      expect(onChange).not.toHaveBeenCalled();
    });

    it("keeps pointer events on the canvas from reaching the node canvas behind it", () => {
      mountLoaded();
      const outer = vi.fn();
      container.addEventListener("pointerdown", outer);
      container.addEventListener("mousedown", outer);
      pointer(canvas(), "pointerdown", ...at(100, 50));
      canvas().dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
      expect(outer).not.toHaveBeenCalled();
    });

    it("ignores the pointer until an image has loaded", () => {
      const onChange = vi.fn<(value: CropValue) => void>();
      mount({ value: baseValue(), onChange });
      drag(at(100, 50), at(110, 60));
      expect(onChange).not.toHaveBeenCalled();
    });
  });

  describe("sidebar presets", () => {
    it("applies an aspect ratio around the current crop centre", () => {
      const { onChange } = mountLoaded();
      sidebarButton("1:1").click();
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 75, top: 25, width: 50, height: 50 }));
    });

    it("clamps a resolution preset to the image", () => {
      const { onChange } = mountLoaded();
      sidebarButton("854×480").click();
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 0, top: 0, width: 200, height: 100 }));
    });

    it("moves the crop to a named position", () => {
      const { onChange } = mountLoaded();
      sidebarButton("Bottom Right").click();
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 100, top: 50, width: 100, height: 50 }));
    });
  });

  describe("zoom and rotation controls", () => {
    it("previews a zoom while sliding and commits it on change", () => {
      const { onChange, onLiveChange } = mountLoaded();
      const [zoom] = sliders();
      if (!zoom) throw new Error("no zoom slider");
      fakes.lastContext().reset();

      zoom.value = "200";
      zoom.dispatchEvent(new Event("input", { bubbles: true }));
      expect(onLiveChange).toHaveBeenLastCalledWith(expect.objectContaining({ zoom: 200 }));
      expect(onChange).not.toHaveBeenCalled();
      expect(fakes.lastContext().count("setLineDash")).toBeGreaterThan(0);
      expect(zoom.parentElement?.textContent).toContain("200%");

      zoom.dispatchEvent(new Event("change", { bubbles: true }));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ zoom: 200, rotate: 0 }));
    });

    it("previews and commits a rotation, and draws the rotated capture area", () => {
      const { onChange } = mountLoaded();
      const [, rotate] = sliders();
      if (!rotate) throw new Error("no rotate slider");
      fakes.lastContext().reset();

      rotate.value = "-45";
      rotate.dispatchEvent(new Event("input", { bubbles: true }));
      rotate.dispatchEvent(new Event("change", { bubbles: true }));

      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ zoom: 100, rotate: -45 }));
      expect(fakes.lastContext().calls).toContainEqual({ method: "rotate", args: [(-45 * Math.PI) / 180] });
      expect(rotate.parentElement?.textContent).toContain("-45°");
    });

    it("resets a slider to its default with its reset button", () => {
      const { handle, onChange } = mountLoaded();
      handle.update({ value: baseValue({ zoom: 250 }), onChange });
      const zoomRow = sliders()[0]?.parentElement;
      const reset = zoomRow?.querySelector<HTMLButtonElement>("button");
      if (!reset) throw new Error("no reset button");
      expect(reset.title).toBe("Reset to 100%");

      reset.click();

      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ zoom: 100 }));
      expect(sliders()[0]?.value).toBe("100");
    });
  });

  describe("disabled state", () => {
    it("ignores the pointer, hides the handles, and disables every control", () => {
      // Given a loaded editor that is disabled
      const { onChange, onLiveChange } = mountLoaded({ disabled: true });
      const ctx = fakes.lastContext();
      ctx.reset();

      // When the user drags, slides, and clicks presets
      drag(at(100, 50), at(110, 60));
      const [zoom] = sliders();
      if (!zoom) throw new Error("no zoom slider");
      zoom.value = "200";
      zoom.dispatchEvent(new Event("input", { bubbles: true }));
      zoom.dispatchEvent(new Event("change", { bubbles: true }));
      sidebarButton("1:1").click();

      // Then nothing is emitted and the controls show as disabled
      expect(onChange).not.toHaveBeenCalled();
      expect(onLiveChange).not.toHaveBeenCalled();
      expect(sliders().every((s) => s.disabled)).toBe(true);
      expect(sidebarButton("1:1").disabled).toBe(true);
      expect(ctx.count("arc")).toBe(0);
    });

    it("re-enables everything when an update clears the flag", () => {
      const { handle, onChange } = mountLoaded({ disabled: true });
      handle.update({ value: baseValue(), onChange, disabled: false });
      fakes.lastContext().reset();
      handle.update({ value: baseValue(), onChange, disabled: false });

      expect(sliders().every((s) => !s.disabled)).toBe(true);
      expect(sidebarButton("1:1").disabled).toBe(false);
      expect(fakes.lastContext().count("arc")).toBe(8);
      drag(at(100, 50), at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });

  describe("locked parameters", () => {
    it("blocks moving and presets when a coordinate is driven by a connection", () => {
      // Given left is locked by an incoming connection
      const { onChange } = mountLoaded({ locked: ["left"] });

      // When the user drags inside the crop and clicks a preset
      drag(at(100, 50), at(110, 60));
      sidebarButton("1:1").click();

      // Then nothing commits, and the status names the connected parameter
      expect(onChange).not.toHaveBeenCalled();
      expect(sidebarButton("1:1").disabled).toBe(true);
      expect(statusText()).toBe("\u{1F512} Connected: left");
      pointer(canvas(), "pointermove", ...at(100, 50));
      expect(canvas().style.cursor).toBe("not-allowed");
    });

    it("still allows handles that do not touch a locked parameter", () => {
      const { onChange } = mountLoaded({ locked: ["left"] });
      drag(at(150, 75), at(170, 90));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ left: 50, width: 120, height: 65 }));
    });

    it("locks only the slider whose parameter is connected", () => {
      const { onChange } = mountLoaded({ locked: ["zoom"] });
      const [zoom, rotate] = sliders();
      expect(zoom?.disabled).toBe(true);
      expect(rotate?.disabled).toBe(false);

      if (!zoom) throw new Error("no zoom slider");
      zoom.value = "300";
      zoom.dispatchEvent(new Event("change", { bubbles: true }));
      expect(onChange).not.toHaveBeenCalled();
    });

    it("unlocks when an update drops the connection", () => {
      const { handle, onChange } = mountLoaded({ locked: ["left", "top"] });
      handle.update({ value: baseValue(), onChange, locked: [] });
      expect(sidebarButton("1:1").disabled).toBe(false);
      drag(at(100, 50), at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });

  describe("cleanup", () => {
    it("removes the editor DOM, stops observing resizes, and forgets the editor", () => {
      // Given a mounted editor observing its canvas area
      const handle = mount({ value: baseValue() });
      const ro = fakes.lastResizeObserver();
      expect(ro.observed).toHaveLength(1);

      // When the loader cleans it up
      handle.cleanup();

      // Then its DOM and resize observation are gone
      expect(editors()).toHaveLength(0);
      expect(ro.disconnected).toBe(true);
      expect((container as HTMLElement & { _cropEditor?: unknown })._cropEditor).toBeUndefined();
    });

    it("registers no listeners outside its own DOM", () => {
      const onWindow = vi.spyOn(window, "addEventListener");
      const onDocument = vi.spyOn(document, "addEventListener");
      const onContainer = vi.spyOn(container, "addEventListener");
      mountLoaded();
      expect(onWindow).not.toHaveBeenCalled();
      expect(onDocument).not.toHaveBeenCalled();
      expect(onContainer).not.toHaveBeenCalled();
    });

    it("re-renders on resize until cleaned up", () => {
      const { handle } = mountLoaded();
      const ro = fakes.lastResizeObserver();
      const ctx = fakes.lastContext();

      ctx.reset();
      ro.trigger();
      expect(ctx.count("drawImage")).toBe(1);

      handle.cleanup();
      ctx.reset();
      ro.trigger();
      expect(ctx.count("drawImage")).toBe(0);
    });

    it("mounts a fresh editor on the same container after cleanup", () => {
      // Given an editor that has been cleaned up
      const first = mount({ value: baseValue() });
      first.cleanup();

      // When the loader mounts again
      const onChange = vi.fn<(value: CropValue) => void>();
      const second = mount({ value: baseValue({ left: 0, top: 0 }), onChange });
      fakes.lastImage().load(200, 100);

      // Then it gets a new, working editor
      expect(second.cleanup).not.toBe(first.cleanup);
      expect(editors()).toHaveLength(1);
      expect(fakes.lastResizeObserver().disconnected).toBe(false);
      expect(statusText()).toBe("x: 0  y: 0  →  100 × 50 px");
      sidebarButton("Reset crop to full image").click();
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });
});
