import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { CropVideoEditorProps, CropVideoValue, LiveCropVideoValue } from "../src/CropVideoEditor/CropVideoEditor";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type DomFakes, installDomFakes, pointer } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<CropVideoEditorProps>;
type Props = Partial<CropVideoEditorProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/CropVideoEditor/CropVideoEditor")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["CropVideoEditor"]);
      bundleDirs.push(outDir);
      return importMount<CropVideoEditorProps>(join(outDir, "CropVideoEditor.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

// A 200×100 video in a container the DOM env reports as 0×0, so the widget falls back to a
// 480×360 area: scale 2.4 and a 480×240 canvas. Video (x, y) sits at canvas (2.4x, 2.4y).
const SCALE = 2.4;
const VIDEO_URL = "https://example.test/clip.mp4?token=1";

// The node's video details plus its synced crop settings: a 100×50 custom box at (50, 25).
function baseValue(overrides: Partial<CropVideoValue> = {}): CropVideoValue {
  return {
    video_url: VIDEO_URL,
    video_path: "/videos/clip.mp4",
    video_width: 200,
    video_height: 100,
    total_frames: 50,
    crop_size: "Custom",
    custom_width: 100,
    custom_height: 50,
    crop_position: "Custom",
    custom_left: 50,
    custom_top: 25,
    ...overrides,
  };
}

function at(ix: number, iy: number): [number, number] {
  return [ix * SCALE, iy * SCALE];
}

// happy-dom never decodes media: give the element its dimensions and fire the metadata event.
function loadMetadata(video: HTMLVideoElement, { width = 200, height = 100, duration = 2 } = {}): void {
  Object.defineProperty(video, "videoWidth", { configurable: true, value: width });
  Object.defineProperty(video, "videoHeight", { configurable: true, value: height });
  Object.defineProperty(video, "duration", { configurable: true, value: duration });
  video.dispatchEvent(new Event("loadedmetadata"));
}

describe.each(implementations)("CropVideoEditor (%s)", (_label, load) => {
  let CropVideoEditor: Mount;
  let fakes: DomFakes;
  let container: HTMLElement;
  let handles: MountedWidget<CropVideoEditorProps>[];

  beforeAll(async () => {
    CropVideoEditor = await load();
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
    const mounted = mountWidget(CropVideoEditor, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  function mountLoaded(props: Props = {}) {
    const onChange = vi.fn<(value: CropVideoValue) => void>();
    const onLiveChange = vi.fn<(value: LiveCropVideoValue) => void>();
    const handle = mount({ value: baseValue(), onChange, onLiveChange, ...props });
    loadMetadata(video());
    return { handle, onChange, onLiveChange };
  }

  const wrapper = () => container.firstElementChild as HTMLElement | null;
  const video = () => {
    const el = container.querySelector("video");
    if (!el) throw new Error("video not mounted");
    return el;
  };
  const canvas = () => {
    const el = container.querySelector("canvas");
    if (!el) throw new Error("canvas not mounted");
    return el;
  };
  const statusText = () => wrapper()?.querySelector(":scope > div:last-child > span")?.textContent ?? "";
  const scrubber = () => {
    const el = container.querySelector<HTMLInputElement>('input[type="range"]');
    if (!el) throw new Error("no frame scrubber");
    return el;
  };
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

  // What a drag commits: the incoming value with the box as custom fields and both presets on Custom.
  function committed(box: { l: number; t: number; w: number; h: number }, value: CropVideoValue = baseValue()) {
    return {
      ...value,
      custom_left: box.l,
      custom_top: box.t,
      custom_width: box.w,
      custom_height: box.h,
      crop_size: "Custom",
      crop_position: "Custom",
    };
  }

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the video, overlay canvas, sidebar, and scrubber", () => {
      const handle = mount({ value: baseValue() });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(container.children).toHaveLength(1);
      expect(wrapper()?.className).toBe("nodrag nowheel");
      expect(container.querySelectorAll("video")).toHaveLength(1);
      expect(container.querySelector("canvas")).not.toBeNull();
      expect(sidebarButton("Reset crop to full video frame")).toBeTruthy();
      expect(scrubber()).toBeTruthy();
      expect(statusText()).toBe("No video connected");
    });

    it("loads the video muted and anonymously, without one when the value has none", () => {
      mount({ value: baseValue() });
      expect(video().muted).toBe(true);
      expect(video().crossOrigin).toBe("anonymous");
      expect(video().src).toBe(VIDEO_URL);

      container.innerHTML = "";
      mount({ value: null });
      expect(video().hasAttribute("src")).toBe(false);
    });

    it("sizes the canvas to fit and shows the crop from the settings once metadata loads", () => {
      mountLoaded();
      expect(canvas().width).toBe(480);
      expect(canvas().height).toBe(240);
      expect(statusText()).toBe("x: 50  y: 25  →  100 × 50 px");
      expect(scrubber().disabled).toBe(false);
      expect(scrubber().max).toBe("49");
      // One render draws the eight handles
      const ctx = fakes.lastContext();
      ctx.reset();
      video().dispatchEvent(new Event("seeked"));
      expect(ctx.count("arc")).toBe(8);
    });

    it("takes the size from the video element when the value carries none, defaulting to the full frame", () => {
      mount({ value: { video_url: VIDEO_URL } });
      loadMetadata(video(), { width: 300, height: 150 });
      expect(statusText()).toBe("x: 0  y: 0  →  300 × 150 px");
    });
  });

  describe("crop settings", () => {
    it("works out the box from a size string and a named position on update", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ crop_size: "100x50", crop_position: "bottom-right" }) });
      expect(statusText()).toBe("x: 100  y: 50  →  100 × 50 px");
    });

    it("clamps a preset larger than the video to the frame", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ crop_size: "Square (1080x1080)", crop_position: "center" }) });
      expect(statusText()).toBe("x: 0  y: 0  →  200 × 100 px");
    });

    it("keeps the in-progress crop while a drag is under way", () => {
      const { handle, onChange } = mountLoaded();
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(110, 60));

      handle.update({ value: baseValue({ custom_left: 0, custom_top: 0 }), onChange });
      pointer(canvas(), "pointerup", ...at(110, 60));

      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ custom_left: 60, custom_top: 35 }));
    });

    it("keeps the video when only the URL's query changes, and loads a new one when its path does", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ video_url: "https://example.test/clip.mp4?token=2" }) });
      expect(video().src).toBe(VIDEO_URL);
      expect(statusText()).toBe("x: 50  y: 25  →  100 × 50 px");

      handle.update({ value: baseValue({ video_url: "https://example.test/other.mp4" }) });
      expect(video().src).toBe("https://example.test/other.mp4");
      expect(statusText()).toBe("No video connected");
    });
  });

  describe("crop drag and resize", () => {
    it("moves the crop by dragging inside it, emitting live changes and committing on release", () => {
      const { onChange, onLiveChange } = mountLoaded();
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(110, 60));

      expect(onLiveChange).toHaveBeenLastCalledWith({ custom_left: 60, custom_top: 35, custom_width: 100, custom_height: 50 });
      expect(onChange).not.toHaveBeenCalled();

      pointer(canvas(), "pointerup", ...at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(onChange).toHaveBeenLastCalledWith(committed({ l: 60, t: 35, w: 100, h: 50 }));

      // And a move past the edge stops at the frame
      drag(at(100, 50), at(190, 95));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ custom_left: 100, custom_top: 50 }));
    });

    it("resizes from the south-east handle", () => {
      const { onChange } = mountLoaded();
      drag(at(150, 75), at(170, 90));
      expect(onChange).toHaveBeenLastCalledWith(committed({ l: 50, t: 25, w: 120, h: 65 }));
    });

    it("keeps the aspect ratio when resizing a corner with shift held", () => {
      const { onChange } = mountLoaded();
      drag(at(150, 75), at(170, 80), { shiftKey: true });
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ custom_width: 120, custom_height: 60 }));
    });

    it("draws a new crop by dragging outside the current one", () => {
      const { onChange } = mountLoaded();
      drag(at(10, 10), at(40, 30));
      expect(onChange).toHaveBeenLastCalledWith(committed({ l: 10, t: 10, w: 30, h: 20 }));
    });

    it("sets a move cursor inside the crop and a crosshair elsewhere", () => {
      mountLoaded();
      pointer(canvas(), "pointermove", ...at(100, 50));
      expect(canvas().style.cursor).toBe("move");
      pointer(canvas(), "pointermove", ...at(150, 75));
      expect(canvas().style.cursor).toBe("crosshair");
      pointer(canvas(), "pointermove", ...at(10, 10));
      expect(canvas().style.cursor).toBe("crosshair");
    });

    it("keeps pointer events on the canvas from reaching the node canvas behind it", () => {
      mountLoaded();
      const outer = vi.fn();
      container.addEventListener("pointerdown", outer);
      container.addEventListener("pointermove", outer);
      pointer(canvas(), "pointerdown", ...at(100, 50));
      pointer(canvas(), "pointermove", ...at(100, 50));
      expect(outer).not.toHaveBeenCalled();
    });
  });

  describe("sidebar presets", () => {
    it("applies an aspect ratio around the current crop centre and commits it as a custom box", () => {
      const { onChange } = mountLoaded();
      sidebarButton("1:1").click();
      expect(onChange).toHaveBeenLastCalledWith(committed({ l: 75, t: 25, w: 50, h: 50 }));

      sidebarButton("Reset crop to full video frame").click();
      expect(onChange).toHaveBeenLastCalledWith(committed({ l: 0, t: 0, w: 200, h: 100 }));
    });
  });

  describe("frame scrubber", () => {
    it("seeks the video to the chosen frame without committing anything", () => {
      const { onChange } = mountLoaded();
      scrubber().value = "10";
      scrubber().dispatchEvent(new Event("input", { bubbles: true }));

      // Frame 10 of 50 in a 2 s video
      expect(video().currentTime).toBeCloseTo(0.4);
      expect(scrubber().parentElement?.textContent).toBe("10 / 50");
      expect(onChange).not.toHaveBeenCalled();

      // And the overlay is redrawn once the seek completes
      const ctx = fakes.lastContext();
      ctx.reset();
      video().dispatchEvent(new Event("seeked"));
      expect(ctx.count("strokeRect")).toBe(1);
    });
  });

  describe("disabled state", () => {
    it("ignores the pointer, the presets, and the scrubber", () => {
      // Given a loaded editor that is disabled
      const { onChange, onLiveChange } = mountLoaded({ disabled: true });

      // When the user drags, clicks a preset, and scrubs
      drag(at(100, 50), at(110, 60));
      sidebarButton("1:1").click();
      scrubber().value = "10";
      scrubber().dispatchEvent(new Event("input", { bubbles: true }));

      // Then nothing is emitted or seeked, and the controls show as disabled
      expect(onChange).not.toHaveBeenCalled();
      expect(onLiveChange).not.toHaveBeenCalled();
      expect(video().currentTime).toBe(0);
      expect(sidebarButton("1:1").disabled).toBe(true);
      expect(scrubber().disabled).toBe(true);
    });

    it("re-enables everything when an update clears the flag", () => {
      const { handle, onChange } = mountLoaded({ disabled: true });
      handle.update({ value: baseValue(), onChange, disabled: false });
      expect(sidebarButton("1:1").disabled).toBe(false);
      expect(scrubber().disabled).toBe(false);
      drag(at(100, 50), at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });

  describe("locked parameters", () => {
    it("blocks moving, drawing, and presets when the position is driven by a connection", () => {
      // Given crop_position is locked, which locks both the left and top edges
      const { onChange } = mountLoaded({ locked: ["crop_position"] });

      // When the user drags inside and outside the crop and clicks a preset
      drag(at(100, 50), at(110, 60));
      drag(at(10, 10), at(40, 30));
      sidebarButton("1:1").click();

      // Then nothing commits, and the status names the locked edges
      expect(onChange).not.toHaveBeenCalled();
      expect(sidebarButton("1:1").disabled).toBe(true);
      expect(statusText()).toBe("\u{1F512} Connected: left, top");
    });

    it("moves only along the axis that is still free", () => {
      const { onChange } = mountLoaded({ locked: ["custom_left"] });
      drag(at(100, 50), at(110, 60));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ custom_left: 50, custom_top: 35 }));
    });

    it("does not resize from a handle whose edge is locked", () => {
      const { onChange } = mountLoaded({ locked: ["crop_size"] });
      drag(at(150, 75), at(170, 90));
      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ custom_width: 100, custom_height: 50 }));
    });

    it("unlocks when an update drops the connection", () => {
      const { handle, onChange } = mountLoaded({ locked: ["crop_position"] });
      handle.update({ value: baseValue(), onChange, locked: [] });
      expect(sidebarButton("1:1").disabled).toBe(false);
      drag(at(100, 50), at(110, 60));
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });

  describe("callbacks and re-mounts", () => {
    it("sends commits and live changes to the callbacks from the latest update only", () => {
      const { handle, onChange: oldChange, onLiveChange: oldLive } = mountLoaded();
      const onChange = vi.fn<(value: CropVideoValue) => void>();
      const onLiveChange = vi.fn<(value: LiveCropVideoValue) => void>();
      handle.update({ value: baseValue(), onChange, onLiveChange });
      drag(at(100, 50), at(110, 60));

      expect(onLiveChange).toHaveBeenCalled();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(oldChange).not.toHaveBeenCalled();
      expect(oldLive).not.toHaveBeenCalled();
    });

    it("reuses the editor on a repeated mount of the same container", () => {
      // Given a mounted, loaded editor
      const first = mountLoaded();

      // When the loader mounts again with new props
      const onChange = vi.fn<(value: CropVideoValue) => void>();
      const second = mount({ value: baseValue({ custom_left: 10, custom_top: 5 }), onChange });

      // Then no second editor is built, the new value shows, and the new callback receives commits
      expect(container.children).toHaveLength(1);
      expect(container.querySelectorAll("video")).toHaveLength(1);
      expect(statusText()).toBe("x: 10  y: 5  →  100 × 50 px");
      expect(second.result).toEqual(first.handle.result);
      sidebarButton("Reset crop to full video frame").click();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(first.onChange).not.toHaveBeenCalled();
    });
  });

  describe("cleanup", () => {
    it("removes the editor DOM, releases the video, stops observing resizes, and forgets the editor", () => {
      // Given a mounted, loaded editor
      mountLoaded();
      const el = video();
      const pause = vi.spyOn(el, "pause");
      const ro = fakes.lastResizeObserver();
      const ctx = fakes.lastContext();

      // When the loader cleans it up
      handles[0]?.cleanup();

      // Then its DOM, video source, and resize observation are gone
      expect(container.children).toHaveLength(0);
      expect(pause).toHaveBeenCalled();
      expect(el.hasAttribute("src")).toBe(false);
      expect(ro.disconnected).toBe(true);
      expect((container as HTMLElement & { _instance?: unknown })._instance).toBeUndefined();
      ctx.reset();
      ro.trigger();
      expect(ctx.calls).toHaveLength(0);
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

    it("mounts a fresh editor on the same container after cleanup", () => {
      const first = mount({ value: baseValue() });
      first.cleanup();

      const onChange = vi.fn<(value: CropVideoValue) => void>();
      mount({ value: baseValue(), onChange });
      loadMetadata(video());

      expect(container.children).toHaveLength(1);
      expect(fakes.lastResizeObserver().disconnected).toBe(false);
      sidebarButton("1:1").click();
      expect(onChange).toHaveBeenCalledTimes(1);
    });
  });
});
