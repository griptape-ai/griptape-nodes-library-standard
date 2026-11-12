import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type {
  LiveMaskPreviewValue,
  MaskAdjustmentPreviewProps,
  MaskPreviewValue,
} from "../src/MaskAdjustmentPreview/MaskAdjustmentPreview";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<MaskAdjustmentPreviewProps>;
type Props = Partial<MaskAdjustmentPreviewProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/MaskAdjustmentPreview/MaskAdjustmentPreview")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["MaskAdjustmentPreview"]);
      bundleDirs.push(outDir);
      return importMount<MaskAdjustmentPreviewProps>(join(outDir, "MaskAdjustmentPreview.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

const ORIGINAL_URL = "https://example.test/original.mp4?t=1";
const MASK_URL = "https://example.test/mask.mp4?t=1";
const SIZE = 7;

// A SIZE×SIZE mask frame, white where `on(x, y)` holds.
function maskFrame(on: (x: number, y: number) => boolean): ImageData {
  const data = new Uint8ClampedArray(SIZE * SIZE * 4);
  for (let y = 0; y < SIZE; y++) {
    for (let x = 0; x < SIZE; x++) {
      const v = on(x, y) ? 255 : 0;
      const i = (y * SIZE + x) * 4;
      data[i] = data[i + 1] = data[i + 2] = v;
      data[i + 3] = 255;
    }
  }
  return new ImageData(data, SIZE, SIZE);
}
const centrePixel = (x: number, y: number) => x === 3 && y === 3;
const centreBlock = (x: number, y: number) => Math.abs(x - 3) <= 1 && Math.abs(y - 3) <= 1;

function count(data: ImageData, pred: (r: number, g: number, b: number, a: number) => boolean): number {
  let n = 0;
  for (let i = 0; i < data.data.length; i += 4) {
    if (pred(data.data[i] ?? 0, data.data[i + 1] ?? 0, data.data[i + 2] ?? 0, data.data[i + 3] ?? 0)) n++;
  }
  return n;
}
const yellow = (r: number, g: number, b: number, a: number) => r === 255 && g === 255 && b === 0 && a === 128;
const white = (r: number, g: number, b: number) => r === 255 && g === 255 && b === 255;

// A 2D context that hands out the current mask frame and records what is drawn and put.
class FakeContext {
  calls: string[] = [];
  puts: ImageData[] = [];
  constructor(private readonly frame: () => ImageData) {}
  clearRect() { this.calls.push("clearRect"); }
  drawImage() { this.calls.push("drawImage"); }
  getImageData() { this.calls.push("getImageData"); return this.frame(); }
  putImageData(data: ImageData) { this.calls.push("putImageData"); this.puts.push(data); }
}

// happy-dom never decodes media: give the element its dimensions and fire the event.
function loadMedia(video: HTMLVideoElement, event: "loadedmetadata" | "loadeddata", duration = 2): void {
  Object.defineProperty(video, "videoWidth", { configurable: true, value: SIZE });
  Object.defineProperty(video, "videoHeight", { configurable: true, value: SIZE });
  Object.defineProperty(video, "duration", { configurable: true, value: duration });
  video.dispatchEvent(new Event(event));
}

describe.each(implementations)("MaskAdjustmentPreview (%s)", (_label, load) => {
  let MaskAdjustmentPreview: Mount;
  let container: HTMLElement;
  let handles: MountedWidget<MaskAdjustmentPreviewProps>[];
  let frame: ImageData;
  let contexts: FakeContext[];
  let frames: FrameRequestCallback[];

  beforeAll(async () => {
    MaskAdjustmentPreview = await load();
  });

  beforeEach(() => {
    frame = maskFrame(centrePixel);
    contexts = [];
    frames = [];
    vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function () {
      const ctx = new FakeContext(() => frame);
      contexts.push(ctx);
      return ctx;
    } as unknown as HTMLCanvasElement["getContext"]);
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(() => {
    for (const h of handles) h.cleanup();
    container.remove();
    document.getElementById("mask-preview-slider-styles")?.remove();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props) {
    const mounted = mountWidget(MaskAdjustmentPreview, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  function baseValue(overrides: Partial<MaskPreviewValue> = {}): MaskPreviewValue {
    return { original_video_url: ORIGINAL_URL, mask_video_url: MASK_URL, current_frame: 0, total_frames: 100, adjustment: 0, ...overrides };
  }

  function flushFrames() {
    const pending = frames.splice(0);
    for (const cb of pending) cb(0);
  }

  // Mounts and loads both videos (or only the mask), then runs the render it schedules.
  function mountLoaded(props: Props = {}) {
    const onChange = vi.fn<(value: MaskPreviewValue) => void>();
    const onLiveChange = vi.fn<(value: LiveMaskPreviewValue) => void>();
    const handle = mount({ value: baseValue(), onChange, onLiveChange, ...props });
    if (original().hasAttribute("src")) loadMedia(original(), "loadedmetadata");
    loadMedia(mask(), "loadedmetadata");
    loadMedia(mask(), "loadeddata");
    flushFrames();
    return { handle, onChange, onLiveChange };
  }

  const videos = () => [...container.querySelectorAll("video")];
  const original = () => {
    const el = videos()[0];
    if (!el) throw new Error("no original video");
    return el;
  };
  const mask = () => {
    const el = videos()[1];
    if (!el) throw new Error("no mask video");
    return el;
  };
  const ranges = () => [...container.querySelectorAll<HTMLInputElement>('input[type="range"]')];
  const adjustmentSlider = () => {
    const el = ranges()[0];
    if (!el) throw new Error("no adjustment slider");
    return el;
  };
  const timeline = () => {
    const el = ranges()[1];
    if (!el) throw new Error("no timeline slider");
    return el;
  };
  const numberInput = () => {
    const el = container.querySelector<HTMLInputElement>('input[type="number"]');
    if (!el) throw new Error("no number input");
    return el;
  };
  const infoText = () => container.querySelector(".mask-adjustment-preview > div:last-child > span")?.textContent ?? "";
  // tmpCvs's context is created first, the overlay canvas's second.
  const tmpCtx = () => contexts[0] as FakeContext;
  const overlayCtx = () => contexts[1] as FakeContext;
  const lastPut = () => {
    const data = tmpCtx().puts.at(-1);
    if (!data) throw new Error("nothing was put");
    return data;
  };

  function slide(el: HTMLInputElement, value: string, ...events: ("input" | "change")[]) {
    el.value = value;
    for (const type of events) el.dispatchEvent(new Event(type, { bubbles: true }));
  }

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the preview, controls, and info bar", () => {
      const handle = mount({ value: baseValue() });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(container.querySelectorAll(".mask-adjustment-preview")).toHaveLength(1);
      expect(videos()).toHaveLength(2);
      expect(container.querySelector("canvas")).not.toBeNull();
      expect(ranges()).toHaveLength(2);
      expect(numberInput().value).toBe("0");
      expect(document.querySelectorAll("#mask-preview-slider-styles")).toHaveLength(1);
      expect(infoText()).toBe("Loading video... | Loading mask...");
    });

    it("loads both videos anonymously and shows the original with the mask hidden", () => {
      mount({ value: baseValue() });
      expect(original().src).toBe(ORIGINAL_URL);
      expect(mask().src).toBe(MASK_URL);
      expect(original().crossOrigin).toBe("anonymous");
      expect(mask().crossOrigin).toBe("anonymous");
      expect(original().style.display).toBe("block");
      expect(mask().style.display).toBe("none");
    });

    it("waits for inputs when the value has no videos", () => {
      mount({ value: null });
      expect(infoText()).toBe("Waiting for video inputs...");
      expect(videos().every((v) => !v.hasAttribute("src"))).toBe(true);
    });

    it("shows the mask itself when there is no original video", () => {
      mount({ value: baseValue({ original_video_url: "" }) });
      expect(original().style.display).toBe("none");
      expect(mask().style.display).toBe("block");
    });
  });

  describe("overlay rendering", () => {
    it("draws the mask as a yellow overlay once both videos have loaded", () => {
      mountLoaded();
      expect(count(lastPut(), yellow)).toBe(1);
      expect(overlayCtx().calls).toContain("drawImage");
      expect(infoText()).toBe("Video: OK | Mask: OK | Overlay active (adj: 0px)");
    });

    it("dilates the mask for a positive adjustment and erodes it for a negative one", () => {
      // Given a single white pixel, a +1 adjustment grows it to a 3×3 block
      const { handle } = mountLoaded({ value: baseValue({ adjustment: 1 }) });
      expect(count(lastPut(), yellow)).toBe(9);

      // And given a 3×3 block, a -1 adjustment shrinks it back to its centre
      frame = maskFrame(centreBlock);
      mask().dispatchEvent(new Event("seeked"));
      handle.update({ value: baseValue({ adjustment: -1 }) });
      flushFrames();
      expect(count(lastPut(), yellow)).toBe(1);
    });

    it("draws the adjusted black-and-white mask in mask-only mode, and nothing without an adjustment", () => {
      const { handle } = mountLoaded({ value: baseValue({ original_video_url: "", adjustment: 2 }) });
      expect(count(lastPut(), white)).toBe(25);
      expect(infoText()).toBe("Mask: OK | Adjusted preview (2px)");

      overlayCtx().calls = [];
      handle.update({ value: baseValue({ original_video_url: "", adjustment: 0 }) });
      flushFrames();
      expect(overlayCtx().calls).not.toContain("drawImage");
      expect(infoText()).toBe("Mask: OK | No adjustment");
    });

    it("coalesces several render requests into one animation frame", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ adjustment: 1 }) });
      slide(adjustmentSlider(), "2", "input");
      slide(adjustmentSlider(), "3", "input");
      expect(frames).toHaveLength(1);
    });
  });

  describe("adjustment controls", () => {
    it("previews on the node while sliding without committing", () => {
      const { onChange, onLiveChange } = mountLoaded();
      slide(adjustmentSlider(), "5", "input");
      flushFrames();

      expect(onLiveChange).toHaveBeenLastCalledWith({ adjustment: 5 });
      expect(onChange).not.toHaveBeenCalled();
      expect(numberInput().value).toBe("5");
      expect(infoText()).toBe("Video: OK | Mask: OK | Overlay active (adj: 5px)");
    });

    it("commits the adjustment with the videos and frame position on release", () => {
      const { onChange } = mountLoaded({ value: { ...baseValue(), extra: "dropped" } });
      slide(adjustmentSlider(), "-3", "input", "change");
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(onChange).toHaveBeenLastCalledWith({
        original_video_url: ORIGINAL_URL,
        mask_video_url: MASK_URL,
        adjustment: -3,
        current_frame: 0,
        total_frames: 60,
      });
    });

    it("commits a typed value on blur or Enter, clamped to the slider's range", () => {
      const { onChange } = mountLoaded();
      numberInput().focus();
      numberInput().value = "40";
      numberInput().dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", bubbles: true }));

      expect(onChange).toHaveBeenLastCalledWith(expect.objectContaining({ adjustment: 25 }));
      expect(adjustmentSlider().value).toBe("25");
      expect(numberInput().value).toBe("25");
    });

    it("restores the current value when the typed one is not a number", () => {
      const { onChange } = mountLoaded({ value: baseValue({ adjustment: 4 }) });
      numberInput().focus();
      numberInput().value = "";
      numberInput().blur();
      expect(onChange).not.toHaveBeenCalled();
      expect(numberInput().value).toBe("4");
    });

    it("applies an adjustment from an update to both controls", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ adjustment: 7 }) });
      expect(adjustmentSlider().value).toBe("7");
      expect(numberInput().value).toBe("7");
    });
  });

  describe("timeline", () => {
    it("seeks both videos to the chosen frame and waits for the seek before the next one", () => {
      // Given loaded 2 s videos at the default 30 fps: 60 frames
      mountLoaded();
      expect(timeline().max).toBe("59");

      // When the user scrubs to frame 30, then again before the seek completes
      slide(timeline(), "30", "input");
      slide(timeline(), "45", "input");

      // Then both videos sit at 1 s and the second scrub waited
      expect(original().currentTime).toBe(1);
      expect(mask().currentTime).toBe(1);
      expect(timeline().parentElement?.textContent).toContain("45 / 60");

      // And once the mask's seek completes, its new frame is captured and drawn
      const puts = tmpCtx().puts.length;
      mask().dispatchEvent(new Event("seeked"));
      flushFrames();
      expect(tmpCtx().puts.length).toBe(puts + 1);
      slide(timeline(), "15", "input");
      expect(mask().currentTime).toBe(0.5);
    });

    it("restores the saved frame once the videos load", () => {
      mountLoaded({ value: baseValue({ current_frame: 30 }) });
      expect(timeline().value).toBe("30");
      expect(original().currentTime).toBe(1);
    });
  });

  describe("disabled and locked", () => {
    it("disables both adjustment controls when the node is disabled", () => {
      mountLoaded({ disabled: true });
      expect(adjustmentSlider().disabled).toBe(true);
      expect(numberInput().disabled).toBe(true);
    });

    it("disables them while the adjustment parameter is connected, and re-enables when the connection goes", () => {
      const { handle } = mountLoaded({ locked: ["adjustment"] });
      expect(adjustmentSlider().disabled).toBe(true);
      expect(numberInput().disabled).toBe(true);

      handle.update({ value: baseValue(), locked: [] });
      expect(adjustmentSlider().disabled).toBe(false);
      expect(numberInput().disabled).toBe(false);
    });
  });

  describe("updates and re-mounts", () => {
    it("switches to mask-only mode and back as the original video comes and goes", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ original_video_url: "" }) });
      expect(original().style.display).toBe("none");
      expect(mask().style.display).toBe("block");

      handle.update({ value: baseValue() });
      expect(original().style.display).toBe("block");
      expect(mask().style.display).toBe("none");
    });

    it("keeps a video whose URL only changed its query, and loads one whose path changed", () => {
      const { handle } = mountLoaded();
      handle.update({ value: baseValue({ mask_video_url: "https://example.test/mask.mp4?t=2" }) });
      expect(mask().src).toBe(MASK_URL);

      handle.update({ value: baseValue({ mask_video_url: "https://example.test/other-mask.mp4" }) });
      expect(mask().src).toBe("https://example.test/other-mask.mp4");
      expect(infoText()).toBe("Video: OK | Loading mask...");
    });

    it("sends commits to the latest callback, and keeps the previous one when an update has none", () => {
      const { handle, onChange: first } = mountLoaded();
      const onChange = vi.fn<(value: MaskPreviewValue) => void>();
      handle.update({ value: baseValue(), onChange });
      slide(adjustmentSlider(), "1", "change");
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(first).not.toHaveBeenCalled();

      handle.update({ value: baseValue(), onChange: undefined });
      slide(adjustmentSlider(), "2", "change");
      expect(onChange).toHaveBeenCalledTimes(2);
    });

    it("reuses the preview on a repeated mount of the same container", () => {
      const first = mountLoaded();
      const second = mount({ value: baseValue({ adjustment: 3 }) });
      expect(container.querySelectorAll(".mask-adjustment-preview")).toHaveLength(1);
      expect(videos()).toHaveLength(2);
      expect(adjustmentSlider().value).toBe("3");
      expect(second.result).toEqual(first.handle.result);
    });

    it("keeps pointer events on the preview from reaching the node canvas behind it", () => {
      mount({ value: baseValue() });
      const outer = vi.fn();
      container.addEventListener("pointerdown", outer);
      container.addEventListener("mousedown", outer);
      const canvas = container.querySelector("canvas") as HTMLCanvasElement;
      canvas.dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
      canvas.dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
      expect(outer).not.toHaveBeenCalled();
    });

    it("reports a video that fails to load", () => {
      mount({ value: baseValue() });
      mask().dispatchEvent(new Event("error"));
      expect(infoText()).toBe("Mask video error: unknown");
    });
  });

  describe("cleanup", () => {
    it("releases both videos, removes the preview DOM, and forgets the preview", () => {
      // Given a mounted, loaded preview
      mountLoaded();
      const [orig, msk] = [original(), mask()];
      const pauses = [vi.spyOn(orig, "pause"), vi.spyOn(msk, "pause")];

      // When the loader cleans it up
      handles[0]?.cleanup();

      // Then the videos are stopped and emptied and the DOM is gone
      expect(pauses.every((p) => p.mock.calls.length === 1)).toBe(true);
      expect(orig.hasAttribute("src")).toBe(false);
      expect(msk.hasAttribute("src")).toBe(false);
      expect(container.children).toHaveLength(0);
      expect((container as HTMLElement & { _maskPreview?: unknown })._maskPreview).toBeUndefined();
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

    it("builds a fresh preview on the same container after cleanup", () => {
      const first = mount({ value: baseValue() });
      first.cleanup();
      mount({ value: baseValue({ adjustment: 2 }) });
      expect(container.querySelectorAll(".mask-adjustment-preview")).toHaveLength(1);
      expect(adjustmentSlider().value).toBe("2");
    });
  });
});
