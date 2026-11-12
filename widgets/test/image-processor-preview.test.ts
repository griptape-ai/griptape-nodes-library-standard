import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { ImageProcessorPreviewProps, PreviewValue } from "../src/ImageProcessorPreview/ImageProcessorPreview";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type DomFakes, FakeImage, installDomFakes } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<ImageProcessorPreviewProps>;
type Props = Partial<ImageProcessorPreviewProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/ImageProcessorPreview/ImageProcessorPreview")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["ImageProcessorPreview"]);
      bundleDirs.push(outDir);
      return importMount<ImageProcessorPreviewProps>(join(outDir, "ImageProcessorPreview.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

const IMAGE_URL = "https://example.test/image.png?v=1";
const VIDEO_URL = "https://example.test/clip.mp4";

// ── A 2D context the preview can read pixels back from ───────────────────────

interface Call {
  method: string;
  args: unknown[];
}

// Every canvas's context records its calls. getImageData hands back the source colour on every pixel,
// or throws the way a tainted canvas does when `blocked` is set; measureText returns fixed metrics.
class PixelCanvases {
  calls = new Map<HTMLCanvasElement, Call[]>();
  source: [number, number, number, number] = [255, 0, 0, 255];
  blocked = false;

  install(): void {
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    vi.mocked(HTMLCanvasElement.prototype.getContext).mockImplementation(function (this: HTMLCanvasElement) {
      return self.context(this);
    } as unknown as HTMLCanvasElement["getContext"]);
  }

  private context(canvas: HTMLCanvasElement): CanvasRenderingContext2D {
    const calls: Call[] = [];
    this.calls.set(canvas, calls);
    const props = new Map<PropertyKey, unknown>();
    const proxy = new Proxy(
      {},
      {
        get: (_target, prop) => {
          if (prop === "canvas") return canvas;
          if (props.has(prop)) return props.get(prop);
          return (...args: unknown[]) => {
            calls.push({ method: String(prop), args });
            if (prop === "getImageData") return this.pixels(Number(args[2]), Number(args[3]));
            if (prop === "measureText") {
              const width = String(args[0]).length * 10;
              return {
                width,
                actualBoundingBoxLeft: 0,
                actualBoundingBoxRight: width,
                actualBoundingBoxAscent: 10,
                actualBoundingBoxDescent: 2,
                fontBoundingBoxAscent: 12,
              };
            }
            return undefined;
          };
        },
        set: (_target, prop, value) => {
          props.set(prop, value);
          return true;
        },
      },
    );
    return proxy as unknown as CanvasRenderingContext2D;
  }

  private pixels(width: number, height: number): ImageData {
    if (this.blocked) throw new Error("The canvas has been tainted by cross-origin data.");
    const data = new Uint8ClampedArray(width * height * 4);
    for (let i = 0; i < data.length; i += 4) data.set(this.source, i);
    return new ImageData(data, width, height);
  }

  of(canvas: HTMLCanvasElement): Call[] {
    const calls = this.calls.get(canvas);
    if (!calls) throw new Error("canvas has no context");
    return calls;
  }

  all(method: string): Call[] {
    return [...this.calls.values()].flat().filter((c) => c.method === method);
  }

  // The last ImageData put on any canvas: the operation's result.
  lastPut(): ImageData {
    const call = this.all("putImageData").at(-1);
    if (!call) throw new Error("nothing was put");
    return call.args[0] as ImageData;
  }
}

class FakeFontFace {
  static instances: FakeFontFace[] = [];
  constructor(
    readonly family: string,
    readonly source: string,
  ) {
    FakeFontFace.instances.push(this);
  }
  load(): Promise<FakeFontFace> {
    return Promise.resolve(this);
  }
}

describe.each(implementations)("ImageProcessorPreview (%s)", (_label, load) => {
  let ImageProcessorPreview: Mount;
  let fakes: DomFakes;
  let canvases: PixelCanvases;
  let videos: HTMLVideoElement[];
  let container: HTMLElement;
  let handles: MountedWidget<ImageProcessorPreviewProps>[];

  beforeAll(async () => {
    ImageProcessorPreview = await load();
  });

  beforeEach(() => {
    fakes = installDomFakes();
    canvases = new PixelCanvases();
    canvases.install();
    // The preview's video element never joins the document, so catch it as it's made.
    videos = [];
    const createElement = document.createElement.bind(document);
    vi.spyOn(document, "createElement").mockImplementation(((tag: string, options?: ElementCreationOptions) => {
      const el = createElement(tag, options);
      if (tag === "video") videos.push(el as HTMLVideoElement);
      return el;
    }) as typeof document.createElement);
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
    const mounted = mountWidget(ImageProcessorPreview, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  // Mounts an image operation and loads its 200×100 source. The DOM env reports the canvas area as
  // 0×0, so the preview fits into 480×360.
  function mountLoaded(value: PreviewValue, props: Props = {}) {
    const onChange = vi.fn<(value: PreviewValue) => void>();
    const onLiveChange = vi.fn<(values: Record<string, never>) => void>();
    const handle = mount({ value: { media: "image", image_url: IMAGE_URL, ...value }, onChange, onLiveChange, ...props });
    const img = fakes.lastImage();
    img.load(200, 100);
    return { handle, img, onChange, onLiveChange };
  }

  const canvas = () => {
    const el = container.querySelector("canvas");
    if (!el) throw new Error("canvas not mounted");
    return el;
  };
  const texts = () => canvases.of(canvas()).filter((c) => c.method === "fillText").map((c) => String(c.args[0]));
  const previews = () => container.querySelectorAll(".image-processor-preview");
  const statusText = () => container.querySelector(".image-processor-preview > div:last-child > span")?.textContent ?? "";
  const scrubber = () => {
    const el = container.querySelector<HTMLInputElement>('input[type="range"]');
    if (!el) throw new Error("no scrubber");
    return el;
  };
  const lastVideo = () => {
    const video = videos.at(-1);
    if (!video) throw new Error("no video was created");
    return video;
  };

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the preview inside the container", () => {
      const handle = mount({ value: { operation: "flip", media: "image" } });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(previews()).toHaveLength(1);
      expect(container.querySelectorAll("canvas")).toHaveLength(1);
      expect(scrubber().max).toBe("1000");
      expect(scrubber().parentElement?.style.display).toBe("none");
      expect(container.textContent).toContain("v1.2.0");
    });

    it("says so when the node's operation has no preview", () => {
      mount({ value: { operation: "no_such_operation" } });
      expect(texts()).toContain("No preview for this node");
      expect(statusText()).toBe("");
    });

    it("asks for an image, or a video for a video node, until one is connected", () => {
      // Given an image node with nothing connected
      const handle = mount({ value: { operation: "flip", media: "image" } });
      expect(texts()).toContain("Connect an image to preview the result");
      expect(statusText()).toBe("No image connected");

      // When it becomes a video node
      handle.update({ value: { operation: "video_eq", media: "video" } });

      // Then it asks for a video and shows the scrubber
      expect(texts()).toContain("Connect a video to preview the result");
      expect(statusText()).toBe("No video connected");
      expect(scrubber().parentElement?.style.display).toBe("flex");
    });
  });

  describe("image sources", () => {
    it("loads the image anonymously and draws the operation fitted into view once it loads", () => {
      mount({ value: { operation: "flip", media: "image", image_url: IMAGE_URL } });
      const img = fakes.lastImage();
      expect(img.crossOrigin).toBe("anonymous");
      expect(img.src).toBe(IMAGE_URL);

      img.load(200, 100);

      expect(canvas().width).toBe(480);
      expect(canvas().height).toBe(240);
      expect(statusText()).toBe("horizontal  →  200 × 100 px");
      expect(canvases.of(canvas())).toContainEqual({ method: "scale", args: [-1, 1] });
      expect(canvases.of(canvas())).toContainEqual({ method: "drawImage", args: [img, 0, 0, 480, 240] });
    });

    it("sizes the view to the operation's output size", () => {
      mountLoaded({ operation: "rotate", rotation: "90° clockwise" });
      expect(canvas().width).toBe(180);
      expect(canvas().height).toBe(360);
      expect(statusText()).toBe("90° clockwise  →  100 × 200 px");
    });

    it("shows the operation's size error instead of a preview", () => {
      mountLoaded({ operation: "resize", resize_mode: "aspect ratio", aspect_ratio: "abc" });
      expect(texts()).toContain('"abc" isn\'t an aspect ratio');
      expect(statusText()).toBe('"abc" isn\'t an aspect ratio');
    });

    it("runs a colour operation on the scaled source's pixels", () => {
      // Given a pure red source
      canvases.source = [255, 0, 0, 255];

      // When it is previewed in grayscale
      mountLoaded({ operation: "grayscale" });

      // Then the result is Pillow's luma of red, drawn over the checkerboard at the view size
      const result = canvases.lastPut();
      expect([result.width, result.height]).toEqual([480, 240]);
      expect([...result.data.slice(0, 4)]).toEqual([76, 76, 76, 255]);
      expect(statusText()).toBe("200 × 100 px");
    });

    it("reads the source's pixels once while settings change", () => {
      const { handle } = mountLoaded({ operation: "levels", shadows: 0 });
      handle.update({ value: { operation: "levels", media: "image", image_url: IMAGE_URL, shadows: 100 } });
      handle.update({ value: { operation: "levels", media: "image", image_url: IMAGE_URL, shadows: 200 } });
      expect(canvases.all("getImageData")).toHaveLength(1);
      expect(canvases.lastPut().data[0]).toBe(255);
    });

    it("draws the plain source and says so when the browser won't hand over its pixels", () => {
      canvases.blocked = true;
      const { img } = mountLoaded({ operation: "grayscale" });
      expect(canvases.all("putImageData")).toHaveLength(0);
      expect(canvases.of(canvas())).toContainEqual({ method: "drawImage", args: [img, 0, 0, 480, 240] });
      expect(statusText()).toBe("200 × 100 px  (preview unavailable for this source)");
    });

    it("shows the message an operation throws", () => {
      mountLoaded({ operation: "display_channel", image_mode: "P" });
      expect(texts()).toContain("The node can't read P images");
      expect(statusText()).toBe("The node can't read P images");
    });

    it("asks for an image again when the image fails to load", () => {
      const { img } = mountLoaded({ operation: "flip" });
      img.fail();
      expect(statusText()).toBe("No image connected");
    });
  });

  describe("generated operations", () => {
    it("generates the image at full size and scales it into view", () => {
      mount({ value: { operation: "color_bars", media: "image", width: 64, height: 32 } });
      const full = canvases.lastPut();
      expect([full.width, full.height]).toEqual([64, 32]);
      expect([...full.data.slice(0, 4)]).toEqual([255, 255, 255, 255]);
      expect(canvas().width).toBe(480);
      expect(canvas().height).toBe(240);
      expect(statusText()).toBe("SMPTE 219-100 Bars  →  64 × 32 px");
    });

    it("shows the message a generated operation throws", () => {
      mount({ value: { operation: "text_image", media: "image", background_color: "not a colour" } });
      expect(texts()).toContain("A colour isn't one the node can read");
    });
  });

  describe("multi-image operations", () => {
    function loadMask(img: FakeImage, width: number, height: number) {
      (img as FakeImage & { complete: boolean }).complete = true;
      img.load(width, height);
    }
    const masks = (n: number) => FakeImage.instances.slice(-n);

    it("asks for masks until some are connected", () => {
      mount({ value: { operation: "combine_masks", images: [] } });
      expect(texts()).toContain("Connect masks to preview the result");
      expect(statusText()).toBe("No masks connected");
    });

    it("combines the masks once every one has loaded", () => {
      // Given two masks, one more that can't be previewed
      mount({
        value: {
          operation: "combine_masks",
          images: [{ url: "https://example.test/a.png", mode: "L" }, { url: "https://example.test/b.png", mode: "L" }],
          skipped: 1,
        },
      });
      const [a, b] = masks(2);
      if (!a || !b) throw new Error("masks not loaded");
      expect(a.src).toBe("https://example.test/a.png");
      expect(statusText()).toBe("Loading");

      // When both load
      loadMask(a, 100, 50);
      loadMask(b, 100, 50);

      // Then the combined mask is drawn, with a note about the skipped one
      expect(statusText()).toBe("2 masks  →  100 × 50 px  (1 more can't be previewed)");
      const out = canvases.lastPut();
      expect([out.width, out.height]).toEqual([480, 240]);
      expect([...out.data.slice(0, 4)]).toEqual([255, 255, 255, 255]);
    });

    it("refuses masks of different sizes", () => {
      mount({ value: { operation: "combine_masks", images: [{ url: "https://example.test/a.png" }, { url: "https://example.test/b.png" }] } });
      const [a, b] = masks(2);
      if (!a || !b) throw new Error("masks not loaded");
      loadMask(a, 100, 50);
      loadMask(b, 50, 50);
      expect(statusText()).toBe("The masks aren't all the same size");
    });
  });

  describe("updates", () => {
    it("redraws with new settings, keeping the image when only its query string changes", () => {
      // Given a loaded flip preview
      const { handle, img } = mountLoaded({ operation: "flip" });

      // When the direction changes and the URL only gains a new query string
      handle.update({ value: { operation: "flip", media: "image", image_url: "https://example.test/image.png?v=2", direction: "vertical" } });

      // Then the same image is redrawn with the new settings
      expect(img.src).toBe(IMAGE_URL);
      expect(statusText()).toBe("vertical  →  200 × 100 px");
    });

    it("loads a new image when the URL changes, and waits for it", () => {
      const { handle, img } = mountLoaded({ operation: "flip" });
      handle.update({ value: { operation: "flip", media: "image", image_url: "https://example.test/other.png" } });
      expect(img.src).toBe("https://example.test/other.png");
      expect(statusText()).toBe("No image connected");
    });
  });

  describe("video sources", () => {
    it("loads the video, shows the frame the scrubber picks, and seeks as it moves", () => {
      // Given a video node whose 10 s, 320×160 video has loaded
      mount({ value: { operation: "video_eq", media: "video", video_url: VIDEO_URL } });
      const video = lastVideo();
      expect(video.src).toBe(VIDEO_URL);
      expect(video.crossOrigin).toBe("anonymous");
      expect(video.muted).toBe(true);
      Object.defineProperty(video, "duration", { configurable: true, value: 10 });
      Object.defineProperty(video, "videoWidth", { configurable: true, value: 320 });
      Object.defineProperty(video, "videoHeight", { configurable: true, value: 160 });
      video.dispatchEvent(new Event("loadeddata"));
      video.dispatchEvent(new Event("seeked"));
      expect(statusText()).toBe("320 × 160 px");

      // When the scrubber moves to the middle
      scrubber().value = "500";
      scrubber().dispatchEvent(new Event("input", { bubbles: true }));

      // Then the video seeks there and the time shows
      expect(video.currentTime).toBe(5);
      expect(scrubber().parentElement?.textContent).toBe("0:05.0 / 0:10.0");
    });

    it("keeps scrubber drags from reaching the node canvas behind it", () => {
      mount({ value: { operation: "video_eq", media: "video", video_url: VIDEO_URL } });
      const outer = vi.fn();
      container.addEventListener("pointerdown", outer);
      container.addEventListener("mousedown", outer);
      scrubber().dispatchEvent(new PointerEvent("pointerdown", { bubbles: true }));
      scrubber().dispatchEvent(new MouseEvent("mousedown", { bubbles: true }));
      expect(outer).not.toHaveBeenCalled();
    });
  });

  describe("fonts", () => {
    it("loads the node's font once and redraws when it arrives", async () => {
      // Given a browser with FontFace
      FakeFontFace.instances = [];
      vi.stubGlobal("FontFace", FakeFontFace);
      const added: unknown[] = [];
      Object.defineProperty(document, "fonts", { configurable: true, value: { add: (face: unknown) => added.push(face) } });

      // When a text preview names its font, twice
      const value = { operation: "text_image", media: "image" as const, font_url: "http://localhost/font.ttf" };
      const handle = mount({ value });
      handle.update({ value });
      const rendersBefore = canvases.all("fillRect").length;
      await Promise.resolve();
      await Promise.resolve();

      // Then the font is loaded once, added to the document, and the preview redrawn
      expect(FakeFontFace.instances).toHaveLength(1);
      expect(FakeFontFace.instances[0]?.family).toBe("GriptapePillowDefault");
      expect(FakeFontFace.instances[0]?.source).toBe('url("http://localhost/font.ttf")');
      expect(added).toEqual([FakeFontFace.instances[0]]);
      expect(canvases.all("fillRect").length).toBeGreaterThan(rendersBefore);
      delete (document as { fonts?: unknown }).fonts;
    });
  });

  describe("commits", () => {
    it("never commits or shows live values, disabled or not", () => {
      for (const disabled of [false, true]) {
        // Given a preview, mounted on a fresh container
        container.replaceChildren();
        const { onChange, onLiveChange, handle } = mountLoaded({ operation: "flip" }, { disabled });

        // When settings change and the scrubber moves
        handle.update({ value: { operation: "flip", media: "image", image_url: IMAGE_URL, direction: "both" }, disabled });
        scrubber().dispatchEvent(new Event("input", { bubbles: true }));

        // Then it only draws
        expect(statusText()).toBe("both  →  200 × 100 px");
        expect(onChange).not.toHaveBeenCalled();
        expect(onLiveChange).not.toHaveBeenCalled();
        handle.cleanup();
      }
    });
  });

  describe("repeated mount on the same container", () => {
    it("reuses the preview and applies the new props", () => {
      // Given a mounted, loaded preview
      const first = mountLoaded({ operation: "flip" });

      // When the loader mounts again on the same container with new props
      const second = mount({ value: { operation: "flip", media: "image", image_url: IMAGE_URL, direction: "vertical" } });

      // Then no second preview is built and the new value is shown
      expect(previews()).toHaveLength(1);
      expect(statusText()).toBe("vertical  →  200 × 100 px");
      expect(second.result).toEqual(first.handle.result);
    });

    it("builds a fresh preview when the previous one was removed from the document", () => {
      mount({ value: { operation: "flip" } });
      const stale = previews()[0];
      stale?.remove();
      mount({ value: { operation: "flip" } });
      expect(previews()).toHaveLength(1);
      expect(previews()[0]).not.toBe(stale);
    });
  });

  describe("cleanup", () => {
    it("removes the preview, stops observing resizes, releases the video, and forgets the preview", () => {
      // Given a mounted video preview
      const handle = mount({ value: { operation: "video_eq", media: "video", video_url: VIDEO_URL } });
      const ro = fakes.lastResizeObserver();
      const video = lastVideo();
      const pause = vi.spyOn(video, "pause");
      expect(ro.observed).toHaveLength(1);

      // When the loader cleans it up
      handle.cleanup();

      // Then its DOM, resize observation, and media player are gone
      expect(previews()).toHaveLength(0);
      expect(ro.disconnected).toBe(true);
      expect(pause).toHaveBeenCalled();
      expect(video.hasAttribute("src")).toBe(false);
      expect((container as HTMLElement & { _imagePreview?: unknown })._imagePreview).toBeUndefined();
    });

    it("registers no listeners outside its own DOM", () => {
      const onWindow = vi.spyOn(window, "addEventListener");
      const onDocument = vi.spyOn(document, "addEventListener");
      const onContainer = vi.spyOn(container, "addEventListener");
      mountLoaded({ operation: "flip" });
      expect(onWindow).not.toHaveBeenCalled();
      expect(onDocument).not.toHaveBeenCalled();
      expect(onContainer).not.toHaveBeenCalled();
    });

    it("re-renders on resize until cleaned up", () => {
      const { handle } = mountLoaded({ operation: "flip" });
      const ro = fakes.lastResizeObserver();
      const draws = () => canvases.all("drawImage").length;

      const before = draws();
      ro.trigger();
      expect(draws()).toBe(before + 1);

      handle.cleanup();
      ro.trigger();
      expect(draws()).toBe(before + 1);
    });

    it("mounts a fresh, working preview on the same container after cleanup", () => {
      const first = mount({ value: { operation: "flip" } });
      first.cleanup();

      const second = mountLoaded({ operation: "flip", direction: "vertical" });

      expect(second.handle.cleanup).not.toBe(first.cleanup);
      expect(previews()).toHaveLength(1);
      expect(fakes.lastResizeObserver().disconnected).toBe(false);
      expect(statusText()).toBe("vertical  →  200 × 100 px");
    });
  });
});
