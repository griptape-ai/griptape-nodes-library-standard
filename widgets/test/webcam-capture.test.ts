import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { GalleryItem, WebcamCaptureProps, WebcamCaptureValue } from "../src/WebcamCapture/WebcamCapture";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type DomFakes, installDomFakes } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<WebcamCaptureProps>;
type Props = Partial<WebcamCaptureProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/WebcamCapture/WebcamCapture")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["WebcamCapture"]);
      bundleDirs.push(outDir);
      return importMount<WebcamCaptureProps>(join(outDir, "WebcamCapture.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

// ── Media fakes: happy-dom has no camera ────────────────────────────────────────

class FakeTrack {
  stopped = false;
  constructor(readonly deviceId: string) {}
  stop(): void {
    this.stopped = true;
  }
  getSettings(): MediaTrackSettings {
    return { deviceId: this.deviceId };
  }
}

class FakeStream {
  constructor(readonly tracks: FakeTrack[]) {}
  getTracks(): FakeTrack[] {
    return this.tracks;
  }
  getVideoTracks(): FakeTrack[] {
    return this.tracks;
  }
}

const DEVICES = [
  { kind: "videoinput", deviceId: "cam-1", label: "Front camera" },
  { kind: "videoinput", deviceId: "cam-2", label: "" },
  { kind: "audioinput", deviceId: "mic-1", label: "Built-in mic" },
];

type MediaRequest = { video: MediaTrackConstraints; audio: unknown };

// navigator.mediaDevices whose getUserMedia answers from a per-test handler.
class FakeMediaDevices extends EventTarget {
  requests: MediaRequest[] = [];
  streams: FakeStream[] = [];
  handler: (req: MediaRequest) => Promise<FakeStream> = (req) => {
    const exact = (req.video.deviceId as { exact?: string } | undefined)?.exact;
    const s = new FakeStream([new FakeTrack(exact ?? "cam-1")]);
    this.streams.push(s);
    return Promise.resolve(s);
  };

  getUserMedia = vi.fn((req: MediaRequest) => {
    this.requests.push(req);
    return this.handler(req);
  });
  enumerateDevices = vi.fn(async () => DEVICES);
}

// Lets the widget's async chains (getUserMedia → enumerateDevices, fetch → then) settle.
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

const JPEG_BYTES = "JPEGDATA";
const DATA_URL = `data:image/jpeg;base64,${btoa(JPEG_BYTES)}`;
const uploadUrl = (seq: number) => `http://localhost:8124/static-uploads/temp/_wc_node_${seq}.jpg`;
const item = (n: number): GalleryItem => ({ url: `https://example.test/${n}.jpg`, _path: `webcam/${n}.jpg` });

function processed(seq: number, items: GalleryItem[]): WebcamCaptureValue {
  return {
    state: "processed",
    gallery_items: items,
    selected_index: items.length - 1,
    gallery_count: items.length,
    _emitSeq: seq,
  };
}

describe.each(implementations)("WebcamCapture (%s)", (_label, load) => {
  let WebcamCapture: Mount;
  let fakes: DomFakes;
  let container: HTMLElement;
  let handles: MountedWidget<WebcamCaptureProps>[];
  let media: FakeMediaDevices;
  let fetchMock: ReturnType<typeof vi.fn<(url: string, init: RequestInit) => Promise<{ ok: boolean; statusText: string }>>>;
  let srcObjectDescriptor: PropertyDescriptor | undefined;

  beforeAll(async () => {
    WebcamCapture = await load();
  });

  beforeEach(() => {
    fakes = installDomFakes();
    vi.spyOn(HTMLCanvasElement.prototype, "toDataURL").mockReturnValue(DATA_URL);
    media = new FakeMediaDevices();
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: media });
    fetchMock = vi.fn(async () => ({ ok: true, statusText: "OK" }));
    vi.stubGlobal("fetch", fetchMock);
    // happy-dom only accepts its own MediaStream here; keep whatever the widget assigns.
    srcObjectDescriptor = Object.getOwnPropertyDescriptor(HTMLMediaElement.prototype, "srcObject");
    const srcObjects = new WeakMap<object, unknown>();
    Object.defineProperty(HTMLMediaElement.prototype, "srcObject", {
      configurable: true,
      get() { return srcObjects.get(this) ?? null; },
      set(v: unknown) { srcObjects.set(this, v); },
    });

    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(() => {
    for (const h of handles) h.cleanup();
    container.remove();
    if (srcObjectDescriptor) Object.defineProperty(HTMLMediaElement.prototype, "srcObject", srcObjectDescriptor);
    delete (navigator as { mediaDevices?: unknown }).mediaDevices;
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props = {}) {
    const mounted = mountWidget(WebcamCapture, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  async function mountStreaming(props: Props = {}) {
    const onChange = vi.fn<(value: WebcamCaptureValue) => void>();
    const handle = mount({ onChange, ...props });
    await flush();
    // Every later update carries the same callback unless a test says otherwise.
    const update = (value: WebcamCaptureValue) => handle.update({ value, onChange });
    return { handle, onChange, update };
  }

  const wrappers = () => container.querySelectorAll(":scope > div");
  const video = () => container.querySelector("video")!;
  const placeholder = () => video().parentElement!.children[0] as HTMLDivElement;
  const shutterBtn = () => video().parentElement!.querySelector("button")!;
  const thumbStrip = () => wrappers()[0]!.children[1] as HTMLDivElement;
  const thumbs = () => [...container.querySelectorAll<HTMLImageElement>("img.wc-thumb")];
  const pendingThumbs = () => [...thumbStrip().querySelectorAll(":scope > div")];
  const clearBtn = () => [...container.querySelectorAll("button")].find((b) => b.textContent?.startsWith("Clear"))!;
  const videoSel = () => container.querySelector("select")!;

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the camera view inside the container", () => {
      const handle = mount();
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(wrappers()).toHaveLength(1);
      expect(wrappers()[0]?.className).toBe("nodrag nowheel");
      expect(placeholder().textContent).toBe("Starting camera…");
      expect(shutterBtn().disabled).toBe(true);
      expect(thumbStrip().hidden).toBe(true);
      expect(clearBtn().hidden).toBe(true);
      expect(document.getElementById("wc-styles")).not.toBeNull();
    });

    it("opens the camera at up to 4K without audio and lists the cameras", async () => {
      // Given a fresh widget
      // When the camera opens
      await mountStreaming();

      // Then the camera was asked for its best resolution, and streams
      expect(media.requests).toEqual([{ video: { width: { ideal: 4096 }, height: { ideal: 2160 } }, audio: false }]);
      expect(video().srcObject).toBe(media.streams[0]);
      expect(placeholder().hidden).toBe(true);
      expect(shutterBtn().disabled).toBe(false);

      // And only cameras are listed, unnamed ones numbered, with the open one selected
      expect([...videoSel().options].map((o) => o.textContent)).toEqual(["Front camera", "Camera 2"]);
      expect(videoSel().value).toBe("cam-1");
    });

    it("says so when the camera can't be opened", async () => {
      media.handler = () => Promise.reject(new DOMException("denied", "NotAllowedError"));
      await mountStreaming();
      expect(placeholder().textContent).toBe("Camera unavailable — check browser permissions.");
      expect(shutterBtn().disabled).toBe(true);
    });

    it("shows the gallery from the value with its selection", () => {
      mount({ value: { state: "idle", gallery_items: [item(1), item(2)], selected_index: -1, gallery_count: 2 } });
      expect(thumbs().map((t) => t.src)).toEqual([item(1).url, item(2).url]);
      expect(thumbs().map((t) => t.title)).toEqual(["Photo 1", "Photo 2"]);
      // selected_index -1 is a number, so it is kept rather than defaulted
      expect(thumbs().filter((t) => t.classList.contains("selected"))).toHaveLength(0);
      expect(clearBtn().hidden).toBe(false);
      expect(clearBtn().textContent).toBe("Clear (2)");
    });

    it("selects the newest photo when the value carries no selection", () => {
      mount({ value: { gallery_items: [item(1), item(2)] } as unknown as WebcamCaptureValue });
      expect(thumbs()[1]?.classList.contains("selected")).toBe(true);
    });
  });

  describe("capture and upload handshake", () => {
    it("captures a JPEG frame, shows it pending, and asks the node for an upload URL", async () => {
      // Given a streaming widget
      const { onChange } = await mountStreaming();

      // When the shutter is clicked
      shutterBtn().click();

      // Then the frame is drawn at the fallback size and encoded once as JPEG
      expect(fakes.lastContext().count("drawImage")).toBe(1);
      expect(fakes.lastContext().calls[0]?.args.slice(1)).toEqual([0, 0, 640, 480]);
      expect(HTMLCanvasElement.prototype.toDataURL).toHaveBeenCalledWith("image/jpeg", 0.95);

      // And a pending thumbnail shows the frame while the node is asked for a URL
      expect(thumbStrip().hidden).toBe(false);
      expect(pendingThumbs()).toHaveLength(1);
      expect(pendingThumbs()[0]?.querySelector("img")?.src).toBe(DATA_URL);
      expect(onChange).toHaveBeenCalledWith({ state: "requesting_upload_url", _emitSeq: 1 });
    });

    it("uploads the capture to the URL the node returns and commits accepted, ignoring a repeat", async () => {
      // Given a capture awaiting its URL
      const { onChange, update } = await mountStreaming();
      shutterBtn().click();

      // When the node replies with the URL, twice
      update({ state: "upload_ready", _uploadUrl: uploadUrl(1), _emitSeq: 1 });
      update({ state: "upload_ready", _uploadUrl: uploadUrl(1), _emitSeq: 1 });
      await flush();

      // Then the decoded JPEG is PUT once and accepted is committed with the same sequence
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, init] = fetchMock.mock.calls[0]!;
      expect(url).toBe(uploadUrl(1));
      expect(init.method).toBe("PUT");
      const body = init.body as Blob;
      expect(body.type).toBe("image/jpeg");
      expect(await body.text()).toBe(JPEG_BYTES);
      expect(onChange).toHaveBeenLastCalledWith({ state: "accepted", _emitSeq: 1 });
    });

    it("still commits accepted when the upload fails, so the pending thumbnail can clear", async () => {
      fetchMock.mockRejectedValue(new Error("network"));
      const { onChange, update } = await mountStreaming();
      shutterBtn().click();
      update({ state: "upload_ready", _uploadUrl: uploadUrl(1), _emitSeq: 1 });
      await flush();
      expect(onChange).toHaveBeenLastCalledWith({ state: "accepted", _emitSeq: 1 });
    });

    it("queues captures and sends the next only once the node has processed the previous", async () => {
      // Given two captures in quick succession
      const { onChange, update } = await mountStreaming();
      shutterBtn().click();
      shutterBtn().click();

      // Then only the first asks for a URL, and both show pending
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(pendingThumbs()).toHaveLength(2);

      // When the first is uploaded and processed
      update({ state: "upload_ready", _uploadUrl: uploadUrl(1), _emitSeq: 1 });
      await flush();
      update(processed(1, [item(1)]));

      // Then the photo joins the gallery, selected, and the second capture asks for its URL
      expect(thumbs().map((t) => t.src)).toEqual([item(1).url]);
      expect(thumbs()[0]?.classList.contains("selected")).toBe(true);
      expect(pendingThumbs()).toHaveLength(1);
      expect(onChange).toHaveBeenLastCalledWith({ state: "requesting_upload_url", _emitSeq: 2 });

      // And a repeated processed for the same sequence changes nothing
      update(processed(1, [item(1)]));
      expect(pendingThumbs()).toHaveLength(1);
      expect(onChange).toHaveBeenCalledTimes(3);
    });

    it("ignores the node's selection_confirmed echo", async () => {
      const { update } = await mountStreaming({
        value: { state: "idle", gallery_items: [item(1), item(2)], selected_index: 1, gallery_count: 2 },
      });
      update({ state: "selection_confirmed", gallery_items: [], selected_index: 0, gallery_count: 0, _emitSeq: 9 });
      expect(thumbs()).toHaveLength(2);
      expect(thumbs()[1]?.classList.contains("selected")).toBe(true);
    });
  });

  describe("gallery", () => {
    it("clicking a thumbnail selects it and commits the selection", async () => {
      // Given a gallery of two photos
      const gallery = [item(1), item(2)];
      const { onChange } = await mountStreaming({
        value: { state: "idle", gallery_items: gallery, selected_index: 1, gallery_count: 2 },
      });

      // When the first is clicked
      thumbs()[0]!.click();

      // Then it is highlighted and the selection is committed with the gallery
      expect(thumbs()[0]?.classList.contains("selected")).toBe(true);
      expect(thumbs()[1]?.classList.contains("selected")).toBe(false);
      expect(onChange).toHaveBeenCalledWith({ state: "selected", selected_index: 0, gallery_items: gallery, _emitSeq: 1 });
    });

    it("clear drops pending captures and asks the node to clear, and idle empties the gallery", async () => {
      // Given a gallery with a capture awaiting its URL
      const { onChange, update } = await mountStreaming({
        value: { state: "idle", gallery_items: [item(1)], selected_index: 0, gallery_count: 1 },
      });
      shutterBtn().click();

      // When Clear is clicked
      clearBtn().click();

      // Then the node is asked to clear
      expect(onChange).toHaveBeenLastCalledWith({ state: "clear_gallery", _emitSeq: 2 });

      // And a late URL for the dropped capture is not uploaded
      update({ state: "upload_ready", _uploadUrl: uploadUrl(1), _emitSeq: 1 });
      await flush();
      expect(fetchMock).not.toHaveBeenCalled();

      // When the node replies idle
      update({ state: "idle", gallery_items: [], selected_index: -1, gallery_count: 0, _emitSeq: 2 });

      // Then the strip and the Clear button hide
      expect(thumbs()).toHaveLength(0);
      expect(pendingThumbs()).toHaveLength(0);
      expect(thumbStrip().hidden).toBe(true);
      expect(clearBtn().hidden).toBe(true);
    });

    it("sends commits to the onChange from the latest update", async () => {
      // Given a widget whose callback was replaced by an update
      const { handle, onChange: oldChange } = await mountStreaming();
      const onChange = vi.fn<(value: WebcamCaptureValue) => void>();
      handle.update({ value: { state: "selection_confirmed", gallery_items: [], selected_index: -1, gallery_count: 0, _emitSeq: 0 }, onChange });

      // When a photo is taken
      shutterBtn().click();

      // Then only the new callback hears it
      expect(onChange).toHaveBeenCalledWith({ state: "requesting_upload_url", _emitSeq: 1 });
      expect(oldChange).not.toHaveBeenCalled();
    });
  });

  describe("camera selection", () => {
    it("reopens the camera on the one picked", async () => {
      // Given a streaming widget
      await mountStreaming();

      // When another camera is picked
      videoSel().value = "cam-2";
      videoSel().dispatchEvent(new Event("change"));
      await flush();

      // Then the old stream stops and the picked camera opens
      expect(media.streams[0]!.tracks[0]!.stopped).toBe(true);
      expect(media.requests.at(-1)?.video.deviceId).toEqual({ exact: "cam-2" });
      expect(video().srcObject).toBe(media.streams[1]);
    });

    it("repopulates the cameras when the devices change", async () => {
      await mountStreaming();
      media.enumerateDevices.mockClear();
      media.dispatchEvent(new Event("devicechange"));
      expect(media.enumerateDevices).toHaveBeenCalledTimes(1);
    });
  });

  describe("repeated mount on the same container", () => {
    it("reuses the widget and applies the new props", async () => {
      // Given a mounted widget
      const first = await mountStreaming();

      // When the loader mounts again with a new callback
      const onChange = vi.fn<(value: WebcamCaptureValue) => void>();
      const second = mount({ value: { state: "idle", gallery_items: [], selected_index: -1, gallery_count: 0 }, onChange });

      // Then no second widget or camera request is made, and the new callback receives commits
      expect(wrappers()).toHaveLength(1);
      expect(media.requests).toHaveLength(1);
      expect(second.result).toEqual(first.handle.result);
      shutterBtn().click();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(first.onChange).not.toHaveBeenCalled();
    });
  });

  describe("cleanup", () => {
    it("removes the widget, stops the camera, and stops listening for devices", async () => {
      // Given a streaming widget
      const { handle } = await mountStreaming();
      const removeListener = vi.spyOn(media, "removeEventListener");

      // When the widget is cleaned up
      handle.cleanup();

      // Then the DOM is gone and the camera is released
      expect(wrappers()).toHaveLength(0);
      expect(media.streams[0]!.tracks[0]!.stopped).toBe(true);
      expect(removeListener).toHaveBeenCalledWith("devicechange", expect.any(Function));

      // And a device change no longer repopulates the selector
      media.enumerateDevices.mockClear();
      media.dispatchEvent(new Event("devicechange"));
      expect(media.enumerateDevices).not.toHaveBeenCalled();
    });

    it("mounts a fresh widget on the same container after cleanup", async () => {
      const first = await mountStreaming();
      const stale = wrappers()[0];
      first.handle.cleanup();
      mount();
      expect(wrappers()).toHaveLength(1);
      expect(wrappers()[0]).not.toBe(stale);
    });
  });
});
