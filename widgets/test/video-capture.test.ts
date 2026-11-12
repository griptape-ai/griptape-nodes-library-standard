import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { VideoCaptureProps, VideoCaptureValue } from "../src/VideoCapture/VideoCapture";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<VideoCaptureProps>;
type Props = Partial<VideoCaptureProps>;

// Each behaviour runs against the TypeScript source and against the bundle the build emits,
// so the bundle is held to the same contract the editor relies on.
const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/VideoCapture/VideoCapture")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["VideoCapture"]);
      bundleDirs.push(outDir);
      return importMount<VideoCaptureProps>(join(outDir, "VideoCapture.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

// ── Media fakes: happy-dom has no camera, microphone, or recorder ───────────────

class FakeTrack {
  stopped = false;
  constructor(readonly kind: "video" | "audio", readonly deviceId: string) {}
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
    return this.tracks.filter((t) => t.kind === "video");
  }
  getAudioTracks(): FakeTrack[] {
    return this.tracks.filter((t) => t.kind === "audio");
  }
}

const DEVICES = [
  { kind: "videoinput", deviceId: "cam-1", label: "Front camera" },
  { kind: "videoinput", deviceId: "cam-2", label: "" },
  { kind: "audioinput", deviceId: "mic-1", label: "Built-in mic" },
  { kind: "audioinput", deviceId: "mic-2", label: "USB mic" },
];

type MediaRequest = { video: unknown; audio: unknown };

// navigator.mediaDevices whose getUserMedia answers from a per-test handler.
class FakeMediaDevices extends EventTarget {
  requests: MediaRequest[] = [];
  streams: FakeStream[] = [];
  handler: (req: MediaRequest) => Promise<FakeStream> = (req) => Promise.resolve(this.stream(req));

  getUserMedia = vi.fn((req: MediaRequest) => {
    this.requests.push(req);
    return this.handler(req);
  });
  enumerateDevices = vi.fn(async () => DEVICES);

  // A stream for the request, on the requested devices or the first of each kind.
  stream(req: MediaRequest): FakeStream {
    const exact = (c: unknown) => (c as { deviceId?: { exact?: string } }).deviceId?.exact;
    const tracks = [new FakeTrack("video", exact(req.video) ?? "cam-1")];
    if (req.audio) tracks.push(new FakeTrack("audio", exact(req.audio) ?? "mic-1"));
    const s = new FakeStream(tracks);
    this.streams.push(s);
    return s;
  }
}

class FakeMediaRecorder {
  static instances: FakeMediaRecorder[] = [];
  static supported: string[] = ["video/webm;codecs=vp9,opus", "video/webm"];
  static isTypeSupported(t: string): boolean {
    return FakeMediaRecorder.supported.includes(t);
  }

  state: "inactive" | "recording" = "inactive";
  timeslice: number | undefined;
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;

  constructor(readonly stream: FakeStream, readonly options: { mimeType: string }) {
    FakeMediaRecorder.instances.push(this);
  }
  start(timeslice?: number): void {
    this.state = "recording";
    this.timeslice = timeslice;
  }
  stop(): void {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["frames"], { type: this.options.mimeType }) });
    this.onstop?.();
  }
}

// Lets the widget's async chains (getUserMedia → enumerateDevices, fetch → then) settle.
async function flush(): Promise<void> {
  for (let i = 0; i < 20; i++) await Promise.resolve();
}

const MIME = "video/webm;codecs=vp9,opus";
const UPLOAD_URL = "http://localhost:8124/static-uploads/temp/_vc_node.webm";

describe.each(implementations)("VideoCapture (%s)", (_label, load) => {
  let VideoCapture: Mount;
  let container: HTMLElement;
  let handles: MountedWidget<VideoCaptureProps>[];
  let media: FakeMediaDevices;
  let frames: FrameRequestCallback[];
  let fetchMock: ReturnType<typeof vi.fn<(url: string, init: RequestInit) => Promise<{ ok: boolean; statusText: string }>>>;
  let srcObjectDescriptor: PropertyDescriptor | undefined;

  beforeAll(async () => {
    VideoCapture = await load();
  });

  beforeEach(() => {
    media = new FakeMediaDevices();
    Object.defineProperty(navigator, "mediaDevices", { configurable: true, value: media });
    FakeMediaRecorder.instances = [];
    vi.stubGlobal("MediaRecorder", FakeMediaRecorder);
    frames = [];
    vi.stubGlobal("requestAnimationFrame", (cb: FrameRequestCallback) => frames.push(cb));
    fetchMock = vi.fn(async () => ({ ok: true, statusText: "OK" }));
    vi.stubGlobal("fetch", fetchMock);
    vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:http://localhost/recording");
    vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => undefined);
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
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  // Mounts with the editor's props filled in around the ones a test cares about.
  function mount(props: Props = {}) {
    const mounted = mountWidget(VideoCapture, container, widgetProps(props));
    const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
    handles.push(handle);
    return handle;
  }

  async function mountStreaming(props: Props = {}) {
    const onChange = vi.fn<(value: VideoCaptureValue) => void>();
    const handle = mount({ value: { state: "idle" }, onChange, ...props });
    await flush();
    return { handle, onChange };
  }

  // Records, stops, and leaves the widget reviewing the recording.
  async function mountReviewing(props: Props = {}) {
    const mounted = await mountStreaming(props);
    recordBtn().click();
    recordBtn().click();
    return mounted;
  }

  // Accepts the recording and lets the next animation frame run.
  function acceptRecording() {
    button("Accept").click();
    for (const cb of frames.splice(0)) cb(0);
  }

  const wrappers = () => container.querySelectorAll(":scope > div");
  const video = () => container.querySelector("video")!;
  const videoWrap = () => video().parentElement!;
  const placeholder = () => videoWrap().children[0] as HTMLDivElement;
  const recBadge = () => videoWrap().children[2] as HTMLDivElement;
  const micBadge = () => videoWrap().children[3] as HTMLButtonElement;
  const encodingOverlay = () => videoWrap().children[4] as HTMLDivElement;
  const recordBtn = () => videoWrap().children[5] as HTMLButtonElement;
  const reviewOverlay = () => videoWrap().children[6] as HTMLDivElement;
  const retryBtn = () => videoWrap().children[7] as HTMLButtonElement;
  const selects = () => [...container.querySelectorAll("select")];
  const button = (label: string) => {
    const btn = [...container.querySelectorAll("button")].find((b) => b.querySelector("span")?.textContent === label);
    if (!btn) throw new Error(`no button labelled ${label}`);
    return btn;
  };

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the camera view inside the container", () => {
      const handle = mount();
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(wrappers()).toHaveLength(1);
      expect(wrappers()[0]?.className).toBe("nodrag nowheel");
      expect(video().muted).toBe(true);
      expect(placeholder().textContent).toBe("Starting camera…");
      expect(selects()).toHaveLength(2);
      expect(reviewOverlay().hidden).toBe(true);
      expect(recordBtn().disabled).toBe(true);
      expect(document.getElementById("vc-styles")).not.toBeNull();
    });

    it("opens the default camera and microphone and lists the devices once streaming", async () => {
      // Given a fresh widget
      // When the camera opens
      await mountStreaming();

      // Then both default devices were requested and the stream is shown
      expect(media.requests).toEqual([{ video: true, audio: true }]);
      expect(video().srcObject).toBe(media.streams[0]);
      expect(placeholder().hidden).toBe(true);
      expect(recordBtn().disabled).toBe(false);
      expect(micBadge().hidden).toBe(true);

      // And the selectors list the devices, unnamed ones numbered, with the open ones selected
      const [videoSel, audioSel] = selects();
      expect([...videoSel!.options].map((o) => o.textContent)).toEqual(["Front camera", "Device 2"]);
      expect([...audioSel!.options].map((o) => o.textContent)).toEqual(["Built-in mic", "USB mic"]);
      expect(videoSel!.value).toBe("cam-1");
      expect(audioSel!.value).toBe("mic-1");
    });

    it("falls back to video only when the microphone can't be opened", async () => {
      // Given a microphone that refuses to open
      media.handler = (req) =>
        req.audio ? Promise.reject(new DOMException("no mic", "NotFoundError")) : Promise.resolve(media.stream(req));
      vi.spyOn(console, "warn").mockImplementation(() => undefined);

      // When the widget mounts
      await mountStreaming();

      // Then it streams video only and offers to retry the microphone
      expect(media.requests.at(-1)).toEqual({ video: true, audio: false });
      expect(placeholder().hidden).toBe(true);
      expect(micBadge().hidden).toBe(false);

      // When a microphone becomes available and the badge is clicked
      media.handler = (req) => Promise.resolve(media.stream(req));
      micBadge().click();
      await flush();

      // Then the camera reopens with the default microphone
      expect(media.streams[0]!.tracks.every((t) => t.stopped)).toBe(true);
      expect(media.requests.at(-1)).toEqual({ video: { deviceId: { exact: "cam-1" } }, audio: true });
      expect(micBadge().hidden).toBe(true);
    });

    it("explains a blocked camera and retries with the default devices", async () => {
      // Given a camera the browser blocks
      media.handler = () => Promise.reject(new DOMException("denied", "NotAllowedError"));
      vi.spyOn(console, "warn").mockImplementation(() => undefined);

      // When the widget mounts
      await mountStreaming();

      // Then the reason is shown with a retry button and no record button
      expect(placeholder().textContent).toBe(
        "Camera access is blocked. Allow camera access in your browser or system privacy settings, then retry.",
      );
      expect(retryBtn().hidden).toBe(false);
      expect(recordBtn().hidden).toBe(true);

      // When the camera becomes available and Retry is clicked
      media.handler = (req) => Promise.resolve(media.stream(req));
      retryBtn().click();
      await flush();

      // Then the default devices open
      expect(media.requests.at(-1)).toEqual({ video: true, audio: true });
      expect(retryBtn().hidden).toBe(true);
      expect(recordBtn().hidden).toBe(false);
    });
  });

  describe("recording", () => {
    it("records the stream with the preferred supported MIME type and counts the time", async () => {
      // Given a streaming widget
      vi.useFakeTimers({ toFake: ["setInterval", "clearInterval"] });
      await mountStreaming();

      // When record is clicked
      recordBtn().click();

      // Then the stream records in 100 ms chunks and the badge counts up
      const rec = FakeMediaRecorder.instances[0]!;
      expect(rec.stream).toBe(media.streams[0]);
      expect(rec.options).toEqual({ mimeType: MIME });
      expect(rec.timeslice).toBe(100);
      expect(recBadge().hidden).toBe(false);
      expect(recBadge().textContent).toBe("● REC 00:00");
      vi.advanceTimersByTime(65_000);
      expect(recBadge().textContent).toBe("● REC 01:05");

      // And the device selectors are locked while recording
      expect(selects().every((s) => s.disabled)).toBe(true);
    });

    it("stops into review: plays the recording back and releases the camera", async () => {
      // Given a recording under way
      await mountStreaming();
      recordBtn().click();

      // When record is clicked again
      recordBtn().click();

      // Then the camera is released and the recording plays back unmuted
      expect(media.streams[0]!.tracks.every((t) => t.stopped)).toBe(true);
      expect(video().srcObject).toBeNull();
      expect(video().src).toBe("blob:http://localhost/recording");
      expect(video().muted).toBe(false);
      expect(recordBtn().hidden).toBe(true);
      expect(recBadge().hidden).toBe(true);
      expect(reviewOverlay().hidden).toBe(false);
      expect(selects().every((s) => s.disabled)).toBe(true);
    });

    it("re-record discards the recording and reopens the camera", async () => {
      // Given a recording in review
      await mountReviewing();

      // When Re-record is clicked
      button("Re-record").click();
      await flush();

      // Then the blob is released and the camera streams again
      expect(URL.revokeObjectURL).toHaveBeenCalledWith("blob:http://localhost/recording");
      expect(media.requests).toHaveLength(2);
      expect(video().srcObject).toBe(media.streams[1]);
      expect(reviewOverlay().hidden).toBe(true);
      expect(recordBtn().hidden).toBe(false);
    });

    it("reopens the camera on the device picked in a selector", async () => {
      // Given a streaming widget
      await mountStreaming();
      const [videoSel] = selects();

      // When another camera is picked
      videoSel!.value = "cam-2";
      videoSel!.dispatchEvent(new Event("change"));
      await flush();

      // Then the old stream stops and the picked camera opens with the current microphone
      expect(media.streams[0]!.tracks.every((t) => t.stopped)).toBe(true);
      expect(media.requests.at(-1)).toEqual({
        video: { deviceId: { exact: "cam-2" } },
        audio: { deviceId: { exact: "mic-1" } },
      });
    });
  });

  describe("upload handshake", () => {
    it("accept asks the node for an upload URL on the next frame and shows the overlay", async () => {
      // Given a recording in review
      const { onChange } = await mountReviewing();

      // When Accept is clicked
      button("Accept").click();

      // Then nothing is sent until the frame, and the review buttons lock under the overlay
      expect(onChange).not.toHaveBeenCalled();
      expect(encodingOverlay().hidden).toBe(false);
      expect(button("Accept").disabled).toBe(true);
      expect(button("Re-record").disabled).toBe(true);

      for (const cb of frames.splice(0)) cb(0);
      expect(onChange).toHaveBeenCalledWith({ state: "requesting_upload_url", _mime: MIME, _emitSeq: 1 });
    });

    it("uploads the recording to the URL the node returns, then commits accepted once", async () => {
      // Given an accepted recording
      const { handle, onChange } = await mountReviewing();
      acceptRecording();

      // When the node replies with an upload URL, twice
      const ready: VideoCaptureValue = { state: "upload_ready", _uploadUrl: UPLOAD_URL, _emitSeq: 1 };
      handle.update({ value: ready, onChange });
      handle.update({ value: ready, onChange });
      await flush();

      // Then the blob is PUT once and accepted is committed with the same sequence
      expect(fetchMock).toHaveBeenCalledTimes(1);
      const [url, init] = fetchMock.mock.calls[0]!;
      expect(url).toBe(UPLOAD_URL);
      expect(init.method).toBe("PUT");
      expect(init.body).toBeInstanceOf(Blob);
      expect((init.body as Blob).type).toBe(MIME);
      expect(onChange).toHaveBeenLastCalledWith({ state: "accepted", _emitSeq: 1 });
    });

    it("a failed upload unlocks the review buttons without committing", async () => {
      // Given an accepted recording whose upload fails
      fetchMock.mockResolvedValue({ ok: false, statusText: "Bad Gateway" });
      const { handle, onChange } = await mountReviewing();
      acceptRecording();

      // When the node replies with an upload URL
      handle.update({ value: { state: "upload_ready", _uploadUrl: UPLOAD_URL, _emitSeq: 1 }, onChange });
      await flush();

      // Then the overlay clears and only the request was committed
      expect(encodingOverlay().hidden).toBe(true);
      expect(button("Accept").disabled).toBe(false);
      expect(onChange).toHaveBeenCalledTimes(1);
    });

    it("processed hides the overlay", async () => {
      const { handle, onChange } = await mountReviewing();
      acceptRecording();
      handle.update({ value: { state: "processed", url: "https://example.test/v.webm", _emitSeq: 1 }, onChange });
      expect(encodingOverlay().hidden).toBe(true);
      expect(button("Re-record").disabled).toBe(false);
    });

    it("an error shows the node's message, then clears after four seconds", async () => {
      // Given an accepted recording
      vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
      const { handle, onChange } = await mountReviewing();
      acceptRecording();

      // When the node reports an error
      handle.update({ value: { state: "error", message: "Upload not found. Please re-record.", _emitSeq: 1 }, onChange });

      // Then the overlay shows it until the timeout
      expect(encodingOverlay().textContent).toBe("Upload not found. Please re-record.");
      expect(encodingOverlay().hidden).toBe(false);
      vi.advanceTimersByTime(4000);
      expect(encodingOverlay().hidden).toBe(true);
    });

    it("sends commits to the onChange from the latest update", async () => {
      // Given a widget whose callback was replaced by an update
      const { handle, onChange: oldChange } = await mountReviewing();
      const onChange = vi.fn<(value: VideoCaptureValue) => void>();
      handle.update({ value: { state: "idle" }, onChange });

      // When the recording is accepted
      acceptRecording();

      // Then only the new callback hears it
      expect(onChange).toHaveBeenCalledWith({ state: "requesting_upload_url", _mime: MIME, _emitSeq: 1 });
      expect(oldChange).not.toHaveBeenCalled();
    });
  });

  describe("repeated mount on the same container", () => {
    it("reuses the widget and applies the new props", async () => {
      // Given a mounted widget
      const first = await mountReviewing();

      // When the loader mounts again with a new callback
      const onChange = vi.fn<(value: VideoCaptureValue) => void>();
      const second = mount({ value: { state: "idle" }, onChange });

      // Then no second widget or camera request is made, and the new callback receives commits
      expect(wrappers()).toHaveLength(1);
      expect(media.requests).toHaveLength(1);
      expect(second.result).toEqual(first.handle.result);
      acceptRecording();
      expect(onChange).toHaveBeenCalledTimes(1);
      expect(first.onChange).not.toHaveBeenCalled();
    });
  });

  describe("cleanup", () => {
    it("removes the widget, stops the camera and the recorder, and stops listening for devices", async () => {
      // Given a recording under way
      const { handle } = await mountStreaming();
      recordBtn().click();
      const rec = FakeMediaRecorder.instances[0]!;
      const removeListener = vi.spyOn(media, "removeEventListener");

      // When the widget is cleaned up
      handle.cleanup();

      // Then the DOM is gone, the recorder stops without entering review, and the tracks stop
      expect(wrappers()).toHaveLength(0);
      expect(rec.state).toBe("inactive");
      expect(rec.onstop).toBeNull();
      expect(media.streams[0]!.tracks.every((t) => t.stopped)).toBe(true);
      expect(removeListener).toHaveBeenCalledWith("devicechange", expect.any(Function));

      // And a device change no longer repopulates the selectors
      media.enumerateDevices.mockClear();
      media.dispatchEvent(new Event("devicechange"));
      expect(media.enumerateDevices).not.toHaveBeenCalled();
    });

    it("stops a camera that opens after cleanup instead of attaching it", async () => {
      // Given a camera request still in flight
      let resolve!: (s: FakeStream) => void;
      media.handler = () => new Promise((r) => { resolve = r; });
      const handle = mount();

      // When the widget is cleaned up before the camera opens
      handle.cleanup();
      const late = new FakeStream([new FakeTrack("video", "cam-1")]);
      resolve(late);
      await flush();

      // Then the late stream is stopped
      expect(late.tracks[0]!.stopped).toBe(true);
    });

    it("mounts a fresh widget on the same container after cleanup", async () => {
      const first = await mountStreaming();
      const stale = wrappers()[0];
      first.handle.cleanup();
      mount();
      expect(wrappers()).toHaveLength(1);
      expect(wrappers()[0]).not.toBe(stale);
      expect(media.requests).toHaveLength(2);
    });
  });
});
