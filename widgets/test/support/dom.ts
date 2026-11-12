// Stand-ins for the browser APIs the DOM test environment lacks or can't drive: a 2D canvas
// context that records draw calls, an Image whose load the test triggers, and a ResizeObserver
// whose callbacks and disconnects the test can see.

import { vi } from "vitest";

export interface DrawCall {
  method: string;
  args: unknown[];
}

export class RecordingContext {
  calls: DrawCall[] = [];

  texts(): string[] {
    return this.calls.filter((c) => c.method === "fillText").map((c) => String(c.args[0]));
  }

  count(method: string): number {
    return this.calls.filter((c) => c.method === method).length;
  }

  reset(): void {
    this.calls = [];
  }
}

// Any method call on the context is recorded and does nothing; property writes are kept.
function recordingContextProxy(rec: RecordingContext): CanvasRenderingContext2D {
  const props = new Map<PropertyKey, unknown>();
  const proxy = new Proxy(
    {},
    {
      get(_target, prop) {
        if (props.has(prop)) return props.get(prop);
        return (...args: unknown[]) => {
          rec.calls.push({ method: String(prop), args });
        };
      },
      set(_target, prop, value) {
        props.set(prop, value);
        return true;
      },
    },
  );
  return proxy as unknown as CanvasRenderingContext2D;
}

export class FakeImage {
  static instances: FakeImage[] = [];

  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  crossOrigin: string | null = null;
  naturalWidth = 0;
  naturalHeight = 0;
  src = "";

  constructor() {
    FakeImage.instances.push(this);
  }

  load(width: number, height: number): void {
    this.naturalWidth = width;
    this.naturalHeight = height;
    this.onload?.();
  }

  fail(): void {
    this.onerror?.();
  }
}

export class FakeResizeObserver {
  static instances: FakeResizeObserver[] = [];

  observed: Element[] = [];
  disconnected = false;

  constructor(private readonly callback: ResizeObserverCallback) {
    FakeResizeObserver.instances.push(this);
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

  // Delivers a resize the way the browser would: only to targets still observed.
  trigger(): void {
    if (this.disconnected || this.observed.length === 0) return;
    this.callback([], this as unknown as ResizeObserver);
  }
}

export interface DomFakes {
  contexts: RecordingContext[];
  lastContext(): RecordingContext;
  lastImage(): FakeImage;
  lastResizeObserver(): FakeResizeObserver;
}

// Installs the fakes for one test. vi.unstubAllGlobals / vi.restoreAllMocks undo them.
export function installDomFakes(): DomFakes {
  FakeImage.instances = [];
  FakeResizeObserver.instances = [];
  const contexts: RecordingContext[] = [];

  vi.stubGlobal("Image", FakeImage);
  vi.stubGlobal("ResizeObserver", FakeResizeObserver);

  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockImplementation(function () {
    const rec = new RecordingContext();
    contexts.push(rec);
    return recordingContextProxy(rec);
  } as unknown as HTMLCanvasElement["getContext"]);

  // Lay the canvas out at its intrinsic size, so client and canvas coordinates match.
  vi.spyOn(HTMLCanvasElement.prototype, "getBoundingClientRect").mockImplementation(function (
    this: HTMLCanvasElement,
  ) {
    return new DOMRect(0, 0, this.width, this.height);
  });

  if (!("setPointerCapture" in Element.prototype)) {
    Object.defineProperty(Element.prototype, "setPointerCapture", {
      configurable: true,
      value: () => undefined,
    });
  }
  vi.spyOn(Element.prototype, "setPointerCapture").mockImplementation(() => undefined);

  return {
    contexts,
    lastContext() {
      const ctx = contexts.at(-1);
      if (!ctx) throw new Error("no canvas context was created");
      return ctx;
    },
    lastImage() {
      const img = FakeImage.instances.at(-1);
      if (!img) throw new Error("no Image was created");
      return img;
    },
    lastResizeObserver() {
      const ro = FakeResizeObserver.instances.at(-1);
      if (!ro) throw new Error("no ResizeObserver was created");
      return ro;
    },
  };
}

export function pointer(
  target: Element,
  type: "pointerdown" | "pointermove" | "pointerup" | "pointercancel",
  x: number,
  y: number,
  init: { shiftKey?: boolean } = {},
): void {
  target.dispatchEvent(
    new PointerEvent(type, {
      bubbles: true,
      cancelable: true,
      clientX: x,
      clientY: y,
      pointerId: 1,
      shiftKey: init.shiftKey ?? false,
    }),
  );
}
