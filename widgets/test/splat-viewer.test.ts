import { join } from "node:path";

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";

import type { SplatViewerProps, SplatViewerState } from "../src/SplatViewer/SplatViewer";
import type { WidgetMount } from "@griptape-ai/nodes-widget-api";
import { buildIntoScratch, importMount, removeDir } from "./support/bundle";
import { FakeResizeObserver } from "./support/dom";
import { type MountedWidget, mountWidget, widgetProps } from "./support/widget";

type Mount = WidgetMount<SplatViewerProps>;
type Props = Partial<SplatViewerProps>;

// The source run swaps the WebGL pieces for recorders: three's WebGLRenderer, and Spark's renderer,
// controls, and splat mesh. Scenes, cameras, and boxes are the real three. The bundle carries the
// real libraries, which can't render in this DOM env, so it runs only the behaviour that needs no
// WebGL.
const fakes = vi.hoisted(() => ({
  renderers: [] as FakeRendererLike[],
  meshes: [] as FakeMeshLike[],
  controls: [] as { update: ReturnType<typeof import("vitest").vi.fn> }[],
  rendererError: null as Error | null,
  meshError: null as Error | null,
}));

interface FakeRendererLike {
  domElement: HTMLCanvasElement;
  loop: (() => void) | null;
  setSize: ReturnType<typeof vi.fn>;
  render: ReturnType<typeof vi.fn>;
  dispose: ReturnType<typeof vi.fn>;
}

interface FakeMeshLike {
  options: { url: string; fileType?: string; onLoad?: () => void };
  rotation: { x: number };
  boundingBox?: unknown;
  dispose: ReturnType<typeof vi.fn>;
  resolve: () => void;
  reject: (err: unknown) => void;
}

vi.mock("three", async (importOriginal) => {
  const actual = await importOriginal<typeof import("three")>();
  class WebGLRenderer {
    domElement = document.createElement("canvas");
    loop: (() => void) | null = null;
    setClearColor = vi.fn();
    setPixelRatio = vi.fn();
    setSize = vi.fn();
    render = vi.fn();
    dispose = vi.fn();
    constructor() {
      if (fakes.rendererError) throw fakes.rendererError;
      fakes.renderers.push(this);
    }
    setAnimationLoop(cb: (() => void) | null) {
      this.loop = cb;
    }
  }
  return { ...actual, WebGLRenderer };
});

vi.mock("@sparkjsdev/spark", async () => {
  const THREE = await vi.importActual<typeof import("three")>("three");
  class SparkRenderer extends THREE.Object3D {}
  class SparkControls {
    update = vi.fn();
    constructor() {
      fakes.controls.push(this);
    }
  }
  class SplatMesh extends THREE.Object3D {
    initialized: Promise<unknown>;
    boundingBox?: InstanceType<typeof THREE.Box3>;
    dispose = vi.fn();
    resolve!: () => void;
    reject!: (err: unknown) => void;
    constructor(public options: FakeMeshLike["options"]) {
      super();
      if (fakes.meshError) throw fakes.meshError;
      this.initialized = new Promise((resolve, reject) => {
        this.resolve = () => resolve(this);
        this.reject = reject;
      });
      fakes.meshes.push(this as unknown as FakeMeshLike);
    }
  }
  return { SparkRenderer, SparkControls, SplatMesh, SplatFileType: { SPLAT: "splat", KSPLAT: "ksplat" } };
});

const implementations: Array<[string, () => Promise<Mount>]> = [
  [
    "source",
    async () => (await import("../src/SplatViewer/SplatViewer")).default as Mount,
  ],
  [
    "bundle",
    async () => {
      const { outDir } = await buildIntoScratch(undefined, ["SplatViewer"]);
      bundleDirs.push(outDir);
      return importMount<SplatViewerProps>(join(outDir, "SplatViewer.js"));
    },
  ],
];
const bundleDirs: string[] = [];

afterAll(async () => {
  await Promise.all(bundleDirs.map(removeDir));
});

const URL_FULL = "https://example.test/world-full.spz";
const URL_100K = "https://example.test/world-100k.spz";

function state(splats: SplatViewerState["splats"]): string {
  return JSON.stringify({ splats });
}

let container: HTMLElement;
let handles: MountedWidget<SplatViewerProps>[];

function setup() {
  beforeEach(() => {
    FakeResizeObserver.instances = [];
    vi.stubGlobal("ResizeObserver", FakeResizeObserver);
    for (const method of ["info", "warn", "error"] as const) vi.spyOn(console, method).mockImplementation(() => undefined);
    container = document.createElement("div");
    document.body.appendChild(container);
    handles = [];
  });

  afterEach(() => {
    for (const h of handles) h.cleanup();
    container.remove();
    vi.useRealTimers();
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });
}

function mountWith(mount: Mount, props: Props) {
  const mounted = mountWidget(mount, container, widgetProps(props));
  const handle = { ...mounted, update: (next: Props) => mounted.update(widgetProps(next)) };
  handles.push(handle);
  return handle;
}

const select = () => container.querySelector("select")!;
const options = () => [...select().options].map((o) => [o.value, o.textContent, o.disabled]);
const stage = () => container.querySelector<HTMLElement>(".splat-viewer-widget > div:last-child")!;
const stageText = () => stage().textContent;

describe.each(implementations)("SplatViewer (%s)", (_label, load) => {
  let SplatViewer: Mount;
  const mount = (props: Props) => mountWith(SplatViewer, props);

  beforeAll(async () => {
    SplatViewer = await load();
  });
  setup();

  describe("mount", () => {
    it("returns a cleanup/update handle and builds the picker and stage", () => {
      const handle = mount({ value: undefined });
      expect(typeof handle.cleanup).toBe("function");
      expect(typeof handle.update).toBe("function");
      expect(container.querySelectorAll(".splat-viewer-widget")).toHaveLength(1);
      expect(container.querySelector(".splat-viewer-widget")!.className).toBe("splat-viewer-widget nodrag nowheel");
      expect(options().map(([value]) => value)).toEqual(["100k", "500k", "full_res"]);
    });

    it("is 640 px tall unless the node gives a height", () => {
      mount({ value: undefined });
      expect(container.querySelector<HTMLElement>(".splat-viewer-widget")!.style.height).toBe("640px");
      mount({ value: undefined, height: 300 });
      expect(container.querySelectorAll(".splat-viewer-widget")).toHaveLength(1);
      expect(container.querySelector<HTMLElement>(".splat-viewer-widget")!.style.height).toBe("300px");
    });

    it("asks for a splat and marks every resolution unwired when nothing is wired", () => {
      mount({ value: state({ "100k": null, "500k": null, full_res: null }) });
      expect(stageText()).toBe("Wire a splat input to render.");
      expect(options()).toEqual([
        ["100k", "100k (not wired)", true],
        ["500k", "500k (not wired)", true],
        ["full_res", "Full Res (not wired)", true],
      ]);
    });

    it("treats a value that isn't valid JSON as empty", () => {
      mount({ value: "{not json" });
      expect(stageText()).toBe("Wire a splat input to render.");
    });
  });

  describe("update", () => {
    it("keeps one viewer and refreshes the picker", () => {
      const handle = mount({ value: undefined });
      handle.update({ value: { splats: {} } });
      expect(container.querySelectorAll(".splat-viewer-widget")).toHaveLength(1);
      expect(stageText()).toBe("Wire a splat input to render.");
    });
  });

  it("keeps pointer, wheel, and key events from reaching the canvas", () => {
    mount({ value: undefined });
    const seen = vi.fn();
    for (const type of ["pointerdown", "mousedown", "wheel", "keydown"]) container.addEventListener(type, seen);
    for (const type of ["pointerdown", "mousedown", "wheel", "keydown"]) stage().dispatchEvent(new Event(type, { bubbles: true }));
    expect(seen).not.toHaveBeenCalled();
  });

  it("cleanup empties the container", () => {
    const handle = mount({ value: undefined });
    handle.cleanup();
    expect(container.childElementCount).toBe(0);
  });
});

// With WebGL stubbed: the scene the widget builds for a wired splat.
describe("SplatViewer (source, stubbed WebGL)", () => {
  let SplatViewer: Mount;
  const mount = (props: Props) => mountWith(SplatViewer, props);

  beforeAll(async () => {
    SplatViewer = (await import("../src/SplatViewer/SplatViewer")).default as Mount;
  });
  setup();
  beforeEach(() => {
    fakes.renderers.length = 0;
    fakes.meshes.length = 0;
    fakes.controls.length = 0;
    fakes.rendererError = null;
    fakes.meshError = null;
  });

  const renderer = () => fakes.renderers.at(-1)!;
  const mesh = () => fakes.meshes.at(-1)!;
  const flush = () => new Promise((r) => setTimeout(r, 0));

  it("loads the highest-fidelity wired splat and marks the rest unwired", () => {
    // Given a state with 100k and full res wired
    mount({ value: state({ "100k": { url: URL_100K }, "500k": null, full_res: { url: URL_FULL } }) });

    // Then full res is chosen and loading
    expect(select().value).toBe("full_res");
    expect(options()).toEqual([
      ["100k", "100k", false],
      ["500k", "500k (not wired)", true],
      ["full_res", "Full Res", false],
    ]);
    expect(mesh().options.url).toBe(URL_FULL);
    expect(mesh().options.fileType).toBeUndefined();
    expect(stage().contains(renderer().domElement)).toBe(true);
    expect(stageText()).toBe("Loading splat…");
  });

  it("accepts the state as an object", () => {
    mount({ value: { splats: { "500k": { url: URL_100K } } } });
    expect(select().value).toBe("500k");
    expect(mesh().options.url).toBe(URL_100K);
  });

  it.each([
    ["https://example.test/a.splat?sig=1", "splat"],
    ["https://example.test/a.ksplat", "ksplat"],
  ])("tells Spark the file type of %s", (url, fileType) => {
    mount({ value: state({ full_res: { url } }) });
    expect(mesh().options.fileType).toBe(fileType);
  });

  it("removes the loading message, flips the splat, and frames it once it loads", async () => {
    // Given a splat whose bounds are a 2-unit cube around the origin
    const { Box3, Vector3 } = await import("three");
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    mesh().boundingBox = new Box3(new Vector3(-1, -1, -1), new Vector3(1, 1, 1));

    // When it loads
    mesh().options.onLoad?.();

    // Then the overlay is gone, the splat is upright, and the camera sits back by 1.5× its size
    expect(stageText()).toBe("");
    expect(mesh().rotation.x).toBe(Math.PI);
    renderer().loop?.();
    const camera = renderer().render.mock.lastCall?.[1] as { position: { z: number }; far: number };
    expect(camera.position.z).toBeCloseTo(3);
    expect(camera.far).toBeCloseTo(300);
  });

  it("shows why a splat failed to load", async () => {
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    mesh().reject(new Error("404"));
    await flush();
    expect(stageText()).toBe("Failed to load splat: 404");
  });

  it("shows the error when WebGL is unavailable", () => {
    fakes.rendererError = new Error("no gl");
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    expect(stageText()).toBe("WebGL error: no gl");
  });

  it("shows the error when Spark can't create the splat", () => {
    fakes.meshError = new Error("bad url");
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    expect(stageText()).toBe("SplatMesh error: bad url");
  });

  it("renders each frame and steers the camera with the controls", () => {
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    renderer().loop?.();
    expect(fakes.controls.at(-1)!.update).toHaveBeenCalledTimes(1);
    expect(renderer().render).toHaveBeenCalledTimes(1);
  });

  it("sizes the renderer from the stage and follows its resizes", () => {
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    const ro = FakeResizeObserver.instances.at(-1)!;
    expect(ro.observed).toContain(stage());
    expect(renderer().setSize).toHaveBeenLastCalledWith(400, 604, false);
    renderer().setSize.mockClear();
    ro.trigger();
    expect(renderer().setSize).toHaveBeenCalledTimes(1);
  });

  it("swaps to the resolution the user picks, tearing down the previous scene", () => {
    // Given full res showing
    mount({ value: state({ "100k": { url: URL_100K }, full_res: { url: URL_FULL } }) });
    const first = renderer();
    const firstMesh = mesh();
    const firstRo = FakeResizeObserver.instances.at(-1)!;

    // When the user picks 100k
    select().value = "100k";
    select().dispatchEvent(new Event("change"));

    // Then the old scene is disposed and the 100k splat loads
    expect(first.loop).toBeNull();
    expect(first.dispose).toHaveBeenCalled();
    expect(firstMesh.dispose).toHaveBeenCalled();
    expect(firstRo.disconnected).toBe(true);
    expect(stage().contains(first.domElement)).toBe(false);
    expect(mesh().options.url).toBe(URL_100K);
  });

  it("keeps the scene when an update brings the same splat, and reloads it when the splat changed", () => {
    // Given a splat showing
    const value = state({ full_res: { url: URL_FULL, meta: { created_at: "t1" } } });
    const handle = mount({ value });

    // When the same state arrives again, nothing reloads
    handle.update({ value });
    expect(fakes.meshes).toHaveLength(1);

    // When the same URL arrives with a new creation time, it reloads
    handle.update({ value: state({ full_res: { url: URL_FULL, meta: { created_at: "t2" } } }) });
    expect(fakes.meshes).toHaveLength(2);
  });

  it("keeps the user's pick across updates while it stays wired, and falls back when it's unwired", () => {
    const both = state({ "100k": { url: URL_100K }, full_res: { url: URL_FULL } });
    const handle = mount({ value: both });
    select().value = "100k";
    select().dispatchEvent(new Event("change"));

    handle.update({ value: both });
    expect(select().value).toBe("100k");

    handle.update({ value: state({ "100k": null, full_res: { url: URL_FULL } }) });
    expect(select().value).toBe("full_res");
    expect(mesh().options.url).toBe(URL_FULL);
  });

  it("shows a placeholder when every splat is unwired, disposing the scene", () => {
    const handle = mount({ value: state({ full_res: { url: URL_FULL } }) });
    const r = renderer();
    handle.update({ value: state({ full_res: null }) });
    expect(r.dispose).toHaveBeenCalled();
    expect(stageText()).toBe("Wire a splat input to render.");
  });

  it("cleanup stops rendering, disconnects the observer, disposes the scene, and empties the container", () => {
    const handle = mount({ value: state({ full_res: { url: URL_FULL } }) });
    const r = renderer();
    const ro = FakeResizeObserver.instances.at(-1)!;

    handle.cleanup();

    expect(r.loop).toBeNull();
    expect(r.dispose).toHaveBeenCalled();
    expect(mesh().dispose).toHaveBeenCalled();
    expect(ro.disconnected).toBe(true);
    expect(container.childElementCount).toBe(0);
  });
});

// The bundled three, for real: it runs in this DOM env and reports that WebGL isn't there.
describe("SplatViewer (bundle, real three)", () => {
  let SplatViewer: Mount;
  const mount = (props: Props) => mountWith(SplatViewer, props);

  beforeAll(async () => {
    const { outDir } = await buildIntoScratch(undefined, ["SplatViewer"]);
    bundleDirs.push(outDir);
    SplatViewer = await importMount<SplatViewerProps>(join(outDir, "SplatViewer.js"));
  });
  setup();

  it("shows a WebGL error for a wired splat when the browser has no WebGL", () => {
    mount({ value: state({ full_res: { url: URL_FULL } }) });
    expect(stageText()).toMatch(/^WebGL error: /);
  });
});
