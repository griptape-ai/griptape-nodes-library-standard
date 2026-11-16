import { type CSSProperties, useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import {
  SplatFileType,
  SplatMesh,
  SparkControls,
  SparkRenderer,
} from "@sparkjsdev/spark";
import { reactWidget } from "@griptape-ai/nodes-widget-api/react";
import type { SplatMap, SplatResolution, SplatViewerValue } from './_types';
import type { WidgetProps } from '@griptape-ai/nodes-widget-api';

export type { SplatEntry, SplatMap, SplatResolution, SplatViewerState, SplatViewerValue } from './_types';

export type SplatViewerProps = WidgetProps<SplatViewerValue>;

// The bounding boxes fitCamera looks for, which a SplatMesh may or may not carry.
type FittableSplat = SplatMesh & {
  boundingBox?: THREE.Box3;
  geometry?: { boundingBox?: THREE.Box3 | null; computeBoundingBox?: () => void };
};

// What the stage shows over the canvas: a grey message, a red load failure, or nothing.
type StageStatus = { kind: "message" | "failed"; text: string } | null;

const VIEWER_HEIGHT = 640;
const PRIORITY: SplatResolution[] = ["full_res", "500k", "100k"];      // auto-select: highest fidelity first
const DISPLAY_ORDER: SplatResolution[] = ["100k", "500k", "full_res"];  // dropdown: matches parameter order
const RESOLUTION_LABELS: Record<SplatResolution, string> = { "100k": "100k", "500k": "500k", "full_res": "Full Res" };
const TAG = "[SplatViewer]";

const stopPropagation = (e: { stopPropagation(): void }) => e.stopPropagation();

// Keeps pointer, wheel and key events from reaching the node editor behind the widget.
const stopNodeEvents = {
  onPointerDown: stopPropagation,
  onMouseDown: stopPropagation,
  onWheel: stopPropagation,
  onKeyDown: stopPropagation,
};

// ---------- styles -----------------------------------------------------------

const barStyle: CSSProperties = {
  display: "flex", alignItems: "center", gap: 8, padding: "4px 8px",
  background: "rgba(0,0,0,0.35)", borderBottom: "1px solid rgba(255,255,255,0.08)",
  flexShrink: 0, zIndex: 2,
};
const labelStyle: CSSProperties = { font: "11px system-ui", color: "#999", whiteSpace: "nowrap" };
const selectStyle: CSSProperties = {
  padding: "2px 4px", font: "11px system-ui",
  background: "#1a1a1a", color: "#ddd",
  border: "1px solid #333", borderRadius: 3,
  cursor: "pointer",
};
const stageStyle: CSSProperties = { flex: 1, position: "relative", overflow: "hidden" };
const canvasHostStyle: CSSProperties = { position: "absolute", inset: 0 };
const overlayBase: CSSProperties = {
  position: "absolute", inset: 0,
  display: "flex", alignItems: "center", justifyContent: "center",
  font: "12px ui-monospace,monospace", pointerEvents: "none",
};
const messageStyle: CSSProperties = { ...overlayBase, zIndex: 1, color: "#888" };
const failedStyle: CSSProperties = { ...overlayBase, color: "#f87171", padding: 12, textAlign: "center" };

// ---------- helpers ----------------------------------------------------------

function parseState(raw: unknown): SplatMap | null {
  if (typeof raw === "string") {
    try { raw = JSON.parse(raw); } catch (e) {
      console.warn(TAG, "JSON.parse failed:", e);
      return null;
    }
  }
  if (!raw || typeof raw !== "object") return null;
  const state = raw as { splats?: unknown };
  if (!state.splats || typeof state.splats !== "object") return null;
  return state.splats as SplatMap;
}

function pickHighest(splats: SplatMap | null): SplatResolution | null {
  if (!splats) return null;
  for (const key of PRIORITY) {
    if (splats[key]?.url) return key;
  }
  return null;
}

// Keeps the previous pick while it's still wired, otherwise falls back to the highest-fidelity wired one.
function resolveKey(previous: SplatResolution | null, splats: SplatMap): SplatResolution | null {
  if (previous && splats[previous]?.url) return previous;
  return pickHighest(splats);
}

function getEntry(splats: SplatMap | null, key: SplatResolution | null) {
  if (!splats || !key) return null;
  const entry = splats[key];
  if (!entry || typeof entry.url !== "string" || !entry.url) return null;
  const meta = (entry.meta && typeof entry.meta === "object") ? entry.meta : {};
  const refreshKey = meta.created_at ?? meta.content_hash ?? entry.url;
  return { url: entry.url, refreshKey };
}

function getFileTypeOverride(url: string) {
  if (!url) return undefined;
  const ext = url.split("?")[0]!.slice(url.lastIndexOf(".") + 1).toLowerCase();
  if (ext === "splat") return SplatFileType.SPLAT;
  if (ext === "ksplat") return SplatFileType.KSPLAT;
  return undefined;
}

function findRealWidth(el: HTMLElement) {
  let node: HTMLElement | null = el;
  while (node) {
    const w = node.getBoundingClientRect().width;
    if (w > 1) return Math.round(w);
    node = node.parentElement;
  }
  return 400;
}

// ---------- scene ------------------------------------------------------------

/**
 * Builds a three.js + Spark scene for `url` inside `host`, sized from `stage`, and returns its
 * teardown. Reports what the stage should show through `setStatus`.
 */
function mountScene(
  url: string,
  stage: HTMLElement,
  host: HTMLElement,
  widthAncestor: HTMLElement | null,
  fallbackHeight: () => number,
  setStatus: (status: StageStatus) => void,
): () => void {
  console.info(TAG, `mountScene url="${url.slice(0, 100)}"`);
  setStatus({ kind: "message", text: "Loading splat…" });

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, 0, 2);

  let renderer: THREE.WebGLRenderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
  } catch (e) {
    console.error(TAG, "WebGLRenderer creation failed:", e);
    setStatus({ kind: "message", text: `WebGL error: ${(e as Error)?.message || e}` });
    return () => {};
  }
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.domElement.style.cssText = "display:block; width:100%; height:100%;";
  host.appendChild(renderer.domElement);

  ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
    renderer.domElement.addEventListener(evt, stopPropagation),
  );

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  let controls: SparkControls | undefined;
  try {
    controls = new SparkControls({ canvas: renderer.domElement });
  } catch (e) {
    console.error(TAG, "SparkControls creation failed:", e);
  }

  const resize = () => {
    const w = stage.clientWidth > 1 ? stage.clientWidth : findRealWidth(stage);
    const h = stage.clientHeight || fallbackHeight();
    console.info(TAG, `resize: ${w}x${h}`);
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };

  const ro = new ResizeObserver(resize);
  ro.observe(stage);
  if (widthAncestor) ro.observe(widthAncestor);

  resize();
  const timers = [setTimeout(resize, 0), setTimeout(resize, 200)];

  let disposed = false;
  let splat: FittableSplat;

  const disposeRenderer = () => {
    disposed = true;
    timers.forEach(clearTimeout);
    renderer.setAnimationLoop(null);
    ro.disconnect();
    try { scene.remove(spark); } catch (_) {}
    renderer.dispose();
    renderer.domElement.remove();
  };

  function fitCamera() {
    try {
      let box: THREE.Box3 | null = null;
      if (splat.boundingBox) {
        box = splat.boundingBox.clone();
      } else if (splat.geometry?.boundingBox) {
        box = splat.geometry.boundingBox.clone();
      } else if (typeof splat.geometry?.computeBoundingBox === "function") {
        splat.geometry.computeBoundingBox();
        box = splat.geometry.boundingBox?.clone() ?? null;
      }
      if (!box) { console.warn(TAG, "fitCamera: no bounding box"); return; }
      box.applyMatrix4(splat.matrixWorld);
      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const maxDim = Math.max(size.x, size.y, size.z);
      if (!Number.isFinite(maxDim) || maxDim <= 0) { console.warn(TAG, "fitCamera: maxDim unusable:", maxDim); return; }
      const distance = maxDim * 1.5;
      camera.position.set(center.x, center.y, center.z + distance);
      camera.near = Math.max(distance / 1000, 0.01);
      camera.far = distance * 100;
      camera.updateProjectionMatrix();
      console.info(TAG, `fitCamera: z=${center.z + distance}`);
    } catch (e) {
      console.warn(TAG, "fitCamera error:", e);
    }
  }

  try {
    splat = new SplatMesh({
      url,
      fileType: getFileTypeOverride(url),
      onLoad: () => {
        console.info(TAG, "SplatMesh onLoad");
        if (disposed) return;
        setStatus(null);
        splat.rotation.x = Math.PI;
        fitCamera();
      },
    }) as FittableSplat;
  } catch (e) {
    console.error(TAG, "SplatMesh constructor threw:", e);
    setStatus({ kind: "message", text: `SplatMesh error: ${(e as Error)?.message || e}` });
    return disposeRenderer;
  }

  splat.initialized
    .then(() => { console.info(TAG, "initialized resolved — re-fitting"); if (!disposed) fitCamera(); })
    .catch((err: unknown) => {
      console.error(TAG, "initialized rejected:", err);
      if (disposed) return;
      setStatus({ kind: "failed", text: `Failed to load splat: ${(err as Error)?.message || err}` });
    });

  scene.add(splat);

  renderer.setAnimationLoop(() => {
    if (disposed) return;
    if (controls) controls.update(camera);
    renderer.render(scene, camera);
  });

  return () => {
    console.info(TAG, "sceneCleanup");
    scene.remove(splat);
    if (typeof splat.dispose === "function") try { splat.dispose(); } catch (_) {}
    disposeRenderer();
  };
}

// ---------- component --------------------------------------------------------

function SplatViewer({ value, height: propsHeight }: SplatViewerProps) {
  const height = (propsHeight && propsHeight > 0) ? propsHeight : VIEWER_HEIGHT;
  const splats = useMemo(() => parseState(value) ?? {}, [value]);

  // The user's pick, re-resolved against each new value so it falls back when it's unwired.
  const [pick, setPick] = useState(() => ({ value, key: resolveKey(null, splats) }));
  let selectedKey = pick.key;
  if (pick.value !== value) {
    selectedKey = resolveKey(pick.key, splats);
    setPick({ value, key: selectedKey });
  }

  const entry = getEntry(splats, selectedKey);
  const url = entry?.url ?? null;
  const refreshKey = entry?.refreshKey ?? null;

  const [status, setStatus] = useState<StageStatus>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const hostRef = useRef<HTMLDivElement>(null);
  const heightRef = useRef(height);
  heightRef.current = height;

  useEffect(() => {
    if (!url) return;
    // The widget's container is the wrapper's grandparent (reactWidget renders into a div inside it);
    // its nearest ancestor with a real width is observed too.
    let widthAncestor = wrapperRef.current?.parentElement?.parentElement?.parentElement ?? null;
    while (widthAncestor && widthAncestor.clientWidth <= 1) widthAncestor = widthAncestor.parentElement;
    return mountScene(url, stageRef.current!, hostRef.current!, widthAncestor, () => heightRef.current - 36, setStatus);
  }, [url, refreshKey]);

  useEffect(() => () => console.info(TAG, "cleanup"), []);

  const placeholder = url ? null : (selectedKey
    ? `"${RESOLUTION_LABELS[selectedKey]}" is not wired. Wire it upstream.`
    : "Wire a splat input to render.");

  return (
    <div
      ref={wrapperRef}
      className="splat-viewer-widget nodrag nowheel"
      style={{
        position: "relative", width: "100%", height,
        overflow: "hidden", display: "flex", flexDirection: "column",
      }}
      {...stopNodeEvents}
    >
      <div style={barStyle} {...stopNodeEvents}>
        <span style={labelStyle}>Resolution</span>
        <select
          style={selectStyle}
          value={selectedKey ?? DISPLAY_ORDER[0]}
          onPointerDown={stopPropagation}
          onChange={(e) => {
            const chosen = e.currentTarget.value as SplatResolution;
            console.info(TAG, `picker changed to "${chosen}"`);
            setPick({ value, key: chosen });
          }}
        >
          {DISPLAY_ORDER.map((key) => {
            const wired = !!splats[key]?.url;
            return (
              <option key={key} value={key} disabled={!wired}>
                {RESOLUTION_LABELS[key] + (wired ? "" : " (not wired)")}
              </option>
            );
          })}
        </select>
      </div>
      <div ref={stageRef} style={stageStyle}>
        <div ref={hostRef} style={canvasHostStyle} />
        {placeholder !== null ? (
          <div style={messageStyle}>{placeholder}</div>
        ) : status ? (
          <div style={status.kind === "failed" ? failedStyle : messageStyle}>{status.text}</div>
        ) : null}
      </div>
    </div>
  );
}

export default reactWidget(SplatViewer);
