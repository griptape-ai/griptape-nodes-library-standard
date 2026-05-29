import * as THREE from "https://esm.sh/three@0.180.0";
import {
  SplatFileType,
  SplatMesh,
  SparkControls,
  SparkRenderer,
} from "https://esm.sh/@sparkjsdev/spark?deps=three@0.180.0";

const VIEWER_HEIGHT = 640;
const PRIORITY = ["full_res", "500k", "100k"];      // auto-select: highest fidelity first
const DISPLAY_ORDER = ["100k", "500k", "full_res"];  // dropdown: matches parameter order
const RESOLUTION_LABELS = { "100k": "100k", "500k": "500k", "full_res": "Full Res" };
const TAG = "[SplatViewer]";

export default function SplatViewer(container, props) {
  console.info(TAG, "mount called", {
    valueSnippet: typeof props.value === "string" ? props.value.slice(0, 120) : props.value,
    propsHeight: props.height,
    containerSize: { w: container.clientWidth, h: container.clientHeight },
  });

  const { value } = props;
  const height = (props.height && props.height > 0) ? props.height : VIEWER_HEIGHT;

  let currentUrl = null;
  let currentRefreshKey = null;
  let sceneCleanup = null;
  let selectedKey = null;   // user's current picker selection
  let currentSplats = {};   // latest parsed splats map
  let selectEl = null;      // reference to the <select> element

  // ---------- outer wrapper -------------------------------------------------

  const wrapper = document.createElement("div");
  wrapper.className = "splat-viewer-widget nodrag nowheel";
  wrapper.style.cssText = `
    position:relative; width:100%; height:${height}px;
    overflow:hidden; display:flex; flex-direction:column;
  `;
  ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
    wrapper.addEventListener(evt, (e) => e.stopPropagation()),
  );
  container.innerHTML = "";
  container.appendChild(wrapper);

  // ---------- picker bar ---------------------------------------------------

  const pickerBar = document.createElement("div");
  pickerBar.style.cssText = `
    display:flex; align-items:center; gap:8px; padding:4px 8px;
    background:rgba(0,0,0,0.35); border-bottom:1px solid rgba(255,255,255,0.08);
    flex-shrink:0; z-index:2;
  `;
  ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
    pickerBar.addEventListener(evt, (e) => e.stopPropagation()),
  );

  const pickerLabel = document.createElement("span");
  pickerLabel.style.cssText = "font:11px system-ui; color:#999; white-space:nowrap;";
  pickerLabel.textContent = "Resolution";
  pickerBar.appendChild(pickerLabel);

  selectEl = document.createElement("select");
  selectEl.style.cssText = `
    padding:2px 4px; font:11px system-ui;
    background:#1a1a1a; color:#ddd;
    border:1px solid #333; border-radius:3px;
    cursor:pointer;
  `;
  DISPLAY_ORDER.forEach((key) => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = RESOLUTION_LABELS[key];
    selectEl.appendChild(opt);
  });
  selectEl.addEventListener("pointerdown", (e) => e.stopPropagation());
  selectEl.addEventListener("change", () => {
    const chosen = selectEl.value;
    console.info(TAG, `picker changed to "${chosen}"`);
    selectedKey = chosen;
    loadSelected();
  });
  pickerBar.appendChild(selectEl);
  wrapper.appendChild(pickerBar);

  // ---------- stage (canvas lives here) ------------------------------------

  const stage = document.createElement("div");
  stage.style.cssText = "flex:1; position:relative; overflow:hidden;";
  wrapper.appendChild(stage);

  // ---------- helpers -------------------------------------------------------

  function parseState(raw) {
    if (typeof raw === "string") {
      try { raw = JSON.parse(raw); } catch (e) {
        console.warn(TAG, "JSON.parse failed:", e);
        return null;
      }
    }
    if (!raw || typeof raw !== "object") return null;
    if (!raw.splats || typeof raw.splats !== "object") return null;
    return raw.splats;
  }

  function pickHighest(splats) {
    if (!splats) return null;
    for (const key of PRIORITY) {
      if (splats[key]?.url) return key;
    }
    return null;
  }

  function getEntry(splats, key) {
    if (!splats || !key) return null;
    const entry = splats[key];
    if (!entry || typeof entry.url !== "string" || !entry.url) return null;
    const meta = (entry.meta && typeof entry.meta === "object") ? entry.meta : {};
    const refreshKey = meta.created_at ?? meta.content_hash ?? entry.url;
    return { url: entry.url, refreshKey };
  }

  function getFileTypeOverride(url) {
    if (!url) return undefined;
    const ext = url.split("?")[0].slice(url.lastIndexOf(".") + 1).toLowerCase();
    if (ext === "splat") return SplatFileType.SPLAT;
    if (ext === "ksplat") return SplatFileType.KSPLAT;
    return undefined;
  }

  function findRealWidth(el) {
    let node = el;
    while (node) {
      const w = node.getBoundingClientRect().width;
      if (w > 1) return Math.round(w);
      node = node.parentElement;
    }
    return 400;
  }

  // ---------- dropdown state ------------------------------------------------

  function refreshDropdown(splats) {
    Array.from(selectEl.options).forEach((opt) => {
      const wired = !!(splats && splats[opt.value]?.url);
      opt.disabled = !wired;
      opt.textContent = RESOLUTION_LABELS[opt.value] + (wired ? "" : " (not wired)");
    });
    if (selectedKey) selectEl.value = selectedKey;
  }

  // ---------- placeholder ---------------------------------------------------

  function showPlaceholder(text) {
    console.info(TAG, "showPlaceholder:", text);
    teardownScene();
    stage.innerHTML = "";
    const msg = document.createElement("div");
    msg.style.cssText = `
      position:absolute; inset:0;
      display:flex; align-items:center; justify-content:center;
      color:#888; font:12px ui-monospace,monospace; pointer-events:none;
    `;
    msg.textContent = text;
    stage.appendChild(msg);
  }

  // ---------- scene lifecycle -----------------------------------------------

  function teardownScene() {
    if (sceneCleanup) {
      try { sceneCleanup(); } catch (_) {}
      sceneCleanup = null;
    }
  }

  function mountScene(url) {
    console.info(TAG, `mountScene url="${url.slice(0, 100)}"`);
    teardownScene();
    stage.innerHTML = "";

    const overlay = document.createElement("div");
    overlay.style.cssText = `
      position:absolute; inset:0; z-index:1;
      display:flex; align-items:center; justify-content:center;
      color:#888; font:12px ui-monospace,monospace; pointer-events:none;
    `;
    overlay.textContent = "Loading splat…";
    stage.appendChild(overlay);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
    camera.position.set(0, 0, 2);

    let renderer;
    try {
      renderer = new THREE.WebGLRenderer({ antialias: false, alpha: true });
    } catch (e) {
      console.error(TAG, "WebGLRenderer creation failed:", e);
      overlay.textContent = `WebGL error: ${e?.message || e}`;
      return;
    }
    renderer.setClearColor(0x000000, 0);
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.domElement.style.cssText = "display:block; width:100%; height:100%;";
    stage.appendChild(renderer.domElement);

    ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
      renderer.domElement.addEventListener(evt, (e) => e.stopPropagation()),
    );

    const spark = new SparkRenderer({ renderer });
    scene.add(spark);

    let controls;
    try {
      controls = new SparkControls({ canvas: renderer.domElement });
    } catch (e) {
      console.error(TAG, "SparkControls creation failed:", e);
    }

    const resize = () => {
      const w = stage.clientWidth > 1 ? stage.clientWidth : findRealWidth(stage);
      const h = stage.clientHeight || (height - 36);
      console.info(TAG, `resize: ${w}x${h}`);
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };

    const ro = new ResizeObserver(resize);
    ro.observe(stage);
    const realWidthAncestor = (() => {
      let n = container.parentElement;
      while (n) { if (n.clientWidth > 1) return n; n = n.parentElement; }
      return null;
    })();
    if (realWidthAncestor) ro.observe(realWidthAncestor);

    resize();
    setTimeout(resize, 0);
    setTimeout(resize, 200);

    let disposed = false;
    let splat;

    function fitCamera() {
      try {
        let box = null;
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
          if (overlay.parentNode) overlay.remove();
          splat.rotation.x = Math.PI;
          fitCamera();
        },
      });
    } catch (e) {
      console.error(TAG, "SplatMesh constructor threw:", e);
      overlay.textContent = `SplatMesh error: ${e?.message || e}`;
      return;
    }

    splat.initialized
      .then(() => { console.info(TAG, "initialized resolved — re-fitting"); if (!disposed) fitCamera(); })
      .catch((err) => {
        console.error(TAG, "initialized rejected:", err);
        if (disposed) return;
        if (overlay.parentNode) overlay.remove();
        const errDiv = document.createElement("div");
        errDiv.style.cssText = `
          position:absolute; inset:0;
          display:flex; align-items:center; justify-content:center;
          color:#f87171; font:12px ui-monospace,monospace;
          padding:12px; text-align:center; pointer-events:none;
        `;
        errDiv.textContent = `Failed to load splat: ${err?.message || err}`;
        stage.appendChild(errDiv);
      });

    scene.add(splat);

    renderer.setAnimationLoop(() => {
      if (disposed) return;
      if (controls) controls.update(camera);
      renderer.render(scene, camera);
    });

    sceneCleanup = () => {
      console.info(TAG, "sceneCleanup");
      disposed = true;
      renderer.setAnimationLoop(null);
      ro.disconnect();
      scene.remove(splat);
      if (typeof splat.dispose === "function") try { splat.dispose(); } catch (_) {}
      try { scene.remove(spark); } catch (_) {}
      renderer.dispose();
      if (renderer.domElement.parentNode) renderer.domElement.parentNode.removeChild(renderer.domElement);
    };
  }

  // ---------- load the currently selected resolution -----------------------

  function loadSelected() {
    const entry = getEntry(currentSplats, selectedKey);
    if (!entry) {
      showPlaceholder(selectedKey
        ? `"${RESOLUTION_LABELS[selectedKey]}" is not wired. Wire it upstream.`
        : "Wire a splat input to render.");
      currentUrl = null;
      currentRefreshKey = null;
      return;
    }
    if (entry.url === currentUrl && entry.refreshKey === currentRefreshKey) {
      console.info(TAG, "loadSelected: short-circuit");
      return;
    }
    currentUrl = entry.url;
    currentRefreshKey = entry.refreshKey;
    mountScene(entry.url);
  }

  // ---------- state application ---------------------------------------------

  function applyState(raw) {
    console.info(TAG, "applyState");
    const splats = parseState(raw) ?? {};
    currentSplats = splats;

    // Refresh dropdown enabled/disabled state.
    refreshDropdown(splats);

    // Determine which key to show:
    // - If previously selected key is still wired, keep it.
    // - Otherwise fall back to highest-fidelity wired.
    if (!selectedKey || !splats[selectedKey]?.url) {
      selectedKey = pickHighest(splats);
      if (selectedKey) selectEl.value = selectedKey;
      console.info(TAG, `applyState: resolved selectedKey="${selectedKey}"`);
    }

    loadSelected();
  }

  applyState(value);

  return {
    cleanup() {
      console.info(TAG, "cleanup");
      teardownScene();
      container.innerHTML = "";
    },
    update(nextProps) {
      console.info(TAG, "update");
      applyState(nextProps?.value);
    },
  };
}
