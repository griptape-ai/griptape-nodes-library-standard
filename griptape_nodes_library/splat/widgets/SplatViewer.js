/**
 * SplatViewer Widget
 *
 * Self-contained Gaussian splat viewer: owns resolution selection, LoD knobs,
 * and the Load trigger button. Receives a `viewer_state` dict from Python:
 *
 *   {
 *     splats: {
 *       "100k":     {url, meta} | null,
 *       "500k":     {url, meta} | null,
 *       "full_res": {url, meta} | null,
 *     },
 *     defaults: { flip_coordinates, enable_lod, lod_scale, max_sh },
 *   }
 *
 * UX flow: user wires splats from upstream or picks local files via the
 * parameter file pickers → widget shows the resolution dropdown (greyed out
 * for resolutions not populated) → user picks a resolution and clicks Load →
 * widget mounts SparkRenderer + SplatMesh and renders. Switching resolution
 * or tweaking options requires another Load click.
 */

import * as THREE from "https://esm.sh/three@0.180.0";
import {
  SparkRenderer,
  SplatMesh,
  SparkControls,
} from "https://esm.sh/@sparkjsdev/spark?deps=three@0.180.0";

const WIDGET_VERSION = "0.5.0";
const VIEWER_HEIGHT = 640;
const STAGE_MESSAGE_CLASS = "splat-stage-overlay-msg";

// Module-level counter to confirm whether the framework is re-invoking the
// widget on value changes. If the widget IS being called multiple times but
// we're treating each as a fresh mount, that's a us bug. If it's only called
// once, that's a framework propagation issue.
let __MOUNT_COUNTER = 0;

const RESOLUTION_KEYS = ["100k", "500k", "full_res"];
const RESOLUTION_LABELS = { "100k": "100k", "500k": "500k", "full_res": "Full Res" };

export default function SplatViewer(container, props) {
  const { value, onChange, disabled } = props;
  const state = parseViewerState(value);

  // UI state — kept in closure so re-renders don't lose user selections.
  let selectedResolution = pickDefaultResolution(state);
  let options = { ...state.defaults };
  let viewerCleanup = null;
  let loadedFor = null; // {resolution, options} of the currently mounted viewer

  __MOUNT_COUNTER += 1;
  const myInvocation = __MOUNT_COUNTER;
  console.log(`[SplatViewer] invocation #${myInvocation}`, {
    valueType: typeof value,
    splatsWired: state.splats ? Object.entries(state.splats).filter(([, v]) => v).map(([k]) => k) : [],
    diagCount: (state.diag || []).length,
    selectedResolution,
  });

  // resolutionSelect is exposed so applyUpdatedState can patch it in-place.
  let resolutionSelect = null;

  const wrapper = el("div", {
    className: "splat-viewer-lab nodrag nowheel",
    style: `
      display: flex; flex-direction: column;
      width: 100%;
      background: #0a0a0a;
      border-radius: 6px;
      overflow: hidden;
      user-select: none;
      box-sizing: border-box;
    `,
  });
  ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
    wrapper.addEventListener(evt, (e) => e.stopPropagation()),
  );
  container.innerHTML = "";
  container.appendChild(wrapper);

  // Build sections.
  const controls = el("div", {
    style: `
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 8px; padding: 10px;
      background: #141414;
      border-bottom: 1px solid #222;
    `,
  });
  controls.appendChild(buildResolutionPicker());
  controls.appendChild(buildLodScaleControl());
  controls.appendChild(buildMaxShControl());
  controls.appendChild(buildFlipControl());
  controls.appendChild(buildEnableLodControl());
  controls.appendChild(buildLoadButton());
  wrapper.appendChild(controls);

  const stage = el("div", {
    className: "splat-viewer-stage",
    style: `
      position: relative;
      width: 100%;
      height: ${VIEWER_HEIGHT}px;
      background: #050505;
    `,
  });
  wrapper.appendChild(stage);

  showStageMessage(
    selectedResolution
      ? `Click "Load" to render ${RESOLUTION_LABELS[selectedResolution]} splat.`
      : "Wire a splat input upstream, then pick a resolution and click Load.",
  );

  // Diagnostics panel — visible when there's a `diag` array on the value.
  // Header line shows widget invocation count: if this stays at #1 after
  // wiring, the framework isn't re-invoking us; if it climbs, the framework
  // IS calling us with new data.
  const dbg = el("div", {
    style: `
      font: 10px ui-monospace, monospace; color: #6c6;
      padding: 6px 10px; background: #0f0f0f; border-top: 1px solid #222;
      max-height: 110px; overflow-y: auto; white-space: pre;
    `,
  });
  const lines = [`widget invocation #${myInvocation}, diag entries: ${(state.diag || []).length}`];
  if (Array.isArray(state.diag) && state.diag.length > 0) {
    state.diag.forEach((d, i) => lines.push(`${i + 1}. ${JSON.stringify(d)}`));
  } else {
    lines.push("(no connection/value events recorded yet)");
  }
  dbg.textContent = lines.join("\n");
  wrapper.appendChild(dbg);

  // ---------- controls -----------------------------------------------------

  function buildResolutionPicker() {
    const w = el("div", { style: "display: flex; flex-direction: column; gap: 4px;" });
    w.appendChild(labelText("Resolution"));
    const sel = el("select", { style: selectStyle() });
    resolutionSelect = sel;
    sel.disabled = !!disabled;
    populateResolutionOptions(sel, state.splats);
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    sel.addEventListener("change", () => {
      selectedResolution = sel.value;
    });
    w.appendChild(sel);
    return w;
  }

  function populateResolutionOptions(sel, splats) {
    const currentVal = sel.value || selectedResolution;
    sel.innerHTML = "";
    RESOLUTION_KEYS.forEach((key) => {
      const opt = document.createElement("option");
      opt.value = key;
      const wired = !!(splats && splats[key]);
      opt.textContent = `${RESOLUTION_LABELS[key]}${wired ? "" : "  (not wired)"}`;
      opt.disabled = !wired;
      if (key === currentVal) opt.selected = true;
      sel.appendChild(opt);
    });
    // If saved selection is now wired but wasn't before, keep it; otherwise pick best available.
    if (!splats[sel.value]) {
      const best = pickDefaultResolution({ splats });
      if (best) sel.value = best;
      selectedResolution = sel.value;
    }
  }

  function buildLodScaleControl() {
    const w = el("div", { style: "display: flex; flex-direction: column; gap: 4px;" });
    const lbl = labelText(`LoD Scale: ${options.lod_scale.toFixed(2)}`);
    w.appendChild(lbl);
    const input = el("input", {
      type: "range",
      min: "0.25",
      max: "2.0",
      step: "0.05",
      value: String(options.lod_scale),
      style: "width: 100%;",
    });
    input.disabled = !!disabled;
    input.addEventListener("pointerdown", (e) => e.stopPropagation());
    input.addEventListener("input", (e) => {
      options.lod_scale = parseFloat(e.target.value);
      lbl.textContent = `LoD Scale: ${options.lod_scale.toFixed(2)}`;
    });
    w.appendChild(input);
    return w;
  }

  function buildMaxShControl() {
    const w = el("div", { style: "display: flex; flex-direction: column; gap: 4px;" });
    w.appendChild(labelText("Max SH"));
    const sel = el("select", { style: selectStyle() });
    sel.disabled = !!disabled;
    [0, 1, 2, 3].forEach((n) => {
      const opt = document.createElement("option");
      opt.value = String(n);
      opt.textContent = `SH${n}${n === 3 ? " (full)" : n === 0 ? " (base color)" : ""}`;
      if (n === options.max_sh) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.addEventListener("pointerdown", (e) => e.stopPropagation());
    sel.addEventListener("change", () => {
      options.max_sh = parseInt(sel.value, 10);
    });
    w.appendChild(sel);
    return w;
  }

  function buildFlipControl() {
    return buildCheckbox("Flip Y/Z (Marble OpenCV)", "flip_coordinates");
  }

  function buildEnableLodControl() {
    return buildCheckbox("Enable LoD", "enable_lod");
  }

  function buildCheckbox(labelTextStr, key) {
    const w = el("label", {
      style: `
        display: flex; align-items: center; gap: 8px;
        font: 12px system-ui; color: #ccc;
        padding-top: 16px; cursor: pointer;
      `,
    });
    const cb = el("input", { type: "checkbox", style: "cursor: pointer;" });
    cb.checked = !!options[key];
    cb.disabled = !!disabled;
    cb.addEventListener("pointerdown", (e) => e.stopPropagation());
    cb.addEventListener("change", () => {
      options[key] = cb.checked;
    });
    const txt = document.createElement("span");
    txt.textContent = labelTextStr;
    w.appendChild(cb);
    w.appendChild(txt);
    return w;
  }

  function buildLoadButton() {
    const w = el("div", { style: "grid-column: 1 / -1; display: flex; gap: 8px; align-items: center;" });
    const btn = el("button", {
      type: "button",
      style: `
        flex: 1; padding: 8px 12px;
        background: #7c3aed; color: #fff;
        border: 1px solid #7c3aed; border-radius: 4px;
        font: 13px system-ui; cursor: pointer;
        opacity: ${disabled ? 0.5 : 1};
      `,
    });
    btn.textContent = "Load";
    btn.disabled = !!disabled;
    btn.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      if (btn.disabled) return;
      const entry = state.splats && state.splats[selectedResolution];
      if (!entry || !entry.url) {
        showStageMessage(
          `${RESOLUTION_LABELS[selectedResolution] || "Selected"} splat is not wired. Wire it from upstream.`,
          "warn",
        );
        return;
      }
      loadInto(entry.url, { ...options }, selectedResolution);
    });
    const status = el("div", {
      style: "font: 11px ui-monospace, monospace; color: #888;",
    });
    status.textContent = `widget v${WIDGET_VERSION}`;
    w.appendChild(btn);
    w.appendChild(status);
    return w;
  }

  // ---------- viewer mount -------------------------------------------------

  function loadInto(url, opts, resolution) {
    if (viewerCleanup) {
      try { viewerCleanup(); } catch (_) { /* noop */ }
      viewerCleanup = null;
    }
    stage.innerHTML = "";
    showStageMessage(`Loading ${RESOLUTION_LABELS[resolution] || ""} splat...`);
    viewerCleanup = mountViewer(stage, url, opts);
    loadedFor = { resolution, options: opts };
  }

  function showStageMessage(text, kind) {
    stage.innerHTML = "";
    const div = el("div", {
      className: STAGE_MESSAGE_CLASS,
      style: `
        position: absolute; inset: 0;
        display: flex; align-items: center; justify-content: center;
        color: ${kind === "warn" ? "#fbbf24" : "#888"};
        font: 12px ui-monospace, monospace;
        pointer-events: none;
        white-space: pre-line; text-align: center; padding: 12px;
      `,
    });
    div.textContent = text;
    stage.appendChild(div);
  }

  // Track the last applied value string so `update` can short-circuit cheaply.
  let lastAppliedValueJson = typeof value === "string" ? value : null;

  function applyUpdatedState(newState) {
    // Update live splats reference so the Load button always uses the latest.
    Object.assign(state.splats, newState.splats || {});

    // Patch resolution dropdown without resetting user's current choice.
    if (resolutionSelect) {
      populateResolutionOptions(resolutionSelect, state.splats);
    }

    // Update diag panel.
    const newDiag = Array.isArray(newState.diag) ? newState.diag : [];
    state.diag = newDiag;
    const newLines = [`diag entries: ${newDiag.length} (live)`];
    if (newDiag.length > 0) {
      newDiag.forEach((d, i) => newLines.push(`${i + 1}. ${JSON.stringify(d)}`));
    } else {
      newLines.push("(no events yet)");
    }
    dbg.textContent = newLines.join("\n");
  }

  function update(nextProps) {
    const next = typeof nextProps?.value === "string" ? nextProps.value : null;
    if (!next || next === lastAppliedValueJson) return;
    lastAppliedValueJson = next;
    let parsed;
    try { parsed = parseViewerState(next); } catch (_) { return; }
    applyUpdatedState(parsed);
  }

  return {
    cleanup() {
      if (viewerCleanup) {
        try { viewerCleanup(); } catch (_) { /* noop */ }
        viewerCleanup = null;
      }
      container.innerHTML = "";
    },
    update,
  };
}

// ---------------------------------------------------------------------------
// Spark mount (canvas + SplatMesh + controls)
// ---------------------------------------------------------------------------

function mountViewer(stage, url, opts) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, 0, 8);

  const spark = new SparkRenderer({ renderer });
  scene.add(spark);

  const canvas = renderer.domElement;
  canvas.style.cssText = "display:block; width:100%; height:100%;";
  stage.appendChild(canvas);
  ["pointerdown", "mousedown", "wheel", "keydown"].forEach((evt) =>
    canvas.addEventListener(evt, (e) => e.stopPropagation()),
  );

  const resize = () => {
    const w = stage.clientWidth || 1;
    const h = stage.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  resize();
  const ro = new ResizeObserver(resize);
  ro.observe(stage);

  let splat = null;
  let disposed = false;

  function clearOverlay() {
    if (disposed) return;
    stage.querySelectorAll(`.${STAGE_MESSAGE_CLASS}`).forEach((n) => n.remove());
  }

  try {
    const splatOpts = {
      url,
      onLoad: () => {
        clearOverlay();
        try { fitCameraToSplat(camera, splat); } catch (_) { /* noop */ }
      },
    };
    if (opts.enable_lod !== false) splatOpts.lod = true;
    if (typeof opts.lod_scale === "number" && opts.lod_scale !== 1.0) {
      splatOpts.lodScale = opts.lod_scale;
    }
    splat = new SplatMesh(splatOpts);
    if (typeof opts.max_sh === "number" && opts.max_sh >= 0 && opts.max_sh <= 3) {
      splat.maxSh = opts.max_sh;
    }
    if (opts.flip_coordinates) {
      splat.scale.set(1, -1, -1);
    }
    scene.add(splat);

    // Also await .initialized so errors surface in the console.
    splat.initialized.catch((err) => {
      if (!disposed) {
        console.error("[SplatViewer] load failed:", err);
        clearOverlay();
        stage.innerHTML = "";
        const errDiv = document.createElement("div");
        errDiv.style.cssText = "position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:#f87171;font:12px ui-monospace,monospace;padding:12px;text-align:center;";
        errDiv.textContent = `Failed to load splat: ${err?.message || err}`;
        stage.appendChild(errDiv);
      }
    });
  } catch (err) {
    console.error("[SplatViewer] init failed:", err);
  }

  const controls = new SparkControls({ canvas });

  renderer.setAnimationLoop(() => {
    if (disposed) return;
    try { controls.update(camera); } catch (_) { /* noop */ }
    renderer.render(scene, camera);
  });

  return () => {
    disposed = true;
    try { renderer.setAnimationLoop(null); } catch (_) { /* noop */ }
    try { ro.disconnect(); } catch (_) { /* noop */ }
    if (splat) {
      try {
        scene.remove(splat);
        if (typeof splat.dispose === "function") splat.dispose();
      } catch (_) { /* noop */ }
    }
    try { scene.remove(spark); } catch (_) { /* noop */ }
    try { renderer.dispose(); } catch (_) { /* noop */ }
    if (canvas.parentNode === stage) stage.removeChild(canvas);
  };
}

function fitCameraToSplat(camera, splat) {
  let box = null;
  if (splat.boundingBox) {
    box = splat.boundingBox.clone();
  } else if (splat.geometry && splat.geometry.boundingBox) {
    box = splat.geometry.boundingBox.clone();
  } else if (splat.geometry && typeof splat.geometry.computeBoundingBox === "function") {
    splat.geometry.computeBoundingBox();
    box = splat.geometry.boundingBox ? splat.geometry.boundingBox.clone() : null;
  }
  if (!box) return;
  box.applyMatrix4(splat.matrixWorld);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(maxDim) || maxDim <= 0) return;
  const fov = (camera.fov * Math.PI) / 180;
  const distance = (maxDim / 2) / Math.tan(fov / 2);
  const padded = distance * 1.6;
  camera.position.copy(center.clone().add(new THREE.Vector3(0, 0, padded)));
  camera.near = Math.max(padded / 1000, 0.01);
  camera.far = padded * 100;
  camera.updateProjectionMatrix();
  camera.lookAt(center);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parseViewerState(value) {
  const empty = {
    splats: { "100k": null, "500k": null, "full_res": null },
    defaults: { flip_coordinates: true, enable_lod: true, lod_scale: 1.0, max_sh: 3 },
    diag: [],
  };
  // viewer_state is now a JSON-serialized string (Python switched away from
  // type="dict" because dict-typed params don't seem to push value updates to
  // mounted widgets at edit time). Parse it on this end.
  if (typeof value === "string") {
    if (!value) return empty;
    try {
      value = JSON.parse(value);
    } catch (_) {
      return empty;
    }
  }
  if (!value || typeof value !== "object") return empty;
  const splatsIn = value.splats && typeof value.splats === "object" ? value.splats : {};
  const defaultsIn = value.defaults && typeof value.defaults === "object" ? value.defaults : {};
  return {
    splats: {
      "100k": normalizeSplatEntry(splatsIn["100k"]),
      "500k": normalizeSplatEntry(splatsIn["500k"]),
      "full_res": normalizeSplatEntry(splatsIn["full_res"]),
    },
    defaults: {
      flip_coordinates:
        typeof defaultsIn.flip_coordinates === "boolean" ? defaultsIn.flip_coordinates : true,
      enable_lod:
        typeof defaultsIn.enable_lod === "boolean" ? defaultsIn.enable_lod : true,
      lod_scale:
        typeof defaultsIn.lod_scale === "number" && Number.isFinite(defaultsIn.lod_scale)
          ? defaultsIn.lod_scale
          : 1.0,
      max_sh:
        typeof defaultsIn.max_sh === "number" && Number.isFinite(defaultsIn.max_sh)
          ? Math.max(0, Math.min(3, Math.round(defaultsIn.max_sh)))
          : 3,
    },
    diag: Array.isArray(value.diag) ? value.diag : [],
  };
}

function normalizeSplatEntry(entry) {
  if (!entry || typeof entry !== "object") return null;
  const url = typeof entry.url === "string" ? entry.url : null;
  if (!url) return null;
  const meta = entry.meta && typeof entry.meta === "object" ? entry.meta : {};
  return { url, meta };
}

function pickDefaultResolution(state) {
  if (state.splats["100k"]) return "100k";
  if (state.splats["500k"]) return "500k";
  if (state.splats["full_res"]) return "full_res";
  return "100k";
}

function selectStyle() {
  return `
    padding: 4px 6px; font: 12px system-ui;
    background: #0e0e0e; color: #ddd;
    border: 1px solid #333; border-radius: 4px;
    cursor: pointer;
  `;
}

function labelText(text) {
  return el("div", {
    style: "font: 11px system-ui; color: #888;",
    textContent: text,
  });
}

function el(tag, attrs) {
  const node = document.createElement(tag);
  if (!attrs) return node;
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "style") node.style.cssText = v;
    else if (k === "textContent") node.textContent = v;
    else if (k === "className") node.className = v;
    else if (k === "disabled") node.disabled = !!v;
    else node.setAttribute(k, v);
  }
  return node;
}

SplatViewer.WIDGET_VERSION = WIDGET_VERSION;
