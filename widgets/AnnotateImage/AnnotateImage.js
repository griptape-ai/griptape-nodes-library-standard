// AnnotateImageSimple — single-image annotation widget
// Layout: toolbar (tools left, settings right) + canvas
// Tools: Select, Paint, Text, Arrow

import { ICON_PATHS, mkIcon } from './_icons.js';
import {
  injectStyles, defaultData,
  DEFAULT_CANVAS_WIDTH, DEFAULT_CANVAS_HEIGHT, DEFAULT_COLOR,
  DEFAULT_PAINT_SIZE, MIN_PAINT_SIZE, MAX_PAINT_SIZE,
  DEFAULT_TEXT_SIZE,  MIN_TEXT_SIZE,  MAX_TEXT_SIZE,
  DEFAULT_ARROW_WIDTH, MIN_ARROW_WIDTH, MAX_ARROW_WIDTH,
  DEFAULT_SHAPE_WIDTH, MIN_SHAPE_WIDTH, MAX_SHAPE_WIDTH,
  SEL_COLOR, SEL_COLOR_RGB, IMP_COLOR, IMP_COLOR_RGB,
  FRAME_FILL_OPACITY, FRAME_BORDER_OPACITY, FRAME_CORNER_OPACITY,
  FRAME_ROT_STEM_OPACITY, ROT_HANDLE_INNER_OPACITY,
  HOVER_OPACITY, LASSO_FILL_OPACITY, LASSO_STROKE_OPACITY, LAYER_HOVER_OPACITY,
  HANDLE_FILL, HANDLE_STROKE_OPACITY,
  LINE_WIDTH_PRIMARY, LINE_WIDTH_SECONDARY, LINE_WIDTH_TERTIARY,
  HANDLE_RADIUS, ROT_HANDLE_RADIUS, CORNER_TICK_LEN,
  DASH_ROT_STEM, DASH_LASSO,
} from './_styles.js';
import {
  decimatePoints,
  strokeBounds, naturalBounds, paintCenter,
  paintTransformPt, paintInvTransformPt, getTransformedCorners,
  defaultCps, snapshotAnn,
  frameCorners, frameTopMid, frameRotHandle,
} from './_geometry.js';
import { createDrawing } from './_drawing.js';
import { createTooltip } from './_tooltip.js';
import { setupHotkeys } from './_hotkeys.js';
import { createSettings } from './_settings.js';
import { createHud, expandGroupSelection } from './_hud.js';

// ── main widget ───────────────────────────────────────────────────────────────

export default function AnnotateImageSimple(container, props) {
  if (container._aisInst?.wrapper?.isConnected) {
    container._aisInst.handleUpdate(props);
    return { cleanup: container._aisInst.cleanup, update: container._aisInst.handleUpdate };
  }

  injectStyles();

  const { onChange } = props;
  const rawValue = (props.value && typeof props.value === "object") ? props.value : {};
  const defTS = defaultData().tool_settings;
  const rawTS = rawValue.tool_settings || {};
  let currentValue = { ...defaultData(), ...rawValue, tool_settings: {
    ...defTS,
    ...rawTS,
    // Deep-merge each tool so new default fields (e.g. taper) survive alongside stored values.
    // Migrate arrow width: old default was 3, new default is 8 — upgrade silently.
    arrow:   { ...defTS.arrow,   ...(rawTS.arrow   || {}), width: (rawTS.arrow?.width === 3 ? 8 : (rawTS.arrow?.width ?? defTS.arrow.width)) },
    rect:    { ...defTS.rect,    ...(rawTS.rect    || {}) },
    ellipse: { ...defTS.ellipse, ...(rawTS.ellipse || {}) },
  } };
  // Migrate old selected_id (single string) to selected_ids (array)
  if (rawValue.selected_id && !rawValue.selected_ids) {
    currentValue.selected_ids = [rawValue.selected_id];
  } else if (!Array.isArray(currentValue.selected_ids)) {
    currentValue.selected_ids = [];
  }

  let activeTool = currentValue.active_tool || "select";
  let toolSettings = { ...currentValue.tool_settings };
  let displayScale = 1;
  let centerOffsetX = 0, centerOffsetY = 0;

  // zoom / pan state
  let viewScale = 1;
  let panX = 0, panY = 0;
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let isAltHeld = false;
  let resetViewBtn = null;

  // ── Tooltip system ────────────────────────────────────────────────────────
  const _tooltip = createTooltip();
  const _addTooltip = _tooltip.addTooltip;

  // unified transform frame (OBB)
  let txFrame = null; // { pivotX, pivotY, rotation, halfW, halfH }
  const _frameActiveTools = ["select", "paint", "rect", "ellipse"];

  // pointer state
  let isPointerDown = false;
  let _mouseIsOver = false;
  let currentStroke = null;
  let currentArrow = null;
  let currentRect = null;
  let currentEllipse = null;
  let dragState = null;
  let lastPtTime = 0, lastPtX = 0, lastPtY = 0, velSmoothed = 0;

  // text edit state
  let textInput = null;
  let textEditId = null;
  let hoverId = null;       // annotation id being hovered
  let hoverGroupId = null;  // group_id of hoverId's annotation (null if ungrouped)
  let marqueePreviewIds = null; // ids that would be selected by the current marquee drag

  // ── import/override helpers ───────────────────────────────────────────────

  // True if id belongs to an imported annotation (not created locally in this session).
  function _isImported(id) {
    return (currentValue.imported_annotations || []).some((a) => a.id === id);
  }

  // Returns imported (with overrides applied, deleted ones skipped) + local annotations.
  // This is the authoritative list for rendering, hit-testing, and selection.
  function _effectiveAnnotations() {
    const imported = currentValue.imported_annotations || [];
    const overrides = currentValue.overrides || {};
    const local = currentValue.annotations || [];
    const merged = imported
      .filter((a) => !overrides[a.id]?.deleted)
      .map((a) => ({ ...a, ...overrides[a.id], _imported: true }));
    return [...merged, ...local];
  }

  // Apply mapFn to the given annotation ids, routing imported ones to overrides.
  function _applyAnnotationMap(selIds, mapFn) {
    const newAnnotations = (currentValue.annotations || []).map((a) => {
      if (!selIds.includes(a.id)) return a;
      return mapFn(a);
    });
    const newOverrides = { ...(currentValue.overrides || {}) };
    for (const imp of (currentValue.imported_annotations || [])) {
      if (!selIds.includes(imp.id)) continue;
      const existing = newOverrides[imp.id] || {};
      const merged = { ...imp, ...existing };
      const updated = mapFn(merged);
      // Store only fields that actually differ from the original imported annotation.
      // This keeps overrides minimal — moving shouldn't override text, color, etc.
      const newOverride = {};
      for (const key of Object.keys(updated)) {
        if (key.startsWith("_")) continue; // skip internal flags like _imported
        if (updated[key] !== imp[key]) newOverride[key] = updated[key];
      }
      newOverrides[imp.id] = newOverride;
    }
    return { annotations: newAnnotations, overrides: newOverrides };
  }

  // Update a single annotation by id, routing imported ones to overrides.
  function _applySingleUpdate(id, mapFn) {
    const { annotations, overrides } = _applyAnnotationMap([id], mapFn);
    currentValue = { ...currentValue, annotations, overrides };
  }

  // Delete a single annotation by id: soft-delete imported ones via override, remove local ones.
  function _deleteAnnotations(ids) {
    const newOverrides = { ...(currentValue.overrides || {}) };
    const importedIds = new Set((currentValue.imported_annotations || []).map((a) => a.id));
    for (const id of ids) {
      if (importedIds.has(id)) {
        newOverrides[id] = { ...(newOverrides[id] || {}), deleted: true };
      }
    }
    currentValue = {
      ...currentValue,
      annotations: (currentValue.annotations || []).filter((a) => !ids.includes(a.id) || importedIds.has(a.id)),
      overrides: newOverrides,
    };
  }

  // image cache — strip query strings so the same base URL always hits the same entry
  const imageCache = {};
  function urlCacheKey(url) { return url ? url.split("?")[0] : url; }
  // Loads an image from url, caches it, and returns a promise resolving to the HTMLImageElement.
  function loadImage(url) {
    const key = urlCacheKey(url);
    return new Promise((resolve, reject) => {
      if (imageCache[key]) { resolve(imageCache[key]); return; }
      const img = new window.Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { imageCache[key] = img; resolve(img); };
      img.onerror = reject;
      img.src = url;
    });
  }

  // ── DOM ───────────────────────────────────────────────────────────────────

  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel ais-wrapper";
  wrapper.style.cssText =
    "display:flex;flex-direction:column;width:100%;height:100%;background:var(--background);border-radius:6px;" +
    "font-family:sans-serif;box-sizing:border-box;overflow:hidden;";

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText =
    "display:flex;align-items:center;gap:4px;padding:5px 8px;" +
    "background:var(--card);border-bottom:1px solid var(--border);flex-shrink:0;flex-wrap:wrap;min-height:38px;";

  // Tool buttons — navigation group
  const NAV_TOOLS = [
    { id: "select",  title: "Select & Move  [V]" },
    { id: "hand",    title: "Pan  [H]" },
    { id: "zoom",    title: "Zoom  [Z]" },
  ];
  // Tool buttons — drawing group
  const DRAW_TOOLS = [
    { id: "paint",   title: "Draw  [D]" },
    { id: "text",    title: "Text  [T]" },
    { id: "arrow",   title: "Arrow  [L]" },
    { id: "rect",    title: "Rectangle  [R]" },
    { id: "ellipse", title: "Ellipse  [O]" },
  ];
  const TOOLS = [...NAV_TOOLS, ...DRAW_TOOLS];
  const toolBtns = {};

  // Creates a toolbar tool button, registers it in toolBtns, and appends it to the toolbar.
  const _mkToolBtn = (t) => {
    const btn = document.createElement("button");
    btn.className = "ais-tool-btn" + (t.id === activeTool ? " active" : "");
    _addTooltip(btn, t.title);
    btn.appendChild(mkIcon(t.id));
    btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); setTool(t.id); btn.blur(); });
    toolbar.appendChild(btn);
    toolBtns[t.id] = btn;
  };

  for (const t of NAV_TOOLS) _mkToolBtn(t);

  // Fit-to-window button sits with the nav group
  resetViewBtn = document.createElement("button");
  resetViewBtn.className = "ais-tool-btn";
  _addTooltip(resetViewBtn, "Fit canvas to window  [F]");
  resetViewBtn.style.opacity = "0.4";
  resetViewBtn.style.pointerEvents = "none";
  resetViewBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
    <path d="M8 3H5a2 2 0 0 0-2 2v3"/>
    <path d="M21 8V5a2 2 0 0 0-2-2h-3"/>
    <path d="M3 16v3a2 2 0 0 0 2 2h3"/>
    <path d="M16 21h3a2 2 0 0 0 2-2v-3"/>
  </svg>`;
  resetViewBtn.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    e.preventDefault();
    resetView();
    resetViewBtn.blur();
  });
  toolbar.appendChild(resetViewBtn);

  // Divider between nav and drawing tools
  const divider = document.createElement("div");
  divider.style.cssText = "width:1px;height:20px;background:var(--border);margin:0 4px;flex-shrink:0;";
  toolbar.appendChild(divider);

  for (const t of DRAW_TOOLS) _mkToolBtn(t);

  // Divider before settings area
  const divider2 = document.createElement("div");
  divider2.style.cssText = "width:1px;height:20px;background:var(--border);margin:0 4px;flex-shrink:0;";
  toolbar.appendChild(divider2);

  // Settings area (right of divider, grows to fill)
  const settingsArea = document.createElement("div");
  settingsArea.style.cssText = "display:flex;align-items:center;gap:6px;flex:1;min-width:0;overflow:hidden;justify-content:flex-end;";
  toolbar.appendChild(settingsArea);
  // While editing text, don't let toolbar controls steal focus from the textarea.
  // Range sliders are excluded so they can still gain focus (the blur handler re-focuses).
  settingsArea.addEventListener("mousedown", (e) => {
    if (!textEditId) return;
    if (e.target.type === "range") return;
    e.preventDefault();
  });

  // Canvas area
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;width:100%;overflow:hidden;background:#111;flex:1 1 0;min-height:0;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;transform-origin:top left;cursor:crosshair;outline:none;" +
    `box-shadow:0 0 0 1px rgba(${SEL_COLOR_RGB},${FRAME_ROT_STEM_OPACITY});`;
  canvas.tabIndex = 0; // focusable so keyboard events naturally target canvas
  canvas.width = DEFAULT_CANVAS_WIDTH;
  canvas.height = DEFAULT_CANVAS_HEIGHT;

  canvasWrap.appendChild(canvas);

  // ── Context HUD (top-center of canvas, select mode only) ─────────────────
  const hudEl = document.createElement("div");
  hudEl.className = "ais-hud";
  hudEl.style.display = "none";
  canvasWrap.appendChild(hudEl);

  // ── Fullscreen HUD (top-right of canvas) ──────────────────────────────────
  const fsHudEl = document.createElement("div");
  fsHudEl.style.cssText =
    "position:absolute;top:10px;right:10px;display:flex;align-items:center;" +
    "background:rgba(18,18,20,0.88);border:1px solid rgba(255,255,255,0.10);" +
    "box-shadow:0 2px 16px rgba(0,0,0,0.55),0 0 0 1px rgba(255,255,255,0.06);" +
    "border-radius:7px;padding:2px;pointer-events:auto;z-index:21;";
  const fsBtn = document.createElement("button");
  fsBtn.style.cssText =
    "display:flex;align-items:center;justify-content:center;width:28px;height:28px;" +
    "padding:0;border:none;border-radius:7px;background:transparent;color:#e0e0e0;" +
    "cursor:pointer;line-height:1;transition:background 0.12s,color 0.12s;flex-shrink:0;";
  fsBtn.addEventListener("mouseenter", () => { fsBtn.style.background = "rgba(255,255,255,0.10)"; fsBtn.style.color = "#fff"; });
  fsBtn.addEventListener("mouseleave", () => { fsBtn.style.background = "transparent"; fsBtn.style.color = "#e0e0e0"; });
  _addTooltip(fsBtn, "Fullscreen");
  fsBtn.appendChild(mkIcon("expand", 15));
  fsBtn.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    if (_modalEl) { _closeModal(); } else { _openModal(); }
    fsBtn.blur();
  });
  fsHudEl.appendChild(fsBtn);
  canvasWrap.appendChild(fsHudEl);

  wrapper.appendChild(toolbar);
  wrapper.appendChild(canvasWrap);
  container.appendChild(wrapper);

  const ctx = canvas.getContext("2d");

  // ── coordinate conversion ─────────────────────────────────────────────────
  // Transform is on the canvas element itself. getBoundingClientRect() returns
  // the visual (scaled) rect, so canvas.width/rect.width = 1/displayScale.
  function screenToCanvas(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (canvas.width / rect.width),
      (e.clientY - rect.top) * (canvas.height / rect.height),
    ];
  }

  // ── canvas scaling ────────────────────────────────────────────────────────
  const resizeObserver = new ResizeObserver(() => applyCanvasScale());
  resizeObserver.observe(canvasWrap);
  applyCanvasScale(); // run synchronously so height is set before framework measures the node

  // ── zoom via scroll wheel ─────────────────────────────────────────────────
  canvasWrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const newVS = Math.max(0.25, Math.min(10, viewScale * factor));
    const rect = canvasWrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const ratio = newVS / viewScale;
    const lx = mx - centerOffsetX;
    const ly = my - centerOffsetY;
    panX = lx - (lx - panX) * ratio;
    panY = ly - (ly - panY) * ratio;
    viewScale = newVS;
    _applyViewTransform();
    const isDefault = viewScale === 1 && panX === 0 && panY === 0;
    resetViewBtn.style.opacity = isDefault ? "0.4" : "1";
    resetViewBtn.style.pointerEvents = isDefault ? "none" : "auto";
  }, { passive: false });

  // ── Alt key state for pan cursor ──────────────────────────────────────────
  // Registered via setupHotkeys below — these callbacks update canvas cursor

  // Base CSS cursor for the active tool when not hovering a specific handle.
  function _currentToolCursor() {
    if (activeTool === "select") return "default";
    if (activeTool === "hand") return "grab";
    if (activeTool === "zoom") return "zoom-in";
    return "crosshair";
  }

  // Applies the current viewScale + pan offset as a CSS transform on the canvas element.
  function _applyViewTransform() {
    const totalScale = displayScale * viewScale;
    canvas.style.transform = `translate(${centerOffsetX + panX}px, ${centerOffsetY + panY}px) scale(${totalScale})`;
  }

  // Fits the canvas to the container, updates displayScale, and recenters the canvas.
  // Called on resize and whenever canvas_width/height change.
  function applyCanvasScale() {
    const cw = currentValue.canvas_width || DEFAULT_CANVAS_WIDTH;
    const ch = currentValue.canvas_height || DEFAULT_CANVAS_HEIGHT;
    const areaW = canvasWrap.clientWidth || 300;
    const areaH = canvasWrap.clientHeight || 200;
    const newScale = Math.min(areaW / cw, areaH / ch);
    centerOffsetX = Math.max(0, (areaW - cw * newScale) / 2);
    centerOffsetY = Math.max(0, (areaH - ch * newScale) / 2);

    const dimsChanged = canvas.width !== cw || canvas.height !== ch;
    if (dimsChanged) {
      canvas.width = cw;
      canvas.height = ch;
    }
    canvas.style.width = cw + "px";
    canvas.style.height = ch + "px";
    if (newScale !== displayScale || dimsChanged) {
      displayScale = newScale;
      _applyViewTransform();
    }
  }

  // Resets zoom and pan to defaults (viewScale=1, panX/Y=0).
  function resetView() {
    viewScale = 1;
    panX = 0;
    panY = 0;
    _applyViewTransform();
    if (resetViewBtn) {
      resetViewBtn.style.opacity = "0.4";
      resetViewBtn.style.pointerEvents = "none";
    }
  }

  // ── settings and HUD module handles (wired at init, before setTool) ─────────
  // Declared as let so rebuildSettings can reference them before the factories run.
  let updateHud = null;
  let dismissLayerPopup = null;
  let buildToolSettings = null;
  let buildAnnotationSettings = null;
  let buildMultiSettings = null;

  // Rebuilds the settings area (right of toolbar) to match the active tool and current selection.
  // Also rebuilds the transform frame and HUD. Pass keepLayerPopup=true to avoid closing the
  // layer order popup when a reorder action triggers a rebuild.
  function rebuildSettings(keepLayerPopup = false) {
    if (!keepLayerPopup) dismissLayerPopup?.();
    _buildTxFrame();
    updateHud?.();
    settingsArea.innerHTML = "";

    if (activeTool === "select") {
      const selIds = currentValue.selected_ids || [];
      if (selIds.length === 1) {
        const selAnn = _effectiveAnnotations().find((a) => a.id === selIds[0]);
        if (selAnn) buildAnnotationSettings(selAnn);
      } else if (selIds.length > 1) {
        buildMultiSettings(selIds);
      }
      return;
    }

    // Paint/arrow/rect/ellipse tool with a single matching annotation selected: show its settings
    if (activeTool === "paint" || activeTool === "arrow" || activeTool === "rect" || activeTool === "ellipse") {
      const selIds = currentValue.selected_ids || [];
      if (selIds.length === 1) {
        const selAnn = _effectiveAnnotations().find((a) => a.id === selIds[0]);
        if (selAnn?.type === activeTool) {
          buildAnnotationSettings(selAnn);
          return;
        }
      }
    }

    // All other tools: always show tool settings (brush size, color, etc.)
    buildToolSettings();
  }

  // Switches the active tool: commits any open text edit, updates button highlight,
  // resets hover state, rebuilds settings, and emits.
  function setTool(id) {
    commitTextEdit();
    hoverId = null; hoverGroupId = null;
    activeTool = id;
    currentValue = { ...currentValue, active_tool: id };
    for (const [tid, btn] of Object.entries(toolBtns)) {
      btn.className = "ais-tool-btn" + (tid === id ? " active" : "");
    }
    canvas.style.cursor = _currentToolCursor();
    canvas.focus({ preventScroll: true });
    rebuildSettings();
    renderCanvas();
    _emit();
  }

  // ── rendering ─────────────────────────────────────────────────────────────
  let renderGen = 0;

  // Schedules a canvas redraw via RAF. Increments renderGen so any in-flight render is cancelled.
  function renderCanvas() {
    const gen = ++renderGen;
    requestAnimationFrame(() => { if (gen === renderGen) _doRender(gen); });
  }

  // Performs the actual canvas redraw: loads image (async, cached), clears, draws annotations,
  // transform frame, in-progress shapes, and the marquee lasso.
  async function _doRender(gen) {
    const cw = canvas.width, ch = canvas.height;
    const imgUrl = currentValue.image_url;

    // Resolve image before touching the canvas.
    // If cached, this is fully synchronous (no await) → no blank-frame flicker.
    let img = null;
    if (imgUrl) {
      const key = urlCacheKey(imgUrl);
      if (imageCache[key]) {
        img = imageCache[key]; // synchronous cache hit
      } else {
        // First load: async, acceptable to have one-time flicker
        try { img = await loadImage(imgUrl); } catch { /* img stays null */ }
        if (gen !== renderGen) return;
        // Auto-size canvas from image on first load
        if (img && (!currentValue.canvas_width || !currentValue.canvas_height)) {
          currentValue = { ...currentValue, canvas_width: img.naturalWidth, canvas_height: img.naturalHeight };
          applyCanvasScale();
          return; // applyCanvasScale triggers another render
        }
      }
    }

    // From here everything is synchronous → no flicker between clear and draw
    if (gen !== renderGen) return;
    ctx.clearRect(0, 0, cw, ch);

    if (img) {
      ctx.drawImage(img, 0, 0, cw, ch);
    } else {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
    }

    if (gen !== renderGen) return;

    // In drawing modes (paint, arrow, rect, ellipse, text) selection state is visual noise —
    // the user is focused on placing new content, not manipulating existing objects.
    // Only show selection highlights in select mode (mirrors tldraw / modern Figma).
    const showSelection = activeTool === "select";

    // Draw committed annotations
    for (const ann of _effectiveAnnotations()) {
      if (ann.id === textEditId) continue; // skip live-edited text
      drawAnnotation(ann, showSelection && (currentValue.selected_ids || []).includes(ann.id));
    }

    // Transform handles are only meaningful in select mode.
    if (showSelection && txFrame && _frameActiveTools.includes(activeTool)) {
      const corners = frameCorners(txFrame);
      const topMid = frameTopMid(txFrame);
      const rh = frameRotHandle(txFrame, displayScale);
      const selIds = currentValue.selected_ids || [];
      const allImported = selIds.length > 0 && selIds.every((id) => _isImported(id));
      const frameColor = allImported ? IMP_COLOR : SEL_COLOR;
      const frameColorRgb = allImported ? IMP_COLOR_RGB : SEL_COLOR_RGB;
      const hs = HANDLE_RADIUS / 2 / displayScale;   // corner handle half-size
      const rhs = ROT_HANDLE_RADIUS / displayScale;
      ctx.save();

      // Subtle fill tint inside selection
      ctx.fillStyle = `rgba(${frameColorRgb},${FRAME_FILL_OPACITY})`;
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath(); ctx.fill();

      // Solid outline — thin and clean
      ctx.strokeStyle = `rgba(${frameColorRgb},${FRAME_BORDER_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_SECONDARY / displayScale;
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath(); ctx.stroke();

      // Corner handles — small white squares
      ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
      for (const [hx, hy] of corners) {
        ctx.fillStyle = HANDLE_FILL;
        ctx.fillRect(hx - hs, hy - hs, hs * 2, hs * 2);
        ctx.strokeStyle = frameColor;
        ctx.strokeRect(hx - hs, hy - hs, hs * 2, hs * 2);
      }

      // Center move handle — only shown in paint mode
      if (activeTool === "paint") {
        const chLen = CORNER_TICK_LEN / displayScale;
        const cpx = txFrame.pivotX, cpy = txFrame.pivotY;
        ctx.fillStyle = HANDLE_FILL;
        ctx.fillRect(cpx - hs, cpy - hs, hs * 2, hs * 2);
        ctx.strokeStyle = frameColor; ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
        ctx.strokeRect(cpx - hs, cpy - hs, hs * 2, hs * 2);
        ctx.strokeStyle = `rgba(${frameColorRgb},${FRAME_CORNER_OPACITY})`; ctx.lineWidth = LINE_WIDTH_SECONDARY / displayScale;
        ctx.beginPath();
        ctx.moveTo(cpx - chLen, cpy); ctx.lineTo(cpx + chLen, cpy);
        ctx.moveTo(cpx, cpy - chLen); ctx.lineTo(cpx, cpy + chLen);
        ctx.stroke();
      }

      // Rotation handle — thin dotted stem + filled circle (hidden for text)
      if (!txFrame.noRotate) {
        ctx.strokeStyle = `rgba(${frameColorRgb},${FRAME_ROT_STEM_OPACITY})`; ctx.lineWidth = LINE_WIDTH_TERTIARY / displayScale;
        ctx.setLineDash(DASH_ROT_STEM.map((v) => v / displayScale));
        ctx.beginPath(); ctx.moveTo(topMid[0], topMid[1]); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = frameColor;
        ctx.beginPath(); ctx.arc(rh[0], rh[1], rhs, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = `rgba(255,255,255,${ROT_HANDLE_INNER_OPACITY})`; ctx.lineWidth = LINE_WIDTH_PRIMARY / displayScale;
        ctx.beginPath(); ctx.arc(rh[0], rh[1], rhs, 0, Math.PI * 2); ctx.stroke();
      }

      ctx.restore();
    }

    // In-progress paint stroke
    if (currentStroke) {
      renderStrokes([currentStroke], 1);
    }

    // In-progress arrow
    if (currentArrow) {
      const ts = toolSettings.arrow;
      drawArrowLine(
        currentArrow.x1, currentArrow.y1, currentArrow.x2, currentArrow.y2,
        ts.color || DEFAULT_COLOR, ts.width || DEFAULT_ARROW_WIDTH,
        null, null, null, null,
        ts.has_start_arrow ?? false, ts.has_end_arrow ?? true, ts.taper ?? false
      );
    }

    // In-progress rect
    if (currentRect) {
      const ts = toolSettings.rect;
      const rx = Math.min(currentRect.x1, currentRect.x2);
      const ry = Math.min(currentRect.y1, currentRect.y2);
      const rw = Math.abs(currentRect.x2 - currentRect.x1);
      const rh = Math.abs(currentRect.y2 - currentRect.y1);
      ctx.save();
      ctx.lineWidth = ts.width || 2;
      ctx.strokeStyle = ts.color || DEFAULT_COLOR;
      if (ts.fill_color) { ctx.fillStyle = ts.fill_color; ctx.fillRect(rx, ry, rw, rh); }
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.restore();
    }

    // In-progress ellipse
    if (currentEllipse) {
      const ts = toolSettings.ellipse;
      const ex = (currentEllipse.x1 + currentEllipse.x2) / 2;
      const ey = (currentEllipse.y1 + currentEllipse.y2) / 2;
      const erx = Math.max(0.5, Math.abs(currentEllipse.x2 - currentEllipse.x1) / 2);
      const ery = Math.max(0.5, Math.abs(currentEllipse.y2 - currentEllipse.y1) / 2);
      ctx.save();
      ctx.lineWidth = ts.width || 2;
      ctx.strokeStyle = ts.color || DEFAULT_COLOR;
      ctx.beginPath();
      ctx.ellipse(ex, ey, erx, ery, 0, 0, Math.PI * 2);
      if (ts.fill_color) { ctx.fillStyle = ts.fill_color; ctx.fill(); }
      ctx.stroke();
      ctx.restore();
    }

    // Marquee selection rectangle
    if (dragState?.type === "marquee") {
      const mx1 = Math.min(dragState.startCx, dragState.x2);
      const my1 = Math.min(dragState.startCy, dragState.y2);
      const mw = Math.abs(dragState.x2 - dragState.startCx);
      const mh = Math.abs(dragState.y2 - dragState.startCy);
      ctx.save();
      ctx.fillStyle = `rgba(${SEL_COLOR_RGB},${LASSO_FILL_OPACITY})`;
      ctx.fillRect(mx1, my1, mw, mh);
      ctx.strokeStyle = `rgba(${SEL_COLOR_RGB},${LASSO_STROKE_OPACITY})`;
      ctx.lineWidth = LINE_WIDTH_SECONDARY / displayScale;
      ctx.setLineDash(DASH_LASSO.map((v) => v / displayScale));
      ctx.strokeRect(mx1, my1, mw, mh);
      ctx.restore();
    }
  }

  // Dispatches to the type-specific draw function for a committed annotation.
  function drawAnnotation(ann, selected) {
    if (ann.type === "paint")   drawPaint(ann, selected);
    else if (ann.type === "text")   drawText(ann, selected);
    else if (ann.type === "arrow")  drawArrowAnnotation(ann, selected);
    else if (ann.type === "rect")   drawRect(ann, selected);
    else if (ann.type === "ellipse") drawEllipse(ann, selected);
  }

  // ── drawing functions (bound to live state via factory) ───────────────────
  const drawing = createDrawing(() => ({ ctx, displayScale, hoverId, hoverGroupId, marqueePreviewIds }));
  const { renderStrokes, drawPaint, drawText, drawArrowLine, drawArrowAnnotation, drawRect, drawEllipse } = drawing;

  // ── hit testing ───────────────────────────────────────────────────────────
  function hitTest(cx, cy) {
    const anns = [..._effectiveAnnotations()].reverse();
    for (const ann of anns) {
      if (ann.type === "text") {
        const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
        ctx.font = `${fontSize}px sans-serif`;
        const lines = (ann.text || "").split("\n");
        const w = Math.max(1, ...lines.map((l) => ctx.measureText(l).width));
        const h = fontSize * 1.2 * lines.length;
        const r = -(ann.rotation || 0);
        const cos = Math.cos(r), sin = Math.sin(r);
        const dx = cx - (ann.x || 0), dy = cy - (ann.y || 0);
        const lx = dx * cos - dy * sin, ly = dx * sin + dy * cos;
        if (lx >= -4 && lx <= w + 4 && ly >= -4 && ly <= h + 4) return ann;
      } else if (ann.type === "arrow") {
        const { cp1x, cp1y, cp2x, cp2y } = defaultCps(ann);
        const tol = Math.max(12 / displayScale, (ann.width || 3) + 6);
        const N = 20;
        for (let i = 0; i <= N; i++) {
          const t = i / N, mt = 1 - t;
          const bx = mt**3*ann.x1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*ann.x2;
          const by = mt**3*ann.y1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ann.y2;
          if (Math.hypot(cx - bx, cy - by) <= tol) return ann;
        }
      } else if (ann.type === "paint") {
        const [lx, ly] = paintInvTransformPt(ann, cx, cy);
        // If already selected, accept click anywhere inside the bounding box
        const isSelected = (currentValue.selected_ids || []).includes(ann.id);
        if (isSelected) {
          const nb = naturalBounds(ann);
          if (nb && lx >= nb.minX && lx <= nb.maxX && ly >= nb.minY && ly <= nb.maxY) return ann;
        }
        for (const stroke of (ann.strokes || [])) {
          const tol = Math.max(12 / displayScale, (stroke.size || DEFAULT_PAINT_SIZE) / 2 + 4);
          for (const pt of (stroke.points || [])) {
            if (Math.hypot(lx - pt[0], ly - pt[1]) <= tol) return ann;
          }
        }
      } else if (ann.type === "rect" || ann.type === "ellipse") {
        const dx = cx - (ann.x || 0), dy = cy - (ann.y || 0);
        const r = -(ann.rotation || 0);
        const cos = Math.cos(r), sin = Math.sin(r);
        const lx = dx * cos - dy * sin, ly = dx * sin + dy * cos;
        const hw = (ann.w || 10) / 2, hh = (ann.h || 10) / 2;
        const tol = Math.max(8 / displayScale, (ann.width || 2) + 4);
        const isSelected = (currentValue.selected_ids || []).includes(ann.id);
        if (ann.type === "rect") {
          if (isSelected || ann.fill_color) {
            if (lx >= -hw && lx <= hw && ly >= -hh && ly <= hh) return ann;
          } else {
            const nearH = Math.abs(Math.abs(lx) - hw) <= tol && ly >= -hh - tol && ly <= hh + tol;
            const nearV = Math.abs(Math.abs(ly) - hh) <= tol && lx >= -hw - tol && lx <= hw + tol;
            if (nearH || nearV) return ann;
          }
        } else {
          const ex = lx / (hw + tol), ey = ly / (hh + tol);
          if (ex * ex + ey * ey <= 1) {
            if (isSelected || ann.fill_color) return ann;
            const exIn = hw > tol ? lx / (hw - tol) : 0, eyIn = hh > tol ? ly / (hh - tol) : 0;
            if (exIn * exIn + eyIn * eyIn >= 1) return ann;
          }
        }
      }
    }
    return null;
  }

  // Returns the canvas-space axis-aligned bounding box for an annotation
  function _getAnnotationBounds(ann) {
    if (ann.type === "text") {
      const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
      ctx.font = `${fontSize}px sans-serif`;
      const w = ctx.measureText(ann.text || "").width;
      const h = fontSize * 1.2;
      const ax = ann.x || 0, ay = ann.y || 0;
      return { minX: ax - 4, minY: ay - 4, maxX: ax + w + 8, maxY: ay + h + 8 };
    } else if (ann.type === "arrow") {
      const { cp1x, cp1y, cp2x, cp2y } = defaultCps(ann);
      const pad = Math.max(8, (ann.width || 3) / 2 + 4);
      const xs = [ann.x1, ann.x2, cp1x, cp2x], ys = [ann.y1, ann.y2, cp1y, cp2y];
      return {
        minX: Math.min(...xs) - pad, minY: Math.min(...ys) - pad,
        maxX: Math.max(...xs) + pad, maxY: Math.max(...ys) + pad,
      };
    } else if (ann.type === "paint") {
      const corners = getTransformedCorners(ann, 10);
      if (!corners.length) return null;
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const [cx, cy] of corners) {
        minX = Math.min(minX, cx); minY = Math.min(minY, cy);
        maxX = Math.max(maxX, cx); maxY = Math.max(maxY, cy);
      }
      return { minX, minY, maxX, maxY };
    } else if (ann.type === "rect" || ann.type === "ellipse") {
      const hw = (ann.w || 10) / 2, hh = (ann.h || 10) / 2;
      const r = ann.rotation || 0;
      const cos = Math.cos(r), sin = Math.sin(r);
      const corners = [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([lx,ly]) =>
        [(ann.x || 0) + lx*cos - ly*sin, (ann.y || 0) + lx*sin + ly*cos]);
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (const [cx, cy] of corners) {
        minX = Math.min(minX, cx); minY = Math.min(minY, cy);
        maxX = Math.max(maxX, cx); maxY = Math.max(maxY, cy);
      }
      const pad = (ann.width || 2) / 2 + 4;
      return { minX: minX - pad, minY: minY - pad, maxX: maxX + pad, maxY: maxY + pad };
    }
    return null;
  }

  // Returns the axis-aligned bounding box enclosing all annotations in selIds, plus center coords.
  function _getGroupBounds(selIds) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let found = false;
    for (const ann of _effectiveAnnotations()) {
      if (!selIds.includes(ann.id)) continue;
      const b = _getAnnotationBounds(ann);
      if (!b) continue;
      found = true;
      minX = Math.min(minX, b.minX); minY = Math.min(minY, b.minY);
      maxX = Math.max(maxX, b.maxX); maxY = Math.max(maxY, b.maxY);
    }
    if (!found) return null;
    return { minX, minY, maxX, maxY, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2 };
  }

  // ── unified transform frame (OBB) ────────────────────────────────────────────
  // The frame is an oriented bounding box (pivot + rotation + half-extents).
  // For a single paint, rotation matches the paint's rotation so handles rotate with it.
  // For groups/text, rotation starts at 0 and accumulates during txRotate drags.
  // Updating txFrame.rotation live during a drag keeps handles glued to content.

  function _buildTxFrame() {
    const selIds = currentValue.selected_ids || [];
    if (!selIds.length || !_frameActiveTools.includes(activeTool)) { txFrame = null; return; }
    const pad = 6 / displayScale;
    // Detect selection change to reset accumulated rotation
    const prevIds = txFrame?._selIds || [];
    const selChanged = selIds.length !== prevIds.length ||
      selIds.some((id, i) => id !== prevIds[i]);
    if (selIds.length === 1) {
      const ann = _effectiveAnnotations().find((a) => a.id === selIds[0]);
      if (!ann) { txFrame = null; return; }
      if (ann.type === "arrow") { txFrame = null; return; } // arrows use endpoint handles
      if (ann.type === "rect" || ann.type === "ellipse") {
        txFrame = {
          pivotX: ann.x || 0, pivotY: ann.y || 0,
          rotation: ann.rotation || 0,
          halfW: (ann.w || 10) / 2 + pad,
          halfH: (ann.h || 10) / 2 + pad,
          _selIds: [...selIds],
        };
        return;
      }
      if (ann.type === "paint") {
        const nb = naturalBounds(ann);
        if (!nb) { txFrame = null; return; }
        const [pcx, pcy] = paintCenter(ann);
        const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1;
        txFrame = {
          pivotX: pcx + (ann.x || 0), pivotY: pcy + (ann.y || 0),
          rotation: ann.rotation || 0,
          halfW: (nb.maxX - nb.minX) / 2 * sx + pad,
          halfH: (nb.maxY - nb.minY) / 2 * sy + pad,
          _selIds: [...selIds],
        };
        return;
      }
      if (ann.type === "text") {
        const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
        const lineHeight = fontSize * 1.2;
        const lines = (ann.text || "").split("\n");
        ctx.save(); ctx.font = `${fontSize}px sans-serif`;
        const textW = Math.max(1, ...lines.map((l) => ctx.measureText(l).width));
        ctx.restore();
        const hw = textW / 2, hh = (lineHeight * lines.length) / 2;
        const r = ann.rotation || 0, cos = Math.cos(r), sin = Math.sin(r);
        txFrame = {
          pivotX: (ann.x || 0) + hw * cos - hh * sin,
          pivotY: (ann.y || 0) + hw * sin + hh * cos,
          rotation: r,
          halfW: hw + pad, halfH: hh + pad,
          _selIds: [...selIds],
        };
        return;
      }
    }
    // If the selection hasn't changed and we already have a frame, keep it.
    // The frame is maintained live during all drag types (txRotate, txScale, translate),
    // so it is always correct at this point. Recomputing from the AABB would give the
    // wrong extents for a group that has been rotated (AABB grows; OBB doesn't).
    if (!selChanged && txFrame) {
      txFrame = { ...txFrame, _selIds: [...selIds] };
      return;
    }
    const gb = _getGroupBounds(selIds);
    if (!gb) { txFrame = null; return; }
    txFrame = {
      pivotX: gb.centerX, pivotY: gb.centerY,
      rotation: 0,
      halfW: (gb.maxX - gb.minX) / 2 + pad,
      halfH: (gb.maxY - gb.minY) / 2 + pad,
      _selIds: [...selIds],
    };
  }

  // Returns true if annotation overlaps the given canvas-space rectangle
  function _annotationIntersectsRect(ann, x1, y1, x2, y2) {
    if (ann.type === "text") {
      const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
      ctx.font = `${fontSize}px sans-serif`;
      const w = ctx.measureText(ann.text || "").width;
      const h = fontSize * 1.2;
      const ax = ann.x || 0, ay = ann.y || 0;
      return !(ax + w < x1 || ax > x2 || ay + h < y1 || ay > y2);
    } else if (ann.type === "arrow") {
      const { cp1x, cp1y, cp2x, cp2y } = defaultCps(ann);
      const N = 12;
      for (let i = 0; i <= N; i++) {
        const t = i / N, mt = 1 - t;
        const bx = mt**3*ann.x1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*ann.x2;
        const by = mt**3*ann.y1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ann.y2;
        if (bx >= x1 && bx <= x2 && by >= y1 && by <= y2) return true;
      }
      return false;
    } else if (ann.type === "paint") {
      for (const stroke of (ann.strokes || [])) {
        for (const pt of (stroke.points || [])) {
          const [px, py] = paintTransformPt(ann, pt[0], pt[1]);
          if (px >= x1 && px <= x2 && py >= y1 && py <= y2) return true;
        }
      }
      return false;
    } else if (ann.type === "rect" || ann.type === "ellipse") {
      const b = _getAnnotationBounds(ann);
      if (!b) return false;
      return !(b.maxX < x1 || b.minX > x2 || b.maxY < y1 || b.minY > y2);
    }
    return false;
  }

  // ── text editing ──────────────────────────────────────────────────────────

  // Overlays a positioned, rotated <textarea> on top of the canvas for inline text editing.
  function startTextEdit(ann) {
    commitTextEdit();
    textEditId = ann.id;
    currentValue = { ...currentValue, selected_ids: [ann.id] };

    const fontSize = Math.max(MIN_TEXT_SIZE, ann.font_size || DEFAULT_TEXT_SIZE);
    const totalScale = displayScale * viewScale;
    textInput = document.createElement("textarea");
    textInput.value = ann.text || "";
    textInput.rows = 1;
    textInput.style.cssText = [
      "position:absolute",
      `left:${(ann.x || 0) * totalScale + centerOffsetX + panX}px`,
      `top:${(ann.y || 0) * totalScale + centerOffsetY + panY}px`,
      "min-width:60px",
      "background:transparent",
      `color:${ann.color || "#ffffff"}`,
      `font-size:${fontSize * totalScale}px`,
      "font-family:sans-serif",
      "border:none",
      "outline:none",
      "resize:none",
      "overflow:hidden",
      "white-space:nowrap",
      "z-index:100",
      "padding:0",
      "margin:0",
      "line-height:1",
      `transform:rotate(${ann.rotation || 0}rad)`,
      "transform-origin:0px 0px",
    ].join(";");

    canvasWrap.appendChild(textInput);
    _autoResizeTextarea();

    textInput.addEventListener("input", () => {
      _autoResizeTextarea();
      _applySingleUpdate(textEditId, (a) => ({ ...a, text: textInput.value }));
    });
    textInput.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commitTextEdit(); canvas.focus({ preventScroll: true }); }
      if (e.key === "Escape") { e.preventDefault(); commitTextEdit(); canvas.focus({ preventScroll: true }); }
    });
    textInput.addEventListener("blur", (e) => {
      // If focus moves to a toolbar control (slider, button, etc.), don't commit —
      // re-focus the textarea after the control interaction so the user can keep typing.
      const toEl = e.relatedTarget;
      if (toEl && toolbar.contains(toEl)) {
        setTimeout(() => { if (textInput) textInput.focus(); }, 0);
        return;
      }
      commitTextEdit();
    });
    textInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    // Defer focus so it fires after the browser's mousedown default action
    // (which would re-focus the canvas and immediately blur/destroy the textarea)
    setTimeout(() => {
      if (!textInput) return;
      textInput.focus();
      const len = textInput.value.length;
      textInput.setSelectionRange(len, len);
    }, 0);
    renderCanvas();
  }

  // Shrinks textarea to 1px then expands to scrollWidth/scrollHeight so it fits its content exactly.
  function _autoResizeTextarea() {
    if (!textInput) return;
    textInput.style.width = "1px";
    textInput.style.width = textInput.scrollWidth + "px";
    textInput.style.height = "auto";
    textInput.style.height = textInput.scrollHeight + "px";
  }

  // Saves the textarea's text to the annotation and removes the textarea.
  // Deletes the annotation if the text is empty; switches to select mode in either case.
  function commitTextEdit() {
    if (!textInput) return;
    const id = textEditId;
    const rawText = textInput.value;
    const text = rawText.trim() ? rawText : "";  // preserve internal/trailing newlines; treat all-whitespace as empty
    textInput.removeEventListener("blur", commitTextEdit);
    textInput.remove();
    textInput = null;
    textEditId = null;
    dragState = null;
    if (id) {
      if (!text) {
        // Empty text: discard the annotation entirely and clear selection.
        _deleteAnnotations([id]);
        currentValue = { ...currentValue, selected_ids: [] };
        setTool("select");
      } else {
        // Text committed: keep it selected and drop into select mode so the user
        // can immediately reposition it — same pattern as tldraw / Figma text tool.
        _applySingleUpdate(id, (a) => ({ ...a, text }));
        currentValue = { ...currentValue, selected_ids: [id] };
        setTool("select");
      }
      _emit();
    }
    rebuildSettings();
    renderCanvas();
  }

  // Creates a new text annotation at canvas position (cx, cy) and immediately opens it for editing.
  function placeNewText(cx, cy) {
    const id = _uid("text");
    const ann = {
      id, type: "text", text: "",
      x: cx, y: cy, rotation: 0,
      color: toolSettings.text.color,
      font_size: toolSettings.text.font_size,
    };
    currentValue = {
      ...currentValue,
      annotations: [...(currentValue.annotations || []), ann],
      selected_ids: [id],
    };
    startTextEdit(ann);
    // emit after startTextEdit so the annotation with the initial text is sent
    _emit();
  }

  // ── pointer events ────────────────────────────────────────────────────────
  // ── hotkeys (all document-level keyboard listeners) ───────────────────────
  const _cleanupHotkeys = setupHotkeys(
    () => ({ mouseIsOver: _mouseIsOver, textEditId, activeTool, currentValue, toolSettings }),
    {
      setTool, resetView, rebuildSettings,
      emit: _emit, renderCanvas,
      deleteAnnotations: _deleteAnnotations,
      setCurrentValue: (v) => { currentValue = v; },
      setTxFrame: (f) => { txFrame = f; },
      applySingleUpdate: _applySingleUpdate,
      effectiveAnnotations: _effectiveAnnotations,
      onPointerDown,
      onAltDown: () => { if (!isAltHeld) { isAltHeld = true; if (!isPointerDown) canvas.style.cursor = "grab"; } },
      onAltUp:   () => { isAltHeld = false; if (!isPanning && !isPointerDown) canvas.style.cursor = _currentToolCursor(); },
      wrapper,
    }
  );

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("mousemove", onMouseHover);
  canvas.addEventListener("mouseleave", () => {
    if (hoverId || hoverGroupId) { hoverId = null; hoverGroupId = null; renderCanvas(); }
  });

  container.addEventListener("mouseenter", () => { _mouseIsOver = true; });
  container.addEventListener("mouseleave", () => { _mouseIsOver = false; });

  // ── pointer events ────────────────────────────────────────────────────────

  // Handles pointerdown: pan, zoom drag, transform handles, hit selection, shape drawing.
  function onPointerDown(e) {
    if (e.button !== 0) return;
    e.stopPropagation();
    canvas.setPointerCapture(e.pointerId);
    canvas.focus({ preventScroll: true });
    isPointerDown = true;

    // Alt + LMB or hand tool → pan
    if (e.altKey || activeTool === "hand") {
      e.preventDefault();
      isPanning = true;
      panStartX = e.clientX - panX;
      panStartY = e.clientY - panY;
      canvas.style.cursor = "grabbing";
      return;
    }

    // Zoom tool — drag right to zoom in, drag left to zoom out, anchored to click point
    if (activeTool === "zoom") {
      e.preventDefault();
      const rect = canvasWrap.getBoundingClientRect();
      dragState = {
        type: "zoom",
        startClientX: e.clientX,
        anchorMx: e.clientX - rect.left,
        anchorMy: e.clientY - rect.top,
        startViewScale: viewScale,
        startPanX: panX,
        startPanY: panY,
      };
      canvas.style.cursor = "ew-resize";
      return;
    }

    const [cx, cy] = screenToCanvas(e);

    // OBB handle check — only in select mode (handles aren't rendered in drawing modes)
    if (activeTool === "select" && txFrame) {
      const handleR = 8 / displayScale;
      const selIds = currentValue.selected_ids || [];
      const corners = frameCorners(txFrame);
      const rh = frameRotHandle(txFrame, displayScale);
      const buildSnapshots = () => {
        const s = {};
        for (const ann of _effectiveAnnotations())
          if (selIds.includes(ann.id)) s[ann.id] = snapshotAnn(ann);
        return s;
      };
      if (!txFrame.noRotate && Math.hypot(cx - rh[0], cy - rh[1]) <= handleR) {
        dragState = { type: "txRotate",
          pivot: { x: txFrame.pivotX, y: txFrame.pivotY },
          origAngle: Math.atan2(cy - txFrame.pivotY, cx - txFrame.pivotX),
          origRotation: txFrame.rotation,
          origSnapshots: buildSnapshots(), selIds: [...selIds] };
        canvas.style.cursor = "grabbing"; renderCanvas(); return;
      }
      const localCornerSigns = [[-1,-1],[1,-1],[1,1],[-1,1]];
      for (let i = 0; i < corners.length; i++) {
        const [hx, hy] = corners[i];
        if (Math.hypot(cx - hx, cy - hy) <= handleR) {
          dragState = { type: "txScale",
            pivot: { x: txFrame.pivotX, y: txFrame.pivotY },
            origFrameRotation: txFrame.rotation,
            origHalfW: txFrame.halfW, origHalfH: txFrame.halfH,
            cornerSignX: localCornerSigns[i][0],
            cornerSignY: localCornerSigns[i][1],
            origSnapshots: buildSnapshots(), selIds: [...selIds] };
          canvas.style.cursor = "grabbing"; renderCanvas(); return;
        }
      }
      // Center move handle — always available in all frame-active tools (especially useful in paint mode)
      if (Math.hypot(cx - txFrame.pivotX, cy - txFrame.pivotY) <= handleR) {
        const origPositions = {};
        for (const id of selIds) {
          const a = _effectiveAnnotations().find((ann) => ann.id === id);
          if (!a) continue;
          if (a.type === "arrow") {
            const cps = defaultCps(a);
            origPositions[id] = { x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2,
              cp1x: cps.cp1x, cp1y: cps.cp1y, cp2x: cps.cp2x, cp2y: cps.cp2y };
          } else origPositions[id] = { x: a.x ?? 0, y: a.y ?? 0 };
        }
        dragState = { type: "translate", startCx: cx, startCy: cy, origPositions,
          origPivotX: txFrame.pivotX, origPivotY: txFrame.pivotY };
        canvas.style.cursor = "grabbing"; renderCanvas(); return;
      }
    }

    if (activeTool === "text") {
      if (textEditId) { commitTextEdit(); return; } // clicking away commits edit
      const hit = hitTest(cx, cy);
      if (hit && hit.type === "text") {
        // Single click on existing text: select it for dragging
        hoverId = null;
        currentValue = { ...currentValue, selected_ids: [hit.id] };
        dragState = { type: "translate", startCx: cx, startCy: cy,
          origPositions: { [hit.id]: { x: hit.x ?? 0, y: hit.y ?? 0 } },
          origPivotX: txFrame?.pivotX, origPivotY: txFrame?.pivotY };
        canvas.style.cursor = "grabbing";
        rebuildSettings();
        renderCanvas();
        return;
      }
      // Click on empty space: place a new text annotation
      placeNewText(cx, cy);
      return;
    }

    if (activeTool === "select") {
      commitTextEdit();
      const handleR = 8 / displayScale;
      const selIds = currentValue.selected_ids || [];

      // Unified transform frame handle detection (single paint, single text, or group)
      if (txFrame) {
        const corners = frameCorners(txFrame);
        const rh = frameRotHandle(txFrame, displayScale);
        const buildSnapshots = () => {
          const s = {};
          for (const ann of _effectiveAnnotations())
            if (selIds.includes(ann.id)) s[ann.id] = snapshotAnn(ann);
          return s;
        };
        // Rotation handle
        if (!txFrame.noRotate && Math.hypot(cx - rh[0], cy - rh[1]) <= handleR) {
          dragState = { type: "txRotate",
            pivot: { x: txFrame.pivotX, y: txFrame.pivotY },
            origAngle: Math.atan2(cy - txFrame.pivotY, cx - txFrame.pivotX),
            origRotation: txFrame.rotation,
            origSnapshots: buildSnapshots(), selIds: [...selIds] };
          canvas.style.cursor = "grabbing"; renderCanvas(); return;
        }
        // Corner scale handles — corner[i] maps to local sign [-/+ hw, -/+ hh]
        const localCornerSigns = [[-1,-1],[1,-1],[1,1],[-1,1]];
        for (let i = 0; i < corners.length; i++) {
          const [hx, hy] = corners[i];
          if (Math.hypot(cx - hx, cy - hy) <= handleR) {
            dragState = { type: "txScale",
              pivot: { x: txFrame.pivotX, y: txFrame.pivotY },
              origFrameRotation: txFrame.rotation,
              origHalfW: txFrame.halfW, origHalfH: txFrame.halfH,
              cornerSignX: localCornerSigns[i][0],
              cornerSignY: localCornerSigns[i][1],
              origSnapshots: buildSnapshots(), selIds: [...selIds] };
            canvas.style.cursor = "grabbing"; renderCanvas(); return;
          }
        }
        // Click anywhere inside the frame body → translate the selection
        const fdx = cx - txFrame.pivotX, fdy = cy - txFrame.pivotY;
        const fcos = Math.cos(-txFrame.rotation), fsin = Math.sin(-txFrame.rotation);
        const flx = fdx * fcos - fdy * fsin, fly = fdx * fsin + fdy * fcos;
        if (Math.abs(flx) <= txFrame.halfW && Math.abs(fly) <= txFrame.halfH) {
          const origPositions = {};
          for (const id of selIds) {
            const a = _effectiveAnnotations().find((ann) => ann.id === id);
            if (!a) continue;
            if (a.type === "arrow") {
              const cps = defaultCps(a);
              origPositions[id] = { x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2,
                cp1x: cps.cp1x, cp1y: cps.cp1y, cp2x: cps.cp2x, cp2y: cps.cp2y };
            } else origPositions[id] = { x: a.x ?? 0, y: a.y ?? 0 };
          }
          dragState = { type: "translate", startCx: cx, startCy: cy, origPositions,
            origPivotX: txFrame.pivotX, origPivotY: txFrame.pivotY };
          canvas.style.cursor = "grabbing"; renderCanvas(); return;
        }
      }

      // Control point handle detection: only when single arrow already selected
      if (selIds.length === 1) {
        const selAnn = _effectiveAnnotations().find((a) => a.id === selIds[0]);
        if (selAnn?.type === "arrow") {
          const cps = defaultCps(selAnn);
          const cpR = Math.max(8 / displayScale, 5);
          for (const [which, hx, hy] of [["cp1", cps.cp1x, cps.cp1y], ["cp2", cps.cp2x, cps.cp2y]]) {
            if (Math.hypot(cx - hx, cy - hy) <= cpR) {
              dragState = { type: "arrowCp", id: selAnn.id, which, startCx: cx, startCy: cy,
                origCp1x: cps.cp1x, origCp1y: cps.cp1y, origCp2x: cps.cp2x, origCp2y: cps.cp2y };
              canvas.style.cursor = "grabbing"; renderCanvas(); return;
            }
          }
        }
      }

      const hit = hitTest(cx, cy);
      if (hit) {
        // Expand to the full group if the hit annotation is grouped
        const hitGroupIds = expandGroupSelection(_effectiveAnnotations(), hit.id);
        let newSelIds;
        if (e.shiftKey) {
          // Shift+click: toggle the entire group in/out as a unit
          const allSelected = hitGroupIds.every((id) => selIds.includes(id));
          newSelIds = allSelected
            ? selIds.filter((id) => !hitGroupIds.includes(id))
            : [...new Set([...selIds, ...hitGroupIds])];
        } else if (hitGroupIds.every((id) => selIds.includes(id))) {
          // All group members already selected: keep selection for drag
          newSelIds = selIds;
        } else {
          // Click on annotation or group: replace selection with group members
          newSelIds = hitGroupIds;
        }
        currentValue = { ...currentValue, selected_ids: newSelIds };
        // Rebuild frame immediately so origPivotX/Y captures the NEW selection's pivot, not the old one
        _buildTxFrame();

        // Arrow endpoint handles (single selection only)
        if (newSelIds.length === 1 && hit.type === "arrow") {
          const arrowHandleR = Math.max(10 / displayScale, 8);
          const nearStart = Math.hypot(cx - hit.x1, cy - hit.y1) <= arrowHandleR;
          const nearEnd   = Math.hypot(cx - hit.x2, cy - hit.y2) <= arrowHandleR;
          if (nearStart || nearEnd) {
            const hitCps = defaultCps(hit);
            dragState = { type: "arrowHandle", id: hit.id,
              arrowHandle: nearStart ? "start" : "end",
              startCx: cx, startCy: cy,
              origX1: hit.x1, origY1: hit.y1, origX2: hit.x2, origY2: hit.y2,
              origCp1x: hitCps.cp1x, origCp1y: hitCps.cp1y, origCp2x: hitCps.cp2x, origCp2y: hitCps.cp2y };
            canvas.style.cursor = "grabbing"; rebuildSettings(); renderCanvas(); return;
          }
        }

        // Multi-translate drag: all currently selected annotations move together
        const origPositions = {};
        for (const id of newSelIds) {
          const a = _effectiveAnnotations().find((ann) => ann.id === id);
          if (!a) continue;
          if (a.type === "arrow") {
            const cps = defaultCps(a);
            origPositions[id] = { x1: a.x1, y1: a.y1, x2: a.x2, y2: a.y2,
              cp1x: cps.cp1x, cp1y: cps.cp1y, cp2x: cps.cp2x, cp2y: cps.cp2y };
          } else origPositions[id] = { x: a.x ?? 0, y: a.y ?? 0 };
        }
        dragState = { type: "translate", startCx: cx, startCy: cy, origPositions,
          origPivotX: txFrame?.pivotX, origPivotY: txFrame?.pivotY };
        canvas.style.cursor = "grabbing";
      } else {
        // Click on empty space: always start marquee; additive when Shift held
        if (!e.shiftKey) currentValue = { ...currentValue, selected_ids: [] };
        dragState = { type: "marquee", startCx: cx, startCy: cy, x2: cx, y2: cy, additive: e.shiftKey };
      }
      rebuildSettings();
      renderCanvas();
      return;
    }

    if (activeTool === "paint") {
      // In paint mode, corner/rotation/center handles are checked above — all other clicks draw.
      // Deselect any selection and start new stroke
      currentValue = { ...currentValue, selected_ids: [] };
      rebuildSettings();
      const sz = toolSettings.paint.size;
      currentStroke = { color: toolSettings.paint.color, size: sz, points: [[cx, cy, sz]] };
      lastPtTime = performance.now(); lastPtX = cx; lastPtY = cy; velSmoothed = 0;
      // Draw initial dot directly without a full re-render
      ctx.save();
      ctx.fillStyle = toolSettings.paint.color;
      ctx.beginPath();
      ctx.arc(cx, cy, sz / 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      return;
    }

    if (activeTool === "arrow") {
      // If a single arrow is already selected, check its handles first
      const selIds = currentValue.selected_ids || [];
      if (selIds.length === 1) {
        const selAnn = _effectiveAnnotations().find((a) => a.id === selIds[0] && a.type === "arrow");
        if (selAnn) {
          const handleR = Math.max(10 / displayScale, 8);
          const nearStart = Math.hypot(cx - selAnn.x1, cy - selAnn.y1) <= handleR;
          const nearEnd   = Math.hypot(cx - selAnn.x2, cy - selAnn.y2) <= handleR;
          if (nearStart || nearEnd) {
            const hitCps = defaultCps(selAnn);
            dragState = { type: "arrowHandle", id: selAnn.id,
              arrowHandle: nearStart ? "start" : "end",
              startCx: cx, startCy: cy,
              origX1: selAnn.x1, origY1: selAnn.y1, origX2: selAnn.x2, origY2: selAnn.y2,
              origCp1x: hitCps.cp1x, origCp1y: hitCps.cp1y, origCp2x: hitCps.cp2x, origCp2y: hitCps.cp2y };
            canvas.style.cursor = "grabbing"; return;
          }
          if (selAnn.is_bezier) {
            const cps = defaultCps(selAnn);
            const cpR = Math.max(8 / displayScale, 5);
            for (const [which, hx, hy] of [["cp1", cps.cp1x, cps.cp1y], ["cp2", cps.cp2x, cps.cp2y]]) {
              if (Math.hypot(cx - hx, cy - hy) <= cpR) {
                dragState = { type: "arrowCp", id: selAnn.id, which, startCx: cx, startCy: cy,
                  origCp1x: cps.cp1x, origCp1y: cps.cp1y, origCp2x: cps.cp2x, origCp2y: cps.cp2y };
                canvas.style.cursor = "grabbing"; return;
              }
            }
          }
        }
      }
      // No handle hit — start drawing a new arrow (deselect first)
      currentValue = { ...currentValue, selected_ids: [] };
      rebuildSettings();
      currentArrow = { x1: cx, y1: cy, x2: cx, y2: cy };
    }

    if (activeTool === "rect") {
      const selIds = currentValue.selected_ids || [];
      const hit = hitTest(cx, cy);
      if (hit && hit.type === "rect") {
        // Click on existing rect: select and start translate drag
        const newSelIds = selIds.includes(hit.id) ? selIds : [hit.id];
        currentValue = { ...currentValue, selected_ids: newSelIds };
        const origPositions = {};
        for (const id of newSelIds) {
          const a = _effectiveAnnotations().find((ann) => ann.id === id);
          if (a) origPositions[id] = { x: a.x ?? 0, y: a.y ?? 0 };
        }
        dragState = { type: "translate", startCx: cx, startCy: cy, origPositions,
          origPivotX: txFrame?.pivotX, origPivotY: txFrame?.pivotY };
        canvas.style.cursor = "grabbing";
        rebuildSettings(); renderCanvas(); return;
      }
      currentValue = { ...currentValue, selected_ids: [] };
      rebuildSettings();
      currentRect = { x1: cx, y1: cy, x2: cx, y2: cy };
    }

    if (activeTool === "ellipse") {
      const selIds = currentValue.selected_ids || [];
      const hit = hitTest(cx, cy);
      if (hit && hit.type === "ellipse") {
        // Click on existing ellipse: select and start translate drag
        const newSelIds = selIds.includes(hit.id) ? selIds : [hit.id];
        currentValue = { ...currentValue, selected_ids: newSelIds };
        const origPositions = {};
        for (const id of newSelIds) {
          const a = _effectiveAnnotations().find((ann) => ann.id === id);
          if (a) origPositions[id] = { x: a.x ?? 0, y: a.y ?? 0 };
        }
        dragState = { type: "translate", startCx: cx, startCy: cy, origPositions,
          origPivotX: txFrame?.pivotX, origPivotY: txFrame?.pivotY };
        canvas.style.cursor = "grabbing";
        rebuildSettings(); renderCanvas(); return;
      }
      currentValue = { ...currentValue, selected_ids: [] };
      rebuildSettings();
      currentEllipse = { x1: cx, y1: cy, x2: cx, y2: cy };
    }
  }

  // Handles pointermove: panning, zoom drag, paint stroke extension, shape preview,
  // transform drag (rotate/scale/translate/arrow handles), and marquee preview.
  function onPointerMove(e) {
    if (!isPointerDown) return;
    e.stopPropagation();

    if (isPanning) {
      panX = e.clientX - panStartX;
      panY = e.clientY - panStartY;
      _applyViewTransform();
      const isDefault = viewScale === 1 && panX === 0 && panY === 0;
      resetViewBtn.style.opacity = isDefault ? "0.4" : "1";
      resetViewBtn.style.pointerEvents = isDefault ? "none" : "auto";
      return;
    }

    if (dragState?.type === "zoom") {
      const dx = e.clientX - dragState.startClientX;
      // ~300px drag = 4× zoom change; exponential feel
      const newVS = Math.max(0.25, Math.min(10, dragState.startViewScale * Math.pow(2, dx / 150)));
      const ratio = newVS / dragState.startViewScale;
      panX = dragState.anchorMx - (dragState.anchorMx - dragState.startPanX) * ratio;
      panY = dragState.anchorMy - (dragState.anchorMy - dragState.startPanY) * ratio;
      viewScale = newVS;
      _applyViewTransform();
      const isDefault = viewScale === 1 && panX === 0 && panY === 0;
      resetViewBtn.style.opacity = isDefault ? "0.4" : "1";
      resetViewBtn.style.pointerEvents = isDefault ? "none" : "auto";
      return;
    }

    const [cx, cy] = screenToCanvas(e);

    if (activeTool === "paint" && currentStroke) {
      const pts = currentStroke.points;
      const last = pts[pts.length - 1];
      if (Math.hypot(cx - last[0], cy - last[1]) < 2) return; // skip micro-moves
      // Velocity-based size
      const now = performance.now();
      const dt = Math.max(1, now - lastPtTime);
      const dist = Math.hypot(cx - lastPtX, cy - lastPtY);
      velSmoothed = velSmoothed * 0.5 + (dist / dt) * 0.5;
      const baseSize = currentStroke.size;
      const sz = Math.max(baseSize * 0.25, baseSize / (1 + velSmoothed * 0.4));
      lastPtTime = now; lastPtX = cx; lastPtY = cy;
      // Draw variable-width segment incrementally: trapezoid + endpoint circle
      const pr = (last[2] ?? baseSize) / 2, cr = sz / 2;
      ctx.save();
      ctx.fillStyle = currentStroke.color;
      const ddx = cx - last[0], ddy = cy - last[1], len = Math.hypot(ddx, ddy);
      if (len > 0) {
        const nx = -ddy / len, ny = ddx / len;
        ctx.beginPath();
        ctx.moveTo(last[0] + nx * pr, last[1] + ny * pr);
        ctx.lineTo(cx + nx * cr, cy + ny * cr);
        ctx.lineTo(cx - nx * cr, cy - ny * cr);
        ctx.lineTo(last[0] - nx * pr, last[1] - ny * pr);
        ctx.closePath();
        ctx.fill();
      }
      ctx.beginPath(); ctx.arc(cx, cy, cr, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      pts.push([cx, cy, sz]);

    } else if (activeTool === "arrow" && currentArrow) {
      currentArrow = { ...currentArrow, x2: cx, y2: cy };
      renderCanvas();

    } else if (activeTool === "rect" && currentRect) {
      currentRect = { ...currentRect, x2: cx, y2: cy };
      renderCanvas();

    } else if (activeTool === "ellipse" && currentEllipse) {
      currentEllipse = { ...currentEllipse, x2: cx, y2: cy };
      renderCanvas();

    } else if (dragState && (activeTool === "select" || activeTool === "paint" || activeTool === "text" || activeTool === "arrow" || activeTool === "rect" || activeTool === "ellipse")) {
      if (dragState.type === "txRotate") {
        const pivot = dragState.pivot;
        const angle = Math.atan2(cy - pivot.y, cx - pivot.x);
        const dAngle = angle - dragState.origAngle;
        const newRotation = dragState.origRotation + dAngle;
        // Update the live frame so rendered handles rotate with content
        if (txFrame) txFrame = { ...txFrame, rotation: newRotation };
        const cos = Math.cos(dAngle), sin = Math.sin(dAngle);
        const { annotations: rotAnns, overrides: rotOvr } = _applyAnnotationMap(dragState.selIds, (a) => {
          const snap = dragState.origSnapshots[a.id];
          if (!snap) return a;
          if (a.type === "paint") {
            const ax = snap.cx + snap.x - pivot.x, ay = snap.cy + snap.y - pivot.y;
            return { ...a,
              x: ax*cos - ay*sin + pivot.x - snap.cx,
              y: ax*sin + ay*cos + pivot.y - snap.cy,
              rotation: snap.rotation + dAngle };
          } else if (a.type === "rect" || a.type === "ellipse") {
            const dx = snap.x - pivot.x, dy = snap.y - pivot.y;
            return { ...a,
              x: dx*cos - dy*sin + pivot.x, y: dx*sin + dy*cos + pivot.y,
              rotation: snap.rotation + dAngle };
          } else if (a.type === "text") {
            const r = snap.rotation + dAngle;
            const fontSize = Math.max(MIN_TEXT_SIZE, snap.font_size || DEFAULT_TEXT_SIZE);
            const lineHeight = fontSize * 1.2;
            const lines = (snap.text || "").split("\n");
            ctx.save(); ctx.font = `${fontSize}px sans-serif`;
            const hw = Math.max(1, ...lines.map((l) => ctx.measureText(l).width)) / 2;
            ctx.restore();
            const hh = (lineHeight * lines.length) / 2;
            // Compute text's original world-space center from snap TL + snap rotation
            const origR = snap.rotation;
            const origCx = snap.x + hw * Math.cos(origR) - hh * Math.sin(origR);
            const origCy = snap.y + hw * Math.sin(origR) + hh * Math.cos(origR);
            // Orbit that center around the group pivot by dAngle
            const dcx = origCx - pivot.x, dcy = origCy - pivot.y;
            const newCx = pivot.x + dcx * cos - dcy * sin;
            const newCy = pivot.y + dcx * sin + dcy * cos;
            // Derive new TL from new center + new rotation
            return { ...a, rotation: r,
              x: newCx - hw * Math.cos(r) + hh * Math.sin(r),
              y: newCy - hw * Math.sin(r) - hh * Math.cos(r) };
          } else if (a.type === "arrow") {
            const d1x = snap.x1 - pivot.x, d1y = snap.y1 - pivot.y;
            const d2x = snap.x2 - pivot.x, d2y = snap.y2 - pivot.y;
            const dc1x = snap.cp1x - pivot.x, dc1y = snap.cp1y - pivot.y;
            const dc2x = snap.cp2x - pivot.x, dc2y = snap.cp2y - pivot.y;
            return { ...a,
              x1: d1x*cos - d1y*sin + pivot.x, y1: d1x*sin + d1y*cos + pivot.y,
              x2: d2x*cos - d2y*sin + pivot.x, y2: d2x*sin + d2y*cos + pivot.y,
              cp1x: dc1x*cos - dc1y*sin + pivot.x, cp1y: dc1x*sin + dc1y*cos + pivot.y,
              cp2x: dc2x*cos - dc2y*sin + pivot.x, cp2y: dc2x*sin + dc2y*cos + pivot.y };
          }
          return a;
        });
        currentValue = { ...currentValue, annotations: rotAnns, overrides: rotOvr };
      } else if (dragState.type === "txScale") {
        const pivot = dragState.pivot;
        // Project mouse into frame's local space (rotate by -frameRotation around pivot)
        const dx = cx - pivot.x, dy = cy - pivot.y;
        const r = -dragState.origFrameRotation;
        const rcos = Math.cos(r), rsin = Math.sin(r);
        const lx = dx*rcos - dy*rsin;  // mouse in frame-local X
        const ly = dx*rsin + dy*rcos;  // mouse in frame-local Y
        // ratioX/Y: how far the mouse is compared to where the corner was
        let ratioX = Math.max(0.05, lx / (dragState.cornerSignX * dragState.origHalfW));
        let ratioY = Math.max(0.05, ly / (dragState.cornerSignY * dragState.origHalfH));
        if (e.shiftKey) { const ratio = Math.sqrt(ratioX * ratioY); ratioX = ratio; ratioY = ratio; }
        // Update live frame size so handles scale with content
        if (txFrame) txFrame = { ...txFrame, halfW: dragState.origHalfW * ratioX, halfH: dragState.origHalfH * ratioY };
        // Helper: scale an anchor point in frame-local space then back to world
        const fcos = Math.cos(dragState.origFrameRotation), fsin = Math.sin(dragState.origFrameRotation);
        const scaleAnchor = (ax, ay) => {
          const adx = ax - pivot.x, ady = ay - pivot.y;
          const alx = adx*rcos - ady*rsin, aly = adx*rsin + ady*rcos;
          const nlx = alx * ratioX, nly = aly * ratioY;
          return [pivot.x + nlx*fcos - nly*fsin, pivot.y + nlx*fsin + nly*fcos];
        };
        const { annotations: scAnns, overrides: scOvr } = _applyAnnotationMap(dragState.selIds, (a) => {
          const snap = dragState.origSnapshots[a.id];
          if (!snap) return a;
          if (a.type === "paint") {
            const [nax, nay] = scaleAnchor(snap.cx + snap.x, snap.cy + snap.y);
            return { ...a, x: nax - snap.cx, y: nay - snap.cy,
              scaleX: snap.scaleX * ratioX, scaleY: snap.scaleY * ratioY };
          } else if (a.type === "rect" || a.type === "ellipse") {
            const [nx, ny] = scaleAnchor(snap.x, snap.y);
            return { ...a, x: nx, y: ny,
              w: Math.max(2, snap.w * ratioX), h: Math.max(2, snap.h * ratioY) };
          } else if (a.type === "text") {
            // Text can only scale uniformly (font_size is one number).
            // Scale the text's world-space center, then recompute TL from new metrics.
            const r = snap.rotation || 0, cos = Math.cos(r), sin = Math.sin(r);
            const origFontSize = Math.max(MIN_TEXT_SIZE, snap.font_size || DEFAULT_TEXT_SIZE);
            const lines = (snap.text || "").split("\n");
            ctx.save(); ctx.font = `${origFontSize}px sans-serif`;
            const origTextW = Math.max(1, ...lines.map((l) => ctx.measureText(l).width));
            ctx.restore();
            const origHw = origTextW / 2, origHh = (origFontSize * 1.2 * lines.length) / 2;
            const [ncx, ncy] = scaleAnchor(snap.x + origHw * cos - origHh * sin,
                                            snap.y + origHw * sin + origHh * cos);
            const ratio = Math.sqrt(ratioX * ratioY);
            const newFontSize = Math.max(MIN_TEXT_SIZE, Math.round(origFontSize * ratio));
            ctx.save(); ctx.font = `${newFontSize}px sans-serif`;
            const newTextW = Math.max(1, ...lines.map((l) => ctx.measureText(l).width));
            ctx.restore();
            const newHw = newTextW / 2, newHh = (newFontSize * 1.2 * lines.length) / 2;
            return { ...a, font_size: newFontSize,
              x: ncx - newHw * cos + newHh * sin,
              y: ncy - newHw * sin - newHh * cos };
          } else if (a.type === "arrow") {
            const [nx1, ny1] = scaleAnchor(snap.x1, snap.y1);
            const [nx2, ny2] = scaleAnchor(snap.x2, snap.y2);
            const [nc1x, nc1y] = scaleAnchor(snap.cp1x, snap.cp1y);
            const [nc2x, nc2y] = scaleAnchor(snap.cp2x, snap.cp2y);
            return { ...a, x1: nx1, y1: ny1, x2: nx2, y2: ny2, cp1x: nc1x, cp1y: nc1y, cp2x: nc2x, cp2y: nc2y };
          }
          return a;
        });
        currentValue = { ...currentValue, annotations: scAnns, overrides: scOvr };
      } else if (dragState.type === "arrowCp") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        _applySingleUpdate(dragState.id, (a) => {
          if (dragState.which === "cp1")
            return { ...a, cp1x: dragState.origCp1x + dx, cp1y: dragState.origCp1y + dy };
          return { ...a, cp2x: dragState.origCp2x + dx, cp2y: dragState.origCp2y + dy };
        });
      } else if (dragState.type === "arrowHandle") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        _applySingleUpdate(dragState.id, (a) => {
          if (dragState.arrowHandle === "start")
            return { ...a, x1: dragState.origX1 + dx, y1: dragState.origY1 + dy,
              cp1x: dragState.origCp1x + dx, cp1y: dragState.origCp1y + dy };
          return { ...a, x2: dragState.origX2 + dx, y2: dragState.origY2 + dy,
            cp2x: dragState.origCp2x + dx, cp2y: dragState.origCp2y + dy };
        });
      } else if (dragState.type === "translate") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        const translateIds = Object.keys(dragState.origPositions);
        const { annotations: trAnns, overrides: trOvr } = _applyAnnotationMap(translateIds, (a) => {
          const orig = dragState.origPositions[a.id];
          if (!orig) return a;
          if (a.type === "arrow")
            return { ...a, x1: orig.x1 + dx, y1: orig.y1 + dy, x2: orig.x2 + dx, y2: orig.y2 + dy,
              cp1x: orig.cp1x + dx, cp1y: orig.cp1y + dy, cp2x: orig.cp2x + dx, cp2y: orig.cp2y + dy };
          return { ...a, x: orig.x + dx, y: orig.y + dy };
        });
        currentValue = { ...currentValue, annotations: trAnns, overrides: trOvr };
        // Move the frame with the selection so handles follow the annotation
        if (txFrame && dragState.origPivotX != null) {
          txFrame = { ...txFrame, pivotX: dragState.origPivotX + dx, pivotY: dragState.origPivotY + dy };
        }
      } else if (dragState.type === "marquee") {
        dragState = { ...dragState, x2: cx, y2: cy };
        // Compute which annotations would be selected so they can be previewed
        const mx1 = Math.min(dragState.startCx, cx), mx2 = Math.max(dragState.startCx, cx);
        const my1 = Math.min(dragState.startCy, cy), my2 = Math.max(dragState.startCy, cy);
        const directHits = _effectiveAnnotations()
          .filter((a) => _annotationIntersectsRect(a, mx1, my1, mx2, my2)).map((a) => a.id);
        const groupsHit = new Set(_effectiveAnnotations()
          .filter((a) => directHits.includes(a.id) && a.group_id).map((a) => a.group_id));
        marqueePreviewIds = new Set([
          ...directHits,
          ..._effectiveAnnotations().filter((a) => groupsHit.has(a.group_id)).map((a) => a.id),
        ]);
      }
      renderCanvas();
    }
  }

  // Handles pointerup: commits in-progress shapes as annotations, finalises drags, emits state.
  function onPointerUp(e) {
    if (!isPointerDown) return;
    isPointerDown = false;
    e.stopPropagation();

    if (isPanning) {
      isPanning = false;
      canvas.style.cursor = isAltHeld ? "grab" : _currentToolCursor();
      return;
    }

    if (dragState?.type === "zoom") {
      dragState = null;
      canvas.style.cursor = _currentToolCursor();
      return;
    }

    const [cx, cy] = screenToCanvas(e);
    canvas.style.cursor = _cursorForPos(cx, cy);

    if (activeTool === "paint" && currentStroke && currentStroke.points.length >= 1) {
      currentStroke.points = decimatePoints(currentStroke.points);
      const stroke = currentStroke;
      currentStroke = null;
      // Each stroke = its own paint annotation with independent transform
      const b = strokeBounds(stroke);
      const paintAnn = {
        id: _uid("paint"), type: "paint",
        strokes: [stroke],
        cx: b ? (b.minX + b.maxX) / 2 : 0,
        cy: b ? (b.minY + b.maxY) / 2 : 0,
        x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0,
      };
      // Paint is a continuous tool — like a brush in Figma/tldraw, you stay in paint
      // mode and keep drawing.  Don't select the stroke or switch tools; the user will
      // switch to select manually when they want to reposition something.
      currentValue = {
        ...currentValue,
        annotations: [...(currentValue.annotations || []), paintAnn],
        selected_ids: [],
      };
      _emit();
      rebuildSettings();
      canvas.focus({ preventScroll: true });
      renderCanvas();

    } else if (activeTool === "arrow" && currentArrow) {
      const arr = currentArrow;
      currentArrow = null;
      if (Math.hypot(arr.x2 - arr.x1, arr.y2 - arr.y1) > 5) {
        const ts = toolSettings.arrow;
        const ann = {
          id: _uid("arrow"), type: "arrow",
          x1: arr.x1, y1: arr.y1, x2: cx, y2: cy,
          cp1x: arr.x1 + (cx - arr.x1) / 3, cp1y: arr.y1 + (cy - arr.y1) / 3,
          cp2x: arr.x1 + (cx - arr.x1) * 2 / 3, cp2y: arr.y1 + (cy - arr.y1) * 2 / 3,
          color: ts.color,
          width: ts.width,
          has_start_arrow: ts.has_start_arrow ?? false,
          has_end_arrow: ts.has_end_arrow ?? true,
          is_bezier: ts.is_bezier ?? false,
          taper: ts.taper ?? false,
        };
        // Discrete object: select it and drop into select mode so the user can
        // immediately reposition/resize — same pattern as tldraw and modern Figma.
        currentValue = {
          ...currentValue,
          annotations: [...(currentValue.annotations || []), ann],
          selected_ids: [ann.id],
        };
        setTool("select");
        _emit();
        rebuildSettings();
      }
      renderCanvas();

    } else if (activeTool === "rect" && currentRect) {
      const r = currentRect;
      currentRect = null;
      if (Math.hypot(r.x2 - r.x1, r.y2 - r.y1) > 5) {
        const ts = toolSettings.rect;
        const ann = {
          id: _uid("rect"), type: "rect",
          x: (r.x1 + r.x2) / 2, y: (r.y1 + r.y2) / 2,
          w: Math.abs(r.x2 - r.x1), h: Math.abs(r.y2 - r.y1),
          rotation: 0,
          color: ts.color, width: ts.width, fill_color: ts.fill_color || "",
        };
        // Discrete object: select and return to select mode (same as arrow/ellipse).
        currentValue = { ...currentValue, annotations: [...(currentValue.annotations || []), ann], selected_ids: [ann.id] };
        setTool("select");
        _emit(); rebuildSettings();
      }
      renderCanvas();

    } else if (activeTool === "ellipse" && currentEllipse) {
      const el = currentEllipse;
      currentEllipse = null;
      if (Math.hypot(el.x2 - el.x1, el.y2 - el.y1) > 5) {
        const ts = toolSettings.ellipse;
        const ann = {
          id: _uid("ellipse"), type: "ellipse",
          x: (el.x1 + el.x2) / 2, y: (el.y1 + el.y2) / 2,
          w: Math.abs(el.x2 - el.x1), h: Math.abs(el.y2 - el.y1),
          rotation: 0,
          color: ts.color, width: ts.width, fill_color: ts.fill_color || "",
        };
        // Discrete object: select and return to select mode (same as arrow/rect).
        currentValue = { ...currentValue, annotations: [...(currentValue.annotations || []), ann], selected_ids: [ann.id] };
        setTool("select");
        _emit(); rebuildSettings();
      }
      renderCanvas();

    } else if (dragState && (activeTool === "select" || activeTool === "paint" || activeTool === "text" || activeTool === "arrow" || activeTool === "rect" || activeTool === "ellipse")) {
      if (dragState.type === "marquee") {
        const x1 = Math.min(dragState.startCx, dragState.x2);
        const y1 = Math.min(dragState.startCy, dragState.y2);
        const x2 = Math.max(dragState.startCx, dragState.x2);
        const y2 = Math.max(dragState.startCy, dragState.y2);
        if (x2 - x1 > 5 || y2 - y1 > 5) {
          const directHits = _effectiveAnnotations()
            .filter((a) => _annotationIntersectsRect(a, x1, y1, x2, y2))
            .map((a) => a.id);
          // Expand: if any member of a group is hit, include all members
          const groupsHit = new Set(_effectiveAnnotations()
            .filter((a) => directHits.includes(a.id) && a.group_id)
            .map((a) => a.group_id));
          const inRect = [...new Set([
            ...directHits,
            ..._effectiveAnnotations().filter((a) => groupsHit.has(a.group_id)).map((a) => a.id),
          ])];
          const merged = dragState.additive
            ? [...new Set([...(currentValue.selected_ids || []), ...inRect])]
            : inRect;
          currentValue = { ...currentValue, selected_ids: merged };
        }
        dragState = null;
        marqueePreviewIds = null;
        rebuildSettings();
        renderCanvas();
        _emit();
      } else {
        dragState = null;
        _buildTxFrame();
        renderCanvas();
        _emit();
      }
    }
  }

  const _ROTATE_CURSOR = `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 20 20'%3E%3Ccircle cx='10' cy='10' r='9' fill='none' stroke='white' stroke-width='2.5'/%3E%3Ccircle cx='10' cy='10' r='9' fill='none' stroke='%23333' stroke-width='1'/%3E%3Cpolygon points='10,1 14,6 10,4 6,6' fill='white' stroke='%23333' stroke-width='0.5'/%3E%3C/svg%3E") 10 10, grab`;

  // Returns the CSS cursor string for canvas position (cx, cy), accounting for which handle
  // (rotate, scale corner, translate body) is under the pointer and the active tool.
  function _cursorForPos(cx, cy) {
    const handleR = 8 / displayScale;
    const selIds = currentValue.selected_ids || [];

    if (_frameActiveTools.includes(activeTool)) {
      if (txFrame) {
        if (!txFrame.noRotate && Math.hypot(cx - frameRotHandle(txFrame, displayScale)[0], cy - frameRotHandle(txFrame, displayScale)[1]) <= handleR) return _ROTATE_CURSOR;
        const corners = frameCorners(txFrame);
        const localCornerSigns = [[-1,-1],[1,-1],[1,1],[-1,1]];
        for (let i = 0; i < corners.length; i++) {
          const [hx, hy] = corners[i];
          if (Math.hypot(cx - hx, cy - hy) <= handleR) {
            const [sx, sy] = localCornerSigns[i];
            const angle = Math.atan2(sy, sx) + txFrame.rotation;
            const deg = ((angle * 180 / Math.PI) % 180 + 180) % 180;
            return deg < 90 ? "nwse-resize" : "nesw-resize";
          }
        }
        if (Math.hypot(cx - txFrame.pivotX, cy - txFrame.pivotY) <= handleR) return "grab";
      }
      if (activeTool === "paint") return "crosshair";
      if (activeTool === "select") {
        if (selIds.length === 1) {
          const sa = _effectiveAnnotations().find((a) => a.id === selIds[0]);
          if (sa?.type === "arrow") {
            if (sa.is_bezier) {
              const cps = defaultCps(sa), cpR = Math.max(8 / displayScale, 5);
              if (Math.hypot(cx - cps.cp1x, cy - cps.cp1y) <= cpR) return "grab";
              if (Math.hypot(cx - cps.cp2x, cy - cps.cp2y) <= cpR) return "grab";
            }
            const ar = Math.max(10 / displayScale, 8);
            if (Math.hypot(cx - sa.x1, cy - sa.y1) <= ar || Math.hypot(cx - sa.x2, cy - sa.y2) <= ar) return "grab";
          }
        }
        if (txFrame) {
          const fdx = cx - txFrame.pivotX, fdy = cy - txFrame.pivotY;
          const fcos = Math.cos(-txFrame.rotation), fsin = Math.sin(-txFrame.rotation);
          if (Math.abs(fdx * fcos - fdy * fsin) <= txFrame.halfW &&
              Math.abs(fdx * fsin + fdy * fcos) <= txFrame.halfH) return "grab";
        }
        return hitTest(cx, cy) ? "grab" : "default";
      }
      // rect/ellipse: grab if hovering the selected shape
      const hit = hitTest(cx, cy);
      if (hit && (currentValue.selected_ids || []).includes(hit.id)) return "grab";
      return "crosshair";
    }

    if (activeTool === "arrow") {
      if (selIds.length === 1) {
        const sa = _effectiveAnnotations().find((a) => a.id === selIds[0] && a.type === "arrow");
        if (sa) {
          const ar = Math.max(10 / displayScale, 8);
          if (Math.hypot(cx - sa.x1, cy - sa.y1) <= ar || Math.hypot(cx - sa.x2, cy - sa.y2) <= ar) return "grab";
          if (sa.is_bezier) {
            const cps = defaultCps(sa), cpR = Math.max(8 / displayScale, 5);
            if (Math.hypot(cx - cps.cp1x, cy - cps.cp1y) <= cpR) return "grab";
            if (Math.hypot(cx - cps.cp2x, cy - cps.cp2y) <= cpR) return "grab";
          }
        }
      }
      return "crosshair";
    }

    if (activeTool === "text") {
      const hit = hitTest(cx, cy);
      return (hit && hit.type === "text") ? "grab" : "crosshair";
    }

    if (activeTool === "zoom") return isAltHeld ? "zoom-out" : "zoom-in";

    return "crosshair";
  }

  // Tracks which annotation is hovered and updates hoverId/hoverGroupId for highlight rendering.
  // Also updates the canvas cursor on every mouse move (not just during drags).
  function onMouseHover(e) {
    if (isPointerDown) return;
    const [cx, cy] = screenToCanvas(e);
    const prevHoverId = hoverId;
    const prevHoverGroupId = hoverGroupId;
    // Hover highlights only make sense in select mode — in drawing modes the user
    // is placing new content, not inspecting existing objects.
    if (activeTool === "select") {
      const hit = hitTest(cx, cy);
      hoverId = hit ? hit.id : null;
      hoverGroupId = hit?.group_id || null;
    } else {
      hoverId = null;
      hoverGroupId = null;
    }
    canvas.style.cursor = _cursorForPos(cx, cy);
    if (hoverId !== prevHoverId || hoverGroupId !== prevHoverGroupId) renderCanvas();
  }

  // Double-click to edit text (works in both text and select tools)
  canvas.addEventListener("dblclick", (e) => {
    if (activeTool !== "select" && activeTool !== "text") return;
    dragState = null; // cancel any drag that started from the first click
    const [cx, cy] = screenToCanvas(e);
    const hit = hitTest(cx, cy);
    if (hit && hit.type === "text") {
      // Refresh annotation from currentValue so we get the latest text
      const fresh = _effectiveAnnotations().find((a) => a.id === hit.id) || hit;
      startTextEdit(fresh);
    }
  });

  // ── emit / uid ─────────────────────────────────────────────────────────────
  let _emitSeq = 0;

  // Increments _emitSeq and calls onChange with the full current state.
  // _emitSeq lets handleUpdate detect and discard stale roundtrip echoes.
  function _emit() {
    _emitSeq++;
    if (onChange) onChange({ ...currentValue, tool_settings: { ...toolSettings }, _emitSeq });
  }

  // Generates a collision-resistant unique id with a human-readable type prefix.
  function _uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  }

  // ── update from props ──────────────────────────────────────────────────────

  // Receives new props from the parent framework. Stale roundtrip echoes (same or older _emitSeq)
  // are ignored for annotation data but still accepted for image URL / dimension changes.
  function handleUpdate(newProps) {
    const rawNv = (newProps?.value && typeof newProps.value === "object") ? newProps.value : {};

    // Stale-roundtrip guard: if we've emitted more recently than this incoming
    // value, the framework is echoing back an old snapshot. Accept only
    // image/dimension fields (set externally by Python), not annotations.
    // Also block full updates during any active pointer drag — local state is
    // authoritative mid-drag and the correct state will be emitted on pointerup.
    const incomingSeq = rawNv._emitSeq || 0;
    if ((isPointerDown) || (incomingSeq > 0 && incomingSeq < _emitSeq)) {
      const urlChanged = (rawNv.image_url || "") !== currentValue.image_url;
      const dimsChanged = (rawNv.canvas_width || 0) !== currentValue.canvas_width ||
                          (rawNv.canvas_height || 0) !== currentValue.canvas_height;
      if (urlChanged || dimsChanged) {
        currentValue = {
          ...currentValue,
          image_url: rawNv.image_url || currentValue.image_url,
          raw_url: rawNv.raw_url || currentValue.raw_url,
          canvas_width: rawNv.canvas_width || currentValue.canvas_width,
          canvas_height: rawNv.canvas_height || currentValue.canvas_height,
        };
        applyCanvasScale();
        renderCanvas();
      }
      return;
    }

    // Fresh update — apply fully
    const nv = { ...defaultData(), ...rawNv };
    if (rawNv.selected_id && !rawNv.selected_ids) {
      nv.selected_ids = [rawNv.selected_id];
    } else if (!Array.isArray(nv.selected_ids)) {
      nv.selected_ids = [];
    }
    const urlChanged = nv.image_url !== currentValue.image_url;
    const dimsChanged = nv.canvas_width !== currentValue.canvas_width ||
                        nv.canvas_height !== currentValue.canvas_height;
    const mergedTS = { ...toolSettings, ...(nv.tool_settings || {}) };
    if (nv.tool_settings?.rect)    mergedTS.rect    = { ...defTS.rect,    ...mergedTS.rect };
    if (nv.tool_settings?.ellipse) mergedTS.ellipse = { ...defTS.ellipse, ...mergedTS.ellipse };
    currentValue = { ...defaultData(), ...nv, tool_settings: mergedTS };
    toolSettings = { ...currentValue.tool_settings };
    activeTool = currentValue.active_tool || activeTool;

    for (const [tid, btn] of Object.entries(toolBtns)) {
      btn.className = "ais-tool-btn" + (tid === activeTool ? " active" : "");
    }
    canvas.style.cursor = _currentToolCursor();
    rebuildSettings(!!document.getElementById("ais-layer-popup"));

    if (dimsChanged || urlChanged) applyCanvasScale();
    renderCanvas();
  }

  // ── Expand modal ───────────────────────────────────────────────────────────

  let _modalEl = null;

  function _openModal() {
    const backdrop = document.createElement("div");
    backdrop.style.cssText =
      "position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,0.6);" +
      "backdrop-filter:blur(6px);-webkit-backdrop-filter:blur(6px);" +
      "display:flex;align-items:center;justify-content:center;";

    const dialog = document.createElement("div");
    dialog.style.cssText =
      "position:relative;width:90%;height:90%;" +
      "background:var(--background);border:1px solid var(--border);border-radius:8px;" +
      "overflow:hidden;display:flex;flex-direction:column;" +
      "box-shadow:0 10px 40px rgba(0,0,0,0.5),0 0 0 1px rgba(255,255,255,0.04);";

    const modalHeader = document.createElement("div");
    modalHeader.style.cssText =
      "display:flex;align-items:center;justify-content:space-between;padding:0 16px;" +
      "height:48px;border-bottom:1px solid var(--border);flex-shrink:0;background:var(--card);";

    const modalTitle = document.createElement("span");
    modalTitle.style.cssText = "font-size:13px;font-weight:600;color:var(--foreground);letter-spacing:0.01em;";
    modalTitle.textContent = "Annotate Image";

    const closeBtn = document.createElement("button");
    closeBtn.style.cssText =
      "display:flex;align-items:center;justify-content:center;width:28px;height:28px;" +
      "padding:0;border:none;border-radius:6px;background:transparent;" +
      "color:var(--muted-foreground);opacity:0.7;cursor:pointer;transition:opacity 0.15s;";
    closeBtn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>`;
    closeBtn.addEventListener("mouseenter", () => { closeBtn.style.opacity = "1"; });
    closeBtn.addEventListener("mouseleave", () => { closeBtn.style.opacity = "0.7"; });
    closeBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); _closeModal(); });

    modalHeader.appendChild(modalTitle);
    modalHeader.appendChild(closeBtn);
    dialog.appendChild(modalHeader);
    dialog.appendChild(wrapper);
    backdrop.appendChild(dialog);
    document.body.appendChild(backdrop);
    _modalEl = backdrop;

    fsBtn.innerHTML = "";
    fsBtn.appendChild(mkIcon("contract", 15));
    requestAnimationFrame(applyCanvasScale);

    backdrop.addEventListener("pointerdown", (e) => {
      if (e.target === backdrop) _closeModal();
    });
  }

  function _closeModal() {
    if (!_modalEl) return;
    container.appendChild(wrapper);
    _modalEl.remove();
    _modalEl = null;
    fsBtn.innerHTML = "";
    fsBtn.appendChild(mkIcon("expand", 15));
    requestAnimationFrame(applyCanvasScale);
  }

  // ── cleanup ────────────────────────────────────────────────────────────────

  // Tears down all event listeners, observers, and DOM nodes. Called when the widget is unmounted.
  function cleanup() {
    _closeModal();
    commitTextEdit();
    dismissLayerPopup?.();
    _tooltip.cleanup();
    _cleanupHotkeys();
    resizeObserver.disconnect();
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerUp);
    canvas.removeEventListener("mousemove", onMouseHover);
    wrapper.remove();
    delete container._aisInst;
  }

  // ── init ──────────────────────────────────────────────────────────────────
  const _settingsMod = createSettings(settingsArea, {
    addTooltip: _addTooltip,
    getState: () => ({ activeTool, toolSettings, currentValue, textEditId, textInput, displayScale, viewScale }),
    setCurrentValue: (v) => { currentValue = v; },
    applySingleUpdate: _applySingleUpdate,
    applyAnnotationMap: _applyAnnotationMap,
    effectiveAnnotations: _effectiveAnnotations,
    autoResizeTextarea: _autoResizeTextarea,
    renderCanvas,
    emit: _emit,
    rebuild: () => rebuildSettings(),
  });
  buildToolSettings = _settingsMod.buildToolSettings;
  buildAnnotationSettings = _settingsMod.buildAnnotationSettings;
  buildMultiSettings = _settingsMod.buildMultiSettings;

  const _hudMod = createHud(hudEl, {
    addTooltip: _addTooltip,
    getState: () => ({ activeTool, currentValue }),
    setCurrentValue: (v) => { currentValue = v; },
    effectiveAnnotations: _effectiveAnnotations,
    applyAnnotationMap: _applyAnnotationMap,
    commitTextEdit,
    uid: _uid,
    emit: _emit,
    renderCanvas,
    rebuildSettings: (keepLayerPopup) => rebuildSettings(keepLayerPopup),
  });
  updateHud = _hudMod.update;
  dismissLayerPopup = _hudMod.dismissLayerPopup;

  setTool(activeTool);
  applyCanvasScale();
  renderCanvas();

  container._aisInst = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
