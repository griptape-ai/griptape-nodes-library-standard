// AnnotateImageSimple — single-image annotation widget
// Layout: toolbar (tools left, settings right) + canvas
// Tools: Select, Paint, Text, Arrow

// ── Lucide SVG icons (inlined paths, MIT licensed) ────────────────────────────

const ICON_PATHS = {
  select:  `<path d="m4 4 7.07 17 2.51-7.39L21 11.07z"/>`,
  paint:   `<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>`,
  text:    `<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>`,
  arrow:   `<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>`,
  rect:    `<rect x="3" y="3" width="18" height="18" rx="2"/>`,
  ellipse: `<ellipse cx="12" cy="12" rx="10" ry="6"/>`,
  trash:   `<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>`,
};

function mkIcon(name, size = 15) {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("width", size);
  svg.setAttribute("height", size);
  svg.setAttribute("fill", "none");
  svg.setAttribute("stroke", "currentColor");
  svg.setAttribute("stroke-width", "2");
  svg.setAttribute("stroke-linecap", "round");
  svg.setAttribute("stroke-linejoin", "round");
  svg.style.cssText = "display:block;flex-shrink:0;pointer-events:none;";
  svg.innerHTML = ICON_PATHS[name] || "";
  return svg;
}

// ── global style injection ────────────────────────────────────────────────────

function injectStyles() {
  const id = "ais-styles";
  let el = document.getElementById(id);
  if (!el) { el = document.createElement("style"); el.id = id; document.head.appendChild(el); }
  el.textContent = `
    .ais-tool-btn { background:transparent; border:none; border-radius:4px; color:var(--muted-foreground); cursor:pointer; width:28px; height:28px; display:flex; align-items:center; justify-content:center; transition:background 0.15s,color 0.15s; flex-shrink:0; padding:0; }
    .ais-tool-btn:hover { background:var(--muted); color:var(--foreground); }
    .ais-tool-btn.active { background:#7a9db8; color:#ffffff; }
    .ais-setting-label { font-size:11px; color:var(--muted-foreground); }
    .ais-range { accent-color:#7a9db8; cursor:pointer; width:80px; min-width:30px; flex-shrink:1; }
    .ais-color-btn { width:22px; height:22px; border-radius:4px; border:2px solid var(--border,#555); cursor:pointer; flex-shrink:0; }
    .ais-color-input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .ais-val-label { font-size:11px; color:var(--foreground); min-width:20px; text-align:right; flex-shrink:0; }
  `;
}

// ── default data ──────────────────────────────────────────────────────────────

function defaultData() {
  return {
    image_url: "",
    raw_url: "",
    canvas_width: 0,
    canvas_height: 0,
    annotations: [],
    active_tool: "select",
    tool_settings: {
      paint:   { color: "#ff0000", size: 8 },
      text:    { color: "#ffffff", font_size: 48 },
      arrow:   { color: "#ff0000", width: 3, has_start_arrow: false, has_end_arrow: true, is_bezier: false },
      rect:    { color: "#ff0000", width: 2, fill_color: "" },
      ellipse: { color: "#ff0000", width: 2, fill_color: "" },
    },
    selected_ids: [],
  };
}

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

  // zoom / pan state
  let viewScale = 1;
  let panX = 0, panY = 0;
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let isAltHeld = false;
  let resetViewBtn = null;

  // unified transform frame (OBB)
  let txFrame = null; // { pivotX, pivotY, rotation, halfW, halfH }
  const _frameActiveTools = ["select", "paint", "rect", "ellipse"];

  // pointer state
  let isPointerDown = false;
  let currentStroke = null;
  let strokeLastMid = null; // tracks last bezier midpoint for incremental draw
  let currentArrow = null;
  let currentRect = null;
  let currentEllipse = null;
  let dragState = null;
  let lastPtTime = 0, lastPtX = 0, lastPtY = 0, velSmoothed = 0;

  // text edit state
  let textInput = null;
  let textEditId = null;
  let hoverId = null;  // annotation id being hovered (text tool)

  // image cache
  const imageCache = {};
  function urlCacheKey(url) { return url ? url.split("?")[0] : url; }
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
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText =
    "display:flex;flex-direction:column;width:100%;background:var(--background);border-radius:6px;" +
    "font-family:sans-serif;box-sizing:border-box;overflow:hidden;";

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText =
    "display:flex;align-items:center;gap:4px;padding:5px 8px;" +
    "background:var(--card);border-bottom:1px solid var(--border);flex-shrink:0;flex-wrap:wrap;min-height:38px;";

  // Tool buttons (left side)
  const TOOLS = [
    { id: "select",  title: "Select & Move" },
    { id: "paint",   title: "Paint" },
    { id: "text",    title: "Text" },
    { id: "arrow",   title: "Arrow" },
    { id: "rect",    title: "Rectangle" },
    { id: "ellipse", title: "Ellipse / Circle" },
  ];
  const toolBtns = {};
  for (const t of TOOLS) {
    const btn = document.createElement("button");
    btn.className = "ais-tool-btn" + (t.id === activeTool ? " active" : "");
    btn.title = t.title;
    btn.appendChild(mkIcon(t.id));
    btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); setTool(t.id); btn.blur(); });
    toolbar.appendChild(btn);
    toolBtns[t.id] = btn;
  }

  // Divider
  const divider = document.createElement("div");
  divider.style.cssText = "width:1px;height:20px;background:var(--border);margin:0 4px;flex-shrink:0;";
  toolbar.appendChild(divider);

  // Reset-view button (dimmed when at default zoom/pan)
  resetViewBtn = document.createElement("button");
  resetViewBtn.className = "ais-tool-btn";
  resetViewBtn.title = "Reset view (fit to window)";
  resetViewBtn.style.opacity = "0.4";
  resetViewBtn.style.pointerEvents = "none";
  resetViewBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
    <rect x="1" y="1" width="12" height="12" rx="1.5"/>
    <path d="M4.5 1v2.5H2M9.5 1v2.5H12M4.5 13v-2.5H2M9.5 13v-2.5H12"/>
  </svg>`;
  resetViewBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); resetView(); resetViewBtn.blur(); });
  toolbar.appendChild(resetViewBtn);

  // Second divider
  const divider2 = document.createElement("div");
  divider2.style.cssText = "width:1px;height:20px;background:var(--border);margin:0 4px;flex-shrink:0;";
  toolbar.appendChild(divider2);

  // Settings area (right of divider, grows to fill)
  const settingsArea = document.createElement("div");
  settingsArea.style.cssText = "display:flex;align-items:center;gap:6px;flex:1;min-width:0;overflow:hidden;justify-content:flex-end;";
  toolbar.appendChild(settingsArea);

  // Canvas area
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;width:100%;overflow:hidden;background:#111;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;transform-origin:top left;cursor:crosshair;outline:none;" +
    "box-shadow:0 0 0 1px rgba(122,157,184,0.35);";
  canvas.tabIndex = 0; // focusable so keyboard events naturally target canvas
  canvas.width = 800;
  canvas.height = 600;
  canvas.style.width = "800px";
  canvas.style.height = "600px";

  canvasWrap.appendChild(canvas);
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
  let resizeRafId = null;
  const resizeObserver = new ResizeObserver(() => {
    if (resizeRafId) cancelAnimationFrame(resizeRafId);
    resizeRafId = requestAnimationFrame(() => { resizeRafId = null; applyCanvasScale(); });
  });
  resizeObserver.observe(canvasWrap);

  // ── zoom via scroll wheel ─────────────────────────────────────────────────
  canvasWrap.addEventListener("wheel", (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    const newVS = Math.max(0.25, Math.min(10, viewScale * factor));
    const rect = canvasWrap.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    const ratio = newVS / viewScale;
    panX = mx - (mx - panX) * ratio;
    panY = my - (my - panY) * ratio;
    viewScale = newVS;
    _applyViewTransform();
    const isDefault = viewScale === 1 && panX === 0 && panY === 0;
    resetViewBtn.style.opacity = isDefault ? "0.4" : "1";
    resetViewBtn.style.pointerEvents = isDefault ? "none" : "auto";
  }, { passive: false });

  // ── Alt key state for pan cursor ──────────────────────────────────────────
  function _onAltDown(e) {
    if (e.key === "Alt" && !isAltHeld) {
      isAltHeld = true;
      if (!isPointerDown) canvas.style.cursor = "grab";
    }
  }
  function _onAltUp(e) {
    if (e.key === "Alt") {
      isAltHeld = false;
      if (!isPanning && !isPointerDown) canvas.style.cursor = _currentToolCursor();
    }
  }
  document.addEventListener("keydown", _onAltDown);
  document.addEventListener("keyup",   _onAltUp);

  function _currentToolCursor() {
    return activeTool === "select" ? "default" : "crosshair";
  }

  function _applyViewTransform() {
    const totalScale = displayScale * viewScale;
    const ch = currentValue.canvas_height || 600;
    canvas.style.transform = `translate(${panX}px, ${panY}px) scale(${totalScale})`;
    canvasWrap.style.height = ch * displayScale + "px";
  }

  function applyCanvasScale() {
    const cw = currentValue.canvas_width || 800;
    const ch = currentValue.canvas_height || 600;
    const areaW = canvasWrap.clientWidth || 300;
    const newScale = areaW / cw;

    const dimsChanged = canvas.width !== cw || canvas.height !== ch;
    if (dimsChanged) {
      canvas.width = cw;
      canvas.height = ch;
      canvas.style.width = cw + "px";
      canvas.style.height = ch + "px";
    }
    if (newScale !== displayScale || dimsChanged) {
      displayScale = newScale;
      _applyViewTransform();
    }
    if (dimsChanged) renderCanvas();
  }

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

  // ── tool settings panel ───────────────────────────────────────────────────
  let colorPickerEl = null;

  function rebuildSettings(keepLayerPopup = false) {
    if (!keepLayerPopup) _dismissLayerPopup();
    _buildTxFrame();
    settingsArea.innerHTML = "";
    colorPickerEl = null;

    if (activeTool === "select") {
      const selIds = currentValue.selected_ids || [];
      if (selIds.length > 0) {
        _buildLayerOrderButton(selIds);
        const sep = document.createElement("div");
        sep.style.cssText = "width:1px;height:16px;background:var(--border);flex-shrink:0;";
        settingsArea.appendChild(sep);
      }
      if (selIds.length === 1) {
        const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
        if (selAnn) _buildAnnotationSettings(selAnn);
      } else if (selIds.length > 1) {
        _buildMultiSettings(selIds);
      } else {
        const hint = document.createElement("span");
        hint.className = "ais-setting-label";
        hint.textContent = "Click to select · Shift+click to add · Drag to select area";
        hint.style.opacity = "0.5";
        settingsArea.appendChild(hint);
      }
      return;
    }

    // Paint/arrow/rect/ellipse tool with a single matching annotation selected: show its settings
    if (activeTool === "paint" || activeTool === "arrow" || activeTool === "rect" || activeTool === "ellipse") {
      const selIds = currentValue.selected_ids || [];
      if (selIds.length === 1) {
        const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
        if (selAnn?.type === activeTool) {
          _buildAnnotationSettings(selAnn);
          return;
        }
      }
    }

    // All other tools: always show tool settings (brush size, color, etc.)
    _buildToolSettings();
  }

  // onChange(color, emit) — emit=false during drag, emit=true on commit
  function _buildColorSwatch(color, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;display:flex;align-items:center;";
    const swatch = document.createElement("div");
    swatch.className = "ais-color-btn"; swatch.style.background = color;
    colorPickerEl = document.createElement("input");
    colorPickerEl.type = "color"; colorPickerEl.value = color;
    colorPickerEl.className = "ais-color-input";
    colorPickerEl.addEventListener("input", () => {
      swatch.style.background = colorPickerEl.value;
      onChange(colorPickerEl.value, false);
    });
    colorPickerEl.addEventListener("change", () => onChange(colorPickerEl.value, true));
    swatch.addEventListener("click", () => colorPickerEl.click());
    wrap.appendChild(swatch); wrap.appendChild(colorPickerEl);
    settingsArea.appendChild(wrap);
  }

  function _fmtNum(v) {
    const n = Number(v);
    if (!isFinite(n)) return "0";
    if (Number.isInteger(n)) return String(n);
    const r = Math.round(n * 100) / 100;
    return Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/0+$/, "");
  }

  function _buildSizeSlider(label, min, max, value, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;align-items:center;gap:3px;flex-shrink:1;min-width:0;";
    const lbl = document.createElement("span");
    lbl.className = "ais-setting-label"; lbl.textContent = label;
    const slider = document.createElement("input");
    slider.type = "range"; slider.className = "ais-range";
    slider.min = min; slider.max = max; slider.value = value;
    const valLbl = document.createElement("span");
    valLbl.className = "ais-val-label"; valLbl.textContent = _fmtNum(value);
    slider.addEventListener("input", () => { const sz = Number(slider.value); valLbl.textContent = _fmtNum(sz); onChange(sz, false); });
    slider.addEventListener("change", () => onChange(Number(slider.value), true));
    wrap.appendChild(lbl); wrap.appendChild(slider); wrap.appendChild(valLbl);
    settingsArea.appendChild(wrap);
  }

  function _buildFillColorSwatch(fillColor, onChangeColor) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;display:flex;align-items:center;gap:2px;";
    const swatch = document.createElement("div");
    swatch.className = "ais-color-btn";
    swatch.title = "Fill color";
    if (fillColor) {
      swatch.style.background = fillColor;
    } else {
      swatch.style.background = "repeating-conic-gradient(#888 0% 25%,#333 0% 50%) 0 0/8px 8px";
    }
    const pickerInput = document.createElement("input");
    pickerInput.type = "color";
    pickerInput.value = fillColor || "#ffffff";
    pickerInput.className = "ais-color-input";
    pickerInput.addEventListener("input", () => {
      swatch.style.background = pickerInput.value;
      onChangeColor(pickerInput.value, false);
    });
    pickerInput.addEventListener("change", () => onChangeColor(pickerInput.value, true));
    swatch.addEventListener("click", () => pickerInput.click());
    const clearBtn = document.createElement("button");
    clearBtn.className = "ais-tool-btn";
    clearBtn.title = "No fill";
    clearBtn.style.cssText = "width:16px;height:16px;font-size:11px;padding:0;";
    clearBtn.textContent = "✕";
    clearBtn.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      swatch.style.background = "repeating-conic-gradient(#888 0% 25%,#333 0% 50%) 0 0/8px 8px";
      onChangeColor("", true);
    });
    wrap.appendChild(swatch);
    wrap.appendChild(pickerInput);
    wrap.appendChild(clearBtn);
    settingsArea.appendChild(wrap);
  }

  function _buildArrowToggles(source, onToggle) {
    const makeToggleBtn = (label, title, active, onClick) => {
      const btn = document.createElement("button");
      btn.className = "ais-tool-btn" + (active ? " active" : "");
      btn.title = title;
      btn.style.cssText = "font-size:14px;font-weight:bold;width:26px;height:26px;line-height:1;";
      btn.textContent = label;
      btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); onClick(); });
      return btn;
    };
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:2px;";
    row.appendChild(makeToggleBtn("←", "Start arrowhead", source.has_start_arrow ?? false, () => {
      onToggle({ has_start_arrow: !(source.has_start_arrow ?? false) });
    }));
    row.appendChild(makeToggleBtn("→", "End arrowhead", source.has_end_arrow ?? true, () => {
      onToggle({ has_end_arrow: !(source.has_end_arrow ?? true) });
    }));
    row.appendChild(makeToggleBtn("⌒", "Bezier curve", source.is_bezier ?? false, () => {
      onToggle({ is_bezier: !(source.is_bezier ?? false) });
    }));
    settingsArea.appendChild(row);
  }

  function _buildToolSettings() {
    const ts = toolSettings[activeTool] || {};
    if (activeTool === "arrow") {
      _buildArrowToggles(toolSettings.arrow, (changes) => {
        toolSettings.arrow = { ...toolSettings.arrow, ...changes };
        currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
        rebuildSettings();
        renderCanvas();
        _emit();
      });
    }
    const isShape = activeTool === "rect" || activeTool === "ellipse";
    const sizeKey = activeTool === "text" ? "font_size"
      : (activeTool === "arrow" || isShape) ? "width"
      : "size";
    const sizeVal = ts[sizeKey] ?? (activeTool === "text" ? 48 : (activeTool === "arrow" || isShape) ? (isShape ? 2 : 3) : 8);
    const sizeMin = activeTool === "text" ? 8 : 1;
    const sizeMax = activeTool === "text" ? 120 : (activeTool === "arrow" || isShape) ? 20 : 80;
    const sizeLbl = (activeTool === "arrow" || isShape) ? "Width" : "Size";
    _buildSizeSlider(sizeLbl, sizeMin, sizeMax, sizeVal, (sz, emit) => {
      toolSettings[activeTool][sizeKey] = sz;
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      renderCanvas();
      if (emit) _emit();
    });
    const color = ts.color || "#ff0000";
    _buildColorSwatch(color, (col, emit) => {
      toolSettings[activeTool].color = col;
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      if (emit) _emit();
    });
    if (isShape) {
      _buildFillColorSwatch(ts.fill_color || "", (col, emit) => {
        toolSettings[activeTool].fill_color = col;
        currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
        renderCanvas();
        if (emit) _emit();
      });
    }
  }

  function _buildAnnotationSettings(ann) {
    if (ann.type === "arrow") {
      _buildArrowToggles(ann, (changes) => {
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => a.id === ann.id ? { ...a, ...changes } : a),
        };
        // Sync arrow-style toggles to tool settings so next arrow uses same style
        toolSettings.arrow = { ...toolSettings.arrow, ...changes };
        currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
        renderCanvas();
        rebuildSettings();
        _emit();
      });
    }

    let color;
    if (ann.type === "paint") {
      color = (ann.strokes && ann.strokes[0]) ? ann.strokes[0].color : "#ff0000";
    } else {
      color = ann.color || "#ff0000";
    }

    if (ann.type === "paint") {
      const baseSize = (ann.strokes && ann.strokes[0]) ? (ann.strokes[0].size ?? 8) : 8;
      const currentSize = Math.max(1, Math.round(baseSize * (ann.sizeScale ?? 1)));
      _buildSizeSlider("Size", 1, 80, currentSize, (sz, emit) => {
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) =>
            a.id === ann.id ? { ...a, sizeScale: sz / baseSize } : a
          ),
        };
        renderCanvas();
        if (emit) _emit();
      });
    }

    const isShape = ann.type === "rect" || ann.type === "ellipse";
    const sizeKey = ann.type === "text" ? "font_size" : (ann.type === "arrow" || isShape) ? "width" : null;
    if (sizeKey) {
      const sizeVal = ann[sizeKey] ?? (ann.type === "text" ? 48 : isShape ? 2 : 3);
      const sizeMin = ann.type === "text" ? 8 : 1;
      const sizeMax = ann.type === "text" ? 120 : 20;
      const sizeLbl = ann.type === "text" ? "Size" : "Width";
      _buildSizeSlider(sizeLbl, sizeMin, sizeMax, sizeVal, (sz, emit) => {
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) =>
            a.id === ann.id ? { ...a, [sizeKey]: sz } : a
          ),
        };
        if (ann.type === "arrow") { toolSettings.arrow.width = sz; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
        if (ann.type === "text") { toolSettings.text.font_size = sz; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
        if (isShape) { toolSettings[ann.type].width = sz; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
        if (textInput && textEditId === ann.id && sizeKey === "font_size") {
          textInput.style.fontSize = sz * displayScale * viewScale + "px"; _autoResizeTextarea();
        }
        renderCanvas();
        if (emit) _emit();
      });
    }

    _buildColorSwatch(color, (col, emit) => {
      currentValue = {
        ...currentValue,
        annotations: currentValue.annotations.map((a) => {
          if (a.id !== ann.id) return a;
          if (a.type === "paint") return { ...a, strokes: (a.strokes || []).map((s) => ({ ...s, color: col })) };
          return { ...a, color: col };
        }),
      };
      if (ann.type === "arrow") { toolSettings.arrow.color = col; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
      if (ann.type === "text") { toolSettings.text.color = col; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
      if (ann.type === "paint") { toolSettings.paint.color = col; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
      if (isShape) { toolSettings[ann.type].color = col; currentValue = { ...currentValue, tool_settings: { ...toolSettings } }; }
      if (textInput && textEditId === ann.id) {
        textInput.style.color = col; textInput.style.borderBottomColor = col;
      }
      renderCanvas();
      if (emit) _emit();
    });

    if (isShape) {
      _buildFillColorSwatch(ann.fill_color || "", (col, emit) => {
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => a.id === ann.id ? { ...a, fill_color: col } : a),
        };
        toolSettings[ann.type].fill_color = col;
        currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
        renderCanvas();
        if (emit) _emit();
      });
    }

  }

  function _buildMultiSettings(selIds) {
    const anns = (currentValue.annotations || []).filter((a) => selIds.includes(a.id));
    // Capture original sizes when the panel is built; slider applies ratio to these originals
    const origSizes = {};
    for (const a of anns) {
      if (a.type === "paint") origSizes[a.id] = a.sizeScale ?? 1;
      else if (a.type === "text") origSizes[a.id] = a.font_size ?? 48;
      else if (a.type === "arrow") origSizes[a.id] = a.width ?? 3;
      else if (a.type === "rect" || a.type === "ellipse") origSizes[a.id] = { w: a.w ?? 100, h: a.h ?? 100 };
    }
    _buildSizeSlider("Scale %", 25, 400, 100, (val, emit) => {
      const ratio = val / 100;
      currentValue = {
        ...currentValue,
        annotations: currentValue.annotations.map((a) => {
          if (!selIds.includes(a.id)) return a;
          if (a.type === "paint") return { ...a, sizeScale: (origSizes[a.id] ?? 1) * ratio };
          if (a.type === "text") return { ...a, font_size: Math.max(8, Math.round((origSizes[a.id] ?? 48) * ratio)) };
          if (a.type === "arrow") return { ...a, width: Math.max(1, (origSizes[a.id] ?? 3) * ratio) };
          if (a.type === "rect" || a.type === "ellipse") {
            const orig = origSizes[a.id] || { w: 100, h: 100 };
            return { ...a, w: Math.max(2, orig.w * ratio), h: Math.max(2, orig.h * ratio) };
          }
          return a;
        }),
      };
      renderCanvas();
      if (emit) _emit();
    });
    let firstColor = "#ff0000";
    for (const a of anns) {
      if (a.type === "paint" && a.strokes?.[0]) { firstColor = a.strokes[0].color; break; }
      if (a.color) { firstColor = a.color; break; }
    }
    _buildColorSwatch(firstColor, (col, emit) => {
      currentValue = {
        ...currentValue,
        annotations: currentValue.annotations.map((a) => {
          if (!selIds.includes(a.id)) return a;
          if (a.type === "paint") return { ...a, strokes: (a.strokes || []).map((s) => ({ ...s, color: col })) };
          return { ...a, color: col };
        }),
      };
      renderCanvas();
      if (emit) _emit();
    });
  }

  function _reorderAnnotations(selIds, action) {
    const anns = [...(currentValue.annotations || [])];
    const selSet = new Set(selIds);
    const sIdxs = anns.map((a, i) => (selSet.has(a.id) ? i : -1)).filter((i) => i >= 0);
    if (!sIdxs.length) return anns;
    if (action === "front") {
      return [...anns.filter((a) => !selSet.has(a.id)), ...anns.filter((a) => selSet.has(a.id))];
    }
    if (action === "back") {
      return [...anns.filter((a) => selSet.has(a.id)), ...anns.filter((a) => !selSet.has(a.id))];
    }
    if (action === "forward") {
      const result = [...anns];
      const idxs = result.map((a, i) => (selSet.has(a.id) ? i : -1)).filter((i) => i >= 0);
      const lastSel = Math.max(...idxs);
      let swapIdx = lastSel + 1;
      while (swapIdx < result.length && selSet.has(result[swapIdx].id)) swapIdx++;
      if (swapIdx < result.length) {
        const [item] = result.splice(swapIdx, 1);
        result.splice(Math.min(...idxs), 0, item);
      }
      return result;
    }
    if (action === "backward") {
      const result = [...anns];
      const idxs = result.map((a, i) => (selSet.has(a.id) ? i : -1)).filter((i) => i >= 0);
      const firstSel = Math.min(...idxs);
      let swapIdx = firstSel - 1;
      while (swapIdx >= 0 && selSet.has(result[swapIdx].id)) swapIdx--;
      if (swapIdx >= 0) {
        const lastSel = Math.max(...idxs);
        const [item] = result.splice(swapIdx, 1);
        result.splice(lastSel, 0, item);
      }
      return result;
    }
    return anns;
  }

  function _dismissLayerPopup() {
    const p = document.getElementById("ais-layer-popup");
    if (p) p.remove();
  }

  function _buildLayerOrderButton(selIds) {
    const btn = document.createElement("button");
    btn.className = "ais-tool-btn";
    btn.title = "Layer order";
    btn.innerHTML = `<svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round">
      <rect x="1" y="1" width="12" height="3" rx="0.5"/>
      <rect x="1" y="5.5" width="12" height="3" rx="0.5"/>
      <rect x="1" y="10" width="12" height="3" rx="0.5"/>
    </svg>`;
    settingsArea.appendChild(btn);

    btn.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      e.preventDefault();
      btn.blur();

      const popup = document.createElement("div");
      popup.id = "ais-layer-popup";
      popup.style.cssText = [
        "position:fixed",
        "background:var(--popover,#1e1e1e)",
        "border:1px solid var(--border,#444)",
        "border-radius:6px",
        "box-shadow:0 4px 16px rgba(0,0,0,0.5)",
        "z-index:10000",
        "overflow:hidden",
        "min-width:160px",
        "font-family:sans-serif",
        "font-size:12px",
      ].join(";");

      const ACTIONS = [
        { label: "↑   Bring Forward",  action: "forward"  },
        { label: "⬆   Bring to Front", action: "front"    },
        { label: "↓   Send Backward",  action: "backward" },
        { label: "⬇   Send to Back",   action: "back"     },
      ];
      for (const { label, action } of ACTIONS) {
        const item = document.createElement("div");
        item.style.cssText = "padding:7px 14px;cursor:pointer;color:var(--foreground,#eee);white-space:nowrap;";
        item.textContent = label;
        item.addEventListener("pointerover",  () => { item.style.background = "rgba(122,157,184,0.18)"; });
        item.addEventListener("pointerout",   () => { item.style.background = ""; });
        item.addEventListener("pointerdown",  (ev) => {
          ev.stopPropagation();
          currentValue = { ...currentValue, annotations: _reorderAnnotations(selIds, action) };
          _emit(); rebuildSettings(true); renderCanvas();
        });
        popup.appendChild(item);
      }

      document.body.appendChild(popup);
      const bRect = btn.getBoundingClientRect();
      // Right-align popup to button, open downward
      popup.style.top  = `${bRect.bottom + 4}px`;
      popup.style.left = `${bRect.right}px`;  // temp; adjust after paint
      requestAnimationFrame(() => {
        const pw = popup.offsetWidth;
        let left = bRect.right - pw;
        if (left < 8) left = 8;
        popup.style.left = `${left}px`;
      });

      // Dismiss on outside click
      const dismiss = (ev) => {
        if (!popup.contains(ev.target)) {
          _dismissLayerPopup();
          document.removeEventListener("pointerdown", dismiss, { capture: true });
        }
      };
      setTimeout(() => document.addEventListener("pointerdown", dismiss, { capture: true }), 0);
    });
  }

  function setTool(id) {
    commitTextEdit();
    hoverId = null;
    activeTool = id;
    currentValue = { ...currentValue, active_tool: id };
    for (const [tid, btn] of Object.entries(toolBtns)) {
      btn.className = "ais-tool-btn" + (tid === id ? " active" : "");
    }
    canvas.style.cursor = _currentToolCursor();
    canvas.focus({ preventScroll: true });
    rebuildSettings();
    renderCanvas();
  }

  // ── rendering ─────────────────────────────────────────────────────────────
  let renderGen = 0;

  function renderCanvas() {
    const gen = ++renderGen;
    requestAnimationFrame(() => { if (gen === renderGen) _doRender(gen); });
  }

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

    // Draw committed annotations
    for (const ann of (currentValue.annotations || [])) {
      if (ann.id === textEditId) continue; // skip live-edited text
      drawAnnotation(ann, (currentValue.selected_ids || []).includes(ann.id));
    }

    // Unified transform frame — same OBB handles for single or group selection
    if (txFrame && _frameActiveTools.includes(activeTool)) {
      const corners = _frameCorners(txFrame);
      const topMid = _frameTopMid(txFrame);
      const rh = _frameRotHandle(txFrame);
      const hw = 5 / displayScale;
      ctx.save();
      // Dashed OBB outline
      ctx.strokeStyle = "rgba(122,157,184,0.8)";
      ctx.lineWidth = 1.5 / displayScale;
      ctx.setLineDash([5 / displayScale, 4 / displayScale]);
      ctx.beginPath();
      ctx.moveTo(corners[0][0], corners[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
      ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
      // Corner scale handles
      for (const [hx, hy] of corners) {
        ctx.fillStyle = "white"; ctx.beginPath(); ctx.arc(hx, hy, hw, 0, Math.PI*2); ctx.fill();
        ctx.strokeStyle = "rgba(122,157,184,0.9)"; ctx.lineWidth = 1.5 / displayScale;
        ctx.beginPath(); ctx.arc(hx, hy, hw, 0, Math.PI*2); ctx.stroke();
      }
      // Rotation handle stem + circle
      ctx.strokeStyle = "rgba(122,157,184,0.5)"; ctx.lineWidth = 1 / displayScale;
      ctx.beginPath(); ctx.moveTo(topMid[0], topMid[1]); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
      ctx.fillStyle = "#7a9db8"; ctx.beginPath(); ctx.arc(rh[0], rh[1], hw, 0, Math.PI*2); ctx.fill();
      ctx.strokeStyle = "white"; ctx.lineWidth = 1.5 / displayScale;
      ctx.beginPath(); ctx.arc(rh[0], rh[1], hw, 0, Math.PI*2); ctx.stroke();
      ctx.restore();
    }

    // In-progress arrow
    if (currentArrow) {
      const ts = toolSettings.arrow;
      drawArrowLine(
        currentArrow.x1, currentArrow.y1, currentArrow.x2, currentArrow.y2,
        ts.color || "#ff0000", ts.width || 3,
        null, null, null, null,
        ts.has_start_arrow ?? false, ts.has_end_arrow ?? true
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
      ctx.strokeStyle = ts.color || "#ff0000";
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
      ctx.strokeStyle = ts.color || "#ff0000";
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
      ctx.fillStyle = "rgba(122,157,184,0.1)";
      ctx.fillRect(mx1, my1, mw, mh);
      ctx.strokeStyle = "rgba(122,157,184,0.8)";
      ctx.lineWidth = 1 / displayScale;
      ctx.setLineDash([4 / displayScale, 3 / displayScale]);
      ctx.strokeRect(mx1, my1, mw, mh);
      ctx.restore();
    }
  }

  function drawAnnotation(ann, selected) {
    if (ann.type === "paint")   drawPaint(ann, selected);
    else if (ann.type === "text")   drawText(ann, selected);
    else if (ann.type === "arrow")  drawArrowAnnotation(ann, selected);
    else if (ann.type === "rect")   drawRect(ann, selected);
    else if (ann.type === "ellipse") drawEllipse(ann, selected);
  }

  function drawPaint(ann, selected) {
    const [cx, cy] = _paintCenter(ann);
    const x = ann.x || 0, y = ann.y || 0;
    const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = ann.rotation || 0;
    ctx.save();
    ctx.translate(cx + x, cy + y);
    ctx.rotate(r);
    ctx.scale(sx, sy);
    ctx.translate(-cx, -cy);
    renderStrokes(ann.strokes || [], ann.sizeScale ?? 1);
    ctx.restore();

  }

  function _strokeBounds(stroke) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const pt of (stroke.points || [])) {
      minX = Math.min(minX, pt[0]); minY = Math.min(minY, pt[1]);
      maxX = Math.max(maxX, pt[0]); maxY = Math.max(maxY, pt[1]);
    }
    return isFinite(minX) ? { minX, minY, maxX, maxY } : null;
  }

  function _naturalBounds(ann) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (const stroke of (ann.strokes || [])) {
      const b = _strokeBounds(stroke);
      if (b) { minX = Math.min(minX, b.minX); minY = Math.min(minY, b.minY); maxX = Math.max(maxX, b.maxX); maxY = Math.max(maxY, b.maxY); }
    }
    return isFinite(minX) ? { minX, minY, maxX, maxY } : null;
  }

  function _paintCenter(ann) {
    if (ann.cx != null && ann.cy != null) return [ann.cx, ann.cy];
    const b = _naturalBounds(ann);
    return b ? [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2] : [0, 0];
  }

  function _paintTransformPt(ann, nx, ny) {
    const [cx, cy] = _paintCenter(ann);
    const x = ann.x || 0, y = ann.y || 0;
    const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = ann.rotation || 0;
    const cos = Math.cos(r), sin = Math.sin(r);
    const lx = (nx - cx) * sx, ly = (ny - cy) * sy;
    return [cx + x + lx * cos - ly * sin, cy + y + lx * sin + ly * cos];
  }

  function _paintInvTransformPt(ann, px, py) {
    const [cx, cy] = _paintCenter(ann);
    const x = ann.x || 0, y = ann.y || 0;
    const sx = ann.scaleX ?? 1, sy = ann.scaleY ?? 1, r = -(ann.rotation || 0);
    const cos = Math.cos(r), sin = Math.sin(r);
    const dx = px - (cx + x), dy = py - (cy + y);
    return [(dx * cos - dy * sin) / sx + cx, (dx * sin + dy * cos) / sy + cy];
  }

  function _getTransformedCorners(ann, pad = 10) {
    const b = _naturalBounds(ann);
    if (!b) return [];
    return [
      [b.minX - pad, b.minY - pad], [b.maxX + pad, b.minY - pad],
      [b.maxX + pad, b.maxY + pad], [b.minX - pad, b.maxY + pad],
    ].map(([nx, ny]) => _paintTransformPt(ann, nx, ny));
  }

  function renderStrokes(strokes, sizeScale = 1) {
    for (const stroke of strokes) {
      const pts = stroke.points || [];
      if (!pts.length) continue;
      const defaultSz = (stroke.size || 8) * sizeScale;
      ctx.fillStyle = stroke.color || "#ff0000";
      // Initial dot
      const r0 = ((pts[0][2] ?? (stroke.size || 8)) * sizeScale) / 2;
      ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], r0, 0, Math.PI * 2); ctx.fill();
      // Variable-width path: filled trapezoid + endpoint circle per segment
      for (let i = 1; i < pts.length; i++) {
        const px = pts[i-1][0], py = pts[i-1][1], pr = ((pts[i-1][2] ?? (stroke.size || 8)) * sizeScale) / 2;
        const qx = pts[i][0],   qy = pts[i][1],   qr = ((pts[i][2]   ?? (stroke.size || 8)) * sizeScale) / 2;
        ctx.beginPath(); ctx.arc(qx, qy, qr, 0, Math.PI * 2); ctx.fill();
        const dx = qx - px, dy = qy - py, len = Math.hypot(dx, dy);
        if (len > 0) {
          const nx = -dy / len, ny = dx / len;
          ctx.beginPath();
          ctx.moveTo(px + nx * pr, py + ny * pr);
          ctx.lineTo(qx + nx * qr, qy + ny * qr);
          ctx.lineTo(qx - nx * qr, qy - ny * qr);
          ctx.lineTo(px - nx * pr, py - ny * pr);
          ctx.closePath(); ctx.fill();
        }
      }
    }
  }

  function decimatePoints(points, minDist = 3) {
    if (points.length <= 2) return points;
    const out = [points[0]];
    for (let i = 1; i < points.length - 1; i++) {
      const prev = out[out.length - 1];
      if (Math.hypot(points[i][0] - prev[0], points[i][1] - prev[1]) >= minDist) out.push(points[i]);
    }
    out.push(points[points.length - 1]);
    return out;
  }

  function drawText(ann, selected) {
    const fontSize = Math.max(8, ann.font_size || 48);
    const lineHeight = fontSize * 1.2;
    const lines = (ann.text || "").split("\n");
    const x = ann.x || 0;
    const y = ann.y || 0;
    ctx.save();
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillStyle = ann.color || "#ffffff";
    ctx.textBaseline = "top";
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], x, y + i * lineHeight);
    }

    const isHovered = ann.id === hoverId && !selected;
    if (isHovered) {
      const w = Math.max(...lines.map((l) => ctx.measureText(l).width));
      const h = lineHeight * lines.length;
      ctx.strokeStyle = "rgba(122,157,184,0.4)";
      ctx.lineWidth = 1 / displayScale;
      ctx.setLineDash([4 / displayScale, 3 / displayScale]);
      ctx.strokeRect(x - 4, y - 4, w + 8, h + 8);
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  function drawArrowAnnotation(ann, selected) {
    const isBezier = ann.is_bezier ?? false;
    const cps = _defaultCps(ann);
    const cp1x = isBezier ? cps.cp1x : null;
    const cp1y = isBezier ? cps.cp1y : null;
    const cp2x = isBezier ? cps.cp2x : null;
    const cp2y = isBezier ? cps.cp2y : null;
    drawArrowLine(ann.x1, ann.y1, ann.x2, ann.y2, ann.color || "#ff0000", ann.width || 3,
      cp1x, cp1y, cp2x, cp2y, ann.has_start_arrow ?? false, ann.has_end_arrow ?? true);
    if (selected) {
      const r = 5 / displayScale;
      ctx.save();
      // Endpoint handles — white fill + slate outline (same as transform corners)
      for (const [ex, ey] of [[ann.x1, ann.y1], [ann.x2, ann.y2]]) {
        ctx.fillStyle = "white";
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = "rgba(122,157,184,0.9)"; ctx.lineWidth = 1.5 / displayScale;
        ctx.beginPath(); ctx.arc(ex, ey, r, 0, Math.PI * 2); ctx.stroke();
      }
      if (isBezier) {
        const cpR = 4 / displayScale;
        // Control point arms (dashed)
        ctx.strokeStyle = "rgba(122,157,184,0.5)";
        ctx.lineWidth = 1 / displayScale;
        ctx.setLineDash([3 / displayScale, 2 / displayScale]);
        ctx.beginPath(); ctx.moveTo(ann.x1, ann.y1); ctx.lineTo(cps.cp1x, cps.cp1y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ann.x2, ann.y2); ctx.lineTo(cps.cp2x, cps.cp2y); ctx.stroke();
        ctx.setLineDash([]);
        // Control point handles (hollow circles)
        ctx.fillStyle = "white";
        ctx.strokeStyle = "rgba(122,157,184,0.9)";
        ctx.lineWidth = 1.5 / displayScale;
        ctx.beginPath(); ctx.arc(cps.cp1x, cps.cp1y, cpR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cps.cp1x, cps.cp1y, cpR, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = "white";
        ctx.beginPath(); ctx.arc(cps.cp2x, cps.cp2y, cpR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(cps.cp2x, cps.cp2y, cpR, 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore();
    }
  }

  function drawArrowLine(x1, y1, x2, y2, color, width, cp1x, cp1y, cp2x, cp2y, hasStartArrow, hasEndArrow) {
    if (cp1x == null) cp1x = x1 + (x2 - x1) / 3;
    if (cp1y == null) cp1y = y1 + (y2 - y1) / 3;
    if (cp2x == null) cp2x = x1 + (x2 - x1) * 2 / 3;
    if (cp2y == null) cp2y = y1 + (y2 - y1) * 2 / 3;
    if (hasEndArrow == null) hasEndArrow = true;
    const w = Math.max(1, width);
    const head = Math.max(15, w * 4);
    // How far back from the tip the base of the triangle sits (cos 30° of head length)
    const setback = head * Math.cos(Math.PI / 6);

    // Compute angles from tangent directions (control points) before adjusting endpoints
    let endAngle = 0, startAngle = 0;
    if (hasEndArrow) {
      endAngle = Math.hypot(x2 - cp2x, y2 - cp2y) < 0.1
        ? Math.atan2(y2 - y1, x2 - x1)
        : Math.atan2(y2 - cp2y, x2 - cp2x);
    }
    if (hasStartArrow) {
      startAngle = Math.hypot(cp1x - x1, cp1y - y1) < 0.1
        ? Math.atan2(y1 - y2, x1 - x2)
        : Math.atan2(y1 - cp1y, x1 - cp1x);
    }

    // Pull line endpoints back to the arrowhead base so the stroke doesn't poke through
    const lx2 = hasEndArrow   ? x2 - setback * Math.cos(endAngle)   : x2;
    const ly2 = hasEndArrow   ? y2 - setback * Math.sin(endAngle)   : y2;
    const lx1 = hasStartArrow ? x1 - setback * Math.cos(startAngle) : x1;
    const ly1 = hasStartArrow ? y1 - setback * Math.sin(startAngle) : y1;

    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = w;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(lx1, ly1); ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, lx2, ly2); ctx.stroke();

    // Arrowheads drawn at the original tip points
    if (hasEndArrow) {
      ctx.beginPath();
      ctx.moveTo(x2, y2);
      ctx.lineTo(x2 - head * Math.cos(endAngle - Math.PI / 6), y2 - head * Math.sin(endAngle - Math.PI / 6));
      ctx.lineTo(x2 - head * Math.cos(endAngle + Math.PI / 6), y2 - head * Math.sin(endAngle + Math.PI / 6));
      ctx.closePath(); ctx.fill();
    }
    if (hasStartArrow) {
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x1 - head * Math.cos(startAngle - Math.PI / 6), y1 - head * Math.sin(startAngle - Math.PI / 6));
      ctx.lineTo(x1 - head * Math.cos(startAngle + Math.PI / 6), y1 - head * Math.sin(startAngle + Math.PI / 6));
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }

  function drawRect(ann, _selected) {
    const hw = (ann.w || 10) / 2, hh = (ann.h || 10) / 2;
    const r = ann.rotation || 0;
    ctx.save();
    ctx.translate(ann.x || 0, ann.y || 0);
    ctx.rotate(r);
    ctx.lineWidth = ann.width || 2;
    ctx.strokeStyle = ann.color || "#ff0000";
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fillRect(-hw, -hh, hw * 2, hh * 2); }
    ctx.strokeRect(-hw, -hh, hw * 2, hh * 2);
    ctx.restore();
  }

  function drawEllipse(ann, _selected) {
    const rx = Math.max(0.5, (ann.w || 10) / 2), ry = Math.max(0.5, (ann.h || 10) / 2);
    const r = ann.rotation || 0;
    ctx.save();
    ctx.translate(ann.x || 0, ann.y || 0);
    ctx.rotate(r);
    ctx.lineWidth = ann.width || 2;
    ctx.strokeStyle = ann.color || "#ff0000";
    ctx.beginPath();
    ctx.ellipse(0, 0, rx, ry, 0, 0, Math.PI * 2);
    if (ann.fill_color) { ctx.fillStyle = ann.fill_color; ctx.fill(); }
    ctx.stroke();
    ctx.restore();
  }

  // ── hit testing ───────────────────────────────────────────────────────────
  function hitTest(cx, cy) {
    const anns = [...(currentValue.annotations || [])].reverse();
    for (const ann of anns) {
      if (ann.type === "text") {
        const fontSize = Math.max(8, ann.font_size || 48);
        ctx.font = `${fontSize}px sans-serif`;
        const w = ctx.measureText(ann.text || "").width;
        const h = fontSize * 1.2;
        const ax = (ann.x || 0) - 4, ay = (ann.y || 0) - 4;
        if (cx >= ax && cx <= ax + w + 8 && cy >= ay && cy <= ay + h + 8) return ann;
      } else if (ann.type === "arrow") {
        const { cp1x, cp1y, cp2x, cp2y } = _defaultCps(ann);
        const tol = Math.max(12 / displayScale, (ann.width || 3) + 6);
        const N = 20;
        for (let i = 0; i <= N; i++) {
          const t = i / N, mt = 1 - t;
          const bx = mt**3*ann.x1 + 3*mt**2*t*cp1x + 3*mt*t**2*cp2x + t**3*ann.x2;
          const by = mt**3*ann.y1 + 3*mt**2*t*cp1y + 3*mt*t**2*cp2y + t**3*ann.y2;
          if (Math.hypot(cx - bx, cy - by) <= tol) return ann;
        }
      } else if (ann.type === "paint") {
        const [lx, ly] = _paintInvTransformPt(ann, cx, cy);
        for (const stroke of (ann.strokes || [])) {
          const tol = Math.max(12 / displayScale, (stroke.size || 8) / 2 + 4);
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
        if (ann.type === "rect") {
          if (ann.fill_color) {
            if (lx >= -hw && lx <= hw && ly >= -hh && ly <= hh) return ann;
          } else {
            const nearH = Math.abs(Math.abs(lx) - hw) <= tol && ly >= -hh - tol && ly <= hh + tol;
            const nearV = Math.abs(Math.abs(ly) - hh) <= tol && lx >= -hw - tol && lx <= hw + tol;
            if (nearH || nearV) return ann;
          }
        } else {
          const ex = lx / (hw + tol), ey = ly / (hh + tol);
          if (ex * ex + ey * ey <= 1) {
            if (ann.fill_color) return ann;
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
      const fontSize = Math.max(8, ann.font_size || 48);
      ctx.font = `${fontSize}px sans-serif`;
      const w = ctx.measureText(ann.text || "").width;
      const h = fontSize * 1.2;
      const ax = ann.x || 0, ay = ann.y || 0;
      return { minX: ax - 4, minY: ay - 4, maxX: ax + w + 8, maxY: ay + h + 8 };
    } else if (ann.type === "arrow") {
      const { cp1x, cp1y, cp2x, cp2y } = _defaultCps(ann);
      const pad = Math.max(8, (ann.width || 3) / 2 + 4);
      const xs = [ann.x1, ann.x2, cp1x, cp2x], ys = [ann.y1, ann.y2, cp1y, cp2y];
      return {
        minX: Math.min(...xs) - pad, minY: Math.min(...ys) - pad,
        maxX: Math.max(...xs) + pad, maxY: Math.max(...ys) + pad,
      };
    } else if (ann.type === "paint") {
      const corners = _getTransformedCorners(ann, 10);
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

  function _getGroupBounds(selIds) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    let found = false;
    for (const ann of (currentValue.annotations || [])) {
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

  function _snapshotAnn(ann) {
    const [cx, cy] = _paintCenter(ann);
    const cps = _defaultCps(ann);
    return {
      cx, cy,
      x: ann.x ?? 0, y: ann.y ?? 0,
      scaleX: ann.scaleX ?? 1, scaleY: ann.scaleY ?? 1,
      rotation: ann.rotation ?? 0,
      x1: ann.x1 ?? 0, y1: ann.y1 ?? 0, x2: ann.x2 ?? 0, y2: ann.y2 ?? 0,
      cp1x: cps.cp1x, cp1y: cps.cp1y, cp2x: cps.cp2x, cp2y: cps.cp2y,
      font_size: ann.font_size ?? 48,
      width: ann.width ?? 3,
      sizeScale: ann.sizeScale ?? 1,
      w: ann.w ?? 100, h: ann.h ?? 100,
    };
  }

  function _defaultCps(ann) {
    const cp1x = ann.cp1x ?? (ann.x1 + ((ann.x2 ?? 0) - (ann.x1 ?? 0)) / 3);
    const cp1y = ann.cp1y ?? (ann.y1 + ((ann.y2 ?? 0) - (ann.y1 ?? 0)) / 3);
    const cp2x = ann.cp2x ?? (ann.x1 + ((ann.x2 ?? 0) - (ann.x1 ?? 0)) * 2 / 3);
    const cp2y = ann.cp2y ?? (ann.y1 + ((ann.y2 ?? 0) - (ann.y1 ?? 0)) * 2 / 3);
    return { cp1x, cp1y, cp2x, cp2y };
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
      const ann = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
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
        const nb = _naturalBounds(ann);
        if (!nb) { txFrame = null; return; }
        const [pcx, pcy] = _paintCenter(ann);
        txFrame = {
          pivotX: pcx + (ann.x || 0), pivotY: pcy + (ann.y || 0),
          rotation: ann.rotation || 0, // always matches paint's own rotation
          halfW: (nb.maxX - nb.minX) / 2 + pad,
          halfH: (nb.maxY - nb.minY) / 2 + pad,
          _selIds: [...selIds],
        };
        return;
      }
      // single text: fall through to AABB
    }
    const gb = _getGroupBounds(selIds);
    if (!gb) { txFrame = null; return; }
    txFrame = {
      pivotX: gb.centerX, pivotY: gb.centerY,
      rotation: selChanged ? 0 : (txFrame?.rotation ?? 0), // reset rotation on new selection
      halfW: (gb.maxX - gb.minX) / 2 + pad,
      halfH: (gb.maxY - gb.minY) / 2 + pad,
      _selIds: [...selIds],
    };
  }

  function _frameCorners(frame) {
    const { pivotX: px, pivotY: py, rotation: r, halfW: hw, halfH: hh } = frame;
    const cos = Math.cos(r), sin = Math.sin(r);
    return [[-hw,-hh],[hw,-hh],[hw,hh],[-hw,hh]].map(([lx,ly]) =>
      [px + lx*cos - ly*sin, py + lx*sin + ly*cos]);
  }

  function _frameTopMid(frame) {
    // local (0, -halfH) → world
    const { pivotX: px, pivotY: py, rotation: r, halfH: hh } = frame;
    return [px + hh*Math.sin(r), py - hh*Math.cos(r)];
  }

  function _frameRotHandle(frame) {
    const [tx, ty] = _frameTopMid(frame);
    const d = 28 / displayScale;
    return [tx + d*Math.sin(frame.rotation), ty - d*Math.cos(frame.rotation)];
  }

  // Returns true if annotation overlaps the given canvas-space rectangle
  function _annotationIntersectsRect(ann, x1, y1, x2, y2) {
    if (ann.type === "text") {
      const fontSize = Math.max(8, ann.font_size || 48);
      ctx.font = `${fontSize}px sans-serif`;
      const w = ctx.measureText(ann.text || "").width;
      const h = fontSize * 1.2;
      const ax = ann.x || 0, ay = ann.y || 0;
      return !(ax + w < x1 || ax > x2 || ay + h < y1 || ay > y2);
    } else if (ann.type === "arrow") {
      const { cp1x, cp1y, cp2x, cp2y } = _defaultCps(ann);
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
          const [px, py] = _paintTransformPt(ann, pt[0], pt[1]);
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
  function startTextEdit(ann) {
    commitTextEdit();
    textEditId = ann.id;
    currentValue = { ...currentValue, selected_ids: [ann.id] };

    const fontSize = Math.max(8, ann.font_size || 48);
    const totalScale = displayScale * viewScale;
    textInput = document.createElement("textarea");
    textInput.value = ann.text || "";
    textInput.rows = 1;
    textInput.style.cssText = [
      "position:absolute",
      `left:${(ann.x || 0) * totalScale + panX}px`,
      `top:${(ann.y || 0) * totalScale + panY}px`,
      "min-width:60px",
      "background:transparent",
      `color:${ann.color || "#ffffff"}`,
      `font-size:${fontSize * totalScale}px`,
      "font-family:sans-serif",
      "border:none",
      `border-bottom:2px solid ${ann.color || "#ffffff"}`,
      "outline:none",
      "resize:none",
      "overflow:hidden",
      "white-space:nowrap",
      "z-index:100",
      "padding:0",
      "margin:0",
      "line-height:1.2",
    ].join(";");

    canvasWrap.appendChild(textInput);
    textInput.focus();
    textInput.select();
    _autoResizeTextarea();

    textInput.addEventListener("input", () => {
      _autoResizeTextarea();
      currentValue = {
        ...currentValue,
        annotations: currentValue.annotations.map((a) =>
          a.id === textEditId ? { ...a, text: textInput.value } : a
        ),
      };
    });
    textInput.addEventListener("keydown", (e) => {
      e.stopPropagation();
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commitTextEdit(); canvas.focus({ preventScroll: true }); }
      if (e.key === "Escape") { e.preventDefault(); commitTextEdit(); canvas.focus({ preventScroll: true }); }
    });
    textInput.addEventListener("blur", commitTextEdit);
    textInput.addEventListener("pointerdown", (e) => e.stopPropagation());
    renderCanvas();
  }

  function _autoResizeTextarea() {
    if (!textInput) return;
    textInput.style.width = "1px";
    textInput.style.width = textInput.scrollWidth + "px";
    textInput.style.height = "auto";
    textInput.style.height = textInput.scrollHeight + "px";
  }

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
        // Remove empty text annotations
        currentValue = {
          ...currentValue,
          selected_ids: [],
          annotations: currentValue.annotations.filter((a) => a.id !== id),
        };
      } else {
        currentValue = {
          ...currentValue,
          selected_ids: [],
          annotations: currentValue.annotations.map((a) =>
            a.id === id ? { ...a, text } : a
          ),
        };
      }
      _emit();
    }
    rebuildSettings();
    renderCanvas();
  }

  function placeNewText(cx, cy) {
    const id = _uid("text");
    const ann = {
      id, type: "text", text: "Text",
      x: cx, y: cy,
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
  // React Flow intercepts Shift+click via a capture-phase listener on its node element,
  // which fires before any bubble-phase handler on our canvas.  The only way to beat it
  // is to register our own listener at the document level in capture phase, which fires
  // first among all elements.  We only act when the event target is inside our widget
  // and Shift is held; everything else is passed through untouched.
  function _shiftInterceptor(e) {
    if (!e.shiftKey || !wrapper.contains(e.target)) return;
    e.stopPropagation();
    e.stopImmediatePropagation();
    // For pointerdown: dispatch into our own handler manually (event never reaches canvas
    // because we stopped propagation before it could).
    if (e.type === "pointerdown" && e.button === 0) onPointerDown(e);
    // mousedown / click: just swallow so React Flow doesn't multi-select the node.
  }
  document.addEventListener("pointerdown", _shiftInterceptor, { capture: true });
  document.addEventListener("mousedown",   _shiftInterceptor, { capture: true });
  document.addEventListener("click",       _shiftInterceptor, { capture: true });

  function _deleteInterceptor(e) {
    if (e.key !== "Delete" && e.key !== "Backspace") return;
    if (!(currentValue.selected_ids || []).length) return;
    if (textEditId) return;
    if (activeTool !== "select" && activeTool !== "arrow" && activeTool !== "rect" && activeTool !== "ellipse") return;
    // Don't steal Delete from text inputs elsewhere on the page
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    e.stopPropagation();
    e.preventDefault();
    const selIds = currentValue.selected_ids;
    currentValue = {
      ...currentValue,
      annotations: (currentValue.annotations || []).filter((a) => !selIds.includes(a.id)),
      selected_ids: [],
    };
    txFrame = null;
    _emit();
    rebuildSettings();
    renderCanvas();
  }
  document.addEventListener("keydown", _deleteInterceptor, { capture: true });

  function _sizeInterceptor(e) {
    if (e.key !== "[" && e.key !== "]") return;
    if (textEditId) return;
    const t = e.target;
    // Allow [ ] through for range/color/checkbox inputs (widget controls) — only block text-entry fields
    const inputType = (t?.type || "").toLowerCase();
    const isTextEntry = t && (
      (t.tagName === "INPUT" && !["range", "color", "checkbox", "radio"].includes(inputType)) ||
      t.tagName === "TEXTAREA" ||
      t.isContentEditable
    );
    if (isTextEntry) return;
    e.preventDefault();
    e.stopPropagation();
    const delta = e.key === "]" ? 1 : -1;
    const selIds = currentValue.selected_ids || [];
    // Adjust selected annotation if one is selected
    if (selIds.length === 1 && (activeTool === "select" || activeTool === "arrow")) {
      const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
      if (selAnn) {
        if (selAnn.type === "paint") {
          const base = selAnn.strokes?.[0]?.size ?? 8;
          const cur = Math.round(base * (selAnn.sizeScale ?? 1));
          const next = Math.max(1, Math.min(80, cur + delta));
          currentValue = { ...currentValue, annotations: currentValue.annotations.map((a) =>
            a.id === selAnn.id ? { ...a, sizeScale: next / base } : a) };
          _emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "text") {
          const next = Math.max(8, Math.min(120, (selAnn.font_size ?? 48) + delta * 2));
          currentValue = { ...currentValue, annotations: currentValue.annotations.map((a) =>
            a.id === selAnn.id ? { ...a, font_size: next } : a) };
          _emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "arrow") {
          const next = Math.max(1, Math.min(20, (selAnn.width ?? 3) + delta));
          currentValue = { ...currentValue, annotations: currentValue.annotations.map((a) =>
            a.id === selAnn.id ? { ...a, width: next } : a) };
          _emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "rect" || selAnn.type === "ellipse") {
          const next = Math.max(1, Math.min(20, (selAnn.width ?? 2) + delta));
          currentValue = { ...currentValue, annotations: currentValue.annotations.map((a) =>
            a.id === selAnn.id ? { ...a, width: next } : a) };
          _emit(); rebuildSettings(); renderCanvas(); return;
        }
      }
    }
    // Otherwise adjust active tool settings
    if (activeTool === "paint") {
      toolSettings.paint.size = Math.max(1, Math.min(80, (toolSettings.paint.size ?? 8) + delta));
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      rebuildSettings(); _emit();
    } else if (activeTool === "arrow") {
      toolSettings.arrow.width = Math.max(1, Math.min(20, (toolSettings.arrow.width ?? 3) + delta));
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      rebuildSettings(); _emit();
    } else if (activeTool === "text") {
      toolSettings.text.font_size = Math.max(8, Math.min(120, (toolSettings.text.font_size ?? 48) + delta * 2));
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      rebuildSettings(); _emit();
    } else if (activeTool === "rect" || activeTool === "ellipse") {
      toolSettings[activeTool].width = Math.max(1, Math.min(20, (toolSettings[activeTool].width ?? 2) + delta));
      currentValue = { ...currentValue, tool_settings: { ...toolSettings } };
      rebuildSettings(); _emit();
    }
  }
  document.addEventListener("keydown", _sizeInterceptor, { capture: true });

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointercancel", onPointerUp);
  canvas.addEventListener("mousemove", onMouseHover);
  canvas.addEventListener("mouseleave", () => {
    if (hoverId) { hoverId = null; renderCanvas(); }
  });

  function onPointerDown(e) {
    if (e.button !== 0) return;
    e.stopPropagation();
    canvas.setPointerCapture(e.pointerId);
    canvas.focus({ preventScroll: true });
    isPointerDown = true;

    // Alt + LMB → pan (overrides all tools)
    if (e.altKey) {
      e.preventDefault();
      isPanning = true;
      panStartX = e.clientX - panX;
      panStartY = e.clientY - panY;
      canvas.style.cursor = "grabbing";
      return;
    }

    const [cx, cy] = screenToCanvas(e);

    // OBB handle check — works in select, rect, and ellipse modes (wherever txFrame is shown)
    if (txFrame && _frameActiveTools.includes(activeTool)) {
      const handleR = 8 / displayScale;
      const selIds = currentValue.selected_ids || [];
      const corners = _frameCorners(txFrame);
      const rh = _frameRotHandle(txFrame);
      const buildSnapshots = () => {
        const s = {};
        for (const ann of (currentValue.annotations || []))
          if (selIds.includes(ann.id)) s[ann.id] = _snapshotAnn(ann);
        return s;
      };
      if (Math.hypot(cx - rh[0], cy - rh[1]) <= handleR) {
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
        const corners = _frameCorners(txFrame);
        const rh = _frameRotHandle(txFrame);
        const buildSnapshots = () => {
          const s = {};
          for (const ann of (currentValue.annotations || []))
            if (selIds.includes(ann.id)) s[ann.id] = _snapshotAnn(ann);
          return s;
        };
        // Rotation handle
        if (Math.hypot(cx - rh[0], cy - rh[1]) <= handleR) {
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
      }

      // Control point handle detection: only when single arrow already selected
      if (selIds.length === 1) {
        const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
        if (selAnn?.type === "arrow") {
          const cps = _defaultCps(selAnn);
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
        let newSelIds;
        if (e.shiftKey) {
          // Shift+click: toggle in/out of selection
          newSelIds = selIds.includes(hit.id)
            ? selIds.filter((id) => id !== hit.id)
            : [...selIds, hit.id];
        } else if (selIds.includes(hit.id)) {
          // Click on already-selected annotation: keep selection for drag
          newSelIds = selIds;
        } else {
          // Click on new annotation: replace selection
          newSelIds = [hit.id];
        }
        currentValue = { ...currentValue, selected_ids: newSelIds };

        // Arrow endpoint handles (single selection only)
        if (newSelIds.length === 1 && hit.type === "arrow") {
          const arrowHandleR = Math.max(10 / displayScale, 8);
          const nearStart = Math.hypot(cx - hit.x1, cy - hit.y1) <= arrowHandleR;
          const nearEnd   = Math.hypot(cx - hit.x2, cy - hit.y2) <= arrowHandleR;
          if (nearStart || nearEnd) {
            const hitCps = _defaultCps(hit);
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
          const a = (currentValue.annotations || []).find((ann) => ann.id === id);
          if (!a) continue;
          if (a.type === "arrow") {
            const cps = _defaultCps(a);
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
      const selIds = currentValue.selected_ids || [];
      // If the selected paint annotation is hit, start translate drag instead of new stroke
      if (selIds.length === 1) {
        const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0] && a.type === "paint");
        if (selAnn) {
          const hit = hitTest(cx, cy);
          if (hit && hit.id === selAnn.id) {
            dragState = { type: "translate", startCx: cx, startCy: cy,
              origPositions: { [selAnn.id]: { x: selAnn.x ?? 0, y: selAnn.y ?? 0 } },
              origPivotX: txFrame?.pivotX, origPivotY: txFrame?.pivotY };
            canvas.style.cursor = "grabbing";
            return;
          }
        }
      }
      // Click on empty space (or different annotation): deselect + start new stroke
      currentValue = { ...currentValue, selected_ids: [] };
      rebuildSettings();
      const sz = toolSettings.paint.size;
      strokeLastMid = null;
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
        const selAnn = (currentValue.annotations || []).find((a) => a.id === selIds[0] && a.type === "arrow");
        if (selAnn) {
          const handleR = Math.max(10 / displayScale, 8);
          const nearStart = Math.hypot(cx - selAnn.x1, cy - selAnn.y1) <= handleR;
          const nearEnd   = Math.hypot(cx - selAnn.x2, cy - selAnn.y2) <= handleR;
          if (nearStart || nearEnd) {
            const hitCps = _defaultCps(selAnn);
            dragState = { type: "arrowHandle", id: selAnn.id,
              arrowHandle: nearStart ? "start" : "end",
              startCx: cx, startCy: cy,
              origX1: selAnn.x1, origY1: selAnn.y1, origX2: selAnn.x2, origY2: selAnn.y2,
              origCp1x: hitCps.cp1x, origCp1y: hitCps.cp1y, origCp2x: hitCps.cp2x, origCp2y: hitCps.cp2y };
            canvas.style.cursor = "grabbing"; return;
          }
          if (selAnn.is_bezier) {
            const cps = _defaultCps(selAnn);
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
          const a = (currentValue.annotations || []).find((ann) => ann.id === id);
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
          const a = (currentValue.annotations || []).find((ann) => ann.id === id);
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
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            if (!dragState.selIds.includes(a.id)) return a;
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
              const dx = snap.x - pivot.x, dy = snap.y - pivot.y;
              return { ...a, x: dx*cos - dy*sin + pivot.x, y: dx*sin + dy*cos + pivot.y };
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
          }),
        };
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
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            if (!dragState.selIds.includes(a.id)) return a;
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
              const [nx, ny] = scaleAnchor(snap.x, snap.y);
              return { ...a, x: nx, y: ny,
                font_size: Math.max(8, Math.round(snap.font_size * (ratioX + ratioY) / 2)) };
            } else if (a.type === "arrow") {
              const [nx1, ny1] = scaleAnchor(snap.x1, snap.y1);
              const [nx2, ny2] = scaleAnchor(snap.x2, snap.y2);
              const [nc1x, nc1y] = scaleAnchor(snap.cp1x, snap.cp1y);
              const [nc2x, nc2y] = scaleAnchor(snap.cp2x, snap.cp2y);
              return { ...a, x1: nx1, y1: ny1, x2: nx2, y2: ny2, cp1x: nc1x, cp1y: nc1y, cp2x: nc2x, cp2y: nc2y };
            }
            return a;
          }),
        };
      } else if (dragState.type === "arrowCp") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            if (a.id !== dragState.id) return a;
            if (dragState.which === "cp1")
              return { ...a, cp1x: dragState.origCp1x + dx, cp1y: dragState.origCp1y + dy };
            return { ...a, cp2x: dragState.origCp2x + dx, cp2y: dragState.origCp2y + dy };
          }),
        };
      } else if (dragState.type === "arrowHandle") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            if (a.id !== dragState.id) return a;
            if (dragState.arrowHandle === "start")
              return { ...a, x1: dragState.origX1 + dx, y1: dragState.origY1 + dy,
                cp1x: dragState.origCp1x + dx, cp1y: dragState.origCp1y + dy };
            return { ...a, x2: dragState.origX2 + dx, y2: dragState.origY2 + dy,
              cp2x: dragState.origCp2x + dx, cp2y: dragState.origCp2y + dy };
          }),
        };
      } else if (dragState.type === "translate") {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            const orig = dragState.origPositions[a.id];
            if (!orig) return a;
            if (a.type === "arrow")
              return { ...a, x1: orig.x1 + dx, y1: orig.y1 + dy, x2: orig.x2 + dx, y2: orig.y2 + dy,
                cp1x: orig.cp1x + dx, cp1y: orig.cp1y + dy, cp2x: orig.cp2x + dx, cp2y: orig.cp2y + dy };
            return { ...a, x: orig.x + dx, y: orig.y + dy };
          }),
        };
        // Move the frame with the selection so handles follow the annotation
        if (txFrame && dragState.origPivotX != null) {
          txFrame = { ...txFrame, pivotX: dragState.origPivotX + dx, pivotY: dragState.origPivotY + dy };
        }
      } else if (dragState.type === "marquee") {
        dragState = { ...dragState, x2: cx, y2: cy };
      }
      renderCanvas();
    }
  }

  function onPointerUp(e) {
    if (!isPointerDown) return;
    isPointerDown = false;
    e.stopPropagation();

    if (isPanning) {
      isPanning = false;
      canvas.style.cursor = isAltHeld ? "grab" : _currentToolCursor();
      return;
    }

    const [cx, cy] = screenToCanvas(e);
    canvas.style.cursor = _cursorForPos(cx, cy);

    if (activeTool === "paint" && currentStroke && currentStroke.points.length >= 1) {
      currentStroke.points = decimatePoints(currentStroke.points);
      strokeLastMid = null;
      const stroke = currentStroke;
      currentStroke = null;
      // Each stroke = its own paint annotation with independent transform
      const b = _strokeBounds(stroke);
      const paintAnn = {
        id: _uid("paint"), type: "paint",
        strokes: [stroke],
        cx: b ? (b.minX + b.maxX) / 2 : 0,
        cy: b ? (b.minY + b.maxY) / 2 : 0,
        x: 0, y: 0, scaleX: 1, scaleY: 1, rotation: 0,
      };
      currentValue = {
        ...currentValue,
        annotations: [...(currentValue.annotations || []), paintAnn],
        selected_ids: [paintAnn.id],
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
        };
        currentValue = {
          ...currentValue,
          annotations: [...(currentValue.annotations || []), ann],
          selected_ids: [ann.id],
        };
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
        currentValue = { ...currentValue, annotations: [...(currentValue.annotations || []), ann], selected_ids: [ann.id] };
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
        currentValue = { ...currentValue, annotations: [...(currentValue.annotations || []), ann], selected_ids: [ann.id] };
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
          const inRect = (currentValue.annotations || [])
            .filter((a) => _annotationIntersectsRect(a, x1, y1, x2, y2))
            .map((a) => a.id);
          const merged = dragState.additive
            ? [...new Set([...(currentValue.selected_ids || []), ...inRect])]
            : inRect;
          currentValue = { ...currentValue, selected_ids: merged };
        }
        dragState = null;
        rebuildSettings();
        renderCanvas();
      } else {
        dragState = null;
        // Rebuild frame from new annotation state.
        // _buildTxFrame reads txFrame?.rotation for groups (preserving accumulated rotation)
        // and ann.rotation for single paint (which was updated live during txRotate).
        _buildTxFrame();
        _emit();
      }
    }
  }

  function _cursorForPos(cx, cy) {
    const handleR = 8 / displayScale;
    const selIds = currentValue.selected_ids || [];

    if (_frameActiveTools.includes(activeTool)) {
      if (txFrame) {
        if (Math.hypot(cx - _frameRotHandle(txFrame)[0], cy - _frameRotHandle(txFrame)[1]) <= handleR) return "grab";
        for (const [hx, hy] of _frameCorners(txFrame))
          if (Math.hypot(cx - hx, cy - hy) <= handleR) return "grab";
      }
      if (activeTool === "select") {
        if (selIds.length === 1) {
          const sa = (currentValue.annotations || []).find((a) => a.id === selIds[0]);
          if (sa?.type === "arrow") {
            if (sa.is_bezier) {
              const cps = _defaultCps(sa), cpR = Math.max(8 / displayScale, 5);
              if (Math.hypot(cx - cps.cp1x, cy - cps.cp1y) <= cpR) return "grab";
              if (Math.hypot(cx - cps.cp2x, cy - cps.cp2y) <= cpR) return "grab";
            }
            const ar = Math.max(10 / displayScale, 8);
            if (Math.hypot(cx - sa.x1, cy - sa.y1) <= ar || Math.hypot(cx - sa.x2, cy - sa.y2) <= ar) return "grab";
          }
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
        const sa = (currentValue.annotations || []).find((a) => a.id === selIds[0] && a.type === "arrow");
        if (sa) {
          const ar = Math.max(10 / displayScale, 8);
          if (Math.hypot(cx - sa.x1, cy - sa.y1) <= ar || Math.hypot(cx - sa.x2, cy - sa.y2) <= ar) return "grab";
          if (sa.is_bezier) {
            const cps = _defaultCps(sa), cpR = Math.max(8 / displayScale, 5);
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

    return "crosshair";
  }

  function onMouseHover(e) {
    if (isPointerDown) return;
    const [cx, cy] = screenToCanvas(e);
    const prevHoverId = hoverId;
    if (activeTool === "text" || activeTool === "select") {
      const hit = hitTest(cx, cy);
      hoverId = (hit && hit.type === "text") ? hit.id : null;
    } else {
      hoverId = null;
    }
    canvas.style.cursor = _cursorForPos(cx, cy);
    if (hoverId !== prevHoverId) renderCanvas();
  }

  // Double-click to edit text (works in both text and select tools)
  canvas.addEventListener("dblclick", (e) => {
    if (activeTool !== "select" && activeTool !== "text") return;
    dragState = null; // cancel any drag that started from the first click
    const [cx, cy] = screenToCanvas(e);
    const hit = hitTest(cx, cy);
    if (hit && hit.type === "text") {
      // Refresh annotation from currentValue so we get the latest text
      const fresh = (currentValue.annotations || []).find((a) => a.id === hit.id) || hit;
      startTextEdit(fresh);
    }
  });

  // ── emit / uid ─────────────────────────────────────────────────────────────
  let _emitSeq = 0;

  function _emit() {
    _emitSeq++;
    if (onChange) onChange({ ...currentValue, tool_settings: { ...toolSettings }, _emitSeq });
  }

  function _uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  }

  // ── update from props ──────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    const rawNv = (newProps?.value && typeof newProps.value === "object") ? newProps.value : {};

    // Stale-roundtrip guard: if we've emitted more recently than this incoming
    // value, the framework is echoing back an old snapshot. Accept only
    // image/dimension fields (set externally by Python), not annotations.
    const incomingSeq = rawNv._emitSeq || 0;
    if (incomingSeq > 0 && incomingSeq < _emitSeq) {
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

  // ── cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    commitTextEdit();
    _dismissLayerPopup();
    resizeObserver.disconnect();
    if (resizeRafId) cancelAnimationFrame(resizeRafId);
    document.removeEventListener("pointerdown", _shiftInterceptor, { capture: true });
    document.removeEventListener("mousedown",   _shiftInterceptor, { capture: true });
    document.removeEventListener("click",       _shiftInterceptor, { capture: true });
    document.removeEventListener("keydown",     _deleteInterceptor, { capture: true });
    document.removeEventListener("keydown",     _sizeInterceptor,   { capture: true });
    document.removeEventListener("keydown",     _onAltDown);
    document.removeEventListener("keyup",       _onAltUp);
    canvas.removeEventListener("pointerdown", onPointerDown);
    canvas.removeEventListener("pointermove", onPointerMove);
    canvas.removeEventListener("pointerup", onPointerUp);
    canvas.removeEventListener("pointercancel", onPointerUp);
    canvas.removeEventListener("mousemove", onMouseHover);
    wrapper.remove();
    delete container._aisInst;
  }

  // ── init ──────────────────────────────────────────────────────────────────
  setTool(activeTool);
  applyCanvasScale();
  renderCanvas();

  container._aisInst = { handleUpdate, cleanup, wrapper };
  return { cleanup, update: handleUpdate };
}
