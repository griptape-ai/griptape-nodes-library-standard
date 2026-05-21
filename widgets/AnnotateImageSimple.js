// AnnotateImageSimple — single-image annotation widget
// Layout: toolbar (tools left, settings right) + canvas
// Tools: Select, Paint, Text, Arrow

// ── Lucide SVG icons (inlined paths, MIT licensed) ────────────────────────────

const ICON_PATHS = {
  select: `<path d="m4 4 7.07 17 2.51-7.39L21 11.07z"/>`,
  paint:  `<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>`,
  text:   `<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>`,
  arrow:  `<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>`,
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
    .ais-tool-btn.active { background:var(--sidebar-primary); color:var(--sidebar-primary-foreground); }
    .ais-setting-label { font-size:11px; color:var(--muted-foreground); }
    .ais-range { accent-color:var(--sidebar-primary); cursor:pointer; width:80px; }
    .ais-color-btn { width:22px; height:22px; border-radius:4px; border:2px solid var(--border,#555); cursor:pointer; flex-shrink:0; }
    .ais-color-input { position:absolute; opacity:0; width:0; height:0; pointer-events:none; }
    .ais-val-label { font-size:11px; color:var(--foreground); min-width:24px; text-align:right; }
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
      paint: { color: "#ff0000", size: 8 },
      text:  { color: "#ffffff", font_size: 48 },
      arrow: { color: "#ff0000", width: 3 },
    },
    selected_id: null,
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
  let currentValue = (props.value && typeof props.value === "object")
    ? { ...defaultData(), ...props.value, tool_settings: { ...defaultData().tool_settings, ...(props.value.tool_settings || {}) } }
    : defaultData();

  let activeTool = currentValue.active_tool || "select";
  let toolSettings = { ...currentValue.tool_settings };
  let displayScale = 1;

  // pointer state
  let isPointerDown = false;
  let currentStroke = null;
  let strokeLastMid = null; // tracks last bezier midpoint for incremental draw
  let currentArrow = null;
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
    { id: "select", title: "Select & Move" },
    { id: "paint",  title: "Paint" },
    { id: "text",   title: "Text" },
    { id: "arrow",  title: "Arrow" },
  ];
  const toolBtns = {};
  for (const t of TOOLS) {
    const btn = document.createElement("button");
    btn.className = "ais-tool-btn" + (t.id === activeTool ? " active" : "");
    btn.title = t.title;
    btn.appendChild(mkIcon(t.id));
    btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); setTool(t.id); });
    toolbar.appendChild(btn);
    toolBtns[t.id] = btn;
  }

  // Divider
  const divider = document.createElement("div");
  divider.style.cssText = "width:1px;height:20px;background:var(--border);margin:0 4px;flex-shrink:0;";
  toolbar.appendChild(divider);

  // Settings area (right of divider, grows to fill)
  const settingsArea = document.createElement("div");
  settingsArea.style.cssText = "display:flex;align-items:center;gap:8px;flex:1;flex-wrap:wrap;justify-content:flex-end;";
  toolbar.appendChild(settingsArea);

  // Canvas area
  const canvasWrap = document.createElement("div");
  canvasWrap.style.cssText = "position:relative;width:100%;overflow:hidden;background:#111;";

  const canvas = document.createElement("canvas");
  canvas.style.cssText = "display:block;transform-origin:top left;cursor:crosshair;";
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
      canvas.style.transform = `scale(${displayScale})`;
      canvasWrap.style.height = ch * displayScale + "px";
    }
    if (dimsChanged) renderCanvas();
  }

  // ── tool settings panel ───────────────────────────────────────────────────
  let colorPickerEl = null;

  function rebuildSettings() {
    settingsArea.innerHTML = "";
    colorPickerEl = null;

    // If an annotation is selected, show its properties (takes priority over tool defaults)
    const selId = currentValue.selected_id;
    const selAnn = selId ? (currentValue.annotations || []).find((a) => a.id === selId) : null;
    if (selAnn) { _buildAnnotationSettings(selAnn); return; }

    if (activeTool === "select") {
      const hint = document.createElement("span");
      hint.className = "ais-setting-label";
      hint.textContent = "Click to select · Drag to move · Dbl-click text to edit";
      hint.style.opacity = "0.5";
      settingsArea.appendChild(hint);
      return;
    }

    _buildToolSettings();
  }

  // onChange(color, emit) — emit=false during drag, emit=true on commit
  function _buildColorSwatch(color, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;display:flex;align-items:center;gap:5px;";
    const lbl = document.createElement("span");
    lbl.className = "ais-setting-label"; lbl.textContent = "Color";
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
    wrap.appendChild(lbl); wrap.appendChild(swatch); wrap.appendChild(colorPickerEl);
    settingsArea.appendChild(wrap);
  }

  function _buildSizeSlider(label, min, max, value, onChange) {
    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;align-items:center;gap:5px;";
    const lbl = document.createElement("span");
    lbl.className = "ais-setting-label"; lbl.textContent = label;
    const slider = document.createElement("input");
    slider.type = "range"; slider.className = "ais-range";
    slider.min = min; slider.max = max; slider.value = value;
    const valLbl = document.createElement("span");
    valLbl.className = "ais-val-label"; valLbl.textContent = value;
    slider.addEventListener("input", () => { const sz = Number(slider.value); valLbl.textContent = sz; onChange(sz, false); });
    slider.addEventListener("change", () => onChange(Number(slider.value), true));
    wrap.appendChild(lbl); wrap.appendChild(slider); wrap.appendChild(valLbl);
    settingsArea.appendChild(wrap);
  }

  function _buildToolSettings() {
    const ts = toolSettings[activeTool] || {};
    const sizeKey = activeTool === "text" ? "font_size" : activeTool === "arrow" ? "width" : "size";
    const sizeVal = ts[sizeKey] ?? (activeTool === "text" ? 48 : activeTool === "arrow" ? 3 : 8);
    const sizeMin = activeTool === "text" ? 8 : 1;
    const sizeMax = activeTool === "text" ? 120 : activeTool === "arrow" ? 20 : 40;
    const sizeLbl = activeTool === "arrow" ? "Width" : "Size";
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
  }

  function _buildAnnotationSettings(ann) {
    let color;
    if (ann.type === "paint") {
      color = (ann.strokes && ann.strokes[0]) ? ann.strokes[0].color : "#ff0000";
    } else {
      color = ann.color || "#ff0000";
    }

    const sizeKey = ann.type === "text" ? "font_size" : ann.type === "arrow" ? "width" : null;
    if (sizeKey) {
      const sizeVal = ann[sizeKey] ?? (ann.type === "text" ? 48 : 3);
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
        if (textInput && textEditId === ann.id && sizeKey === "font_size") {
          textInput.style.fontSize = sz * displayScale + "px"; _autoResizeTextarea();
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
      if (textInput && textEditId === ann.id) {
        textInput.style.color = col; textInput.style.borderBottomColor = col;
      }
      renderCanvas();
      if (emit) _emit();
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
    canvas.style.cursor = id === "select" ? "default" : "crosshair";
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
    ctx.clearRect(0, 0, cw, ch);

    // Background image
    const imgUrl = currentValue.image_url;
    if (imgUrl) {
      try {
        const alreadyCached = !!imageCache[urlCacheKey(imgUrl)];
        const img = await loadImage(imgUrl);
        if (gen !== renderGen) return;

        // On first load, set canvas dimensions from image natural size
        if (!alreadyCached && (!currentValue.canvas_width || !currentValue.canvas_height)) {
          currentValue = {
            ...currentValue,
            canvas_width: img.naturalWidth,
            canvas_height: img.naturalHeight,
          };
          applyCanvasScale();
          return; // applyCanvasScale triggers another render via dimsChanged
        }

        ctx.drawImage(img, 0, 0, cw, ch);
      } catch {
        ctx.fillStyle = "#222";
        ctx.fillRect(0, 0, cw, ch);
      }
    } else {
      ctx.fillStyle = "#1a1a1a";
      ctx.fillRect(0, 0, cw, ch);
    }

    if (gen !== renderGen) return;

    // Draw committed annotations
    for (const ann of (currentValue.annotations || [])) {
      if (ann.id === textEditId) continue; // skip live-edited text
      drawAnnotation(ann, ann.id === currentValue.selected_id);
    }

    // In-progress arrow
    if (currentArrow) {
      drawArrowLine(
        currentArrow.x1, currentArrow.y1,
        currentArrow.x2, currentArrow.y2,
        toolSettings.arrow.color || "#ff0000",
        toolSettings.arrow.width || 3
      );
    }
  }

  function drawAnnotation(ann, selected) {
    if (ann.type === "paint")  drawPaint(ann, selected);
    else if (ann.type === "text")  drawText(ann, selected);
    else if (ann.type === "arrow") drawArrowAnnotation(ann, selected);
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
    renderStrokes(ann.strokes || []);
    ctx.restore();

    if (selected && activeTool === "select") {
      const corners = _getTransformedCorners(ann);
      if (corners.length === 4) {
        const hw = 5 / displayScale;
        // Bounding box
        ctx.save();
        ctx.strokeStyle = "rgba(79,142,247,0.8)";
        ctx.lineWidth = 1.5 / displayScale;
        ctx.setLineDash([4 / displayScale, 3 / displayScale]);
        ctx.beginPath();
        ctx.moveTo(corners[0][0], corners[0][1]);
        for (let i = 1; i < 4; i++) ctx.lineTo(corners[i][0], corners[i][1]);
        ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);
        // Corner scale handles
        for (const [hx, hy] of corners) {
          ctx.fillStyle = "white"; ctx.beginPath(); ctx.arc(hx, hy, hw, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = "rgba(79,142,247,0.9)"; ctx.lineWidth = 1.5 / displayScale;
          ctx.beginPath(); ctx.arc(hx, hy, hw, 0, Math.PI * 2); ctx.stroke();
        }
        // Rotation handle
        const rh = _getRotationHandle(ann, corners);
        if (rh) {
          const topMid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
          ctx.strokeStyle = "rgba(79,142,247,0.5)"; ctx.lineWidth = 1 / displayScale;
          ctx.beginPath(); ctx.moveTo(topMid[0], topMid[1]); ctx.lineTo(rh[0], rh[1]); ctx.stroke();
          ctx.fillStyle = "#4f8ef7"; ctx.beginPath(); ctx.arc(rh[0], rh[1], hw, 0, Math.PI * 2); ctx.fill();
          ctx.strokeStyle = "white"; ctx.lineWidth = 1.5 / displayScale;
          ctx.beginPath(); ctx.arc(rh[0], rh[1], hw, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.restore();
      }
    }
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

  function _getRotationHandle(ann, corners) {
    if (corners.length < 3) return null;
    const topMid = [(corners[0][0] + corners[1][0]) / 2, (corners[0][1] + corners[1][1]) / 2];
    const center = [(corners[0][0] + corners[2][0]) / 2, (corners[0][1] + corners[2][1]) / 2];
    const dx = topMid[0] - center[0], dy = topMid[1] - center[1];
    const len = Math.hypot(dx, dy) || 1;
    const d = 28 / displayScale;
    return [topMid[0] + dx / len * d, topMid[1] + dy / len * d];
  }

  function renderStrokes(strokes) {
    for (const stroke of strokes) {
      const pts = stroke.points || [];
      if (!pts.length) continue;
      const defaultSz = stroke.size || 8;
      ctx.fillStyle = stroke.color || "#ff0000";
      // Initial dot
      ctx.beginPath();
      ctx.arc(pts[0][0], pts[0][1], (pts[0][2] ?? defaultSz) / 2, 0, Math.PI * 2);
      ctx.fill();
      // Variable-width path: filled trapezoid + endpoint circle per segment
      for (let i = 1; i < pts.length; i++) {
        const px = pts[i-1][0], py = pts[i-1][1], pr = (pts[i-1][2] ?? defaultSz) / 2;
        const qx = pts[i][0],   qy = pts[i][1],   qr = (pts[i][2]   ?? defaultSz) / 2;
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
    ctx.save();
    ctx.font = `${fontSize}px sans-serif`;
    ctx.fillStyle = ann.color || "#ffffff";
    ctx.textBaseline = "top";
    ctx.fillText(ann.text || "", ann.x || 0, ann.y || 0);

    const isHovered = ann.id === hoverId;
    if (selected || isHovered) {
      const w = ctx.measureText(ann.text || "").width;
      const h = fontSize * 1.2;
      ctx.strokeStyle = selected ? "rgba(79,142,247,0.85)" : "rgba(79,142,247,0.4)";
      ctx.lineWidth = (selected ? 1.5 : 1) / displayScale;
      ctx.setLineDash([4 / displayScale, 3 / displayScale]);
      ctx.strokeRect((ann.x || 0) - 4, (ann.y || 0) - 4, w + 8, h + 8);
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  function drawArrowAnnotation(ann, selected) {
    drawArrowLine(ann.x1, ann.y1, ann.x2, ann.y2, ann.color || "#ff0000", ann.width || 3);
    if (selected) {
      const r = 5 / displayScale;
      ctx.fillStyle = "rgba(79,142,247,0.9)";
      ctx.beginPath(); ctx.arc(ann.x1, ann.y1, r, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(ann.x2, ann.y2, r, 0, Math.PI * 2); ctx.fill();
    }
  }

  function drawArrowLine(x1, y1, x2, y2, color, width) {
    const w = Math.max(1, width);
    ctx.save();
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = w;
    ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    const angle = Math.atan2(y2 - y1, x2 - x1);
    const head = Math.max(15, w * 4);
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(angle - Math.PI / 6), y2 - head * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(x2 - head * Math.cos(angle + Math.PI / 6), y2 - head * Math.sin(angle + Math.PI / 6));
    ctx.closePath(); ctx.fill();
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
        const dx = ann.x2 - ann.x1, dy = ann.y2 - ann.y1;
        const len = Math.sqrt(dx * dx + dy * dy);
        if (len < 1) continue;
        const t = Math.max(0, Math.min(1, ((cx - ann.x1) * dx + (cy - ann.y1) * dy) / (len * len)));
        const px = ann.x1 + t * dx, py = ann.y1 + t * dy;
        // 12 screen-pixel minimum tolerance so thin arrows stay clickable at small display sizes
        const tol = Math.max(12 / displayScale, (ann.width || 3) + 6);
        if (Math.sqrt((cx - px) ** 2 + (cy - py) ** 2) <= tol) return ann;
      } else if (ann.type === "paint") {
        const [lx, ly] = _paintInvTransformPt(ann, cx, cy);
        for (const stroke of (ann.strokes || [])) {
          const tol = Math.max(12 / displayScale, (stroke.size || 8) / 2 + 4);
          for (const pt of (stroke.points || [])) {
            if (Math.hypot(lx - pt[0], ly - pt[1]) <= tol) return ann;
          }
        }
      }
    }
    return null;
  }

  // ── text editing ──────────────────────────────────────────────────────────
  function startTextEdit(ann) {
    commitTextEdit();
    textEditId = ann.id;
    currentValue = { ...currentValue, selected_id: ann.id };

    const fontSize = Math.max(8, ann.font_size || 48);
    textInput = document.createElement("textarea");
    textInput.value = ann.text || "";
    textInput.rows = 1;
    textInput.style.cssText = [
      "position:absolute",
      `left:${(ann.x || 0) * displayScale}px`,
      `top:${(ann.y || 0) * displayScale}px`,
      "min-width:60px",
      "background:transparent",
      `color:${ann.color || "#ffffff"}`,
      `font-size:${fontSize * displayScale}px`,
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
      if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); commitTextEdit(); }
      if (e.key === "Escape") { e.preventDefault(); commitTextEdit(); }
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
    const text = textInput.value.trim();
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
          selected_id: null,
          annotations: currentValue.annotations.filter((a) => a.id !== id),
        };
      } else {
        currentValue = {
          ...currentValue,
          selected_id: null,
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
      selected_id: id,
    };
    startTextEdit(ann);
    // emit after startTextEdit so the annotation with the initial text is sent
    _emit();
  }

  // ── pointer events ────────────────────────────────────────────────────────
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
    isPointerDown = true;
    const [cx, cy] = screenToCanvas(e);

    if (activeTool === "text") {
      if (textEditId) { commitTextEdit(); return; } // clicking away commits edit
      const hit = hitTest(cx, cy);
      if (hit && hit.type === "text") {
        // Single click on existing text: select it for dragging
        hoverId = null;
        currentValue = { ...currentValue, selected_id: hit.id };
        dragState = { id: hit.id, startCx: cx, startCy: cy,
          origX: hit.x ?? 0, origY: hit.y ?? 0 };
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

      // Check transform handles on currently selected paint annotation first
      const selId = currentValue.selected_id;
      const selAnn = selId ? (currentValue.annotations || []).find((a) => a.id === selId) : null;
      if (selAnn && selAnn.type === "paint") {
        const corners = _getTransformedCorners(selAnn);
        const rh = corners.length >= 3 ? _getRotationHandle(selAnn, corners) : null;
        const [pcx, pcy] = _paintCenter(selAnn);
        const center = [pcx + (selAnn.x || 0), pcy + (selAnn.y || 0)];
        if (rh && Math.hypot(cx - rh[0], cy - rh[1]) <= handleR) {
          dragState = { id: selAnn.id, type: "rotate",
            centerX: center[0], centerY: center[1],
            origAngle: Math.atan2(cy - center[1], cx - center[0]),
            origRotation: selAnn.rotation || 0 };
          renderCanvas(); return;
        }
        for (let i = 0; i < corners.length; i++) {
          if (Math.hypot(cx - corners[i][0], cy - corners[i][1]) <= handleR) {
            const origDist = Math.hypot(cx - center[0], cy - center[1]);
            dragState = { id: selAnn.id, type: "scale",
              centerX: center[0], centerY: center[1],
              origDist: origDist || 1,
              origScaleX: selAnn.scaleX ?? 1, origScaleY: selAnn.scaleY ?? 1 };
            renderCanvas(); return;
          }
        }
      }

      // Regular hit test
      const hit = hitTest(cx, cy);
      currentValue = { ...currentValue, selected_id: hit ? hit.id : null };
      if (hit) {
        if (hit.type === "arrow") {
          const arrowHandleR = Math.max(10 / displayScale, 8);
          const nearStart = Math.hypot(cx - hit.x1, cy - hit.y1) <= arrowHandleR;
          const nearEnd   = Math.hypot(cx - hit.x2, cy - hit.y2) <= arrowHandleR;
          if (nearStart) {
            dragState = { id: hit.id, arrowHandle: "start", startCx: cx, startCy: cy,
              origX1: hit.x1, origY1: hit.y1, origX2: hit.x2, origY2: hit.y2 };
          } else if (nearEnd) {
            dragState = { id: hit.id, arrowHandle: "end", startCx: cx, startCy: cy,
              origX1: hit.x1, origY1: hit.y1, origX2: hit.x2, origY2: hit.y2 };
          } else {
            dragState = { id: hit.id, startCx: cx, startCy: cy,
              origX1: hit.x1, origY1: hit.y1, origX2: hit.x2, origY2: hit.y2 };
          }
        } else if (hit.type === "paint") {
          dragState = { id: hit.id, startCx: cx, startCy: cy,
            origX: hit.x || 0, origY: hit.y || 0 };
        } else {
          dragState = { id: hit.id, startCx: cx, startCy: cy,
            origX: hit.x ?? 0, origY: hit.y ?? 0 };
        }
      } else {
        dragState = null;
      }
      rebuildSettings();
      renderCanvas();
      return;
    }

    if (activeTool === "paint") {
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
      currentArrow = { x1: cx, y1: cy, x2: cx, y2: cy };
    }
  }

  function onPointerMove(e) {
    if (!isPointerDown) return;
    e.stopPropagation();
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

    } else if (dragState && (activeTool === "select" || activeTool === "text")) {
      if (dragState.type === "rotate") {
        const angle = Math.atan2(cy - dragState.centerY, cx - dragState.centerX);
        const newRotation = dragState.origRotation + (angle - dragState.origAngle);
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) =>
            a.id === dragState.id ? { ...a, rotation: newRotation } : a
          ),
        };
      } else if (dragState.type === "scale") {
        const dist = Math.hypot(cx - dragState.centerX, cy - dragState.centerY);
        const ratio = Math.max(0.05, dist / dragState.origDist);
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) =>
            a.id === dragState.id
              ? { ...a, scaleX: dragState.origScaleX * ratio, scaleY: dragState.origScaleY * ratio }
              : a
          ),
        };
      } else {
        const dx = cx - dragState.startCx, dy = cy - dragState.startCy;
        currentValue = {
          ...currentValue,
          annotations: currentValue.annotations.map((a) => {
            if (a.id !== dragState.id) return a;
            if (a.type === "arrow") {
              if (dragState.arrowHandle === "start")
                return { ...a, x1: dragState.origX1 + dx, y1: dragState.origY1 + dy };
              if (dragState.arrowHandle === "end")
                return { ...a, x2: dragState.origX2 + dx, y2: dragState.origY2 + dy };
              return { ...a, x1: dragState.origX1 + dx, y1: dragState.origY1 + dy,
                x2: dragState.origX2 + dx, y2: dragState.origY2 + dy };
            }
            return { ...a, x: (dragState.origX || 0) + dx, y: (dragState.origY || 0) + dy };
          }),
        };
      }
      renderCanvas();
    }
  }

  function onPointerUp(e) {
    if (!isPointerDown) return;
    isPointerDown = false;
    e.stopPropagation();
    const [cx, cy] = screenToCanvas(e);

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
        selected_id: paintAnn.id,
      };
      _emit();
      rebuildSettings();
      renderCanvas();

    } else if (activeTool === "arrow" && currentArrow) {
      const arr = currentArrow;
      currentArrow = null;
      if (Math.hypot(arr.x2 - arr.x1, arr.y2 - arr.y1) > 5) {
        const ann = {
          id: _uid("arrow"), type: "arrow",
          x1: arr.x1, y1: arr.y1, x2: cx, y2: cy,
          color: toolSettings.arrow.color,
          width: toolSettings.arrow.width,
        };
        currentValue = {
          ...currentValue,
          annotations: [...(currentValue.annotations || []), ann],
          selected_id: ann.id,
        };
        _emit();
        rebuildSettings();
      }
      renderCanvas();

    } else if (dragState && (activeTool === "select" || activeTool === "text")) {
      dragState = null;
      _emit();
    }
  }

  // Hover highlight for text tool (and select tool cursors)
  function onMouseHover(e) {
    if (isPointerDown) return;
    if (activeTool !== "text" && activeTool !== "select") {
      if (hoverId) { hoverId = null; renderCanvas(); }
      return;
    }
    const [cx, cy] = screenToCanvas(e);
    const hit = hitTest(cx, cy);
    const newId = (hit && hit.type === "text") ? hit.id : null;
    if (newId !== hoverId) {
      hoverId = newId;
      // Show pointer cursor when hovering over text in either tool
      if (activeTool === "select") canvas.style.cursor = newId ? "move" : "default";
      if (activeTool === "text")   canvas.style.cursor = newId ? "move" : "crosshair";
      renderCanvas();
    }
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
  function _emit() {
    if (onChange) onChange({ ...currentValue, tool_settings: { ...toolSettings } });
  }

  function _uid(prefix) {
    return `${prefix}-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 7)}`;
  }

  // ── update from props ──────────────────────────────────────────────────────
  function handleUpdate(newProps) {
    if (newProps?.onChange) { /* onChange captured at init */ }
    const nv = (newProps?.value && typeof newProps.value === "object") ? newProps.value : defaultData();
    const urlChanged = nv.image_url !== currentValue.image_url;
    const dimsChanged = nv.canvas_width !== currentValue.canvas_width ||
                        nv.canvas_height !== currentValue.canvas_height;
    currentValue = { ...defaultData(), ...nv, tool_settings: { ...toolSettings, ...(nv.tool_settings || {}) } };
    toolSettings = { ...currentValue.tool_settings };
    activeTool = currentValue.active_tool || activeTool;

    // Update tool buttons
    for (const [tid, btn] of Object.entries(toolBtns)) {
      btn.className = "ais-tool-btn" + (tid === activeTool ? " active" : "");
    }
    canvas.style.cursor = activeTool === "select" ? "default" : "crosshair";
    rebuildSettings();

    if (dimsChanged || urlChanged) applyCanvasScale();
    renderCanvas();
  }

  // ── cleanup ────────────────────────────────────────────────────────────────
  function cleanup() {
    commitTextEdit();
    resizeObserver.disconnect();
    if (resizeRafId) cancelAnimationFrame(resizeRafId);
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
