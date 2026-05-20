/**
 * AnnotateImage Widget
 *
 * An in-node canvas editor that displays image layers and lets users annotate
 * with paint strokes, text, and arrows. The side panel (tool settings + layers)
 * lives inline to the right of the canvas inside the widget itself — no floating
 * document.body panels, no RAF positioning loop.
 *
 * Tools: select (move/click layers), paint (freehand), text (click-to-place), arrow (drag)
 *
 * State is stored in the annotation_data ParameterDict and synced to Python
 * via onChange on every significant user action.
 */

const TOOLS = [
  { id: "select", icon: "select", title: "Select & Move" },
  { id: "paint",  icon: "paint",  title: "Paint" },
  { id: "text",   icon: "text",   title: "Text" },
  { id: "arrow",  icon: "arrow",  title: "Arrow" },
];

const DEFAULT_TOOL_SETTINGS = {
  paint: { color: "#ff0000", size: 10 },
  text:  { color: "#000000", font_size: 24, font: "Arial" },
  arrow: { color: "#ff0000", width: 3 },
};

// ── Lucide SVG icons (inlined paths, MIT licensed) ────────────────────────────

const ICON_PATHS = {
  select:  `<path d="m4 4 7.07 17 2.51-7.39L21 11.07z"/>`,
  paint:   `<path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z"/><path d="m15 5 4 4"/>`,
  text:    `<polyline points="4 7 4 4 20 4 20 7"/><line x1="9" x2="15" y1="20" y2="20"/><line x1="12" x2="12" y1="4" y2="20"/>`,
  arrow:   `<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/>`,
  layers:  `<path d="m12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>`,
  eye:     `<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>`,
  eyeOff:  `<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" x2="22" y1="2" y2="22"/>`,
  trash:   `<path d="M3 6h18"/><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/><line x1="10" x2="10" y1="11" y2="17"/><line x1="14" x2="14" y1="11" y2="17"/>`,
  plus:    `<path d="M5 12h14"/><path d="M12 5v14"/>`,
  image:   `<rect width="18" height="18" x="3" y="3" rx="2" ry="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>`,
};

function icon(name, size = 14) {
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

// ── style injection ───────────────────────────────────────────────────────────

function injectGlobalStyles() {
  // Always overwrite so CSS changes take effect after hot-reload without refresh.
  let style = document.getElementById("annotate-image-styles");
  if (!style) {
    style = document.createElement("style");
    style.id = "annotate-image-styles";
    document.head.appendChild(style);
  }
  // Colors pull from the app's CSS custom properties so the widget matches the theme.
  style.textContent = `
    .ai-tool-btn { background:transparent; border:none; border-radius:4px; color:var(--muted-foreground); cursor:pointer; width:28px; height:28px; display:flex; align-items:center; justify-content:center; transition:background 0.15s,color 0.15s; flex-shrink:0; padding:0; }
    .ai-tool-btn:hover { background:var(--muted); color:var(--foreground); }
    .ai-tool-btn.active { background:var(--sidebar-primary); color:var(--sidebar-primary-foreground); }
    .ai-layers-btn { margin-left:auto; background:transparent; border:1px solid var(--border); border-radius:4px; color:var(--muted-foreground); cursor:pointer; padding:0 8px; height:26px; font-size:11px; white-space:nowrap; display:flex; align-items:center; gap:5px; transition:background 0.15s,color 0.15s; flex-shrink:0; }
    .ai-layers-btn:hover { background:var(--muted); color:var(--foreground); }
    .ai-layers-btn.active { background:var(--sidebar-primary); border-color:var(--sidebar-primary); color:var(--sidebar-primary-foreground); }
    .ai-canvas { display:block; transform-origin:top left; cursor:crosshair; }
    .ai-canvas.select-cursor { cursor:default; }
    .ai-text-input { position:absolute; background:var(--popover); border:1.5px solid var(--sidebar-primary); outline:none; padding:2px 6px; border-radius:3px; min-width:100px; color:var(--foreground); font-size:14px; }
    .ai-panel-header { display:flex; align-items:center; justify-content:space-between; padding:6px 10px; background:var(--card); border-bottom:1px solid var(--border); font-size:11px; font-weight:600; color:var(--muted-foreground); text-transform:uppercase; letter-spacing:0.04em; }
    .ai-panel-body { padding:8px 10px; }
    .ai-layer-row { display:flex; flex-direction:column; padding:5px 6px; margin-bottom:2px; border-radius:4px; cursor:pointer; border:1px solid transparent; }
    .ai-layer-row.selected { background:color-mix(in srgb, var(--sidebar-primary) 15%, transparent); border-color:color-mix(in srgb, var(--sidebar-primary) 50%, transparent); }
    .ai-layer-row:not(.selected):hover { background:var(--muted); }
    .ai-layer-top { display:flex; align-items:center; gap:5px; }
    .ai-layer-name { flex:1; font-size:11px; color:var(--foreground); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .ai-layer-type-icon { color:var(--muted-foreground); flex-shrink:0; display:flex; }
    .ai-icon-btn { background:none; border:none; cursor:pointer; padding:2px; color:var(--muted-foreground); display:flex; align-items:center; flex-shrink:0; border-radius:3px; opacity:0.6; }
    .ai-icon-btn:hover { opacity:1; color:var(--foreground); }
    .ai-opacity-row { display:flex; align-items:center; gap:6px; margin-top:4px; padding:0 2px; }
    .ai-opacity-label { font-size:10px; color:var(--muted-foreground); min-width:40px; }
    .ai-opacity-slider { flex:1; height:3px; accent-color:var(--sidebar-primary); cursor:pointer; }
    .ai-setting-row { display:flex; align-items:center; justify-content:space-between; margin-bottom:8px; }
    .ai-setting-label { font-size:11px; color:var(--muted-foreground); }
    .ai-color-input { width:28px; height:20px; border:none; background:none; cursor:pointer; border-radius:3px; overflow:hidden; padding:0; }
    .ai-size-col { display:flex; flex-direction:column; gap:4px; margin-bottom:8px; }
    .ai-size-header { display:flex; justify-content:space-between; align-items:baseline; }
    .ai-size-value { font-size:11px; color:var(--foreground); }
    .ai-size-slider { width:100%; accent-color:var(--sidebar-primary); cursor:pointer; }
    .ai-select-info { font-size:11px; color:var(--muted-foreground); display:flex; flex-direction:column; gap:3px; }
    .ai-select-info-row { display:flex; justify-content:space-between; }
    .ai-select-hint { font-size:11px; color:var(--muted-foreground); opacity:0.6; }
  `;
}

// ── main widget ───────────────────────────────────────────────────────────────

export default function AnnotateImage(container, props) {
  // Early-return if already mounted in this container.
  if (container._instance) {
    container._instance.handleUpdate(props);
    return {
      cleanup: container._instance.cleanup,
      update: container._instance.handleUpdate,
    };
  }

  injectGlobalStyles();

  // ── state ────────────────────────────────────────────────────────────────

  let onChangeFn = props.onChange;
  let currentValue = normalizeValue(props.value);
  let activeTool = currentValue.active_tool || "select";
  let displayScale = 1;
  let isPointerDown = false;
  let currentStroke = null;
  let currentArrow = null;
  let dragState = null;
  const imageCache = {};

  function normalizeValue(v) {
    const base = {
      canvas_width: 1920,
      canvas_height: 1080,
      layers: [],
      active_tool: "select",
      tool_settings: { ...DEFAULT_TOOL_SETTINGS },
      selected_layer_id: null,
      layers_panel_open: true,
    };
    if (!v || typeof v !== "object") return base;
    return {
      ...base,
      ...v,
      tool_settings: { ...base.tool_settings, ...(v.tool_settings || {}) },
    };
  }

  // ── DOM ───────────────────────────────────────────────────────────────────

  const wrapper = document.createElement("div");
  wrapper.className = "nodrag nowheel";
  wrapper.style.cssText = "display:flex;flex-direction:column;width:100%;background:var(--background);border-radius:6px;font-family:sans-serif;box-sizing:border-box;overflow:hidden;";

  // Toolbar
  const toolbar = document.createElement("div");
  toolbar.style.cssText = "display:flex;align-items:center;gap:3px;padding:5px 8px;background:var(--card);border-bottom:1px solid var(--border);flex-shrink:0;";

  const toolButtons = {};
  TOOLS.forEach(({ id, icon: iconName, title }) => {
    const btn = document.createElement("button");
    btn.className = "ai-tool-btn" + (id === activeTool ? " active" : "");
    btn.title = title;
    btn.appendChild(icon(iconName, 15));
    btn.addEventListener("pointerdown", (e) => {
      e.stopPropagation();
      setActiveTool(id);
    });
    toolbar.appendChild(btn);
    toolButtons[id] = btn;
  });

  const layersToggle = document.createElement("button");
  layersToggle.className = "ai-layers-btn" + (currentValue.layers_panel_open ? " active" : "");
  layersToggle.appendChild(icon("layers", 13));
  const layersToggleLabel = document.createElement("span");
  layersToggleLabel.textContent = "Layers";
  layersToggle.appendChild(layersToggleLabel);
  layersToggle.addEventListener("pointerdown", (e) => {
    e.stopPropagation();
    currentValue = { ...currentValue, layers_panel_open: !currentValue.layers_panel_open };
    layersToggle.className = "ai-layers-btn" + (currentValue.layers_panel_open ? " active" : "");
    sidePanel.style.display = currentValue.layers_panel_open ? "flex" : "none";
    emitChange();
  });
  toolbar.appendChild(layersToggle);

  // Content row: canvas area (flex:1) + inline side panel (shrinkable)
  const content = document.createElement("div");
  content.style.cssText = "display:flex;flex-direction:row;min-width:0;align-items:stretch;";

  const canvasArea = document.createElement("div");
  canvasArea.style.cssText = "flex:1 1 0;min-width:0;overflow:hidden;position:relative;";

  const canvasWrapper = document.createElement("div");
  canvasWrapper.style.cssText = "position:relative;overflow:hidden;width:100%;";

  const canvas = document.createElement("canvas");
  canvas.className = "ai-canvas" + (activeTool === "select" ? " select-cursor" : "");
  canvasWrapper.appendChild(canvas);
  canvasArea.appendChild(canvasWrapper);

  // Inline side panel (tool settings section + divider + layers section)
  const sidePanel = buildSidePanelEl();
  sidePanel.style.display = currentValue.layers_panel_open ? "flex" : "none";

  content.appendChild(canvasArea);
  content.appendChild(sidePanel);

  wrapper.appendChild(toolbar);
  wrapper.appendChild(content);
  container.appendChild(wrapper);

  // ── canvas scaling ────────────────────────────────────────────────────────

  const resizeObserver = new ResizeObserver(() => applyCanvasScale());
  resizeObserver.observe(canvasArea);

  function applyCanvasScale() {
    const cw = currentValue.canvas_width || 1920;
    const ch = currentValue.canvas_height || 1080;
    const areaW = canvasArea.clientWidth || 300;
    displayScale = areaW / cw;

    canvas.width = cw;
    canvas.height = ch;
    canvas.style.width = cw + "px";
    canvas.style.height = ch + "px";
    canvas.style.transform = `scale(${displayScale})`;
    canvasWrapper.style.height = ch * displayScale + "px";

    renderCanvas();
  }

  function screenToCanvas(e) {
    const rect = canvas.getBoundingClientRect();
    return [
      (e.clientX - rect.left) * (canvas.width / rect.width),
      (e.clientY - rect.top) * (canvas.height / rect.height),
    ];
  }

  // ── rendering ─────────────────────────────────────────────────────────────

  const ctx = canvas.getContext("2d");

  function initLayerDimensions(layer, img) {
    const nw = img.naturalWidth;
    const nh = img.naturalHeight;

    let cw = currentValue.canvas_width || 0;
    let ch = currentValue.canvas_height || 0;
    let canvasSizeChanged = false;

    if (!cw || !ch) {
      cw = nw;
      ch = nh;
      canvasSizeChanged = true;
    }

    const scaleX = nw > cw ? cw / nw : 1.0;
    const scaleY = nh > ch ? ch / nh : 1.0;
    const fitScale = Math.min(scaleX, scaleY);

    const updatedLayer = {
      ...layer,
      width: nw,
      height: nh,
      scaleX: fitScale,
      scaleY: fitScale,
      x: cw / 2,
      y: ch / 2,
    };

    const layers = (currentValue.layers || []).map((l) =>
      l.id === layer.id ? updatedLayer : l
    );

    currentValue = {
      ...currentValue,
      canvas_width: cw,
      canvas_height: ch,
      layers,
    };

    if (canvasSizeChanged) {
      applyCanvasScale();
    } else {
      renderCanvas();
    }
    emitChange();
    rebuildLayersPanel();
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      if (imageCache[url]) { resolve(imageCache[url]); return; }
      const img = new window.Image();
      img.crossOrigin = "anonymous";
      img.onload = () => { imageCache[url] = img; resolve(img); };
      img.onerror = reject;
      img.src = url;
    });
  }

  function renderCanvas() {
    const cw = currentValue.canvas_width || 1920;
    const ch = currentValue.canvas_height || 1080;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, cw, ch);

    const layers = [...(currentValue.layers || [])].sort(
      (a, b) => (a.order ?? 0) - (b.order ?? 0)
    );

    const renderPromises = layers
      .filter((l) => l.visible !== false)
      .map((layer) => renderLayer(layer));

    Promise.all(renderPromises).then(() => {
      renderInProgress();
      if (activeTool === "select" && currentValue.selected_layer_id) {
        const sel = layers.find((l) => l.id === currentValue.selected_layer_id);
        if (sel) renderSelectionHandles(sel);
      }
    });
  }

  async function renderLayer(layer) {
    ctx.save();
    ctx.globalAlpha = layer.opacity ?? 1;

    if (layer.type === "image") {
      try {
        const alreadyCached = !!imageCache[layer.url];
        const img = await loadImage(layer.url);

        if (!alreadyCached) {
          initLayerDimensions(layer, img);
          return;
        }

        const bw = layer.width || img.naturalWidth;
        const bh = layer.height || img.naturalHeight;
        const w = bw * (layer.scaleX ?? 1);
        const h = bh * (layer.scaleY ?? 1);
        const x = (layer.x ?? currentValue.canvas_width / 2) - w / 2;
        const y = (layer.y ?? currentValue.canvas_height / 2) - h / 2;

        if (layer.rotation) {
          const cx = x + w / 2;
          const cy = y + h / 2;
          ctx.translate(cx, cy);
          ctx.rotate((layer.rotation * Math.PI) / 180);
          ctx.drawImage(img, -w / 2, -h / 2, w, h);
        } else {
          ctx.drawImage(img, x, y, w, h);
        }
      } catch {
        // skip failed image
      }
    } else if (layer.type === "paint") {
      renderStrokes(layer.strokes || [], layer.x ?? 0, layer.y ?? 0);
    } else if (layer.type === "text") {
      renderText(layer);
    } else if (layer.type === "arrow") {
      renderArrow(layer.x1, layer.y1, layer.x2, layer.y2, layer.color || "#ff0000", layer.width || 3);
    }

    ctx.restore();
  }

  function renderStrokes(strokes, ox = 0, oy = 0) {
    strokes.forEach((stroke) => {
      const pts = stroke.points || [];
      if (pts.length < 1) return;
      ctx.beginPath();
      ctx.strokeStyle = stroke.color || "#ff0000";
      ctx.lineWidth = stroke.size || 10;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      if (pts.length === 1) {
        ctx.arc(pts[0][0] + ox, pts[0][1] + oy, (stroke.size || 10) / 2, 0, Math.PI * 2);
        ctx.fillStyle = stroke.color || "#ff0000";
        ctx.fill();
      } else {
        ctx.moveTo(pts[0][0] + ox, pts[0][1] + oy);
        for (let i = 1; i < pts.length; i++) {
          ctx.lineTo(pts[i][0] + ox, pts[i][1] + oy);
        }
        ctx.stroke();
      }
    });
  }

  function renderText(layer) {
    const size = layer.font_size || 24;
    const font = layer.font || "Arial";
    ctx.fillStyle = layer.color || "#000000";
    ctx.font = `${size}px "${font}", sans-serif`;
    ctx.fillText(layer.text || "", layer.x || 0, layer.y || 0);
  }

  function renderArrow(x1, y1, x2, y2, color, width) {
    if (x1 == null || x2 == null) return;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineCap = "round";
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const angle = Math.atan2(y2 - y1, x2 - x1);
    const headLen = Math.max(15, width * 4);
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.moveTo(x2, y2);
    ctx.lineTo(
      x2 - headLen * Math.cos(angle - Math.PI / 6),
      y2 - headLen * Math.sin(angle - Math.PI / 6)
    );
    ctx.lineTo(
      x2 - headLen * Math.cos(angle + Math.PI / 6),
      y2 - headLen * Math.sin(angle + Math.PI / 6)
    );
    ctx.closePath();
    ctx.fill();
  }

  function renderInProgress() {
    if (currentStroke && currentStroke.points.length >= 1) {
      ctx.save();
      ctx.globalAlpha = 1;
      renderStrokes([currentStroke]);
      ctx.restore();
    }
    if (currentArrow) {
      ctx.save();
      ctx.globalAlpha = 1;
      renderArrow(
        currentArrow.x1, currentArrow.y1,
        currentArrow.x2, currentArrow.y2,
        currentValue.tool_settings?.arrow?.color || "#ff0000",
        currentValue.tool_settings?.arrow?.width || 3
      );
      ctx.restore();
    }
  }

  function renderSelectionHandles(layer) {
    if (layer.type !== "image") return;
    const bw = layer.width || 100;
    const bh = layer.height || 100;
    const w = bw * (layer.scaleX || 1);
    const h = bh * (layer.scaleY || 1);
    const x = (layer.x ?? 0) - w / 2;
    const y = (layer.y ?? 0) - h / 2;

    ctx.save();
    ctx.strokeStyle = "#00aaff";
    ctx.lineWidth = 2 / displayScale;
    ctx.setLineDash([6 / displayScale, 4 / displayScale]);
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    const hs = 8 / displayScale;
    [[x, y], [x + w, y], [x, y + h], [x + w, y + h]].forEach(([hx, hy]) => {
      ctx.fillStyle = "#00aaff";
      ctx.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
    });
    ctx.restore();
  }

  // ── pointer events ────────────────────────────────────────────────────────

  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointerleave", onPointerUp);

  function onPointerDown(e) {
    e.stopPropagation();
    canvas.setPointerCapture(e.pointerId);
    const [cx, cy] = screenToCanvas(e);
    isPointerDown = true;

    if (activeTool === "select") {
      const hit = hitTestLayer(cx, cy);
      if (hit) {
        currentValue = { ...currentValue, selected_layer_id: hit.id };
        if (hit.type === "arrow") {
          dragState = { layerId: hit.id, startCx: cx, startCy: cy,
            origX1: hit.x1, origY1: hit.y1, origX2: hit.x2, origY2: hit.y2 };
        } else {
          dragState = { layerId: hit.id, startCx: cx, startCy: cy,
            origX: hit.x ?? 0, origY: hit.y ?? 0 };
        }
      } else {
        currentValue = { ...currentValue, selected_layer_id: null };
        dragState = null;
      }
      renderCanvas();
      rebuildLayersPanel();
      updateToolSettingsPanel();

    } else if (activeTool === "paint") {
      currentStroke = {
        color: currentValue.tool_settings?.paint?.color || "#ff0000",
        size: currentValue.tool_settings?.paint?.size || 10,
        points: [[cx, cy]],
      };

    } else if (activeTool === "arrow") {
      currentArrow = { x1: cx, y1: cy, x2: cx, y2: cy };
    }
  }

  function onPointerMove(e) {
    if (!isPointerDown) return;
    e.stopPropagation();
    const [cx, cy] = screenToCanvas(e);

    if (activeTool === "paint" && currentStroke) {
      currentStroke.points.push([cx, cy]);
      renderCanvas();
    } else if (activeTool === "arrow" && currentArrow) {
      currentArrow = { ...currentArrow, x2: cx, y2: cy };
      renderCanvas();
    } else if (activeTool === "select" && dragState) {
      const dx = cx - dragState.startCx;
      const dy = cy - dragState.startCy;
      const layers = (currentValue.layers || []).map((l) => {
        if (l.id !== dragState.layerId) return l;
        if (l.type === "arrow") {
          return { ...l,
            x1: dragState.origX1 + dx, y1: dragState.origY1 + dy,
            x2: dragState.origX2 + dx, y2: dragState.origY2 + dy };
        }
        // image, paint, text all translate via x/y
        return { ...l, x: dragState.origX + dx, y: dragState.origY + dy };
      });
      currentValue = { ...currentValue, layers };
      renderCanvas();
    }
  }

  function onPointerUp(e) {
    if (!isPointerDown) return;
    isPointerDown = false;
    e.stopPropagation();

    if (activeTool === "paint" && currentStroke && currentStroke.points.length >= 1) {
      const stroke = currentStroke;
      currentStroke = null;
      commitPaintStroke(stroke);

    } else if (activeTool === "arrow" && currentArrow) {
      const arr = currentArrow;
      currentArrow = null;
      const dx = arr.x2 - arr.x1;
      const dy = arr.y2 - arr.y1;
      if (Math.sqrt(dx * dx + dy * dy) > 5) {
        commitArrow(arr);
      } else {
        renderCanvas();
      }

    } else if (activeTool === "select" && dragState) {
      dragState = null;
      emitChange();
      updateToolSettingsPanel();
      renderCanvas();
    }
  }

  canvas.addEventListener("click", (e) => {
    if (activeTool !== "text") return;
    e.stopPropagation();
    const [cx, cy] = screenToCanvas(e);
    placeTextInput(cx, cy);
  });

  // ── tool actions ──────────────────────────────────────────────────────────

  function pointToSegmentDist(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1, dy = y2 - y1;
    const lenSq = dx * dx + dy * dy;
    if (lenSq === 0) return Math.hypot(px - x1, py - y1);
    const t = Math.max(0, Math.min(1, ((px - x1) * dx + (py - y1) * dy) / lenSq));
    return Math.hypot(px - (x1 + t * dx), py - (y1 + t * dy));
  }

  function hitTestLayer(cx, cy) {
    const layers = [...(currentValue.layers || [])].sort(
      (a, b) => (b.order ?? 0) - (a.order ?? 0)
    );
    for (const layer of layers) {
      if (!layer.visible) continue;
      if (layer.type === "image") {
        const w = (layer.width || 100) * (layer.scaleX || 1);
        const h = (layer.height || 100) * (layer.scaleY || 1);
        const lx = (layer.x ?? 0) - w / 2;
        const ly = (layer.y ?? 0) - h / 2;
        if (cx >= lx && cx <= lx + w && cy >= ly && cy <= ly + h) return layer;
      } else if (layer.type === "paint") {
        const ox = layer.x ?? 0, oy = layer.y ?? 0;
        for (const stroke of (layer.strokes || [])) {
          const r = (stroke.size || 10) / 2 + 4;
          for (const [px, py] of (stroke.points || [])) {
            if (Math.hypot(cx - (px + ox), cy - (py + oy)) <= r) return layer;
          }
        }
      } else if (layer.type === "text") {
        const size = layer.font_size || 24;
        const textW = (layer.text || "").length * size * 0.6;
        const lx = layer.x ?? 0, ly = layer.y ?? 0;
        if (cx >= lx && cx <= lx + textW && cy >= ly - size && cy <= ly + 4) return layer;
      } else if (layer.type === "arrow") {
        const dist = pointToSegmentDist(cx, cy, layer.x1, layer.y1, layer.x2, layer.y2);
        if (dist <= 10) return layer;
      }
    }
    return null;
  }

  function commitPaintStroke(stroke) {
    const layers = currentValue.layers || [];
    const selectedPaint = layers.find(
      (l) => l.id === currentValue.selected_layer_id && l.type === "paint"
    );

    let newLayers;
    let newSelectedId = currentValue.selected_layer_id;

    if (selectedPaint) {
      newLayers = layers.map((l) =>
        l.id === selectedPaint.id
          ? { ...l, strokes: [...(l.strokes || []), stroke] }
          : l
      );
    } else {
      const newLayer = {
        id: `paint-${Date.now()}`,
        type: "paint",
        name: "Paint Layer",
        visible: true,
        opacity: 1.0,
        order: layers.length,
        strokes: [stroke],
      };
      newLayers = [...layers, newLayer];
      newSelectedId = newLayer.id;
    }

    currentValue = { ...currentValue, layers: newLayers, selected_layer_id: newSelectedId };
    emitChange();
    renderCanvas();
    rebuildLayersPanel();
  }

  function commitArrow(arr) {
    const layers = currentValue.layers || [];
    const newLayer = {
      id: `arrow-${Date.now()}`,
      type: "arrow",
      name: "Arrow",
      visible: true,
      opacity: 1.0,
      order: layers.length,
      x1: arr.x1, y1: arr.y1,
      x2: arr.x2, y2: arr.y2,
      color: currentValue.tool_settings?.arrow?.color || "#ff0000",
      width: currentValue.tool_settings?.arrow?.width || 3,
    };
    currentValue = {
      ...currentValue,
      layers: [...layers, newLayer],
      selected_layer_id: newLayer.id,
    };
    emitChange();
    renderCanvas();
    rebuildLayersPanel();
  }

  function placeTextInput(cx, cy) {
    const fontSize = currentValue.tool_settings?.text?.font_size || 24;
    const fontFamily = currentValue.tool_settings?.text?.font || "Arial";
    const textColor = currentValue.tool_settings?.text?.color || "#000000";

    const input = document.createElement("input");
    input.type = "text";
    input.className = "ai-text-input nodrag nowheel";
    input.placeholder = "Type text…";
    input.style.left = cx * displayScale + "px";
    input.style.top = (cy - fontSize) * displayScale + "px";
    input.style.fontSize = fontSize * displayScale + "px";
    input.style.fontFamily = fontFamily;
    input.style.color = textColor;
    canvasWrapper.appendChild(input);
    input.focus();

    function commit() {
      const text = input.value.trim();
      input.remove();
      if (!text) return;
      const layers = currentValue.layers || [];
      const newLayer = {
        id: `text-${Date.now()}`,
        type: "text",
        name: text.substring(0, 20),
        visible: true,
        opacity: 1.0,
        order: layers.length,
        text,
        x: cx,
        y: cy,
        color: textColor,
        font_size: fontSize,
        font: fontFamily,
      };
      currentValue = {
        ...currentValue,
        layers: [...layers, newLayer],
        selected_layer_id: newLayer.id,
      };
      emitChange();
      renderCanvas();
      rebuildLayersPanel();
    }

    input.addEventListener("keydown", (ke) => {
      ke.stopPropagation();
      if (ke.key === "Enter") commit();
      if (ke.key === "Escape") input.remove();
    });
    let blurCommitted = false;
    input.addEventListener("blur", () => { if (!blurCommitted) { blurCommitted = true; commit(); } });
  }

  // ── tool switching ────────────────────────────────────────────────────────

  function setActiveTool(toolId) {
    activeTool = toolId;
    currentValue = { ...currentValue, active_tool: toolId };
    Object.keys(toolButtons).forEach((id) => {
      toolButtons[id].className = "ai-tool-btn" + (id === toolId ? " active" : "");
    });
    canvas.className = "ai-canvas" + (toolId === "select" ? " select-cursor" : "");
    updateToolSettingsPanel();
    emitChange();
  }

  // ── inline side panel (tool settings section + layers section) ────────────

  function buildSidePanelEl() {
    const panel = document.createElement("div");
    // flex-direction:column so tsHeader/tsBody/divider/lHeader/lBody stack vertically
    panel.style.cssText = "flex-direction:column;flex:0 1 200px;min-width:140px;box-sizing:border-box;background:var(--card);border-left:1px solid var(--border);overflow-y:auto;max-height:600px;";
    panel.addEventListener("pointerdown", (e) => e.stopPropagation());
    panel.addEventListener("wheel", (e) => e.stopPropagation());

    // Tool settings section
    const tsHeader = document.createElement("div");
    tsHeader.className = "ai-panel-header";

    const tsBody = document.createElement("div");
    tsBody.className = "ai-panel-body";
    tsBody.style.paddingBottom = "4px";

    // Divider
    const divider = document.createElement("div");
    divider.style.cssText = "border-top:1px solid var(--border);";

    // Layers section header
    const lHeader = document.createElement("div");
    lHeader.className = "ai-panel-header";
    const lTitle = document.createElement("span");
    lTitle.textContent = "Layers";
    const addBtn = document.createElement("button");
    addBtn.title = "Add new paint layer";
    addBtn.style.cssText = "display:flex;align-items:center;gap:3px;font-size:11px;color:var(--muted-foreground);cursor:pointer;background:none;border:none;padding:0;border-radius:3px;";
    addBtn.appendChild(icon("plus", 11));
    addBtn.appendChild(Object.assign(document.createElement("span"), { textContent: "Paint" }));
    addBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); addPaintLayer(); });
    lHeader.appendChild(lTitle);
    lHeader.appendChild(addBtn);

    // Layers list body
    const lBody = document.createElement("div");
    lBody.className = "ai-layers-list";

    panel.appendChild(tsHeader);
    panel.appendChild(tsBody);
    panel.appendChild(divider);
    panel.appendChild(lHeader);
    panel.appendChild(lBody);

    panel._tsHeader = tsHeader;
    panel._tsBody = tsBody;
    panel._lBody = lBody;
    return panel;
  }

  function rebuildLayersPanel() {
    const body = sidePanel._lBody;
    body.innerHTML = "";

    const sorted = [...(currentValue.layers || [])].sort(
      (a, b) => (b.order ?? 0) - (a.order ?? 0)
    );

    sorted.forEach((layer) => {
      const isSelected = layer.id === currentValue.selected_layer_id;
      const row = document.createElement("div");
      row.className = "ai-layer-row" + (isSelected ? " selected" : "");

      const rowTop = document.createElement("div");
      rowTop.className = "ai-layer-top";

      const visBtn = document.createElement("button");
      visBtn.className = "ai-icon-btn";
      visBtn.title = "Toggle visibility";
      visBtn.appendChild(icon(layer.visible !== false ? "eye" : "eyeOff", 12));
      visBtn.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        toggleLayerVisibility(layer.id);
      });

      const typeIconEl = document.createElement("span");
      typeIconEl.className = "ai-layer-type-icon";
      const typeIconName = { image: "image", paint: "paint", text: "text", arrow: "arrow" }[layer.type] || "image";
      typeIconEl.appendChild(icon(typeIconName, 11));

      const name = document.createElement("span");
      name.className = "ai-layer-name";
      name.textContent = layer.name || "Layer";

      rowTop.appendChild(visBtn);
      rowTop.appendChild(typeIconEl);
      rowTop.appendChild(name);

      if (layer.type !== "image") {
        const delBtn = document.createElement("button");
        delBtn.className = "ai-icon-btn";
        delBtn.title = "Remove layer";
        delBtn.appendChild(icon("trash", 12));
        delBtn.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          deleteLayer(layer.id);
        });
        rowTop.appendChild(delBtn);
      }

      row.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        currentValue = { ...currentValue, selected_layer_id: layer.id };
        rebuildLayersPanel();
        updateToolSettingsPanel();
        renderCanvas();
      });

      row.appendChild(rowTop);

      const opRow = document.createElement("div");
      opRow.className = "ai-opacity-row";
      const opLabel = document.createElement("span");
      opLabel.className = "ai-opacity-label";
      opLabel.textContent = "Opacity";
      const opSlider = document.createElement("input");
      opSlider.type = "range";
      opSlider.className = "ai-opacity-slider";
      opSlider.min = 0; opSlider.max = 100; opSlider.step = 1;
      opSlider.value = Math.round((layer.opacity ?? 1) * 100);
      opSlider.addEventListener("pointerdown", (e) => e.stopPropagation());
      opSlider.addEventListener("input", (e) => {
        const layers = (currentValue.layers || []).map((l) =>
          l.id === layer.id ? { ...l, opacity: e.target.value / 100 } : l
        );
        currentValue = { ...currentValue, layers };
        renderCanvas();
      });
      opSlider.addEventListener("change", () => emitChange());
      opRow.appendChild(opLabel);
      opRow.appendChild(opSlider);
      row.appendChild(opRow);

      body.appendChild(row);
    });
  }

  function updateToolSettingsPanel() {
    const header = sidePanel._tsHeader;
    const body = sidePanel._tsBody;
    const toolLabels = { select: "Select", paint: "Paint", text: "Text", arrow: "Arrow" };
    header.textContent = toolLabels[activeTool] || "Tool Settings";
    body.innerHTML = "";

    const ts = currentValue.tool_settings || {};

    if (activeTool === "select") {
      const sel = (currentValue.layers || []).find((l) => l.id === currentValue.selected_layer_id);
      if (sel) {
        const info = document.createElement("div");
        info.className = "ai-select-info";
        const rows = [
          ["X", Math.round(sel.x ?? 0)],
          ["Y", Math.round(sel.y ?? 0)],
        ];
        if (sel.type === "image") {
          rows.push(["Scale X", ((sel.scaleX ?? 1)).toFixed(2)]);
          rows.push(["Scale Y", ((sel.scaleY ?? 1)).toFixed(2)]);
          rows.push(["Rotation", `${Math.round(sel.rotation ?? 0)}°`]);
        }
        rows.forEach(([label, val]) => {
          const r = document.createElement("div");
          r.className = "ai-select-info-row";
          r.innerHTML = `<span>${label}</span><span style="color:var(--foreground)">${val}</span>`;
          info.appendChild(r);
        });
        body.appendChild(info);
      } else {
        const hint = document.createElement("div");
        hint.className = "ai-select-hint";
        hint.textContent = "Click a layer to select it.";
        body.appendChild(hint);
      }

    } else if (activeTool === "paint") {
      body.appendChild(makeColorRow("Color", ts.paint?.color || "#ff0000", (v) => {
        updateToolSetting("paint", "color", v);
      }));
      body.appendChild(makeSizeRow("Size", ts.paint?.size || 10, 1, 100, (v) => {
        updateToolSetting("paint", "size", v);
      }));

    } else if (activeTool === "text") {
      body.appendChild(makeColorRow("Color", ts.text?.color || "#000000", (v) => {
        updateToolSetting("text", "color", v);
      }));
      body.appendChild(makeSizeRow("Font Size", ts.text?.font_size || 24, 8, 200, (v) => {
        updateToolSetting("text", "font_size", v);
      }));

    } else if (activeTool === "arrow") {
      body.appendChild(makeColorRow("Color", ts.arrow?.color || "#ff0000", (v) => {
        updateToolSetting("arrow", "color", v);
      }));
      body.appendChild(makeSizeRow("Width", ts.arrow?.width || 3, 1, 20, (v) => {
        updateToolSetting("arrow", "width", v);
      }));
    }
  }

  // ── panel control helpers ─────────────────────────────────────────────────

  function makeColorRow(label, value, onInput) {
    const row = document.createElement("div");
    row.className = "ai-setting-row";
    const lbl = document.createElement("span");
    lbl.className = "ai-setting-label";
    lbl.textContent = label;
    const picker = document.createElement("input");
    picker.type = "color";
    picker.className = "ai-color-input";
    picker.value = value;
    picker.addEventListener("pointerdown", (e) => e.stopPropagation());
    picker.addEventListener("input", (e) => onInput(e.target.value));
    picker.addEventListener("change", () => emitChange());
    row.appendChild(lbl);
    row.appendChild(picker);
    return row;
  }

  function makeSizeRow(label, value, min, max, onInput) {
    const col = document.createElement("div");
    col.className = "ai-size-col";
    const hdr = document.createElement("div");
    hdr.className = "ai-size-header";
    const lbl = document.createElement("span");
    lbl.className = "ai-setting-label";
    lbl.textContent = label;
    const val = document.createElement("span");
    val.className = "ai-size-value";
    val.textContent = value;
    hdr.appendChild(lbl);
    hdr.appendChild(val);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.className = "ai-size-slider";
    slider.min = min; slider.max = max; slider.step = 1; slider.value = value;
    slider.addEventListener("pointerdown", (e) => e.stopPropagation());
    slider.addEventListener("input", (e) => {
      val.textContent = e.target.value;
      onInput(parseInt(e.target.value, 10));
    });
    slider.addEventListener("change", () => emitChange());
    col.appendChild(hdr);
    col.appendChild(slider);
    return col;
  }

  function updateToolSetting(tool, key, value) {
    currentValue = {
      ...currentValue,
      tool_settings: {
        ...(currentValue.tool_settings || {}),
        [tool]: {
          ...(currentValue.tool_settings?.[tool] || {}),
          [key]: value,
        },
      },
    };
  }

  // ── layer operations ──────────────────────────────────────────────────────

  function addPaintLayer() {
    const layers = currentValue.layers || [];
    const newLayer = {
      id: `paint-${Date.now()}`,
      type: "paint",
      name: "Paint Layer",
      visible: true,
      opacity: 1.0,
      order: layers.length,
      strokes: [],
    };
    currentValue = {
      ...currentValue,
      layers: [...layers, newLayer],
      selected_layer_id: newLayer.id,
    };
    emitChange();
    rebuildLayersPanel();
    renderCanvas();
  }

  function toggleLayerVisibility(layerId) {
    const layers = (currentValue.layers || []).map((l) =>
      l.id === layerId ? { ...l, visible: l.visible === false } : l
    );
    currentValue = { ...currentValue, layers };
    emitChange();
    renderCanvas();
    rebuildLayersPanel();
  }

  function deleteLayer(layerId) {
    const layers = (currentValue.layers || []).filter((l) => l.id !== layerId);
    const selected =
      currentValue.selected_layer_id === layerId ? null : currentValue.selected_layer_id;
    currentValue = { ...currentValue, layers, selected_layer_id: selected };
    emitChange();
    renderCanvas();
    rebuildLayersPanel();
  }

  // ── emit change ───────────────────────────────────────────────────────────

  function emitChange() {
    onChangeFn?.(currentValue);
  }

  // ── update (framework callback on server-side value changes) ──────────────

  function handleUpdate(newProps) {
    onChangeFn = newProps.onChange;
    const newVal = newProps.value;
    if (!newVal) return;

    const prevW = currentValue.canvas_width;
    const prevH = currentValue.canvas_height;
    currentValue = normalizeValue(newVal);
    activeTool = currentValue.active_tool || "select";

    layersToggle.className = "ai-layers-btn" + (currentValue.layers_panel_open ? " active" : "");
    sidePanel.style.display = currentValue.layers_panel_open ? "flex" : "none";

    Object.keys(toolButtons).forEach((id) => {
      toolButtons[id].className = "ai-tool-btn" + (id === activeTool ? " active" : "");
    });
    canvas.className = "ai-canvas" + (activeTool === "select" ? " select-cursor" : "");

    if (currentValue.canvas_width !== prevW || currentValue.canvas_height !== prevH) {
      applyCanvasScale();
    } else {
      renderCanvas();
    }

    rebuildLayersPanel();
    updateToolSettingsPanel();
  }

  // ── cleanup ───────────────────────────────────────────────────────────────

  function cleanup() {
    resizeObserver.disconnect();
    wrapper.remove();
    delete container._instance;
  }

  // ── init ──────────────────────────────────────────────────────────────────

  container._instance = { handleUpdate, cleanup, wrapper };
  applyCanvasScale();
  setActiveTool(activeTool);
  rebuildLayersPanel();
  updateToolSettingsPanel();

  return { cleanup, update: handleUpdate };
}
