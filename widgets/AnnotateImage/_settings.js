// Toolbar settings panel builders for the AnnotateImage widget.
// createSettings(settingsArea, deps) → { buildToolSettings, buildAnnotationSettings, buildMultiSettings }

import { mkIcon } from './_icons.js';
import {
  DEFAULT_COLOR,
  DEFAULT_PAINT_SIZE, MIN_PAINT_SIZE, MAX_PAINT_SIZE,
  DEFAULT_TEXT_SIZE,  MIN_TEXT_SIZE,  MAX_TEXT_SIZE,
  DEFAULT_ARROW_WIDTH, MIN_ARROW_WIDTH, MAX_ARROW_WIDTH,
  DEFAULT_SHAPE_WIDTH, MIN_SHAPE_WIDTH, MAX_SHAPE_WIDTH,
} from './_styles.js';

export function createSettings(settingsArea, {
  addTooltip,
  getState,         // () => { activeTool, toolSettings, currentValue, textEditId, textInput, displayScale, viewScale }
  setCurrentValue,  // (v) => void
  applySingleUpdate,
  applyAnnotationMap,
  effectiveAnnotations,
  autoResizeTextarea,
  renderCanvas,
  emit,
  rebuild,          // rebuildSettings()
}) {
  let colorPickerEl = null;

  // Appends a color swatch + hidden <input type=color> to settingsArea.
  // onChange(color, doEmit) — doEmit=false during drag, true on commit.
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

  // Formats a number for display in slider labels.
  function _fmtNum(v) {
    const n = Number(v);
    if (!isFinite(n)) return "0";
    if (Number.isInteger(n)) return String(n);
    const r = Math.round(n * 100) / 100;
    return Number.isInteger(r) ? String(r) : r.toFixed(2).replace(/0+$/, "");
  }

  // Appends a labeled range slider + value readout to settingsArea.
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

  // Appends a color swatch + opacity slider + clear button, encoding result as #rrggbbaa.
  // colorTip / clearTip control the tooltip labels so it works for both fill and background.
  function _buildAlphaColorSwatch(color, colorTip, clearTip, onChange) {
    let hexRgb = "#ffffff";
    let alpha = 100;
    if (color && color.length === 9) {
      hexRgb = color.slice(0, 7);
      alpha = Math.round(parseInt(color.slice(7, 9), 16) / 255 * 100);
    } else if (color && color.length === 7) {
      hexRgb = color;
    }
    const buildHexa = () => hexRgb + Math.round(alpha / 100 * 255).toString(16).padStart(2, "0");
    const wrap = document.createElement("div");
    wrap.style.cssText = "position:relative;display:flex;align-items:center;gap:2px;";
    const swatch = document.createElement("div");
    swatch.className = "ais-color-btn";
    addTooltip(swatch, colorTip);
    swatch.style.background = color || "repeating-conic-gradient(#888 0% 25%,#333 0% 50%) 0 0/8px 8px";
    const pickerInput = document.createElement("input");
    pickerInput.type = "color"; pickerInput.value = hexRgb; pickerInput.className = "ais-color-input";
    pickerInput.addEventListener("input", () => { hexRgb = pickerInput.value; swatch.style.background = buildHexa(); onChange(buildHexa(), false); });
    pickerInput.addEventListener("change", () => onChange(buildHexa(), true));
    swatch.addEventListener("click", () => pickerInput.click());
    const opacitySlider = document.createElement("input");
    opacitySlider.type = "range"; opacitySlider.className = "ais-range";
    opacitySlider.min = 0; opacitySlider.max = 100; opacitySlider.value = alpha;
    opacitySlider.style.width = "50px";
    opacitySlider.addEventListener("input", () => { alpha = Number(opacitySlider.value); swatch.style.background = buildHexa(); onChange(buildHexa(), false); });
    opacitySlider.addEventListener("change", () => onChange(buildHexa(), true));
    const clearBtn = document.createElement("button");
    clearBtn.className = "ais-tool-btn"; addTooltip(clearBtn, clearTip);
    clearBtn.style.cssText = "width:16px;height:16px;font-size:11px;padding:0;";
    clearBtn.textContent = "✕";
    clearBtn.addEventListener("pointerdown", (e) => { e.stopPropagation(); swatch.style.background = "repeating-conic-gradient(#888 0% 25%,#333 0% 50%) 0 0/8px 8px"; onChange("", true); });
    wrap.appendChild(swatch); wrap.appendChild(pickerInput); wrap.appendChild(opacitySlider); wrap.appendChild(clearBtn);
    settingsArea.appendChild(wrap);
  }

  // Appends left/center/right alignment toggle buttons to settingsArea (text only).
  function _buildTextAlignButtons(currentAlign, onChange) {
    const ALIGNS = [
      { value: "left",   icon: "align-left",   title: "Align left"   },
      { value: "center", icon: "align-center",  title: "Align center" },
      { value: "right",  icon: "align-right",   title: "Align right"  },
    ];
    const row = document.createElement("div");
    row.style.cssText = "display:flex;align-items:center;gap:2px;";
    for (const a of ALIGNS) {
      const btn = document.createElement("button");
      btn.className = "ais-toggle-btn" + (currentAlign === a.value ? " active" : "");
      addTooltip(btn, a.title);
      btn.style.cssText = "width:26px;height:26px;";
      btn.appendChild(mkIcon(a.icon, 14));
      btn.addEventListener("pointerdown", (e) => {
        e.stopPropagation();
        row.querySelectorAll(".ais-toggle-btn").forEach((b) => b.classList.remove("active"));
        btn.classList.add("active");
        onChange(a.value);
      });
      row.appendChild(btn);
    }
    settingsArea.appendChild(row);
  }

  // Appends arrow-specific toggles (start/end arrowhead, bezier, taper) to settingsArea.
  // source is either a tool-settings object or a single arrow annotation.
  function _buildArrowToggles(source, onToggle) {
    const makeToggleBtn = (content, title, active, onClick) => {
      const btn = document.createElement("button");
      btn.className = "ais-toggle-btn" + (active ? " active" : "");
      addTooltip(btn, title);
      btn.style.cssText = "font-size:14px;font-weight:bold;width:26px;height:26px;line-height:1;";
      if (typeof content === "string") { btn.textContent = content; } else { btn.appendChild(content); }
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
    row.appendChild(makeToggleBtn(mkIcon("bezier", 14), "Bezier curve", source.is_bezier ?? false, () => {
      onToggle({ is_bezier: !(source.is_bezier ?? false) });
    }));
    // Taper: variable-width stroke, thin at tail and full-width at arrowhead.
    row.appendChild(makeToggleBtn(mkIcon("taper", 14), "Taper stroke width", source.taper ?? false, () => {
      onToggle({ taper: !(source.taper ?? false) });
    }));
    settingsArea.appendChild(row);
  }

  // Builds settings for the active drawing tool (no annotation selected): size, color, fill, arrow toggles.
  function buildToolSettings() {
    const { activeTool, toolSettings } = getState();
    const ts = toolSettings[activeTool] || {};
    if (activeTool === "arrow") {
      _buildArrowToggles(toolSettings.arrow, (changes) => {
        const s = getState();
        s.toolSettings.arrow = { ...s.toolSettings.arrow, ...changes };
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        rebuild();
        renderCanvas();
        emit();
      });
    }
    const isShape = activeTool === "rect" || activeTool === "ellipse";
    if (activeTool === "text") {
      _buildTextAlignButtons(ts.text_align || "left", (align) => {
        const s = getState();
        s.toolSettings.text.text_align = align;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        if (s.textEditId) {
          s.textInput.style.textAlign = align;
          setCurrentValue({
            ...getState().currentValue,
            annotations: getState().currentValue.annotations.map((a) =>
              a.id === s.textEditId ? { ...a, text_align: align } : a
            ),
          });
        }
        renderCanvas();
        emit();
      });
    }
    const sizeKey = activeTool === "text" ? "font_size"
      : (activeTool === "arrow" || isShape) ? "width"
      : "size";
    const sizeVal = ts[sizeKey] ?? (activeTool === "text" ? DEFAULT_TEXT_SIZE : (activeTool === "arrow") ? DEFAULT_ARROW_WIDTH : isShape ? DEFAULT_SHAPE_WIDTH : DEFAULT_PAINT_SIZE);
    const sizeMin = activeTool === "text" ? MIN_TEXT_SIZE : (activeTool === "arrow" || isShape) ? MIN_ARROW_WIDTH : MIN_PAINT_SIZE;
    const sizeMax = activeTool === "text" ? MAX_TEXT_SIZE : (activeTool === "arrow" || isShape) ? MAX_ARROW_WIDTH : MAX_PAINT_SIZE;
    const sizeLbl = (activeTool === "arrow" || isShape) ? "Width" : "Size";
    _buildSizeSlider(sizeLbl, sizeMin, sizeMax, sizeVal, (sz, doEmit) => {
      const s = getState();
      s.toolSettings[activeTool][sizeKey] = sz;
      setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
      // While editing text, apply font size change live to the textarea and annotation
      if (activeTool === "text" && s.textEditId) {
        s.textInput.style.fontSize = sz * s.displayScale * s.viewScale + "px";
        autoResizeTextarea();
        setCurrentValue({
          ...getState().currentValue,
          annotations: getState().currentValue.annotations.map((a) =>
            a.id === s.textEditId ? { ...a, font_size: sz } : a
          ),
        });
      }
      renderCanvas();
      if (doEmit) emit();
    });
    const color = ts.color || DEFAULT_COLOR;
    _buildColorSwatch(color, (col, doEmit) => {
      const s = getState();
      s.toolSettings[activeTool].color = col;
      setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
      if (activeTool === "text" && s.textEditId) {
        s.textInput.style.color = col;
        setCurrentValue({
          ...getState().currentValue,
          annotations: getState().currentValue.annotations.map((a) =>
            a.id === s.textEditId ? { ...a, color: col } : a
          ),
        });
        renderCanvas();
      }
      if (doEmit) emit();
    });
    if (isShape) {
      _buildAlphaColorSwatch(ts.fill_color || "", "Fill color", "No fill", (col, doEmit) => {
        const s = getState();
        s.toolSettings[activeTool].fill_color = col;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        renderCanvas();
        if (doEmit) emit();
      });
    }
    if (activeTool === "text") {
      _buildAlphaColorSwatch(ts.bg_color || "", "Background color", "No background", (col, doEmit) => {
        const s = getState();
        s.toolSettings.text.bg_color = col;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        if (s.textEditId) {
          setCurrentValue({
            ...getState().currentValue,
            annotations: getState().currentValue.annotations.map((a) =>
              a.id === s.textEditId ? { ...a, bg_color: col } : a
            ),
          });
          if (s.textInput) s.textInput.style.background = col || "transparent";
        }
        renderCanvas();
        if (doEmit) emit();
      });
    }
  }

  // Builds settings for a single selected annotation (size/width, color, fill, arrow toggles).
  // Changes are written back to the annotation and synced to tool_settings so the next
  // new annotation inherits the same style.
  function buildAnnotationSettings(ann) {
    if (ann.type === "arrow") {
      _buildArrowToggles(ann, (changes) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, ...changes }));
        // Sync arrow-style toggles to tool settings so next arrow uses same style
        const s = getState();
        s.toolSettings.arrow = { ...s.toolSettings.arrow, ...changes };
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        renderCanvas();
        rebuild();
        emit();
      });
    }

    let color;
    if (ann.type === "paint") {
      color = (ann.strokes && ann.strokes[0]) ? ann.strokes[0].color : DEFAULT_COLOR;
    } else {
      color = ann.color || DEFAULT_COLOR;
    }

    if (ann.type === "paint") {
      const baseSize = (ann.strokes && ann.strokes[0]) ? (ann.strokes[0].size ?? DEFAULT_PAINT_SIZE) : DEFAULT_PAINT_SIZE;
      const currentSize = Math.max(MIN_PAINT_SIZE, Math.round(baseSize * (ann.sizeScale ?? 1)));
      _buildSizeSlider("Size", MIN_PAINT_SIZE, MAX_PAINT_SIZE, currentSize, (sz, doEmit) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, sizeScale: sz / baseSize }));
        const s = getState();
        s.toolSettings.paint.size = sz;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        renderCanvas();
        if (doEmit) emit();
      });
    }

    const isShape = ann.type === "rect" || ann.type === "ellipse";
    if (ann.type === "text") {
      _buildTextAlignButtons(ann.text_align || "left", (align) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, text_align: align }));
        const s = getState();
        s.toolSettings.text.text_align = align;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        if (s.textInput && s.textEditId === ann.id) s.textInput.style.textAlign = align;
        renderCanvas();
        emit();
      });
    }
    const sizeKey = ann.type === "text" ? "font_size" : (ann.type === "arrow" || isShape) ? "width" : null;
    if (sizeKey) {
      const sizeVal = ann[sizeKey] ?? (ann.type === "text" ? DEFAULT_TEXT_SIZE : isShape ? DEFAULT_SHAPE_WIDTH : DEFAULT_ARROW_WIDTH);
      const sizeMin = ann.type === "text" ? MIN_TEXT_SIZE : MIN_ARROW_WIDTH;
      const sizeMax = ann.type === "text" ? MAX_TEXT_SIZE : MAX_ARROW_WIDTH;
      const sizeLbl = ann.type === "text" ? "Size" : "Width";
      _buildSizeSlider(sizeLbl, sizeMin, sizeMax, sizeVal, (sz, doEmit) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, [sizeKey]: sz }));
        const s = getState();
        if (ann.type === "arrow") { s.toolSettings.arrow.width = sz; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
        if (ann.type === "text")  { s.toolSettings.text.font_size = sz; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
        if (isShape) { s.toolSettings[ann.type].width = sz; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
        const s2 = getState();
        if (s2.textInput && s2.textEditId === ann.id && sizeKey === "font_size") {
          s2.textInput.style.fontSize = sz * s2.displayScale * s2.viewScale + "px";
          autoResizeTextarea();
        }
        renderCanvas();
        if (doEmit) emit();
      });
    }

    _buildColorSwatch(color, (col, doEmit) => {
      applySingleUpdate(ann.id, (a) => {
        if (a.type === "paint") return { ...a, strokes: (a.strokes || []).map((s) => ({ ...s, color: col })) };
        return { ...a, color: col };
      });
      const s = getState();
      if (ann.type === "arrow") { s.toolSettings.arrow.color = col; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
      if (ann.type === "text")  { s.toolSettings.text.color = col;  setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
      if (ann.type === "paint") { s.toolSettings.paint.color = col; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
      if (isShape) { s.toolSettings[ann.type].color = col; setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } }); }
      const s2 = getState();
      if (s2.textInput && s2.textEditId === ann.id) {
        s2.textInput.style.color = col;
        s2.textInput.style.borderBottomColor = col;
      }
      renderCanvas();
      if (doEmit) emit();
    });

    if (isShape) {
      _buildAlphaColorSwatch(ann.fill_color || "", "Fill color", "No fill", (col, doEmit) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, fill_color: col }));
        const s = getState();
        s.toolSettings[ann.type].fill_color = col;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        renderCanvas();
        if (doEmit) emit();
      });
    }
    if (ann.type === "text") {
      _buildAlphaColorSwatch(ann.bg_color || "", "Background color", "No background", (col, doEmit) => {
        applySingleUpdate(ann.id, (a) => ({ ...a, bg_color: col }));
        const s = getState();
        s.toolSettings.text.bg_color = col;
        setCurrentValue({ ...s.currentValue, tool_settings: { ...s.toolSettings } });
        if (s.textInput && s.textEditId === ann.id) s.textInput.style.background = col || "transparent";
        renderCanvas();
        if (doEmit) emit();
      });
    }
  }

  // Builds settings for a multi-selection: a Scale% slider (line widths only, not text/dimensions)
  // and a color swatch that paints all selected annotations at once.
  function buildMultiSettings(selIds) {
    const anns = effectiveAnnotations().filter((a) => selIds.includes(a.id));
    // Capture original sizes when the panel is built; slider applies ratio to these originals
    const origSizes = {};
    for (const a of anns) {
      if (a.type === "paint") origSizes[a.id] = a.sizeScale ?? 1;
      else if (a.type === "arrow") origSizes[a.id] = a.width ?? 3;
      else if (a.type === "rect" || a.type === "ellipse") origSizes[a.id] = a.width ?? DEFAULT_SHAPE_WIDTH;
    }
    _buildSizeSlider("Scale %", 25, 400, 100, (val, doEmit) => {
      const ratio = val / 100;
      const { annotations, overrides } = applyAnnotationMap(selIds, (a) => {
        if (a.type === "paint") return { ...a, sizeScale: (origSizes[a.id] ?? 1) * ratio };
        if (a.type === "text") return a;
        if (a.type === "arrow") return { ...a, width: Math.max(1, (origSizes[a.id] ?? 3) * ratio) };
        if (a.type === "rect" || a.type === "ellipse") return { ...a, width: Math.max(1, (origSizes[a.id] ?? DEFAULT_SHAPE_WIDTH) * ratio) };
        return a;
      });
      setCurrentValue({ ...getState().currentValue, annotations, overrides });
      renderCanvas();
      if (doEmit) emit();
    });
    let firstColor = DEFAULT_COLOR;
    for (const a of anns) {
      if (a.type === "paint" && a.strokes?.[0]) { firstColor = a.strokes[0].color; break; }
      if (a.color) { firstColor = a.color; break; }
    }
    _buildColorSwatch(firstColor, (col, doEmit) => {
      const { annotations, overrides } = applyAnnotationMap(selIds, (a) => {
        if (a.type === "paint") return { ...a, strokes: (a.strokes || []).map((s) => ({ ...s, color: col })) };
        return { ...a, color: col };
      });
      setCurrentValue({ ...getState().currentValue, annotations, overrides });
      renderCanvas();
      if (doEmit) emit();
    });

    const shapeAnns = anns.filter((a) => a.type === "rect" || a.type === "ellipse");
    if (shapeAnns.length) {
      const firstFill = shapeAnns.find((a) => a.fill_color)?.fill_color || "";
      _buildAlphaColorSwatch(firstFill, "Fill color", "No fill", (col, doEmit) => {
        const shapeIds = shapeAnns.map((a) => a.id);
        const { annotations, overrides } = applyAnnotationMap(shapeIds, (a) => ({ ...a, fill_color: col }));
        setCurrentValue({ ...getState().currentValue, annotations, overrides });
        renderCanvas();
        if (doEmit) emit();
      });
    }
  }

  return { buildToolSettings, buildAnnotationSettings, buildMultiSettings };
}
