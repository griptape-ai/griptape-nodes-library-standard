import {
  DEFAULT_PAINT_SIZE, MIN_PAINT_SIZE, MAX_PAINT_SIZE,
  DEFAULT_TEXT_SIZE,  MIN_TEXT_SIZE,  MAX_TEXT_SIZE,
  DEFAULT_ARROW_WIDTH, MIN_ARROW_WIDTH, MAX_ARROW_WIDTH,
  DEFAULT_SHAPE_WIDTH, MIN_SHAPE_WIDTH, MAX_SHAPE_WIDTH,
} from './_styles.js';

// Keyboard shortcut handling for the annotation widget.
//
// setupHotkeys(getState, actions) registers all document-level key listeners
// and returns a cleanup() function to remove them.
//
// getState() must return:
//   { mouseIsOver, textEditId, activeTool, currentValue, toolSettings }
//
// actions must contain:
//   setTool(id), resetView(), rebuildSettings(), emit(), renderCanvas(),
//   deleteAnnotations(ids), setCurrentValue(v), setTxFrame(f),
//   applySingleUpdate(id, fn), effectiveAnnotations(), onPointerDown(e),
//   commitTextEdit(), wrapper (DOM element for shift-click containment check

const TOOL_HOTKEYS = { v: "select", h: "hand", z: "zoom", d: "paint", t: "text", l: "arrow", r: "rect", o: "ellipse" };

export function setupHotkeys(getState, actions) {
  const {
    setTool, resetView, rebuildSettings, emit, renderCanvas,
    deleteAnnotations, setCurrentValue, setTxFrame,
    applySingleUpdate, effectiveAnnotations, onPointerDown, wrapper,
  } = actions;

  function _onAltDown(e) {
    if (!getState().mouseIsOver) return;
    if (e.key === "Alt") actions.onAltDown();
  }

  function _onAltUp(e) {
    if (e.key === "Alt") actions.onAltUp();
  }

  function _shiftInterceptor(e) {
    if (!e.shiftKey || !wrapper.contains(e.target)) return;
    e.stopPropagation();
    e.stopImmediatePropagation();
    if (e.type === "pointerdown" && e.button === 0) onPointerDown(e);
  }

  function _deleteInterceptor(e) {
    const { mouseIsOver, textEditId, activeTool, currentValue } = getState();
    if (!mouseIsOver) return;
    if (e.key !== "Delete" && e.key !== "Backspace") return;
    // Never intercept while focus is in a real text input (typing in a form field, not our canvas)
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    // Always block the key from reaching the node framework when mouse is over the canvas,
    // even if there's nothing to delete — prevents accidental node deletion.
    e.stopPropagation();
    e.preventDefault();
    if (textEditId) return;
    if (!(currentValue.selected_ids || []).length) return;
    if (activeTool !== "select" && activeTool !== "arrow" && activeTool !== "rect" && activeTool !== "ellipse") return;
    deleteAnnotations(currentValue.selected_ids);
    setCurrentValue({ ...getState().currentValue, selected_ids: [] });
    setTxFrame(null);
    emit();
    rebuildSettings();
    renderCanvas();
  }

  function _sizeInterceptor(e) {
    const { mouseIsOver, textEditId, activeTool, currentValue, toolSettings } = getState();
    if (!mouseIsOver) return;
    if (e.key !== "[" && e.key !== "]") return;
    if (textEditId) return;
    const t = e.target;
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
    if (selIds.length === 1 && (activeTool === "select" || activeTool === "arrow")) {
      const selAnn = effectiveAnnotations().find((a) => a.id === selIds[0]);
      if (selAnn) {
        if (selAnn.type === "paint") {
          const base = selAnn.strokes?.[0]?.size ?? DEFAULT_PAINT_SIZE;
          const cur = Math.round(base * (selAnn.sizeScale ?? 1));
          const next = Math.max(MIN_PAINT_SIZE, Math.min(MAX_PAINT_SIZE, cur + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, sizeScale: next / base }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "text") {
          const next = Math.max(MIN_TEXT_SIZE, Math.min(MAX_TEXT_SIZE, (selAnn.font_size ?? DEFAULT_TEXT_SIZE) + delta * 2));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, font_size: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "arrow") {
          const next = Math.max(MIN_ARROW_WIDTH, Math.min(MAX_ARROW_WIDTH, (selAnn.width ?? DEFAULT_ARROW_WIDTH) + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, width: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "rect" || selAnn.type === "ellipse") {
          const next = Math.max(MIN_SHAPE_WIDTH, Math.min(MAX_SHAPE_WIDTH, (selAnn.width ?? DEFAULT_SHAPE_WIDTH) + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, width: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
      }
    }
    // Adjust active tool settings
    if (activeTool === "paint") {
      toolSettings.paint.size = Math.max(MIN_PAINT_SIZE, Math.min(MAX_PAINT_SIZE, (toolSettings.paint.size ?? DEFAULT_PAINT_SIZE) + delta));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "arrow") {
      toolSettings.arrow.width = Math.max(MIN_ARROW_WIDTH, Math.min(MAX_ARROW_WIDTH, (toolSettings.arrow.width ?? DEFAULT_ARROW_WIDTH) + delta));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "text") {
      toolSettings.text.font_size = Math.max(MIN_TEXT_SIZE, Math.min(MAX_TEXT_SIZE, (toolSettings.text.font_size ?? DEFAULT_TEXT_SIZE) + delta * 2));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "rect" || activeTool === "ellipse") {
      toolSettings[activeTool].width = Math.max(MIN_SHAPE_WIDTH, Math.min(MAX_SHAPE_WIDTH, (toolSettings[activeTool].width ?? DEFAULT_SHAPE_WIDTH) + delta));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    }
  }

  function _toolHotkeyInterceptor(e) {
    const { mouseIsOver, textEditId } = getState();
    if (!mouseIsOver) return;
    if (textEditId) return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const key = e.key.toLowerCase();
    if (key === "f") { e.stopPropagation(); resetView(); return; }
    const tool = TOOL_HOTKEYS[key];
    if (!tool) return;
    e.stopPropagation();
    setTool(tool);
  }

  document.addEventListener("keydown", _onAltDown);
  document.addEventListener("keyup",   _onAltUp);
  document.addEventListener("pointerdown", _shiftInterceptor, { capture: true });
  document.addEventListener("mousedown",   _shiftInterceptor, { capture: true });
  document.addEventListener("click",       _shiftInterceptor, { capture: true });
  document.addEventListener("keydown", _deleteInterceptor,    { capture: true });
  document.addEventListener("keydown", _sizeInterceptor,      { capture: true });
  document.addEventListener("keydown", _toolHotkeyInterceptor, { capture: true });

  return function cleanup() {
    document.removeEventListener("keydown", _onAltDown);
    document.removeEventListener("keyup",   _onAltUp);
    document.removeEventListener("pointerdown", _shiftInterceptor, { capture: true });
    document.removeEventListener("mousedown",   _shiftInterceptor, { capture: true });
    document.removeEventListener("click",       _shiftInterceptor, { capture: true });
    document.removeEventListener("keydown", _deleteInterceptor,    { capture: true });
    document.removeEventListener("keydown", _sizeInterceptor,      { capture: true });
    document.removeEventListener("keydown", _toolHotkeyInterceptor, { capture: true });
  };
}
