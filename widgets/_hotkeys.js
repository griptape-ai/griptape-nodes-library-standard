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
    if (!(currentValue.selected_ids || []).length) return;
    if (textEditId) return;
    if (activeTool !== "select" && activeTool !== "arrow" && activeTool !== "rect" && activeTool !== "ellipse") return;
    const t = e.target;
    if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
    e.stopPropagation();
    e.preventDefault();
    deleteAnnotations(currentValue.selected_ids);
    setCurrentValue({ ...currentValue, selected_ids: [] });
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
          const base = selAnn.strokes?.[0]?.size ?? 8;
          const cur = Math.round(base * (selAnn.sizeScale ?? 1));
          const next = Math.max(1, Math.min(80, cur + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, sizeScale: next / base }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "text") {
          const next = Math.max(8, Math.min(120, (selAnn.font_size ?? 48) + delta * 2));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, font_size: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "arrow") {
          const next = Math.max(1, Math.min(20, (selAnn.width ?? 3) + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, width: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
        if (selAnn.type === "rect" || selAnn.type === "ellipse") {
          const next = Math.max(1, Math.min(20, (selAnn.width ?? 2) + delta));
          applySingleUpdate(selAnn.id, (a) => ({ ...a, width: next }));
          emit(); rebuildSettings(); renderCanvas(); return;
        }
      }
    }
    // Adjust active tool settings
    if (activeTool === "paint") {
      toolSettings.paint.size = Math.max(1, Math.min(80, (toolSettings.paint.size ?? 8) + delta));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "arrow") {
      toolSettings.arrow.width = Math.max(1, Math.min(20, (toolSettings.arrow.width ?? 3) + delta));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "text") {
      toolSettings.text.font_size = Math.max(8, Math.min(120, (toolSettings.text.font_size ?? 48) + delta * 2));
      setCurrentValue({ ...currentValue, tool_settings: { ...toolSettings } });
      rebuildSettings(); emit();
    } else if (activeTool === "rect" || activeTool === "ellipse") {
      toolSettings[activeTool].width = Math.max(1, Math.min(20, (toolSettings[activeTool].width ?? 2) + delta));
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
