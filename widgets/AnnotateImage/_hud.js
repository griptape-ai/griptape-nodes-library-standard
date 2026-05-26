// Context HUD (floating action bar) and associated action system for the AnnotateImage widget.
// createHud(hudEl, deps) → { update, dismissLayerPopup }
// expandGroupSelection(anns, hitId) — pure helper, exported for use in pointer handlers

import { IMP_COLOR, IMP_COLOR_RGB, SEL_COLOR_RGB, LAYER_HOVER_OPACITY } from './_styles.js';

// Given the full effective annotation list and a hit annotation id, returns all IDs in
// the same group, or [hitId] if the annotation is ungrouped.
export function expandGroupSelection(anns, hitId) {
  const gid = anns.find((a) => a.id === hitId)?.group_id;
  if (!gid) return [hitId];
  return anns.filter((a) => a.group_id === gid).map((a) => a.id);
}

export function createHud(hudEl, {
  addTooltip,
  getState,           // () => { activeTool, currentValue }
  setCurrentValue,    // (v) => void
  effectiveAnnotations,
  applyAnnotationMap,
  commitTextEdit,
  uid,
  emit,
  renderCanvas,
  rebuildSettings,
}) {

  // ── confirm popup ─────────────────────────────────────────────────────────

  let actionPopup = null;

  function _dismissActionPopup() {
    if (actionPopup) { actionPopup.remove(); actionPopup = null; }
    document.removeEventListener("pointerdown", _outsideActionHandler, true);
  }

  function _outsideActionHandler(e) {
    if (actionPopup && !actionPopup.contains(e.target)) _dismissActionPopup();
  }

  // Shows a fixed-position confirm/cancel popup anchored below anchorEl.
  function _showActionPopup(anchorEl, message, confirmLabel, confirmStyle, onConfirm) {
    if (actionPopup) { _dismissActionPopup(); return; }
    const rect = anchorEl.getBoundingClientRect();
    actionPopup = document.createElement("div");
    actionPopup.style.cssText =
      "position:fixed;z-index:99999;background:var(--card);border:1px solid var(--border);" +
      "border-radius:8px;padding:12px 14px;box-shadow:0 4px 16px rgba(0,0,0,0.4);" +
      "display:flex;flex-direction:column;gap:10px;min-width:200px;" +
      `left:${rect.left}px;top:${rect.bottom + 6}px;`;
    actionPopup.innerHTML = `
      <div style="font-size:12px;color:var(--foreground);line-height:1.4;">
        ${message}<br>
        <span style="color:var(--muted-foreground);font-size:11px;">This cannot be undone.</span>
      </div>
      <div style="display:flex;gap:8px;justify-content:flex-end;">
        <button id="_ais-cancel" style="font-size:11px;padding:3px 10px;border-radius:5px;border:1px solid var(--border);background:var(--muted);color:var(--muted-foreground);cursor:pointer;">Cancel</button>
        <button id="_ais-confirm" style="font-size:11px;padding:3px 10px;border-radius:5px;border:none;${confirmStyle};cursor:pointer;">${confirmLabel}</button>
      </div>`;
    document.body.appendChild(actionPopup);
    actionPopup.querySelector("#_ais-cancel").addEventListener("pointerdown", (e) => { e.stopPropagation(); _dismissActionPopup(); });
    actionPopup.querySelector("#_ais-confirm").addEventListener("pointerdown", (e) => { e.stopPropagation(); _dismissActionPopup(); onConfirm(); });
    setTimeout(() => document.addEventListener("pointerdown", _outsideActionHandler, true), 0);
  }

  // ── action executors ──────────────────────────────────────────────────────

  function _executeDeleteSelected() {
    commitTextEdit();
    const { currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    if (!selIds.length) return;
    const importedIds = (currentValue.imported_annotations || []).map((a) => a.id);
    const newOverrides = { ...(currentValue.overrides || {}) };
    for (const id of selIds) {
      if (importedIds.includes(id)) newOverrides[id] = { ...(newOverrides[id] || {}), deleted: true };
    }
    const newAnnotations = (currentValue.annotations || []).filter((a) => !selIds.includes(a.id));
    setCurrentValue({ ...currentValue, annotations: newAnnotations, overrides: newOverrides, selected_ids: [] });
    emit(); rebuildSettings(); renderCanvas();
  }

  function _executeDeleteAll() {
    commitTextEdit();
    const { currentValue } = getState();
    const importedIds = (currentValue.imported_annotations || []).map((a) => a.id);
    const newOverrides = { ...(currentValue.overrides || {}) };
    for (const id of importedIds) newOverrides[id] = { ...(newOverrides[id] || {}), deleted: true };
    setCurrentValue({ ...currentValue, annotations: [], overrides: newOverrides, selected_ids: [] });
    emit(); rebuildSettings(); renderCanvas();
  }

  function _executeResetSelected() {
    const { currentValue } = getState();
    const selId = (currentValue.selected_ids || [])[0];
    if (!selId) return;
    const newOverrides = { ...(currentValue.overrides || {}) };
    delete newOverrides[selId];
    setCurrentValue({ ...currentValue, overrides: newOverrides });
    emit(); rebuildSettings(); renderCanvas();
  }

  function _executeResetAll() {
    const { currentValue } = getState();
    setCurrentValue({ ...currentValue, overrides: {} });
    emit(); rebuildSettings(); renderCanvas();
  }

  // ── action eligibility checks ─────────────────────────────────────────────

  function _canDeleteSelected() { return (getState().currentValue.selected_ids || []).length > 0; }

  // True when a single imported annotation is selected AND has local overrides to discard.
  function _canResetSelected() {
    const { currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    const overrides = currentValue.overrides || {};
    const importedIds = new Set((currentValue.imported_annotations || []).map((a) => a.id));
    return selIds.length === 1 && importedIds.has(selIds[0]) &&
      overrides[selIds[0]] && Object.keys(overrides[selIds[0]]).length > 0;
  }

  function _canResetAll() { return Object.keys(getState().currentValue.overrides || {}).length > 0; }

  // ── group / ungroup ───────────────────────────────────────────────────────

  // Returns the shared group_id when ALL selected annotations are in the same group; else null.
  function _selectionGroupId() {
    const { currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    if (!selIds.length) return null;
    const anns = effectiveAnnotations();
    const gid = anns.find((a) => a.id === selIds[0])?.group_id;
    if (!gid) return null;
    return selIds.every((id) => anns.find((a) => a.id === id)?.group_id === gid) ? gid : null;
  }

  function _canGroup() {
    const selIds = getState().currentValue.selected_ids || [];
    return selIds.length >= 2 && !_selectionGroupId();
  }

  // True when any selected annotation belongs to a group (so ungrouping makes sense).
  function _canUngroup() {
    const selIds = getState().currentValue.selected_ids || [];
    return effectiveAnnotations().some((a) => selIds.includes(a.id) && a.group_id);
  }

  function _executeGroup() {
    const { currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    if (selIds.length < 2) return;
    const gid = uid("grp");
    const { annotations, overrides } = applyAnnotationMap(selIds, (a) => ({ ...a, group_id: gid }));
    setCurrentValue({ ...currentValue, annotations, overrides });
    emit(); update(); renderCanvas();
  }

  function _executeUngroup() {
    const { currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    const { annotations, overrides } = applyAnnotationMap(selIds, (a) => {
      const b = { ...a }; delete b.group_id; return b;
    });
    setCurrentValue({ ...currentValue, annotations, overrides });
    emit(); update(); renderCanvas();
  }

  // Action descriptors — single source of truth for icon, label, enabled, run
  const ACTION_DESCS = [
    {
      id: "deleteSelected", label: "Delete selected", color: null,
      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>`,
      isEnabled: _canDeleteSelected,
      trigger: () => _executeDeleteSelected(),
    },
    {
      id: "deleteAll", label: "Delete all annotations", color: null,
      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/><path d="M3 6h18" fill="none"/><path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" fill="none"/></svg>`,
      isEnabled: () => true,
      trigger: (anchor) => _showActionPopup(anchor, "Delete all annotations?", "Delete all",
        "background:var(--destructive);color:#fff", _executeDeleteAll),
    },
    {
      id: "resetSelected", label: "Reset overrides for selected", color: IMP_COLOR,
      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/></svg>`,
      isEnabled: _canResetSelected,
      trigger: () => _executeResetSelected(),
    },
    {
      id: "resetAll", label: "Reset all overrides", color: IMP_COLOR,
      icon: `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 2v6h6"/><path d="M21 12A9 9 0 0 0 6 5.3L3 8"/><path d="M21 22v-6h-6"/><path d="M3 12a9 9 0 0 0 15 6.7l3-2.7"/></svg>`,
      isEnabled: _canResetAll,
      trigger: (anchor) => _showActionPopup(anchor, "Reset all overrides?", "Reset all",
        `background:${IMP_COLOR};color:#fff`, _executeResetAll),
    },
  ];

  // ── layer order ───────────────────────────────────────────────────────────

  // Removes the layer-order popup from the DOM (identified by its fixed id).
  function _dismissLayerPopup() {
    const p = document.getElementById("ais-layer-popup");
    if (p) p.remove();
  }

  // Pure function: returns a new annotation array with selIds repositioned per action.
  // action: "front" | "back" | "forward" | "backward". Only affects local annotations.
  function _reorderAnnotations(anns, selIds, action) {
    const selSet = new Set(selIds);
    if (action === "front") {
      return [...anns.filter((a) => !selSet.has(a.id)), ...anns.filter((a) => selSet.has(a.id))];
    }
    if (action === "back") {
      return [...anns.filter((a) => selSet.has(a.id)), ...anns.filter((a) => !selSet.has(a.id))];
    }
    if (action === "forward") {
      const result = [...anns];
      const idxs = result.map((a, i) => (selSet.has(a.id) ? i : -1)).filter((i) => i >= 0);
      if (!idxs.length) return anns;
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
      if (!idxs.length) return anns;
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

  // Creates a "layer order" button that opens a popup menu (Bring to Front / Forward / Backward / Back).
  function _buildLayerOrderButton(selIds, container, btnClass = "ais-hud-btn") {
    const btn = document.createElement("button");
    btn.className = btnClass;
    addTooltip(btn, "Layer order");
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
      <path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83z"/>
      <path d="M2 12a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 12"/>
      <path d="M2 17a1 1 0 0 0 .58.91l8.6 3.91a2 2 0 0 0 1.65 0l8.58-3.9A1 1 0 0 0 22 17"/>
    </svg>`;
    container.appendChild(btn);

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
        { label: "Bring to Front", action: "front",    icon: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M5 3h14"/><path d="m18 13-6-6-6 6"/><path d="M12 7v14"/></svg>' },
        { label: "Bring Forward",  action: "forward",  icon: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m5 12 7-7 7 7"/><path d="M12 19V5"/></svg>' },
        { label: "Send Backward",  action: "backward", icon: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5v14"/><path d="m19 12-7 7-7-7"/></svg>' },
        { label: "Send to Back",   action: "back",     icon: '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 17V3"/><path d="m6 11 6 6 6-6"/><path d="M19 21H5"/></svg>' },
      ];
      for (const { label, action, icon } of ACTIONS) {
        const item = document.createElement("div");
        item.style.cssText = "padding:6px 14px;cursor:pointer;color:var(--foreground,#eee);white-space:nowrap;display:flex;align-items:center;gap:8px;";
        item.innerHTML = `<span style="flex-shrink:0;display:flex;align-items:center;">${icon}</span><span>${label}</span>`;
        item.addEventListener("pointerover",  () => { item.style.background = `rgba(${SEL_COLOR_RGB},${LAYER_HOVER_OPACITY})`; });
        item.addEventListener("pointerout",   () => { item.style.background = ""; });
        item.addEventListener("pointerdown",  (ev) => {
          ev.stopPropagation();
          const { currentValue } = getState();
          const newAnns = _reorderAnnotations(currentValue.annotations || [], selIds, action);
          setCurrentValue({ ...currentValue, annotations: newAnns });
          emit(); rebuildSettings(true); renderCanvas();
        });
        popup.appendChild(item);
      }

      document.body.appendChild(popup);
      const bRect = btn.getBoundingClientRect();
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

  // ── HUD rebuild ───────────────────────────────────────────────────────────

  // Rebuilds the floating HUD bar above the canvas. Shown in select mode when something
  // is selected; hidden otherwise. Contains group/ungroup, layer order, delete, and
  // reset-override buttons.
  function update() {
    const { activeTool, currentValue } = getState();
    const selIds = currentValue.selected_ids || [];
    const hasSelection = activeTool === "select" && selIds.length > 0;
    if (!hasSelection) { hudEl.style.display = "none"; return; }

    hudEl.innerHTML = "";

    function _hudBtn(desc, extraClass = "") {
      const btn = document.createElement("button");
      btn.className = "ais-hud-btn" + (extraClass ? " " + extraClass : "");
      btn.innerHTML = desc.icon;
      addTooltip(btn, desc.label);
      btn.addEventListener("pointerdown", (e) => { e.stopPropagation(); e.preventDefault(); desc.trigger(btn); });
      hudEl.appendChild(btn);
    }
    function _hudSep() { const s = document.createElement("div"); s.className = "ais-hud-sep"; hudEl.appendChild(s); }

    // Group / ungroup — contextual to selection
    const _groupIcon   = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7V5c0-1.1.9-2 2-2h2"/><path d="M17 3h2c1.1 0 2 .9 2 2v2"/><path d="M21 17v2c0 1.1-.9 2-2 2h-2"/><path d="M7 21H5c-1.1 0-2-.9-2-2v-2"/><rect width="7" height="5" x="7" y="7" rx="1"/><rect width="7" height="5" x="10" y="12" rx="1"/></svg>';
    const _ungroupIcon = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="8" height="6" x="5" y="4" rx="1"/><rect width="8" height="6" x="11" y="14" rx="1"/></svg>';
    if (_canGroup())   _hudBtn({ label: "Group",   icon: _groupIcon,   trigger: _executeGroup });
    if (_canUngroup()) _hudBtn({ label: "Ungroup", icon: _ungroupIcon, trigger: _executeUngroup });
    if (_canGroup() || _canUngroup()) _hudSep();

    // Layer order — always shown when something is selected
    _buildLayerOrderButton(selIds, hudEl, "ais-hud-btn");
    _hudSep();

    // Delete selected + delete all
    _hudBtn(ACTION_DESCS.find((d) => d.id === "deleteSelected"), "danger");
    _hudBtn(ACTION_DESCS.find((d) => d.id === "deleteAll"), "danger");

    // Reset overrides for selected — only when applicable
    if (_canResetSelected()) {
      _hudSep();
      _hudBtn(ACTION_DESCS.find((d) => d.id === "resetSelected"), "imp");
    }

    // Reset all overrides — only when any overrides exist
    if (_canResetAll()) {
      _hudSep();
      _hudBtn(ACTION_DESCS.find((d) => d.id === "resetAll"), "imp");
    }

    hudEl.style.display = "flex";
  }

  return { update, dismissLayerPopup: _dismissLayerPopup };
}
